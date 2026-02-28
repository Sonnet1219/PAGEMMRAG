#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Iterator

import pypdfium2 as pdfium
import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from transformers.utils.import_utils import is_flash_attn_2_available
from transformers.utils.logging import disable_progress_bar

from colpali_engine.models import ColQwen2, ColQwen2Processor

# Keep CLI output readable. Set HF_HUB_DISABLE_PROGRESS_BARS=0 to re-enable.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
disable_progress_bar()

VECTOR_DIM = 128
DEFAULT_COLLECTION = "colqwen2_pages"


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if device.startswith("cpu"):
        return torch.float32
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def resolve_attn(attn: str) -> str | None:
    if attn == "auto":
        return "flash_attention_2" if is_flash_attn_2_available() else None
    if attn == "none":
        return None
    return attn


def batched(items: list[Any], batch_size: int) -> Iterator[list[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def slugify(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return value or "pdf"


def make_doc_id(pdf_path: Path, user_doc_id: str) -> str:
    if user_doc_id:
        return user_doc_id
    digest = hashlib.sha1(str(pdf_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{slugify(pdf_path.stem)}-{digest}"


def make_point_id(doc_id: str, page_number: int) -> int:
    digest = hashlib.sha1(f"{doc_id}:{page_number}".encode("utf-8")).hexdigest()[:16]
    return int(digest, 16)


def resolve_qdrant_mode(
    qdrant_url: str,
    qdrant_local_path: str,
    fallback_local_path: Path,
) -> tuple[str, str]:
    if qdrant_url and qdrant_local_path:
        raise ValueError("Use only one of --qdrant-url or --qdrant-local-path.")

    if qdrant_url:
        return ("remote", qdrant_url)

    local_path = Path(qdrant_local_path).expanduser().resolve() if qdrant_local_path else fallback_local_path.resolve()
    local_path.mkdir(parents=True, exist_ok=True)
    return ("local", str(local_path))


def make_qdrant_client(mode: str, value: str, api_key: str) -> QdrantClient:
    if mode == "remote":
        return QdrantClient(url=value, api_key=(api_key or None))
    return QdrantClient(path=value)


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    recreate: bool,
) -> None:
    if client.collection_exists(collection_name) and recreate:
        client.delete_collection(collection_name)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=VECTOR_DIM,
                distance=qmodels.Distance.COSINE,
                multivector_config=qmodels.MultiVectorConfig(
                    comparator=qmodels.MultiVectorComparator.MAX_SIM
                ),
            ),
        )
        return

    info = client.get_collection(collection_name)
    vectors_cfg = info.config.params.vectors
    if not isinstance(vectors_cfg, qmodels.VectorParams):
        raise ValueError(
            f"Collection `{collection_name}` uses named vectors, expected single vector config."
        )
    if vectors_cfg.size != VECTOR_DIM:
        raise ValueError(
            f"Collection `{collection_name}` vector size is {vectors_cfg.size}, expected {VECTOR_DIM}."
        )
    if vectors_cfg.multivector_config is None:
        raise ValueError(
            f"Collection `{collection_name}` has no multivector_config. It must use MAX_SIM."
        )
    if vectors_cfg.multivector_config.comparator != qmodels.MultiVectorComparator.MAX_SIM:
        raise ValueError(
            f"Collection `{collection_name}` comparator is {vectors_cfg.multivector_config.comparator}, expected MAX_SIM."
        )


def delete_doc_points(client: QdrantClient, collection_name: str, doc_id: str) -> None:
    doc_filter = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="doc_id",
                match=qmodels.MatchValue(value=doc_id),
            )
        ]
    )
    client.delete(collection_name=collection_name, points_selector=doc_filter, wait=True)


def load_model_and_processor(args: argparse.Namespace, model_name: str) -> tuple[ColQwen2, ColQwen2Processor]:
    dtype = resolve_dtype(args.dtype, args.device)
    attn_impl = resolve_attn(args.attn)
    print(f"Loading model: {model_name}")
    print(f"Device: {args.device}, dtype: {dtype}, attn: {attn_impl}")
    model = ColQwen2.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=args.device,
        attn_implementation=attn_impl,
    ).eval()
    processor = ColQwen2Processor.from_pretrained(model_name)
    return model, processor


def trim_embeddings(embeddings: torch.Tensor, attention_mask: torch.Tensor) -> list[torch.Tensor]:
    trimmed: list[torch.Tensor] = []
    embeds_cpu = embeddings.detach().float().cpu()
    masks_cpu = attention_mask.detach().bool().cpu()
    for i in range(embeds_cpu.shape[0]):
        token_vectors = embeds_cpu[i][masks_cpu[i]].contiguous()
        if token_vectors.shape[0] == 0:
            raise RuntimeError("Found empty embedding after masking. Check processor outputs.")
        trimmed.append(token_vectors)
    return trimmed


def encode_images(
    model: ColQwen2,
    processor: ColQwen2Processor,
    image_paths: list[Path],
    batch_size: int,
) -> list[torch.Tensor]:
    all_embeddings: list[torch.Tensor] = []
    total = len(image_paths)
    for batch_idx, path_batch in enumerate(batched(image_paths, batch_size), start=1):
        images = [Image.open(path).convert("RGB") for path in path_batch]
        batch_inputs = processor.process_images(images)
        attention_mask = batch_inputs["attention_mask"]
        batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}

        with torch.no_grad():
            embeds = model(**batch_inputs)
        all_embeddings.extend(trim_embeddings(embeds, attention_mask))
        print(f"Encoded image batch {batch_idx}: {len(all_embeddings)}/{total}")

    return all_embeddings


def encode_queries(
    model: ColQwen2,
    processor: ColQwen2Processor,
    queries: list[str],
) -> list[torch.Tensor]:
    batch_inputs = processor.process_queries(queries)
    attention_mask = batch_inputs["attention_mask"]
    batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}
    with torch.no_grad():
        embeds = model(**batch_inputs)
    return trim_embeddings(embeds, attention_mask)


def render_pdf_to_images(pdf_path: Path, pages_dir: Path, dpi: int, max_pages: int) -> list[Path]:
    scale = dpi / 72.0
    doc = pdfium.PdfDocument(str(pdf_path))
    page_count = len(doc)
    limit = page_count if max_pages <= 0 else min(page_count, max_pages)
    image_paths: list[Path] = []

    print(f"Rendering {limit} page(s) from {pdf_path} at {dpi} DPI...")
    for i in range(limit):
        page = doc[i]
        pil_image = page.render(scale=scale).to_pil()
        page.close()

        page_path = pages_dir / f"page_{i + 1:04d}.png"
        pil_image.save(page_path, format="PNG")
        image_paths.append(page_path)

        if (i + 1) % 10 == 0 or i + 1 == limit:
            print(f"Rendered pages: {i + 1}/{limit}")

    doc.close()
    return image_paths


def save_index_meta(
    meta_path: Path,
    pdf_path: Path,
    doc_id: str,
    model_name: str,
    collection_name: str,
    qdrant_mode: str,
    qdrant_value: str,
    page_image_paths: list[Path],
    page_embeddings: list[torch.Tensor],
) -> None:
    token_lengths = [int(t.shape[0]) for t in page_embeddings]
    created_at_utc = utc_now()
    meta = {
        "format_version": "2.0-qdrant",
        "created_at_utc": created_at_utc,
        "pdf_path": str(pdf_path.resolve()),
        "doc_id": doc_id,
        "model_name": model_name,
        "collection_name": collection_name,
        "qdrant_mode": qdrant_mode,
        "qdrant_value": qdrant_value,
        "pages_dir": str((meta_path.parent / "pages").resolve()),
        "page_count": len(page_embeddings),
        "embedding_dim": VECTOR_DIM,
        "token_count_min": min(token_lengths) if token_lengths else 0,
        "token_count_max": max(token_lengths) if token_lengths else 0,
        "token_count_avg": (sum(token_lengths) / len(token_lengths)) if token_lengths else 0.0,
        "page_image_paths": [str(p.resolve()) for p in page_image_paths],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2))


def upsert_page_embeddings(
    client: QdrantClient,
    collection_name: str,
    doc_id: str,
    pdf_path: Path,
    model_name: str,
    page_image_paths: list[Path],
    page_embeddings: list[torch.Tensor],
    batch_size: int,
) -> None:
    indexed_at = utc_now()
    total = len(page_embeddings)
    for batch_idx, start in enumerate(range(0, total, batch_size), start=1):
        end = min(start + batch_size, total)
        points: list[qmodels.PointStruct] = []
        for i in range(start, end):
            page_number = i + 1
            point_id = make_point_id(doc_id=doc_id, page_number=page_number)
            vector = page_embeddings[i].tolist()
            payload = {
                "doc_id": doc_id,
                "pdf_path": str(pdf_path.resolve()),
                "page_number": page_number,
                "page_image_path": str(page_image_paths[i].resolve()),
                "model_name": model_name,
                "token_count": int(page_embeddings[i].shape[0]),
                "indexed_at_utc": indexed_at,
            }
            points.append(qmodels.PointStruct(id=point_id, vector=vector, payload=payload))

        client.upsert(collection_name=collection_name, points=points, wait=True)
        print(f"Uploaded batch {batch_idx}: {end}/{total} page embeddings")


def run_index(args: argparse.Namespace) -> None:
    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    index_dir = Path(args.index_dir).expanduser().resolve()
    pages_dir = index_dir / "pages"
    meta_path = index_dir / "index_meta.json"
    doc_id = make_doc_id(pdf_path, args.doc_id)

    if args.qdrant_url and args.qdrant_local_path:
        raise ValueError("Use only one of --qdrant-url or --qdrant-local-path.")

    if index_dir.exists() and any(index_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Index directory is not empty: {index_dir}. Use --overwrite to rebuild."
        )
    if args.overwrite and index_dir.exists():
        shutil.rmtree(index_dir)

    pages_dir.mkdir(parents=True, exist_ok=True)

    page_image_paths = render_pdf_to_images(
        pdf_path=pdf_path,
        pages_dir=pages_dir,
        dpi=args.dpi,
        max_pages=args.max_pages,
    )
    if not page_image_paths:
        raise RuntimeError("No pages rendered from PDF.")

    model, processor = load_model_and_processor(args, args.model_name)
    page_embeddings = encode_images(
        model=model,
        processor=processor,
        image_paths=page_image_paths,
        batch_size=args.batch_size,
    )

    qdrant_mode, qdrant_value = resolve_qdrant_mode(
        qdrant_url=args.qdrant_url,
        qdrant_local_path=args.qdrant_local_path,
        fallback_local_path=index_dir / "qdrant_storage",
    )
    client = make_qdrant_client(qdrant_mode, qdrant_value, args.qdrant_api_key)

    ensure_collection(
        client=client,
        collection_name=args.collection,
        recreate=args.recreate_collection,
    )

    if not args.keep_existing_doc_points:
        delete_doc_points(client=client, collection_name=args.collection, doc_id=doc_id)

    upsert_page_embeddings(
        client=client,
        collection_name=args.collection,
        doc_id=doc_id,
        pdf_path=pdf_path,
        model_name=args.model_name,
        page_image_paths=page_image_paths,
        page_embeddings=page_embeddings,
        batch_size=args.upload_batch_size,
    )

    save_index_meta(
        meta_path=meta_path,
        pdf_path=pdf_path,
        doc_id=doc_id,
        model_name=args.model_name,
        collection_name=args.collection,
        qdrant_mode=qdrant_mode,
        qdrant_value=qdrant_value,
        page_image_paths=page_image_paths,
        page_embeddings=page_embeddings,
    )

    print(f"Indexed doc_id: {doc_id}")
    print(f"Qdrant collection: {args.collection}")
    print(f"Qdrant mode: {qdrant_mode} ({qdrant_value})")
    print(f"Saved metadata: {meta_path}")
    print(f"Saved page images in: {pages_dir}")


def load_meta(index_dir: Path) -> dict[str, Any]:
    meta_path = index_dir / "index_meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def resolve_search_qdrant_connection(
    args: argparse.Namespace,
    index_dir: Path,
    meta: dict[str, Any],
) -> tuple[str, str]:
    if args.qdrant_url and args.qdrant_local_path:
        raise ValueError("Use only one of --qdrant-url or --qdrant-local-path.")

    if args.qdrant_url:
        return ("remote", args.qdrant_url)
    if args.qdrant_local_path:
        return ("local", str(Path(args.qdrant_local_path).expanduser().resolve()))
    if meta.get("qdrant_mode") == "remote":
        return ("remote", str(meta.get("qdrant_value", "")))
    if meta.get("qdrant_mode") == "local" and meta.get("qdrant_value"):
        return ("local", str(meta["qdrant_value"]))
    return ("local", str((index_dir / "qdrant_storage").resolve()))


def doc_id_filter(doc_id: str) -> qmodels.Filter:
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="doc_id",
                match=qmodels.MatchValue(value=doc_id),
            )
        ]
    )


def run_search(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir).expanduser().resolve()
    meta = load_meta(index_dir)
    if not meta:
        raise FileNotFoundError(
            f"Metadata file not found in {index_dir}. Run `index` first."
        )

    collection_name = args.collection or str(meta.get("collection_name", DEFAULT_COLLECTION))
    doc_id = args.doc_id or str(meta.get("doc_id", ""))
    if not doc_id:
        raise ValueError(
            "doc_id is empty. Pass --doc-id or ensure index_meta.json contains it."
        )

    model_name = args.model_name or str(meta.get("model_name", "vidore/colqwen2-v1.0"))

    qdrant_mode, qdrant_value = resolve_search_qdrant_connection(args, index_dir, meta)
    client = make_qdrant_client(qdrant_mode, qdrant_value, args.qdrant_api_key)

    model, processor = load_model_and_processor(args, model_name)
    query_embeddings = encode_queries(model=model, processor=processor, queries=args.query)

    all_results: list[dict[str, Any]] = []
    query_filter = doc_id_filter(doc_id=doc_id)

    for query_idx, (query_text, query_embed) in enumerate(zip(args.query, query_embeddings)):
        response = client.query_points(
            collection_name=collection_name,
            query=query_embed.tolist(),
            query_filter=query_filter,
            limit=args.top_k,
            with_payload=True,
            with_vectors=False,
            score_threshold=(args.score_threshold if args.score_threshold is not None else None),
        )

        hits: list[dict[str, Any]] = []
        print(f"\nQuery[{query_idx}]: {query_text}")
        for rank, point in enumerate(response.points, start=1):
            payload = point.payload or {}
            page_number = int(payload.get("page_number", -1))
            image_path = str(payload.get("page_image_path", ""))
            hit = {
                "rank": rank,
                "score": float(point.score),
                "point_id": str(point.id),
                "page_number": page_number,
                "image_path": image_path,
            }
            hits.append(hit)
            print(f"  Top{rank}: page {page_number}, score={point.score:.4f}, image={image_path}")

        all_results.append({"query": query_text, "hits": hits})

    if args.copy_best and all_results and all_results[0]["hits"]:
        best_image = all_results[0]["hits"][0].get("image_path", "")
        if not best_image:
            raise RuntimeError("Top-1 hit has no image_path in payload.")
        target = Path(args.copy_best).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_image, target)
        print(f"\nCopied best page image to: {target}")

    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "index_dir": str(index_dir),
            "collection_name": collection_name,
            "doc_id": doc_id,
            "qdrant_mode": qdrant_mode,
            "qdrant_value": qdrant_value,
            "model_name": model_name,
            "queries": args.query,
            "results": all_results,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))
        print(f"Saved search results JSON: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end PDF -> ColQwen2 multi-vector index -> Qdrant retrieval pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Build ColQwen2 multi-vector index from a PDF.")
    index_parser.add_argument("--pdf", required=True, help="Input PDF path.")
    index_parser.add_argument("--index-dir", required=True, help="Output index directory.")
    index_parser.add_argument("--model-name", default="vidore/colqwen2-v1.0")
    index_parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    index_parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    index_parser.add_argument(
        "--attn",
        choices=["auto", "none", "sdpa", "eager", "flash_attention_2"],
        default="auto",
    )
    index_parser.add_argument("--batch-size", type=int, default=4, help="Image embedding batch size.")
    index_parser.add_argument("--dpi", type=int, default=144, help="PDF rendering DPI.")
    index_parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Max pages to index. 0 means all pages.",
    )
    index_parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Qdrant collection name.",
    )
    index_parser.add_argument(
        "--doc-id",
        default="",
        help="Logical document id inside Qdrant. Default is derived from PDF path.",
    )
    index_parser.add_argument(
        "--qdrant-url",
        default="",
        help="Remote Qdrant URL, e.g. http://localhost:6333.",
    )
    index_parser.add_argument(
        "--qdrant-api-key",
        default="",
        help="Qdrant API key for remote mode.",
    )
    index_parser.add_argument(
        "--qdrant-local-path",
        default="",
        help="Local Qdrant storage path. If empty, uses <index-dir>/qdrant_storage.",
    )
    index_parser.add_argument(
        "--upload-batch-size",
        type=int,
        default=8,
        help="Qdrant upsert batch size in pages.",
    )
    index_parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Drop and recreate Qdrant collection before indexing.",
    )
    index_parser.add_argument(
        "--keep-existing-doc-points",
        action="store_true",
        help="Keep existing points for the same doc_id (default overwrites by deleting old points).",
    )
    index_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing index directory.")
    index_parser.set_defaults(func=run_index)

    search_parser = subparsers.add_parser("search", help="Run query against Qdrant indexed PDF pages.")
    search_parser.add_argument("--index-dir", required=True, help="Index directory created by `index`.")
    search_parser.add_argument("--query", nargs="+", required=True, help="One or more query texts.")
    search_parser.add_argument("--top-k", type=int, default=3)
    search_parser.add_argument(
        "--model-name",
        default="",
        help="Optional override model name. Default uses model from index.",
    )
    search_parser.add_argument(
        "--collection",
        default="",
        help="Optional Qdrant collection override. Default reads from index metadata.",
    )
    search_parser.add_argument(
        "--doc-id",
        default="",
        help="Optional document id override. Default reads from index metadata.",
    )
    search_parser.add_argument(
        "--qdrant-url",
        default="",
        help="Remote Qdrant URL override.",
    )
    search_parser.add_argument(
        "--qdrant-api-key",
        default="",
        help="Qdrant API key for remote mode.",
    )
    search_parser.add_argument(
        "--qdrant-local-path",
        default="",
        help="Local Qdrant storage path override.",
    )
    search_parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    search_parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    search_parser.add_argument(
        "--attn",
        choices=["auto", "none", "sdpa", "eager", "flash_attention_2"],
        default="auto",
    )
    search_parser.add_argument("--score-threshold", type=float, default=None)
    search_parser.add_argument("--output-json", default="", help="Optional output JSON for ranked hits.")
    search_parser.add_argument(
        "--copy-best",
        default="",
        help="Optional path to copy top-1 page image of first query.",
    )
    search_parser.set_defaults(func=run_search)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
