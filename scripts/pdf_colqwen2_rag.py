#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator

import pypdfium2 as pdfium
import torch
from PIL import Image, ImageDraw, ImageOps
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
DEFAULT_PATCH_COLLECTION = "colqwen2_patches"
DEFAULT_PATCH_TYPES = ["title", "text", "interline_equation", "image", "list", "table"]


def parse_rgb_color(value: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid --pad-color `{value}`. Expected format: R,G,B")
    rgb: list[int] = []
    for part in parts:
        channel = int(part)
        if channel < 0 or channel > 255:
            raise ValueError(f"Invalid color channel `{channel}` in --pad-color `{value}`.")
        rgb.append(channel)
    return (rgb[0], rgb[1], rgb[2])


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


def load_named_meta(index_dir: Path, filename: str) -> dict[str, Any]:
    meta_path = index_dir / filename
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


def patch_query_filter(doc_id: str, patch_types: list[str]) -> qmodels.Filter:
    must: list[qmodels.FieldCondition] = []
    if doc_id:
        must.append(
            qmodels.FieldCondition(
                key="doc_id",
                match=qmodels.MatchValue(value=doc_id),
            )
        )
    if patch_types:
        must.append(
            qmodels.FieldCondition(
                key="patch_type",
                match=qmodels.MatchAny(any=patch_types),
            )
        )
    return qmodels.Filter(must=must)


def run_search(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir).expanduser().resolve()
    meta = load_named_meta(index_dir, "index_meta.json")
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


def run_patch_search(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir).expanduser().resolve()
    meta = load_named_meta(index_dir, "patch_index_meta.json")
    if not meta:
        raise FileNotFoundError(
            f"Patch index metadata file not found in {index_dir}. Run `patch-index` first."
        )

    collection_name = args.collection or str(meta.get("collection_name", DEFAULT_PATCH_COLLECTION))
    doc_id = args.doc_id or str(meta.get("doc_id", ""))
    if not doc_id:
        raise ValueError(
            "doc_id is empty. Pass --doc-id or ensure patch_index_meta.json contains it."
        )
    model_name = args.model_name or str(meta.get("model_name", "vidore/colqwen2-v1.0"))
    patch_types = sorted({str(v).strip() for v in args.patch_types if str(v).strip()})

    qdrant_mode, qdrant_value = resolve_search_qdrant_connection(args, index_dir, meta)
    client = make_qdrant_client(qdrant_mode, qdrant_value, args.qdrant_api_key)

    model, processor = load_model_and_processor(args, model_name)
    query_embeddings = encode_queries(model=model, processor=processor, queries=args.query)

    all_results: list[dict[str, Any]] = []
    query_filter = patch_query_filter(doc_id=doc_id, patch_types=patch_types)

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
        print(f"\nPatch Query[{query_idx}]: {query_text}")
        for rank, point in enumerate(response.points, start=1):
            payload = point.payload or {}
            image_path = str(payload.get("patch_image_path", ""))
            hit = {
                "rank": rank,
                "score": float(point.score),
                "point_id": str(point.id),
                "patch_number": int(payload.get("patch_number", -1)),
                "patch_type": str(payload.get("patch_type", "")),
                "page_idx": int(payload.get("page_idx", -1)),
                "block_idx": int(payload.get("block_idx", -1)),
                "bbox_pdf_points": payload.get("bbox_pdf_points", []),
                "image_path": image_path,
            }
            hits.append(hit)
            print(
                "  Top{rank}: type={patch_type}, page_idx={page_idx}, block_idx={block_idx}, "
                "score={score:.4f}, image={image}".format(
                    rank=rank,
                    patch_type=hit["patch_type"] or "unknown",
                    page_idx=hit["page_idx"],
                    block_idx=hit["block_idx"],
                    score=point.score,
                    image=image_path,
                )
            )

        all_results.append({"query": query_text, "hits": hits})

    if args.copy_best and all_results and all_results[0]["hits"]:
        best_image = all_results[0]["hits"][0].get("image_path", "")
        if not best_image:
            raise RuntimeError("Top-1 hit has no patch_image_path in payload.")
        target = Path(args.copy_best).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_image, target)
        print(f"\nCopied best patch image to: {target}")

    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "index_dir": str(index_dir),
            "collection_name": collection_name,
            "doc_id": doc_id,
            "patch_types": patch_types,
            "qdrant_mode": qdrant_mode,
            "qdrant_value": qdrant_value,
            "model_name": model_name,
            "queries": args.query,
            "results": all_results,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))
        print(f"Saved patch search results JSON: {out_path}")


def pad_patch_image(
    image: Image.Image,
    max_aspect_ratio: float,
    min_short_edge: int,
    pad_color: tuple[int, int, int],
) -> Image.Image:
    width, height = image.size
    long_side = max(width, height)
    short_side = min(width, height)
    target_short = max(min_short_edge, int(math.ceil(long_side / max_aspect_ratio)))

    if short_side >= target_short:
        return image

    if width < height:
        total_pad = target_short - width
        left = total_pad // 2
        right = total_pad - left
        return ImageOps.expand(image, border=(left, 0, right, 0), fill=pad_color)

    total_pad = target_short - height
    top = total_pad // 2
    bottom = total_pad - top
    return ImageOps.expand(image, border=(0, top, 0, bottom), fill=pad_color)


def save_preview_grid(
    items: list[dict[str, Any]],
    output_path: Path,
    thumb_width: int,
    thumb_height: int,
    cols: int,
    label_key: str,
) -> None:
    if not items:
        return

    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb_width, rows * thumb_height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    for i, item in enumerate(items):
        source = Path(str(item["file"]))
        image = Image.open(source).convert("RGB")
        image.thumbnail((thumb_width - 10, thumb_height - 28))

        cell_x = (i % cols) * thumb_width
        cell_y = (i // cols) * thumb_height
        x = cell_x + (thumb_width - image.width) // 2
        y = cell_y + 4
        canvas.paste(image, (x, y))

        label = str(item[label_key])
        draw.text((cell_x + 4, cell_y + thumb_height - 18), label, fill=(30, 30, 30))

    canvas.save(output_path, format="JPEG", quality=90)


def run_patch(args: argparse.Namespace) -> None:
    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    middle_json_path = Path(args.middle_json).expanduser().resolve()
    if not middle_json_path.exists():
        raise FileNotFoundError(f"middle.json not found: {middle_json_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = (Path.cwd() / "outputs" / f"mineru_patch_layout_v2_{timestamp}").resolve()

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite to rebuild."
        )
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    raw_dir = output_dir / "para_blocks_raw"
    padded_dir = output_dir / "para_blocks_padded"
    raw_dir.mkdir(parents=True, exist_ok=True)
    padded_dir.mkdir(parents=True, exist_ok=True)

    middle_data = json.loads(middle_json_path.read_text())
    pdf_info = middle_data.get("pdf_info")
    if not isinstance(pdf_info, list) or not pdf_info:
        raise ValueError(f"Invalid middle.json: `{middle_json_path}` has no `pdf_info`.")

    patch_types = set(args.types)
    scale = args.dpi / 72.0
    pad_color = parse_rgb_color(args.pad_color)

    type_counts: dict[str, int] = {}
    size_buckets: dict[str, int] = {
        "tiny_raw<=8": 0,
        "small_raw<=16": 0,
        "normal_raw>16": 0,
    }
    patches: list[dict[str, Any]] = []

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        if len(doc) < len(pdf_info):
            raise RuntimeError(
                f"PDF pages ({len(doc)}) is fewer than middle.json pages ({len(pdf_info)})."
            )

        for page_idx, page_info_item in enumerate(pdf_info):
            blocks = page_info_item.get("para_blocks", [])
            if not blocks:
                continue

            page = doc[page_idx]
            page_image = page.render(scale=scale).to_pil().convert("RGB")
            page.close()
            page_width, page_height = page_image.size

            for block_idx, block in enumerate(blocks):
                block_type = str(block.get("type", "unknown"))
                if block_type not in patch_types:
                    continue

                bbox = block.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue

                x0, y0, x1, y1 = [float(v) for v in bbox]
                x0 -= args.margin_pt
                y0 -= args.margin_pt
                x1 += args.margin_pt
                y1 += args.margin_pt

                px0 = int(math.floor(x0 * scale))
                py0 = int(math.floor(y0 * scale))
                px1 = int(math.ceil(x1 * scale))
                py1 = int(math.ceil(y1 * scale))

                px0 = max(0, min(px0, page_width - 1))
                py0 = max(0, min(py0, page_height - 1))
                px1 = max(px0 + 1, min(px1, page_width))
                py1 = max(py0 + 1, min(py1, page_height))

                raw_patch = page_image.crop((px0, py0, px1, py1))
                raw_width, raw_height = raw_patch.size

                min_raw_side = min(raw_width, raw_height)
                if min_raw_side <= 8:
                    size_buckets["tiny_raw<=8"] += 1
                elif min_raw_side <= 16:
                    size_buckets["small_raw<=16"] += 1
                else:
                    size_buckets["normal_raw>16"] += 1

                filename = f"p{page_idx + 1:03d}_b{block_idx:03d}_{block_type}.png"
                raw_path = raw_dir / filename
                raw_patch.save(raw_path, format="PNG")

                padded_patch = pad_patch_image(
                    image=raw_patch,
                    max_aspect_ratio=args.max_aspect_ratio,
                    min_short_edge=args.min_short_edge,
                    pad_color=pad_color,
                )
                padded_path = padded_dir / filename
                padded_patch.save(padded_path, format="PNG")

                type_counts[block_type] = type_counts.get(block_type, 0) + 1

                patches.append(
                    {
                        "file": str(padded_path.resolve()),
                        "raw_file": str(raw_path.resolve()),
                        "label": filename,
                        "mixed_label": f"p{page_idx + 1}-{block_type}",
                        "page_idx": page_idx,
                        "block_idx": block_idx,
                        "type": block_type,
                        "bbox_pdf_points": [float(v) for v in bbox],
                        "crop_px_raw": [px0, py0, px1, py1],
                        "raw_size": [raw_width, raw_height],
                        "padded_size": [padded_patch.size[0], padded_patch.size[1]],
                    }
                )
    finally:
        doc.close()

    meta = {
        "format_version": "1.0-mineru-layout-patch",
        "created_at_utc": utc_now(),
        "pdf_path": str(pdf_path),
        "middle_json_path": str(middle_json_path),
        "output_dir": str(output_dir),
        "raw_dir": str(raw_dir),
        "padded_dir": str(padded_dir),
        "dpi": args.dpi,
        "scale": scale,
        "margin_pt": args.margin_pt,
        "max_aspect_ratio": args.max_aspect_ratio,
        "min_short_edge": args.min_short_edge,
        "pad_color": list(pad_color),
        "types": sorted(patch_types),
        "count": len(patches),
        "type_counts": type_counts,
        "size_buckets_raw": size_buckets,
        "patches": patches,
    }
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2))

    if args.preview:
        type_order = ["title", "text", "interline_equation", "image", "table", "list"]
        for block_type in type_order:
            samples = [p for p in patches if p["type"] == block_type][: args.preview_per_type]
            if not samples:
                continue
            save_preview_grid(
                items=samples,
                output_path=output_dir / f"preview_{block_type}.jpg",
                thumb_width=args.preview_thumb_width,
                thumb_height=args.preview_thumb_height,
                cols=args.preview_cols,
                label_key="label",
            )

        mixed_samples = patches[: args.preview_mixed]
        if mixed_samples:
            save_preview_grid(
                items=mixed_samples,
                output_path=output_dir / "preview_mixed.jpg",
                thumb_width=args.preview_thumb_width,
                thumb_height=args.preview_thumb_height,
                cols=args.preview_mixed_cols,
                label_key="mixed_label",
            )

    print(f"Generated patch dataset in: {output_dir}")
    print(f"Total patches: {len(patches)}")
    print(f"Type counts: {type_counts}")
    print(f"Saved metadata: {meta_path}")


def resolve_mineru_bin(user_value: str) -> str:
    if user_value:
        return user_value
    venv_mineru_bin = (Path.cwd() / ".venv_mineru" / "bin" / "mineru").resolve()
    if venv_mineru_bin.exists():
        return str(venv_mineru_bin)
    return "mineru"


def locate_mineru_ocr_artifacts(output_dir: Path, pdf_path: Path) -> dict[str, str]:
    expected_ocr_dir = output_dir / pdf_path.stem / "ocr"
    if expected_ocr_dir.exists():
        middle_candidates = sorted(expected_ocr_dir.glob("*_middle.json"))
    else:
        middle_candidates = sorted(output_dir.glob("**/*_middle.json"))

    if not middle_candidates:
        raise FileNotFoundError(
            f"Cannot find `*_middle.json` under MinerU output directory: {output_dir}"
        )

    middle_json = middle_candidates[0].resolve()
    ocr_dir = middle_json.parent.resolve()
    stem = middle_json.name.replace("_middle.json", "")

    artifacts = {
        "ocr_dir": str(ocr_dir),
        "middle_json": str(middle_json),
    }

    maybe_content = ocr_dir / f"{stem}_content_list.json"
    maybe_model = ocr_dir / f"{stem}_model.json"
    maybe_md = ocr_dir / f"{stem}.md"
    if maybe_content.exists():
        artifacts["content_list_json"] = str(maybe_content.resolve())
    if maybe_model.exists():
        artifacts["model_json"] = str(maybe_model.resolve())
    if maybe_md.exists():
        artifacts["markdown"] = str(maybe_md.resolve())

    return artifacts


def execute_mineru_ocr(args: argparse.Namespace) -> dict[str, str]:
    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = (Path.cwd() / "outputs" / f"mineru_ocr_full_{timestamp}").resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite to rebuild."
        )
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mineru_bin = resolve_mineru_bin(args.mineru_bin)
    cmd: list[str] = [
        mineru_bin,
        "-p",
        str(pdf_path),
        "-o",
        str(output_dir),
        "-m",
        args.method,
        "-b",
        args.backend,
        "-d",
        args.mineru_device,
        "-f",
        "true" if args.formula else "false",
        "-t",
        "true" if args.table else "false",
    ]
    if args.lang:
        cmd.extend(["-l", args.lang])
    if args.start_page >= 0:
        cmd.extend(["-s", str(args.start_page)])
    if args.end_page >= 0:
        cmd.extend(["-e", str(args.end_page)])

    print(f"Running MinerU OCR command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    artifacts = locate_mineru_ocr_artifacts(output_dir=output_dir, pdf_path=pdf_path)
    run_meta = {
        "format_version": "1.0-mineru-ocr-run",
        "created_at_utc": utc_now(),
        "pdf_path": str(pdf_path),
        "mineru_bin": mineru_bin,
        "command": cmd,
        "output_dir": str(output_dir),
        "artifacts": artifacts,
    }
    run_meta_path = output_dir / "ocr_run_meta.json"
    run_meta_path.write_text(json.dumps(run_meta, ensure_ascii=True, indent=2))

    result = {
        "pdf_path": str(pdf_path),
        "output_dir": str(output_dir),
        "run_meta_path": str(run_meta_path.resolve()),
        **artifacts,
    }
    return result


def run_ocr(args: argparse.Namespace) -> None:
    result = execute_mineru_ocr(args)
    print(f"MinerU OCR output dir: {result['output_dir']}")
    print(f"middle.json: {result['middle_json']}")
    if "content_list_json" in result:
        print(f"content_list.json: {result['content_list_json']}")
    if "markdown" in result:
        print(f"markdown: {result['markdown']}")
    print(f"Saved OCR run metadata: {result['run_meta_path']}")


def make_patch_point_id(doc_id: str, patch_key: str) -> int:
    digest = hashlib.sha1(f"{doc_id}:{patch_key}".encode("utf-8")).hexdigest()[:16]
    return int(digest, 16)


def load_patch_image_records(patch_dir: Path) -> tuple[list[Path], dict[str, Any], dict[str, dict[str, Any]], str]:
    patch_dir = patch_dir.expanduser().resolve()
    if not patch_dir.exists():
        raise FileNotFoundError(f"Patch directory not found: {patch_dir}")

    image_dir = patch_dir
    meta_path = patch_dir / "meta.json"
    if (patch_dir / "para_blocks_padded").is_dir():
        image_dir = patch_dir / "para_blocks_padded"
    elif patch_dir.name == "para_blocks_padded" and (patch_dir.parent / "meta.json").exists():
        meta_path = patch_dir.parent / "meta.json"

    if not image_dir.is_dir():
        raise NotADirectoryError(f"Patch image directory is not a directory: {image_dir}")

    patch_meta: dict[str, Any] = {}
    patch_record_map: dict[str, dict[str, Any]] = {}
    if meta_path.exists():
        patch_meta = json.loads(meta_path.read_text())
        patches = patch_meta.get("patches", [])
        if isinstance(patches, list):
            for rec in patches:
                if not isinstance(rec, dict):
                    continue
                file_path = rec.get("file", "")
                if not file_path:
                    continue
                patch_record_map[str(Path(file_path).expanduser().resolve())] = rec

    image_paths = sorted([p.resolve() for p in image_dir.glob("*.png")])
    if not image_paths:
        image_paths = sorted([p.resolve() for p in image_dir.glob("*.jpg")])
    if not image_paths:
        image_paths = sorted([p.resolve() for p in image_dir.glob("*.jpeg")])
    if not image_paths:
        raise RuntimeError(f"No patch images found in: {image_dir}")

    return image_paths, patch_meta, patch_record_map, str(meta_path.resolve()) if meta_path.exists() else ""


def make_patch_doc_id(
    patch_dir: Path,
    patch_meta: dict[str, Any],
    user_doc_id: str,
) -> str:
    if user_doc_id:
        return user_doc_id
    pdf_path_raw = patch_meta.get("pdf_path", "")
    if pdf_path_raw:
        pdf_path = Path(str(pdf_path_raw))
        if pdf_path.exists():
            return make_doc_id(pdf_path, "")
    digest = hashlib.sha1(str(patch_dir.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{slugify(patch_dir.stem)}-{digest}"


def save_patch_index_meta(
    meta_path: Path,
    patch_dir: Path,
    patch_meta_path: str,
    doc_id: str,
    model_name: str,
    collection_name: str,
    qdrant_mode: str,
    qdrant_value: str,
    image_paths: list[Path],
    image_embeddings: list[torch.Tensor],
) -> None:
    token_lengths = [int(t.shape[0]) for t in image_embeddings]
    payload = {
        "format_version": "1.0-qdrant-patch-index",
        "created_at_utc": utc_now(),
        "patch_dir": str(patch_dir.resolve()),
        "patch_meta_path": patch_meta_path,
        "doc_id": doc_id,
        "model_name": model_name,
        "collection_name": collection_name,
        "qdrant_mode": qdrant_mode,
        "qdrant_value": qdrant_value,
        "patch_count": len(image_embeddings),
        "embedding_dim": VECTOR_DIM,
        "token_count_min": min(token_lengths) if token_lengths else 0,
        "token_count_max": max(token_lengths) if token_lengths else 0,
        "token_count_avg": (sum(token_lengths) / len(token_lengths)) if token_lengths else 0.0,
        "patch_image_paths": [str(p.resolve()) for p in image_paths],
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))


def upsert_patch_embeddings(
    client: QdrantClient,
    collection_name: str,
    doc_id: str,
    model_name: str,
    image_paths: list[Path],
    image_embeddings: list[torch.Tensor],
    patch_record_map: dict[str, dict[str, Any]],
    batch_size: int,
) -> None:
    indexed_at = utc_now()
    total = len(image_embeddings)
    for batch_idx, start in enumerate(range(0, total, batch_size), start=1):
        end = min(start + batch_size, total)
        points: list[qmodels.PointStruct] = []
        for i in range(start, end):
            patch_path = image_paths[i].resolve()
            patch_key = patch_path.name
            point_id = make_patch_point_id(doc_id=doc_id, patch_key=patch_key)
            vector = image_embeddings[i].tolist()
            patch_record = patch_record_map.get(str(patch_path), {})
            payload = {
                "doc_id": doc_id,
                "patch_number": i + 1,
                "patch_image_path": str(patch_path),
                "patch_file_name": patch_key,
                "patch_type": patch_record.get("type", ""),
                "page_idx": patch_record.get("page_idx", -1),
                "block_idx": patch_record.get("block_idx", -1),
                "bbox_pdf_points": patch_record.get("bbox_pdf_points", []),
                "model_name": model_name,
                "token_count": int(image_embeddings[i].shape[0]),
                "indexed_at_utc": indexed_at,
            }
            points.append(qmodels.PointStruct(id=point_id, vector=vector, payload=payload))

        client.upsert(collection_name=collection_name, points=points, wait=True)
        print(f"Uploaded patch batch {batch_idx}: {end}/{total}")


def run_patch_index(args: argparse.Namespace) -> None:
    patch_dir = Path(args.patch_dir).expanduser().resolve()
    index_dir = Path(args.index_dir).expanduser().resolve()

    if index_dir.exists() and any(index_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Index directory is not empty: {index_dir}. Use --overwrite to rebuild."
        )
    if args.overwrite and index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    image_paths, patch_meta, patch_record_map, patch_meta_path = load_patch_image_records(patch_dir)
    doc_id = make_patch_doc_id(patch_dir=patch_dir, patch_meta=patch_meta, user_doc_id=args.doc_id)

    model, processor = load_model_and_processor(args, args.model_name)
    image_embeddings = encode_images(
        model=model,
        processor=processor,
        image_paths=image_paths,
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

    upsert_patch_embeddings(
        client=client,
        collection_name=args.collection,
        doc_id=doc_id,
        model_name=args.model_name,
        image_paths=image_paths,
        image_embeddings=image_embeddings,
        patch_record_map=patch_record_map,
        batch_size=args.upload_batch_size,
    )

    meta_path = index_dir / "patch_index_meta.json"
    save_patch_index_meta(
        meta_path=meta_path,
        patch_dir=patch_dir,
        patch_meta_path=patch_meta_path,
        doc_id=doc_id,
        model_name=args.model_name,
        collection_name=args.collection,
        qdrant_mode=qdrant_mode,
        qdrant_value=qdrant_value,
        image_paths=image_paths,
        image_embeddings=image_embeddings,
    )

    print(f"Indexed patch doc_id: {doc_id}")
    print(f"Patch count: {len(image_paths)}")
    print(f"Qdrant collection: {args.collection}")
    print(f"Qdrant mode: {qdrant_mode} ({qdrant_value})")
    print(f"Saved patch index metadata: {meta_path}")


def run_pipeline(args: argparse.Namespace) -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_outputs = (Path.cwd() / "outputs").resolve()

    ocr_output_dir = (
        Path(args.ocr_output_dir).expanduser().resolve()
        if args.ocr_output_dir
        else (base_outputs / f"mineru_ocr_full_{timestamp}")
    )
    patch_output_dir = (
        Path(args.patch_output_dir).expanduser().resolve()
        if args.patch_output_dir
        else (base_outputs / f"mineru_patch_layout_v2_{timestamp}")
    )
    index_dir = (
        Path(args.index_dir).expanduser().resolve()
        if args.index_dir
        else (base_outputs / f"mineru_patch_index_{timestamp}")
    )

    ocr_args = argparse.Namespace(
        pdf=args.pdf,
        output_dir=str(ocr_output_dir),
        method=args.method,
        backend=args.backend,
        mineru_device=args.mineru_device,
        lang=args.lang,
        start_page=args.start_page,
        end_page=args.end_page,
        formula=args.formula,
        table=args.table,
        mineru_bin=args.mineru_bin,
        overwrite=args.overwrite,
    )
    ocr_result = execute_mineru_ocr(ocr_args)

    patch_args = argparse.Namespace(
        pdf=args.pdf,
        middle_json=ocr_result["middle_json"],
        output_dir=str(patch_output_dir),
        types=args.types,
        dpi=args.patch_dpi,
        margin_pt=args.margin_pt,
        max_aspect_ratio=args.max_aspect_ratio,
        min_short_edge=args.min_short_edge,
        pad_color=args.pad_color,
        preview=args.preview,
        preview_per_type=args.preview_per_type,
        preview_mixed=args.preview_mixed,
        preview_cols=args.preview_cols,
        preview_mixed_cols=args.preview_mixed_cols,
        preview_thumb_width=args.preview_thumb_width,
        preview_thumb_height=args.preview_thumb_height,
        overwrite=args.overwrite,
    )
    run_patch(patch_args)

    if not args.skip_index:
        patch_index_args = argparse.Namespace(
            patch_dir=str(patch_output_dir),
            index_dir=str(index_dir),
            model_name=args.model_name,
            device=args.device,
            dtype=args.dtype,
            attn=args.attn,
            batch_size=args.batch_size,
            collection=args.collection,
            doc_id=args.doc_id,
            qdrant_url=args.qdrant_url,
            qdrant_api_key=args.qdrant_api_key,
            qdrant_local_path=args.qdrant_local_path,
            upload_batch_size=args.upload_batch_size,
            recreate_collection=args.recreate_collection,
            keep_existing_doc_points=args.keep_existing_doc_points,
            overwrite=args.overwrite,
        )
        run_patch_index(patch_index_args)

    print("\nPipeline summary")
    print(f"- OCR output: {ocr_output_dir}")
    print(f"- Patch output: {patch_output_dir}")
    if args.skip_index:
        print("- Patch index: skipped (--skip-index)")
    else:
        print(f"- Patch index: {index_dir}")


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

    ocr_parser = subparsers.add_parser(
        "ocr",
        help="Run MinerU OCR and generate original OCR artifacts (middle/model/content_list/md).",
    )
    ocr_parser.add_argument("--pdf", required=True, help="Input PDF path.")
    ocr_parser.add_argument(
        "--output-dir",
        default="",
        help="OCR output directory. Default: ./outputs/mineru_ocr_full_<timestamp>",
    )
    ocr_parser.add_argument(
        "--method",
        choices=["auto", "txt", "ocr"],
        default="ocr",
        help="MinerU parse method.",
    )
    ocr_parser.add_argument(
        "--backend",
        choices=["pipeline", "vlm-http-client", "hybrid-http-client", "vlm-auto-engine", "hybrid-auto-engine"],
        default="pipeline",
        help="MinerU backend.",
    )
    ocr_parser.add_argument(
        "--mineru-device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device passed to MinerU (e.g. cuda:0, cpu).",
    )
    ocr_parser.add_argument("--lang", default="", help="Optional OCR language code.")
    ocr_parser.add_argument(
        "--start-page",
        type=int,
        default=0,
        help="Start page (0-based).",
    )
    ocr_parser.add_argument(
        "--end-page",
        type=int,
        default=-1,
        help="End page (0-based). -1 means to the end.",
    )
    ocr_parser.add_argument(
        "--formula",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable formula parsing in MinerU.",
    )
    ocr_parser.add_argument(
        "--table",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable table parsing in MinerU.",
    )
    ocr_parser.add_argument(
        "--mineru-bin",
        default="",
        help="MinerU executable path. Default prefers ./.venv_mineru/bin/mineru if exists.",
    )
    ocr_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory.")
    ocr_parser.set_defaults(func=run_ocr)

    patch_parser = subparsers.add_parser(
        "patch",
        help="Create MinerU-layout-aligned patch images from middle.json.",
    )
    patch_parser.add_argument("--pdf", required=True, help="Input PDF path.")
    patch_parser.add_argument(
        "--middle-json",
        required=True,
        help="MinerU middle.json path (e.g. <doc>/ocr/<name>_middle.json).",
    )
    patch_parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Default: ./outputs/mineru_patch_layout_v2_<timestamp>",
    )
    patch_parser.add_argument(
        "--types",
        nargs="+",
        default=DEFAULT_PATCH_TYPES,
        choices=DEFAULT_PATCH_TYPES,
        help="Patch block types to export.",
    )
    patch_parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Render DPI for patch extraction.",
    )
    patch_parser.add_argument(
        "--margin-pt",
        type=float,
        default=2.0,
        help="Expand bbox margin in PDF point units before cropping.",
    )
    patch_parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=120.0,
        help="Max long/short aspect ratio after padding.",
    )
    patch_parser.add_argument(
        "--min-short-edge",
        type=int,
        default=12,
        help="Minimum short edge after padding.",
    )
    patch_parser.add_argument(
        "--pad-color",
        default="255,255,255",
        help="Padding color in R,G,B format.",
    )
    patch_parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate preview contact sheets.",
    )
    patch_parser.add_argument(
        "--preview-per-type",
        type=int,
        default=20,
        help="Max samples per type in preview_<type>.jpg.",
    )
    patch_parser.add_argument(
        "--preview-mixed",
        type=int,
        default=24,
        help="Max samples in preview_mixed.jpg.",
    )
    patch_parser.add_argument(
        "--preview-cols",
        type=int,
        default=4,
        help="Columns for per-type preview grids.",
    )
    patch_parser.add_argument(
        "--preview-mixed-cols",
        type=int,
        default=6,
        help="Columns for mixed preview grid.",
    )
    patch_parser.add_argument(
        "--preview-thumb-width",
        type=int,
        default=320,
        help="Preview thumbnail cell width.",
    )
    patch_parser.add_argument(
        "--preview-thumb-height",
        type=int,
        default=200,
        help="Preview thumbnail cell height.",
    )
    patch_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory.")
    patch_parser.set_defaults(func=run_patch)

    patch_index_parser = subparsers.add_parser(
        "patch-index",
        help="Embed patch images with ColQwen2 and index them into Qdrant.",
    )
    patch_index_parser.add_argument(
        "--patch-dir",
        required=True,
        help="Patch directory (either patch root with para_blocks_padded/ or an image directory).",
    )
    patch_index_parser.add_argument("--index-dir", required=True, help="Output index directory.")
    patch_index_parser.add_argument("--model-name", default="vidore/colqwen2-v1.0")
    patch_index_parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    patch_index_parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    patch_index_parser.add_argument(
        "--attn",
        choices=["auto", "none", "sdpa", "eager", "flash_attention_2"],
        default="auto",
    )
    patch_index_parser.add_argument("--batch-size", type=int, default=4, help="Patch embedding batch size.")
    patch_index_parser.add_argument(
        "--collection",
        default=DEFAULT_PATCH_COLLECTION,
        help="Qdrant collection name for patches.",
    )
    patch_index_parser.add_argument("--doc-id", default="", help="Logical document id inside Qdrant.")
    patch_index_parser.add_argument("--qdrant-url", default="", help="Remote Qdrant URL override.")
    patch_index_parser.add_argument("--qdrant-api-key", default="", help="Qdrant API key for remote mode.")
    patch_index_parser.add_argument("--qdrant-local-path", default="", help="Local Qdrant storage path override.")
    patch_index_parser.add_argument("--upload-batch-size", type=int, default=16, help="Qdrant upsert batch size.")
    patch_index_parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Drop and recreate Qdrant collection before indexing.",
    )
    patch_index_parser.add_argument(
        "--keep-existing-doc-points",
        action="store_true",
        help="Keep existing points for the same doc_id.",
    )
    patch_index_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing index directory.")
    patch_index_parser.set_defaults(func=run_patch_index)

    patch_search_parser = subparsers.add_parser(
        "patch-search",
        help="Run query against Qdrant indexed patch images.",
    )
    patch_search_parser.add_argument(
        "--index-dir",
        required=True,
        help="Patch index directory created by `patch-index`.",
    )
    patch_search_parser.add_argument("--query", nargs="+", required=True, help="One or more query texts.")
    patch_search_parser.add_argument("--top-k", type=int, default=5)
    patch_search_parser.add_argument(
        "--patch-types",
        nargs="+",
        default=[],
        help="Optional patch_type filters (e.g. text table image).",
    )
    patch_search_parser.add_argument(
        "--model-name",
        default="",
        help="Optional override model name. Default uses model from patch index.",
    )
    patch_search_parser.add_argument(
        "--collection",
        default="",
        help="Optional Qdrant collection override. Default reads from patch index metadata.",
    )
    patch_search_parser.add_argument(
        "--doc-id",
        default="",
        help="Optional document id override. Default reads from patch index metadata.",
    )
    patch_search_parser.add_argument(
        "--qdrant-url",
        default="",
        help="Remote Qdrant URL override.",
    )
    patch_search_parser.add_argument(
        "--qdrant-api-key",
        default="",
        help="Qdrant API key for remote mode.",
    )
    patch_search_parser.add_argument(
        "--qdrant-local-path",
        default="",
        help="Local Qdrant storage path override.",
    )
    patch_search_parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    patch_search_parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    patch_search_parser.add_argument(
        "--attn",
        choices=["auto", "none", "sdpa", "eager", "flash_attention_2"],
        default="auto",
    )
    patch_search_parser.add_argument("--score-threshold", type=float, default=None)
    patch_search_parser.add_argument("--output-json", default="", help="Optional output JSON for ranked hits.")
    patch_search_parser.add_argument(
        "--copy-best",
        default="",
        help="Optional path to copy top-1 patch image of first query.",
    )
    patch_search_parser.set_defaults(func=run_patch_search)

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run full pipeline: MinerU OCR -> layout patch -> ColQwen2 patch index.",
    )
    pipeline_parser.add_argument("--pdf", required=True, help="Input PDF path.")
    pipeline_parser.add_argument("--ocr-output-dir", default="", help="OCR output directory override.")
    pipeline_parser.add_argument("--patch-output-dir", default="", help="Patch output directory override.")
    pipeline_parser.add_argument("--index-dir", default="", help="Patch index directory override.")
    pipeline_parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Run only OCR + patch stages, skip Qdrant patch indexing.",
    )
    pipeline_parser.add_argument(
        "--method",
        choices=["auto", "txt", "ocr"],
        default="ocr",
        help="MinerU parse method.",
    )
    pipeline_parser.add_argument(
        "--backend",
        choices=["pipeline", "vlm-http-client", "hybrid-http-client", "vlm-auto-engine", "hybrid-auto-engine"],
        default="pipeline",
        help="MinerU backend.",
    )
    pipeline_parser.add_argument(
        "--mineru-device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device passed to MinerU.",
    )
    pipeline_parser.add_argument("--lang", default="", help="Optional OCR language code.")
    pipeline_parser.add_argument("--start-page", type=int, default=0, help="Start page (0-based).")
    pipeline_parser.add_argument("--end-page", type=int, default=-1, help="End page (0-based). -1 means all pages.")
    pipeline_parser.add_argument(
        "--formula",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable formula parsing in MinerU.",
    )
    pipeline_parser.add_argument(
        "--table",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable table parsing in MinerU.",
    )
    pipeline_parser.add_argument(
        "--mineru-bin",
        default="",
        help="MinerU executable path. Default prefers ./.venv_mineru/bin/mineru if exists.",
    )
    pipeline_parser.add_argument(
        "--types",
        nargs="+",
        default=DEFAULT_PATCH_TYPES,
        choices=DEFAULT_PATCH_TYPES,
        help="Patch block types to export.",
    )
    pipeline_parser.add_argument("--patch-dpi", type=int, default=300, help="Patch extraction render DPI.")
    pipeline_parser.add_argument("--margin-pt", type=float, default=2.0, help="Patch bbox margin in PDF points.")
    pipeline_parser.add_argument("--max-aspect-ratio", type=float, default=120.0, help="Patch max aspect ratio.")
    pipeline_parser.add_argument("--min-short-edge", type=int, default=12, help="Patch min short edge after padding.")
    pipeline_parser.add_argument("--pad-color", default="255,255,255", help="Patch padding color in R,G,B.")
    pipeline_parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate patch preview contact sheets.",
    )
    pipeline_parser.add_argument("--preview-per-type", type=int, default=20)
    pipeline_parser.add_argument("--preview-mixed", type=int, default=24)
    pipeline_parser.add_argument("--preview-cols", type=int, default=4)
    pipeline_parser.add_argument("--preview-mixed-cols", type=int, default=6)
    pipeline_parser.add_argument("--preview-thumb-width", type=int, default=320)
    pipeline_parser.add_argument("--preview-thumb-height", type=int, default=200)
    pipeline_parser.add_argument("--model-name", default="vidore/colqwen2-v1.0")
    pipeline_parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    pipeline_parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    pipeline_parser.add_argument(
        "--attn",
        choices=["auto", "none", "sdpa", "eager", "flash_attention_2"],
        default="auto",
    )
    pipeline_parser.add_argument("--batch-size", type=int, default=4, help="Patch embedding batch size.")
    pipeline_parser.add_argument(
        "--collection",
        default=DEFAULT_PATCH_COLLECTION,
        help="Qdrant collection name for patch indexing.",
    )
    pipeline_parser.add_argument("--doc-id", default="", help="Logical document id inside Qdrant.")
    pipeline_parser.add_argument("--qdrant-url", default="", help="Remote Qdrant URL override.")
    pipeline_parser.add_argument("--qdrant-api-key", default="", help="Qdrant API key for remote mode.")
    pipeline_parser.add_argument("--qdrant-local-path", default="", help="Local Qdrant storage path override.")
    pipeline_parser.add_argument("--upload-batch-size", type=int, default=16, help="Qdrant upsert batch size.")
    pipeline_parser.add_argument("--recreate-collection", action="store_true")
    pipeline_parser.add_argument("--keep-existing-doc-points", action="store_true")
    pipeline_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    pipeline_parser.set_defaults(func=run_pipeline)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
