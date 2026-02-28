#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from transformers.utils.logging import disable_progress_bar

from colpali_engine.models import ColQwen2, ColQwen2Processor

# Keep logs readable during model loading; set HF_HUB_DISABLE_PROGRESS_BARS=0 to re-enable.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
disable_progress_bar()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local ColQwen2 embeddings.")
    parser.add_argument("--model-name", default="vidore/colqwen2-v1.0")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (e.g. cuda:0, cpu).",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Torch dtype for model weights.",
    )
    parser.add_argument(
        "--attn",
        choices=["auto", "none", "sdpa", "eager", "flash_attention_2"],
        default="auto",
        help="Attention backend. 'auto' uses flash_attention_2 when available.",
    )
    parser.add_argument(
        "--image",
        nargs="*",
        default=[],
        help="One or more image file paths.",
    )
    parser.add_argument(
        "--query",
        nargs="*",
        default=[],
        help="One or more text queries.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output JSON path. Saves pooled embeddings and scores.",
    )
    return parser.parse_args()


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


def load_images(paths: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        images.append(Image.open(path).convert("RGB"))
    return images


def main() -> None:
    args = parse_args()
    if not args.image and not args.query:
        raise ValueError("At least one --image or --query is required.")

    dtype = resolve_dtype(args.dtype, args.device)
    attn_impl = resolve_attn(args.attn)

    print(f"Loading model: {args.model_name}")
    print(f"Device: {args.device}, dtype: {dtype}, attn: {attn_impl}")

    model = ColQwen2.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map=args.device,
        attn_implementation=attn_impl,
    ).eval()
    processor = ColQwen2Processor.from_pretrained(args.model_name)

    result: dict[str, object] = {
        "model_name": args.model_name,
        "device": str(model.device),
        "dtype": str(dtype),
    }

    query_embeddings = None
    image_embeddings = None

    with torch.no_grad():
        if args.image:
            images = load_images(args.image)
            batch_images = processor.process_images(images).to(model.device)
            image_embeddings = model(**batch_images)
            print(f"image_embeddings.shape = {tuple(image_embeddings.shape)}")
            result["image_embedding_shape"] = list(image_embeddings.shape)
            result["image_embeddings_mean"] = (
                image_embeddings.mean(dim=1).float().cpu().tolist()
            )

        if args.query:
            batch_queries = processor.process_queries(args.query).to(model.device)
            query_embeddings = model(**batch_queries)
            print(f"query_embeddings.shape = {tuple(query_embeddings.shape)}")
            result["query_embedding_shape"] = list(query_embeddings.shape)
            result["query_embeddings_mean"] = (
                query_embeddings.mean(dim=1).float().cpu().tolist()
            )

    if query_embeddings is not None and image_embeddings is not None:
        scores = processor.score_multi_vector(query_embeddings, image_embeddings)
        print(f"scores.shape = {tuple(scores.shape)}")
        print(scores)
        result["scores_shape"] = list(scores.shape)
        result["scores"] = scores.float().cpu().tolist()

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2))
        print(f"Saved output JSON: {output_path}")


if __name__ == "__main__":
    main()
