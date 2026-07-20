#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Sequence

import chromadb
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp",
    ".bmp", ".tif", ".tiff",
}

DEFAULT_DB = "./chroma_image_db"
DEFAULT_COLLECTION = "product_images"
DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OpenCLIPEmbedder:
    def __init__(
        self,
        model_name: str,
        pretrained: str,
        device: str,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = choose_device(device)

        print(f"Device: {self.device}")
        print(f"Loading OpenCLIP: {model_name} / {pretrained}")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def _autocast(self):
        if self.device.startswith("cuda"):
            return torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            )
        return nullcontext()

    def embed_images(
        self,
        image_paths: Sequence[Path],
    ) -> tuple[list[Path], list[list[float]]]:
        valid_paths: list[Path] = []
        tensors: list[torch.Tensor] = []

        for image_path in image_paths:
            try:
                with Image.open(image_path) as image:
                    tensors.append(
                        self.preprocess(image.convert("RGB"))
                    )
                    valid_paths.append(image_path)
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                print(
                    f"Skipping unreadable image {image_path}: {exc}",
                    file=sys.stderr,
                )

        if not tensors:
            return [], []

        batch = torch.stack(tensors).to(self.device)

        with torch.inference_mode(), self._autocast():
            vectors = self.model.encode_image(batch)
            vectors = F.normalize(vectors.float(), p=2, dim=-1)

        return valid_paths, vectors.cpu().numpy().tolist()

    def embed_image(self, image_path: Path) -> list[float]:
        paths, embeddings = self.embed_images([image_path])
        if not paths:
            raise RuntimeError(f"Could not embed image: {image_path}")
        return embeddings[0]

    def embed_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Text query cannot be empty.")

        tokens = self.tokenizer([text]).to(self.device)

        with torch.inference_mode(), self._autocast():
            vector = self.model.encode_text(tokens)
            vector = F.normalize(vector.float(), p=2, dim=-1)

        return vector[0].cpu().numpy().tolist()


def discover_images(root: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def batches(
    items: Sequence[Path],
    batch_size: int,
) -> Iterable[Sequence[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def make_id(relative_path: str) -> str:
    return hashlib.sha256(
        relative_path.encode("utf-8")
    ).hexdigest()


def create_client(db_path: Path):
    db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_path))


def validate_collection(
    collection,
    model_name: str,
    pretrained: str,
) -> None:
    metadata = collection.metadata or {}

    if metadata.get("openclip_model") != model_name:
        raise RuntimeError(
            "OpenCLIP model mismatch. Use the same model used for indexing."
        )

    if metadata.get("openclip_pretrained") != pretrained:
        raise RuntimeError(
            "OpenCLIP checkpoint mismatch. "
            "Use the same checkpoint used for indexing."
        )


def get_collection(
    client,
    name: str,
    model_name: str,
    pretrained: str,
    create: bool,
    reset: bool = False,
):
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    if create:
        collection = client.get_or_create_collection(
            name=name,
            embedding_function=None,
            metadata={
                "openclip_model": model_name,
                "openclip_pretrained": pretrained,
                "metric": "cosine",
            },
            configuration={
                "hnsw": {
                    "space": "cosine",
                }
            },
        )
    else:
        try:
            collection = client.get_collection(
                name=name,
                embedding_function=None,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Collection '{name}' was not found. "
                "Run the index command first."
            ) from exc

    validate_collection(collection, model_name, pretrained)
    return collection


def build_results(response: dict) -> list[dict]:
    results: list[dict] = []

    for rank, (record_id, metadata, distance) in enumerate(
        zip(
            response["ids"][0],
            response["metadatas"][0],
            response["distances"][0],
        ),
        start=1,
    ):
        metadata = metadata or {}
        cosine_distance = float(distance)
        cosine_similarity = max(
            -1.0,
            min(1.0, 1.0 - cosine_distance),
        )

        results.append(
            {
                "rank": rank,
                "id": record_id,
                "path": metadata.get("path", ""),
                "relative_path": metadata.get("relative_path", ""),
                "filename": metadata.get("filename", ""),
                "category": metadata.get("category", ""),
                "cosine_similarity": cosine_similarity,
                "similarity_percent": cosine_similarity * 100.0,
                "cosine_distance": cosine_distance,
            }
        )

    return results


def print_results(
    query: str,
    results: list[dict],
    json_output: bool,
) -> None:
    if json_output:
        print(
            json.dumps(
                {
                    "query": query,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(f"\nQuery: {query}\n")

    for result in results:
        print(
            f"{result['rank']}. "
            f"Similarity: {result['cosine_similarity']:.6f} "
            f"({result['similarity_percent']:.2f}%)"
        )
        print(f"   Image: {result['path']}")
        print(f"   Category: {result['category']}")
        print()

    if results:
        best = results[0]
        print("CLOSEST MATCH")
        print(f"Image: {best['path']}")
        print(
            f"Similarity: {best['cosine_similarity']:.6f} "
            f"({best['similarity_percent']:.2f}%)"
        )


def index_command(args: argparse.Namespace) -> None:
    image_root = Path(args.images).expanduser().resolve()
    db_path = Path(args.db).expanduser().resolve()

    if not image_root.is_dir():
        raise RuntimeError(f"Image folder not found: {image_root}")

    image_paths = discover_images(image_root)
    if not image_paths:
        raise RuntimeError("No supported images found.")

    embedder = OpenCLIPEmbedder(
        args.model,
        args.pretrained,
        args.device,
    )

    client = create_client(db_path)
    collection = get_collection(
        client,
        args.collection,
        args.model,
        args.pretrained,
        create=True,
        reset=args.reset,
    )

    total_batches = (
        len(image_paths) + args.batch_size - 1
    ) // args.batch_size

    for batch_paths in tqdm(
        batches(image_paths, args.batch_size),
        total=total_batches,
        desc="Indexing",
        unit="batch",
    ):
        valid_paths, embeddings = embedder.embed_images(batch_paths)
        if not valid_paths:
            continue

        ids = []
        metadatas = []

        for path in valid_paths:
            relative_path = path.relative_to(image_root).as_posix()
            ids.append(make_id(relative_path))
            metadatas.append(
                {
                    "path": str(path),
                    "relative_path": relative_path,
                    "filename": path.name,
                    "category": (
                        path.parent.name
                        if path.parent != image_root
                        else ""
                    ),
                }
            )

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print(f"\nCollection records: {collection.count()}")
    print(f"Database saved at: {db_path}")


def search_by_image_command(args: argparse.Namespace) -> None:
    query_path = Path(args.image).expanduser().resolve()
    if not query_path.is_file():
        raise RuntimeError(f"Query image not found: {query_path}")

    client = create_client(Path(args.db).expanduser().resolve())
    collection = get_collection(
        client,
        args.collection,
        args.model,
        args.pretrained,
        create=False,
    )

    embedder = OpenCLIPEmbedder(
        args.model,
        args.pretrained,
        args.device,
    )

    response = collection.query(
        query_embeddings=[embedder.embed_image(query_path)],
        n_results=min(args.top_k, collection.count()),
        include=["metadatas", "distances"],
    )

    print_results(
        str(query_path),
        build_results(response),
        args.json,
    )


def search_by_text_command(args: argparse.Namespace) -> None:
    client = create_client(Path(args.db).expanduser().resolve())
    collection = get_collection(
        client,
        args.collection,
        args.model,
        args.pretrained,
        create=False,
    )

    embedder = OpenCLIPEmbedder(
        args.model,
        args.pretrained,
        args.device,
    )

    response = collection.query(
        query_embeddings=[embedder.embed_text(args.text)],
        n_results=min(args.top_k, collection.count()),
        include=["metadatas", "distances"],
    )

    print_results(
        args.text,
        build_results(response),
        args.json,
    )


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--device", default="auto")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenCLIP image and text search with ChromaDB."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    index_parser = commands.add_parser("index")
    add_common(index_parser)
    index_parser.add_argument("--images", required=True)
    index_parser.add_argument("--batch-size", type=int, default=32)
    index_parser.add_argument("--reset", action="store_true")
    index_parser.set_defaults(function=index_command)

    image_parser = commands.add_parser("image-search")
    add_common(image_parser)
    image_parser.add_argument("--image", required=True)
    image_parser.add_argument("--top-k", type=int, default=5)
    image_parser.add_argument("--json", action="store_true")
    image_parser.set_defaults(function=search_by_image_command)

    text_parser = commands.add_parser("text-search")
    add_common(text_parser)
    text_parser.add_argument("--text", required=True)
    text_parser.add_argument("--top-k", type=int, default=5)
    text_parser.add_argument("--json", action="store_true")
    text_parser.set_defaults(function=search_by_text_command)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.function(args)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
