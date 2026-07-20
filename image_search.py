#!/usr/bin/env python3
r"""
Local image similarity search with OpenCLIP and ChromaDB.

Index product images:
    python image_search.py index --images "D:\product_images"

Search with a query image:
    python image_search.py search --image "D:\query.jpg" --top-k 5

Rebuild the database:
    python image_search.py index --images "D:\product_images" --reset
"""

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
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

DEFAULT_DB_PATH = "./chroma_image_db"
DEFAULT_COLLECTION = "product_images"
DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
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

        self.model, _, self.preprocess = (
            open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
            )
        )

        self.model = self.model.to(self.device)
        self.model.eval()

    def _autocast(self):
        if self.device.startswith("cuda"):
            return torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            )
        return nullcontext()

    def embed_batch(
        self,
        image_paths: Sequence[Path],
    ) -> tuple[list[Path], list[list[float]]]:
        valid_paths: list[Path] = []
        tensors: list[torch.Tensor] = []

        for image_path in image_paths:
            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    tensors.append(self.preprocess(image))
                    valid_paths.append(image_path)
            except (
                OSError,
                ValueError,
                UnidentifiedImageError,
            ) as exc:
                print(
                    f"Skipping unreadable image: {image_path} ({exc})",
                    file=sys.stderr,
                )

        if not tensors:
            return [], []

        batch = torch.stack(tensors).to(self.device)

        with torch.inference_mode(), self._autocast():
            vectors = self.model.encode_image(batch)
            vectors = F.normalize(
                vectors.float(),
                p=2,
                dim=-1,
            )

        return valid_paths, vectors.cpu().numpy().tolist()

    def embed_one(self, image_path: Path) -> list[float]:
        valid_paths, embeddings = self.embed_batch([image_path])

        if not valid_paths:
            raise RuntimeError(
                f"Could not create embedding for {image_path}"
            )

        return embeddings[0]


def discover_images(directory: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    )


def batches(
    items: Sequence[Path],
    batch_size: int,
) -> Iterable[Sequence[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def make_id(relative_path: str) -> str:
    return hashlib.sha256(
        relative_path.encode("utf-8")
    ).hexdigest()


def create_client(db_path: Path):
    db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_path))


def create_collection(
    client,
    collection_name: str,
    model_name: str,
    pretrained: str,
    reset: bool,
):
    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"Deleted collection: {collection_name}")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
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

    validate_collection(
        collection,
        model_name,
        pretrained,
    )
    return collection


def load_collection(
    client,
    collection_name: str,
    model_name: str,
    pretrained: str,
):
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=None,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Collection '{collection_name}' does not exist. "
            "Run the index command first."
        ) from exc

    validate_collection(
        collection,
        model_name,
        pretrained,
    )
    return collection


def validate_collection(
    collection,
    model_name: str,
    pretrained: str,
) -> None:
    metadata = collection.metadata or {}

    stored_model = metadata.get("openclip_model")
    stored_pretrained = metadata.get("openclip_pretrained")

    if stored_model is None or stored_pretrained is None:
        raise RuntimeError(
            "The collection does not contain model information. "
            "Rebuild it with --reset."
        )

    if stored_model != model_name:
        raise RuntimeError(
            "Model mismatch. "
            f"Database uses '{stored_model}', "
            f"but command uses '{model_name}'."
        )

    if stored_pretrained != pretrained:
        raise RuntimeError(
            "Checkpoint mismatch. "
            f"Database uses '{stored_pretrained}', "
            f"but command uses '{pretrained}'."
        )


def index_command(args: argparse.Namespace) -> None:
    image_root = Path(args.images).expanduser().resolve()
    db_path = Path(args.db).expanduser().resolve()

    if not image_root.is_dir():
        raise RuntimeError(
            f"Image directory not found: {image_root}"
        )

    image_paths = discover_images(image_root)

    if not image_paths:
        raise RuntimeError(
            f"No supported images found in {image_root}"
        )

    print(f"Images found: {len(image_paths)}")
    print(f"Database: {db_path}")

    embedder = OpenCLIPEmbedder(
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
    )

    client = create_client(db_path)
    collection = create_collection(
        client=client,
        collection_name=args.collection,
        model_name=args.model,
        pretrained=args.pretrained,
        reset=args.reset,
    )

    indexed = 0
    skipped = 0

    total_batches = (
        len(image_paths) + args.batch_size - 1
    ) // args.batch_size

    for image_batch in tqdm(
        batches(image_paths, args.batch_size),
        total=total_batches,
        desc="Indexing",
        unit="batch",
    ):
        valid_paths, embeddings = embedder.embed_batch(
            image_batch
        )

        skipped += len(image_batch) - len(valid_paths)

        if not valid_paths:
            continue

        ids: list[str] = []
        metadatas: list[dict] = []

        for path in valid_paths:
            relative_path = path.relative_to(
                image_root
            ).as_posix()

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

        indexed += len(valid_paths)

    print()
    print(f"Indexed/upserted: {indexed}")
    print(f"Skipped: {skipped}")
    print(f"Total records: {collection.count()}")
    print(f"Saved database: {db_path}")


def search_command(args: argparse.Namespace) -> None:
    query_path = Path(args.image).expanduser().resolve()
    db_path = Path(args.db).expanduser().resolve()

    if not query_path.is_file():
        raise RuntimeError(
            f"Query image not found: {query_path}"
        )

    client = create_client(db_path)
    collection = load_collection(
        client=client,
        collection_name=args.collection,
        model_name=args.model,
        pretrained=args.pretrained,
    )

    count = collection.count()

    if count == 0:
        raise RuntimeError(
            "The collection is empty. Index images first."
        )

    embedder = OpenCLIPEmbedder(
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
    )

    query_embedding = embedder.embed_one(query_path)

    response = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(args.top_k, count),
        include=[
            "metadatas",
            "distances",
        ],
    )

    ids = response["ids"][0]
    metadatas = response["metadatas"][0]
    distances = response["distances"][0]

    results: list[dict] = []

    for rank, (record_id, metadata, distance) in enumerate(
        zip(ids, metadatas, distances),
        start=1,
    ):
        metadata = metadata or {}
        cosine_distance = float(distance)

        cosine_similarity = 1.0 - cosine_distance
        cosine_similarity = max(
            -1.0,
            min(1.0, cosine_similarity),
        )

        results.append(
            {
                "rank": rank,
                "id": record_id,
                "path": metadata.get("path", ""),
                "relative_path": metadata.get(
                    "relative_path",
                    "",
                ),
                "filename": metadata.get("filename", ""),
                "category": metadata.get("category", ""),
                "cosine_similarity": cosine_similarity,
                "similarity_percent": (
                    cosine_similarity * 100.0
                ),
                "cosine_distance": cosine_distance,
            }
        )

    if args.json:
        print(
            json.dumps(
                {
                    "query_image": str(query_path),
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print()
    print(f"Query: {query_path}")
    print()

    for result in results:
        print(
            f"{result['rank']}. "
            f"Similarity: "
            f"{result['cosine_similarity']:.6f} "
            f"({result['similarity_percent']:.2f}%)"
        )
        print(f"   Image: {result['path']}")
        print(f"   Category: {result['category']}")
        print(
            f"   Cosine distance: "
            f"{result['cosine_distance']:.6f}"
        )
        print()

    best = results[0]

    print("CLOSEST MATCH")
    print(f"Image: {best['path']}")
    print(
        f"Similarity: "
        f"{best['cosine_similarity']:.6f} "
        f"({best['similarity_percent']:.2f}%)"
    )


def add_common_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=(
            "Persistent Chroma database directory. "
            f"Default: {DEFAULT_DB_PATH}"
        ),
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=(
            "Chroma collection name. "
            f"Default: {DEFAULT_COLLECTION}"
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenCLIP model. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--pretrained",
        default=DEFAULT_PRETRAINED,
        help=(
            "OpenCLIP pretrained checkpoint. "
            f"Default: {DEFAULT_PRETRAINED}"
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, or mps",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Store OpenCLIP image embeddings in ChromaDB "
            "and search using cosine similarity."
        )
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    index_parser = commands.add_parser(
        "index",
        help="Index all images in a directory.",
    )
    add_common_arguments(index_parser)
    index_parser.add_argument(
        "--images",
        required=True,
        help="Folder containing product images.",
    )
    index_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size. Default: 32",
    )
    index_parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and rebuild the collection.",
    )
    index_parser.set_defaults(function=index_command)

    search_parser = commands.add_parser(
        "search",
        help="Search using one input image.",
    )
    add_common_arguments(search_parser)
    search_parser.add_argument(
        "--image",
        required=True,
        help="Input/query image path.",
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of closest matches. Default: 5",
    )
    search_parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output.",
    )
    search_parser.set_defaults(function=search_command)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if getattr(args, "batch_size", 1) < 1:
        parser.error("--batch-size must be at least 1")

    if getattr(args, "top_k", 1) < 1:
        parser.error("--top-k must be at least 1")

    try:
        args.function(args)
        return 0
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
