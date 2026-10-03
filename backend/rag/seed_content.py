"""Embed `sample_materials/` into the ChromaDB `learning_materials` collection.

Run from the project root:
    python -m backend.rag.seed_content            # add/update
    python -m backend.rag.seed_content --reset    # wipe collection first
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from backend import config
from backend.rag import chroma_store
from backend.schemas import ProgrammeState

MATERIALS_DIR = Path(__file__).resolve().parent / "sample_materials"

# Chunking: one chunk per file keeps the seed set inspectable. Bump this to split
# long documents (see rag_fundamentals.md chunking section for the trade-offs).
MAX_CHARS = 4000


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta: dict[str, str] = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, parts[2].strip()


def chunk(text: str, size: int = MAX_CHARS) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks, current = [], ""
    for para in re.split(r"\n\s*\n", text):
        if len(current) + len(para) + 2 > size and current:
            chunks.append(current.strip())
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


def main(reset: bool = False) -> int:
    if config.missing_api_key():
        print("ERROR: AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT missing — set them in .env", file=sys.stderr)
        return 1

    files = sorted(MATERIALS_DIR.glob("*.md"))
    if not files:
        print(f"No .md files found in {MATERIALS_DIR}", file=sys.stderr)
        return 1

    if reset:
        chroma_store.reset_collection()
        print("Collection reset.")

    # Imported here so a missing SDK doesn't break `--help`.
    from backend.agents.base_agent import embed

    scratch = ProgrammeState(run_id="seed", manager_request="seed", status="running")

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []

    for path in files:
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        pieces = chunk(f"# {meta.get('title', path.stem)}\n\n{body}")
        for i, piece in enumerate(pieces):
            ids.append(f"{path.stem}::{i}")
            docs.append(piece)
            metas.append(
                {
                    "source": meta.get("title", path.stem),
                    "file": path.name,
                    "topic": meta.get("topic", "general"),
                    "format": meta.get("format", "document"),
                    "level": meta.get("level", "beginner"),
                    "duration_hours": meta.get("duration_hours", ""),
                    "chunk": i,
                }
            )

    print(f"Embedding {len(docs)} chunk(s) from {len(files)} file(s) with {config.EMBED_MODEL}...")
    vectors = embed(scratch, docs, agent="seed")
    chroma_store.upsert(ids, docs, vectors, metas)

    print(f"Done. Collection '{config.COLLECTION_NAME}' now holds {chroma_store.count()} chunk(s).")
    print(f"Persisted at: {config.CHROMA_PERSIST_DIR}")
    for entry in scratch.cost_log:
        print(f"  cost: {entry['total_tokens']} tokens ~ ${entry['cost_usd']:.6f}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the learning-materials vector store.")
    parser.add_argument("--reset", action="store_true", help="delete the collection first")
    args = parser.parse_args()
    raise SystemExit(main(reset=args.reset))
