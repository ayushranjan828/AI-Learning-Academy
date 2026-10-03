"""ChromaDB persistent client for the `learning_materials` collection.

Embeddings are always supplied explicitly (Azure OpenAI embedding deployment, or the local
fallback in base_agent), so the collection's own embedding function is never invoked.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

# Must be set before chromadb imports its telemetry module, otherwise chroma 0.5.x
# spams "Failed to send telemetry event" on every call.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "chromadb.telemetry.product.NoopTelemetryClient")
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

import chromadb
from chromadb.config import Settings

from backend import config

log = logging.getLogger("rag")

_client: chromadb.ClientAPI | None = None


def client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        config.ensure_dirs()
        # On Windows, antivirus can briefly lock chromadb's migration .sql files while it
        # opens them, raising a transient PermissionError — retry rather than fail startup.
        for attempt in range(1, 6):
            try:
                _client = chromadb.PersistentClient(
                    path=str(config.CHROMA_PERSIST_DIR),
                    settings=Settings(anonymized_telemetry=False, allow_reset=True),
                )
                break
            except PermissionError as exc:
                if attempt == 5:
                    raise
                log.info("chroma locked (%s), retrying in %ss", exc.filename, attempt)
                time.sleep(attempt)
    return _client


def collection() -> chromadb.Collection:
    return client().get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def count() -> int:
    try:
        return collection().count()
    except Exception as exc:  # collection missing / corrupt store
        log.warning("chroma count failed: %s", exc)
        return 0


def upsert(
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
) -> None:
    collection().upsert(
        ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
    )


def query(embedding: list[float], n_results: int | None = None) -> list[dict[str, Any]]:
    """Return top-k matches as flat dicts: {id, document, source, topic, level, format, distance}."""
    k = n_results or config.RAG_TOP_K
    col = collection()
    total = col.count()
    if total == 0:
        return []

    res = col.query(
        query_embeddings=[embedding],
        n_results=min(k, total),
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict[str, Any]] = []
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    for i, doc_id in enumerate(ids):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        hits.append(
            {
                "id": doc_id,
                "source": meta.get("source", doc_id),
                "topic": meta.get("topic", "unknown"),
                "level": meta.get("level", "unknown"),
                "format": meta.get("format", "unknown"),
                "distance": round(float(dists[i]), 4) if i < len(dists) else None,
                "excerpt": (docs[i] or "")[:900] if i < len(docs) else "",
            }
        )
    return hits


def reset_collection() -> None:
    try:
        client().delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
