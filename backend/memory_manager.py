"""
memory_manager.py — Hybrid long-term memory using ChromaDB + Google text-embedding-004.

Responsibilities:
  • embed_text(text)                  → Generate embedding via Gemini API
  • store_interaction(...)            → Save Q&A pair to ChromaDB
  • retrieve_relevant_history(...)    → Top-K cosine-similar past turns
  • delete_user_memory(user_id)       → GDPR-style data wipe
"""

import os
import time
import logging
from typing import Optional, Any

import chromadb
import google.generativeai as genai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
COLLECTION_NAME    = "chat_memory"
EMBEDDING_MODEL    = "models/text-embedding-004"
TOP_K_RESULTS      = 3
MAX_EMBED_RETRIES  = 3
EMBED_RETRY_DELAY  = 1.5   # seconds

# ---------------------------------------------------------------------------
# ChromaDB client (module-level singleton)
# ---------------------------------------------------------------------------

os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

_chroma_client: Any = None
_collection    = None


def _get_collection():
    """Lazy-initialize ChromaDB client and collection."""
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB collection '{COLLECTION_NAME}' ready at {CHROMA_PERSIST_DIR}")
    return _collection


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def embed_text(text: str) -> list[float]:
    """
    Generate a vector embedding for `text` using Google's text-embedding-004.
    Retries up to MAX_EMBED_RETRIES times on transient errors.
    """
    for attempt in range(1, MAX_EMBED_RETRIES + 1):
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type="RETRIEVAL_DOCUMENT",
            )
            return result["embedding"]
        except Exception as exc:
            logger.warning(f"Embedding attempt {attempt} failed: {exc}")
            if attempt < MAX_EMBED_RETRIES:
                time.sleep(EMBED_RETRY_DELAY * attempt)
            else:
                raise RuntimeError(f"embed_text failed after {MAX_EMBED_RETRIES} retries: {exc}") from exc
    return []  # unreachable


def embed_query(text: str) -> list[float]:
    """
    Generate a query-optimised embedding (task_type=RETRIEVAL_QUERY).
    Used when searching ChromaDB so the model knows this is a lookup.
    """
    for attempt in range(1, MAX_EMBED_RETRIES + 1):
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type="RETRIEVAL_QUERY",
            )
            return result["embedding"]
        except Exception as exc:
            logger.warning(f"Query embedding attempt {attempt} failed: {exc}")
            if attempt < MAX_EMBED_RETRIES:
                time.sleep(EMBED_RETRY_DELAY * attempt)
            else:
                raise RuntimeError(f"embed_query failed after {MAX_EMBED_RETRIES} retries: {exc}") from exc
    return []


# ---------------------------------------------------------------------------
# Core memory operations
# ---------------------------------------------------------------------------

def store_interaction(
    user_id: str,
    user_message: str,
    bot_response: str,
    message_id: int,
    timestamp: Optional[str] = None,
) -> None:
    """
    Embed and store a (user_message, bot_response) pair in ChromaDB.

    The document text is a concatenation of both turns so that retrieval
    can match on either the question or answer semantics.
    """
    collection = _get_collection()

    doc_text = f"User: {user_message}\nAssistant: {bot_response}"
    doc_id   = f"{user_id}_{message_id}"

    try:
        embedding = embed_text(doc_text)
        collection.upsert(
            ids        = [doc_id],
            embeddings = [embedding],
            documents  = [doc_text],
            metadatas  = [{
                "user_id":   user_id,
                "timestamp": timestamp or str(int(time.time())),
                "msg_id":    str(message_id),
            }],
        )
        logger.debug(f"Stored interaction {doc_id} in ChromaDB")
    except Exception as exc:
        logger.error(f"Failed to store interaction {doc_id}: {exc}")
        # Non-fatal: chat still works even if vector store write fails


def retrieve_relevant_history(
    user_id: str,
    query_text: str,
    top_k: int = TOP_K_RESULTS,
) -> list[dict]:
    """
    Perform cosine-similarity search in ChromaDB for `query_text`,
    filtered to the given `user_id`.

    Returns a list of dicts:
        { 'document': str, 'score': float, 'metadata': dict }
    ordered by relevance (most relevant first).
    """
    collection = _get_collection()

    try:
        query_embedding = embed_query(query_text)
        results = collection.query(
            query_embeddings = [query_embedding],
            n_results        = top_k,
            where            = {"user_id": user_id},
            include          = ["documents", "metadatas", "distances"],
        )

        hits = []
        raw_docs = results.get("documents") or [[]]
        raw_metas = results.get("metadatas") or [[]]
        raw_dists = results.get("distances") or [[]]
        docs = raw_docs[0] if raw_docs else []
        metadatas = raw_metas[0] if raw_metas else []
        distances = raw_dists[0] if raw_dists else []

        for doc, meta, dist in zip(docs, metadatas, distances):
            # ChromaDB cosine distance → similarity score (0–1, higher = more similar)
            score = 1.0 - dist
            hits.append({
                "document": doc,
                "score":    round(score, 4),
                "metadata": meta,
            })

        hits.sort(key=lambda x: x["score"], reverse=True)
        logger.debug(f"Retrieved {len(hits)} relevant history items for user {user_id}")
        return hits

    except Exception as exc:
        logger.warning(f"ChromaDB retrieval failed for user {user_id}: {exc}")
        return []   # Graceful degradation: return empty list, chat still works


def format_rag_context(hits: list[dict]) -> str:
    """
    Format retrieved history hits into a readable string block
    ready to be injected into the Gemini prompt.
    """
    if not hits:
        return ""

    lines = ["--- Relevant Past Conversations (for context) ---"]
    for i, hit in enumerate(hits, 1):
        lines.append(f"[Memory {i} | Relevance: {hit['score']:.2f}]")
        lines.append(hit["document"])
        lines.append("")
    lines.append("--- End of Retrieved Memories ---")
    return "\n".join(lines)


def delete_user_memory(user_id: str) -> int:
    """
    Delete all ChromaDB records for a user_id.
    Returns the number of records deleted.
    """
    collection = _get_collection()
    try:
        existing = collection.get(where={"user_id": user_id})
        ids = existing.get("ids", [])
        if ids:
            collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} ChromaDB records for user {user_id}")
        return len(ids)
    except Exception as exc:
        logger.error(f"Failed to delete memory for user {user_id}: {exc}")
        return 0


def get_collection_stats() -> dict:
    """Return basic stats about the ChromaDB collection."""
    collection = _get_collection()
    return {
        "collection_name": COLLECTION_NAME,
        "total_documents":  collection.count(),
        "persist_dir":     CHROMA_PERSIST_DIR,
    }
