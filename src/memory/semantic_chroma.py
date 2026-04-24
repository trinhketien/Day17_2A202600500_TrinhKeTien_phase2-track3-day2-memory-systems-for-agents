"""
Backend 4: ChromaDB Semantic Memory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Storage   : ChromaDB (Docker HttpClient OR local PersistentClient fallback)
• Scope     : Persistent, cross-session
• Embedding : OpenAI text-embedding-3-small
• Use case  : Cosine-similarity memory search — "Have I asked this before?"
"""
import logging
import os
from typing import List

from langchain_openai import OpenAIEmbeddings

from .models import ContextItem, ContextPriority, MemoryEntry, MemoryType

logger = logging.getLogger(__name__)
COLLECTION_NAME = "agent_memory"
SIMILARITY_THRESHOLD = 0.60  # Only return memories with cosine similarity ≥ 60%

try:
    import chromadb as _chromadb  # type: ignore
    _CHROMA_OK = True
except ImportError:
    _CHROMA_OK = False


class SemanticChromaMemory:
    """ChromaDB semantic memory — auto-falls back to local PersistentClient if Docker is unavailable."""

    def __init__(
        self,
        user_id: str,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.user_id     = user_id
        self._collection = None
        self.embeddings  = OpenAIEmbeddings(model=embedding_model)

        if not _CHROMA_OK:
            logger.warning("'chromadb' not installed. Semantic memory disabled.")
            return

        # Try Docker HttpClient first
        try:
            client = _chromadb.HttpClient(host=chroma_host, port=chroma_port)
            client.heartbeat()
            self._collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"ChromaDB connected via Docker at {chroma_host}:{chroma_port}")
            return
        except Exception:
            pass  # Docker not running — fall through to local

        # Fallback: local PersistentClient (no Docker needed)
        try:
            chroma_dir = os.path.join("data", "chroma")
            os.makedirs(chroma_dir, exist_ok=True)
            client = _chromadb.PersistentClient(path=chroma_dir)
            self._collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"ChromaDB using local PersistentClient → {chroma_dir}")
        except Exception as exc:
            logger.warning(f"ChromaDB unavailable ({exc}). Semantic memory disabled.")

    # ──────────────────────────────────────────────
    # Write
    # ──────────────────────────────────────────────
    def save(self, entry: MemoryEntry) -> bool:
        if not self._collection:
            return False
        try:
            text = f"Q: {entry.query}\nA: {entry.response[:500]}"
            embedding = self.embeddings.embed_query(text)
            self._collection.add(
                ids=[entry.turn_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{
                    "user_id":    self.user_id,
                    "session_id": entry.session_id,
                    "timestamp":  entry.timestamp,
                    "query":      entry.query[:500],
                    "response":   entry.response[:500],
                }],
            )
            return True
        except Exception as exc:
            logger.error(f"Chroma save: {exc}")
            return False

    # ──────────────────────────────────────────────
    # Read
    # ──────────────────────────────────────────────
    def search(self, query: str, k: int = 3) -> List[ContextItem]:
        """Return top-k semantically similar memories for this user."""
        if not self._collection:
            return []
        try:
            count = self._collection.count()
            if count == 0:
                return []

            query_embedding = self.embeddings.embed_query(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, count),
                where={"user_id": self.user_id},
            )

            items: List[ContextItem] = []
            if results and results.get("documents"):
                docs      = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                for doc, meta, dist in zip(docs, metadatas, distances):
                    similarity = 1.0 - dist  # cosine distance → similarity
                    if similarity >= SIMILARITY_THRESHOLD:
                        items.append(
                            ContextItem(
                                content=f"[Semantic Memory] {doc}",
                                priority=ContextPriority.MEDIUM,
                                source=MemoryType.SEMANTIC,
                                relevance_score=similarity,
                                metadata={
                                    "original_query": meta.get("query", ""),
                                    "timestamp":      meta.get("timestamp", ""),
                                    "similarity":     f"{similarity:.2f}",
                                },
                            )
                        )
            return items
        except Exception as exc:
            logger.error(f"Chroma search: {exc}")
            return []

    def clear(self) -> None:
        if not self._collection:
            return
        try:
            res = self._collection.get(where={"user_id": self.user_id})
            if res["ids"]:
                self._collection.delete(ids=res["ids"])
            logger.info(f"ChromaDB cleared for user {self.user_id}")
        except Exception as exc:
            logger.error(f"Chroma clear: {exc}")

    @property
    def is_connected(self) -> bool:
        return self._collection is not None

    @property
    def total_vectors(self) -> int:
        if not self._collection:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0
