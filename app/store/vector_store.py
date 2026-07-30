"""ChromaDB wrapper: semantic search over the user's solved-problem history.

Why a vector store at all
-------------------------
Topic counting answers "how many graph problems have I solved". It cannot answer
"show me problems like the tree ones I kept failing", because a problem tagged
``trees`` and one tagged ``graph traversal`` are neighbours in meaning and
strangers to a string match. Embedding ``name + topics + difficulty`` lets the
tutor retrieve by *similarity*, which is what makes the "why am I weak at X"
explanations able to cite concrete problems the user actually attempted.

Why ChromaDB specifically
-------------------------
It runs embedded (a local directory, zero infrastructure) for development, and
the *same* client API talks to a standalone server for the Docker Compose setup.
That means one code path, switched by ``CHROMA_MODE``, instead of writing a
throwaway local implementation and a separate production one.

``chromadb`` is imported lazily inside :meth:`VectorStore._get_collection` so
that importing this module — which the API and the test suite both do — does not
drag in the embedding runtime until something actually needs to search.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from app.config import Settings, get_settings
from app.models import ProblemRecord

logger = logging.getLogger(__name__)


class VectorStore:
    """Stores one embedded document per problem the user has interacted with."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: Any = None
        self._collection: Any = None
        self._lock = threading.Lock()

    # -- wiring ------------------------------------------------------------
    def _build_embedding_function(self) -> Any:
        from chromadb.utils import embedding_functions

        if self._settings.embedding_backend == "chroma-default":
            # Chroma's bundled ONNX MiniLM. Smaller install, no torch.
            return embedding_functions.DefaultEmbeddingFunction()

        # Local sentence-transformers. Free, offline, and no per-query cost —
        # the deciding factor, since a full sync embeds thousands of problems.
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self._settings.embedding_model
        )

    def _get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection

        with self._lock:
            if self._collection is not None:
                return self._collection

            import chromadb

            settings = self._settings
            if settings.chroma_mode == "http":
                self._client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
                logger.info(
                    "connected to Chroma server",
                    extra={"host": settings.chroma_host, "port": settings.chroma_port},
                )
            else:
                settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
                logger.info(
                    "opened embedded Chroma store",
                    extra={"path": str(settings.chroma_persist_dir)},
                )

            self._collection = self._client.get_or_create_collection(
                name=settings.chroma_collection,
                embedding_function=self._build_embedding_function(),
                # Cosine similarity is the right metric for sentence embeddings;
                # Chroma's default is L2, which would rank longer documents oddly.
                metadata={"hnsw:space": "cosine"},
            )
            return self._collection

    # -- writes ------------------------------------------------------------
    def upsert_records(self, records: list[ProblemRecord]) -> int:
        """Index (or re-index) problem records.

        ``upsert`` rather than ``add`` so a re-sync updates existing rows instead
        of erroring on duplicate ids — syncs are expected to overlap heavily.
        """
        if not records:
            return 0

        collection = self._get_collection()
        ids = [record.problem_id for record in records]
        documents = [record.to_document() for record in records]
        metadatas = [
            {
                "source": record.source,
                "name": record.name,
                "url": record.url,
                # Chroma metadata values must be scalars, so the tag list is
                # flattened. Pipe-delimited with sentinels at both ends makes
                # exact-tag filtering possible later without partial matches.
                "tags": "|" + "|".join(record.tags) + "|" if record.tags else "",
                "rating": record.rating or 0,
                "difficulty": record.difficulty or "Unknown",
                "solved": record.solved,
                "attempt_count": record.attempt_count,
                "solved_at": record.solved_at.isoformat() if record.solved_at else "",
            }
            for record in records
        ]

        # Chroma has a per-call batch ceiling; chunking keeps large first syncs
        # from failing outright.
        batch_size = 500
        for start in range(0, len(ids), batch_size):
            stop = start + batch_size
            collection.upsert(
                ids=ids[start:stop],
                documents=documents[start:stop],
                metadatas=metadatas[start:stop],
            )

        logger.info("indexed problem records", extra={"count": len(ids)})
        return len(ids)

    # -- reads -------------------------------------------------------------
    def search(
        self, query: str, *, limit: int = 5, source: str | None = None, solved: bool | None = None
    ) -> list[dict]:
        """Semantic search over indexed problems, with optional metadata filters."""
        collection = self._get_collection()

        conditions = []
        if source:
            conditions.append({"source": source})
        if solved is not None:
            conditions.append({"solved": solved})
        where = None
        if len(conditions) == 1:
            where = conditions[0]
        elif conditions:
            where = {"$and": conditions}

        result = collection.query(query_texts=[query], n_results=limit, where=where)

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[dict] = []
        for index, document in enumerate(documents):
            metadata = dict(metadatas[index]) if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else None
            tags = [tag for tag in str(metadata.get("tags", "")).split("|") if tag]
            hits.append(
                {
                    "name": metadata.get("name", ""),
                    "url": metadata.get("url", ""),
                    "source": metadata.get("source", ""),
                    "tags": tags,
                    "rating": metadata.get("rating") or None,
                    "difficulty": metadata.get("difficulty"),
                    "solved": bool(metadata.get("solved")),
                    # Cosine distance -> similarity, so bigger is better in the API.
                    "similarity": round(1.0 - distance, 4) if distance is not None else None,
                    "document": document,
                }
            )
        return hits

    def count(self) -> int:
        try:
            return int(self._get_collection().count())
        except Exception:  # noqa: BLE001 - an empty/unavailable store reports zero
            logger.warning("could not read vector store count", exc_info=True)
            return 0

    def reset(self) -> None:
        """Drop and recreate the collection. Used by ``POST /sync?reset=true``."""
        with self._lock:
            if self._client is None:
                self._get_collection()
            self._client.delete_collection(self._settings.chroma_collection)
            self._collection = None
        logger.info("vector store reset", extra={"collection": self._settings.chroma_collection})


_store_singleton: VectorStore | None = None
_singleton_lock = threading.Lock()


def get_vector_store() -> VectorStore:
    """Process-wide vector store. One instance keeps the embedding model resident."""
    global _store_singleton
    if _store_singleton is None:
        with _singleton_lock:
            if _store_singleton is None:
                _store_singleton = VectorStore()
    return _store_singleton
