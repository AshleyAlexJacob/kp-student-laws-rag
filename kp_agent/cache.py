"""Semantic caching layer using Qdrant for KP Education Laws Agent.

Stores and retrieves natural language queries and their verified legal answers
in a dedicated Qdrant collection to reduce latency and model invocation costs.
"""

import time
import uuid
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from kp_agent.config import Settings, get_settings
from kp_agent.logger import get_logger
from kp_agent.models import ModelFactory

logger = get_logger(__name__)


class SemanticCache:
    """Semantic vector cache backed by a secondary Qdrant collection."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[QdrantClient] = None,
        collection_name: str = "kp_laws_cache",
        similarity_threshold: float = 0.92,
    ) -> None:
        """Initialize semantic cache with collection parameters.

        Args:
            settings: Application settings. Defaults to singleton.
            client: Optional shared QdrantClient instance.
            collection_name: Qdrant collection dedicated to query semantic cache.
            similarity_threshold: Cosine similarity score threshold (0.0 - 1.0) for hit acceptance.
        """
        self.settings = settings or get_settings()
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold  # 0.92 threshold for strict semantic equivalence

        # Use passed client or instantiate new connection
        if client is not None:
            self.client = client
        else:
            self.client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
                timeout=10.0,  # 10s timeout
            )

        # Embedding model for converting questions to semantic vectors
        self.embeddings = ModelFactory.get_embeddings(self.settings)

        # Ensure cache collection exists
        self.ensure_collection()

    def ensure_collection(self) -> None:
        """Create the cache collection in Qdrant if it does not yet exist.

        Raises:
            RuntimeError: If collection verification or creation fails.
        """
        try:
            if not self.client.collection_exists(collection_name=self.collection_name):
                logger.info(
                    "Creating semantic cache collection '%s' with dim=%d",
                    self.collection_name,
                    self.settings.embedding_dimension,
                )
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.settings.embedding_dimension,  # Matches embedding model dimension
                        distance=Distance.COSINE,  # Cosine metric for semantic similarity
                    ),
                )
        except Exception as exc:
            logger.error("Failed to verify or create cache collection '%s': %s", self.collection_name, exc, exc_info=True)
            raise RuntimeError(f"SemanticCache collection error: {exc}") from exc

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Look up a semantically equivalent query in the cache.

        Args:
            query: User's question string.

        Returns:
            Optional[Dict[str, Any]]: Cached response dict containing answer, sources, and score,
            or None if no match meets the similarity threshold.
        """
        clean_query = query.strip()
        if not clean_query:
            return None

        try:
            query_vector = self.embeddings.embed_query(clean_query)

            # Query the closest existing cache point
            if hasattr(self.client, "query_points"):
                cache_resp = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=1,  # inspect top single match
                    with_payload=True,
                )
                search_results = cache_resp.points
            else:
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=1,  # inspect top single match
                )

            if not search_results:
                logger.warning("Cache miss (no entries found) for query: '%s'", clean_query[:60])
                return None

            top_hit = search_results[0]
            if top_hit.score >= self.similarity_threshold:
                logger.info(
                    "Cache hit for query '%s' (similarity=%.4f >= %.2f)",
                    clean_query[:60],
                    top_hit.score,
                    self.similarity_threshold,
                )
                payload = top_hit.payload or {}
                return {
                    "answer": payload.get("answer", ""),
                    "sources": payload.get("sources", []),
                    "cached_query": payload.get("query", ""),
                    "similarity": top_hit.score,
                }

            # Score is below threshold
            logger.warning(
                "Cache miss (similarity %.4f < %.2f) for query: '%s'",
                top_hit.score,
                self.similarity_threshold,
                clean_query[:60],
            )
            return None

        except Exception as exc:
            logger.error("Error checking semantic cache: %s", exc, exc_info=True)
            return None  # Fail gracefully to allow normal agent invocation

    def set(self, query: str, answer: str, sources: Optional[List[Dict[str, Any]]] = None) -> None:
        """Store a verified query-response pair in the semantic cache.

        Args:
            query: User query text.
            answer: Generated or verified response text.
            sources: List of legal source chunk metadata.
        """
        clean_query = query.strip()
        clean_answer = answer.strip()
        if not clean_query or not clean_answer:
            return

        try:
            query_vector = self.embeddings.embed_query(clean_query)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"cache_{clean_query}"))

            point = PointStruct(
                id=point_id,
                vector=query_vector,
                payload={
                    "query": clean_query,
                    "answer": clean_answer,
                    "sources": sources or [],
                    "timestamp": time.time(),
                },
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )
            logger.info("Successfully stored query in semantic cache: '%s'", clean_query[:60])

        except Exception as exc:
            logger.error("Failed to store entry in semantic cache: %s", exc, exc_info=True)

