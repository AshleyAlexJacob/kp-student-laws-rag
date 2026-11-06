"""Search vector database tools for KP Education Laws Agent.

Provides the LawRetriever class for querying indexed statutory legal chunks
in Qdrant with semantic vector search and metadata filtering (level, year_from),
and exposes the LangChain tool `search_laws`.
"""

from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from kp_agent.config import Settings, get_settings
from kp_agent.logger import get_logger
from kp_agent.models import ModelFactory

logger = get_logger(__name__)


class LawRetriever:
    """Manages vector search retrieval against the Qdrant legal knowledge base."""

    def __init__(self, settings: Optional[Settings] = None, client: Optional[QdrantClient] = None) -> None:
        """Initialize the LawRetriever with database connections and embedding model.

        Args:
            settings: Application settings containing Qdrant credentials and collection names.
            client: Optional existing QdrantClient instance for connection reuse.
        """
        self.settings = settings or get_settings()

        # Initialize Qdrant client connection
        if client is not None:
            self.client = client
        else:
            logger.info("Connecting to Qdrant at %s", self.settings.qdrant_url)
            self.client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
                timeout=10.0,  # 10 seconds client connection timeout
            )

        # Initialize the active provider's embedding model
        self.embeddings = ModelFactory.get_embeddings(self.settings)

    def _build_filter(self, level: Optional[str] = None, year_from: Optional[int] = None) -> Optional[qmodels.Filter]:
        """Construct Qdrant query filter based on level and year constraints.

        Args:
            level: Educational tier filter (e.g. 'primary', 'secondary', 'higher', 'technical').
            year_from: Earliest year of statutory enactment.

        Returns:
            Optional[qmodels.Filter]: Qdrant boolean filter or None if no filters specified.
        """
        must_clauses: List[qmodels.FieldCondition] = []

        if level:
            clean_level = level.strip().lower()
            must_clauses.append(
                qmodels.FieldCondition(
                    key="level",
                    match=qmodels.MatchValue(value=clean_level),  # exact match on indexed keyword
                )
            )

        if year_from is not None:
            must_clauses.append(
                qmodels.FieldCondition(
                    key="year",
                    range=qmodels.Range(gte=int(year_from)),  # integer range filter on law enactment year
                )
            )

        if not must_clauses:
            return None

        return qmodels.Filter(must=must_clauses)

    def search_chunks(
        self,
        query: str,
        level: Optional[str] = None,
        year_from: Optional[int] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search the Qdrant database for statutory chunks matching the query.

        Args:
            query: Natural language legal question or topic.
            level: Optional education level filter.
            year_from: Optional minimum enactment year filter.
            limit: Maximum number of relevant chunks to retrieve.

        Returns:
            List[Dict[str, Any]]: List of dictionary results containing text, payload, and similarity score.

        Raises:
            RuntimeError: If vector generation or Qdrant search encounters an operational error.
        """
        clean_query = query.strip()
        if not clean_query:
            logger.warning("Empty query passed to search_chunks.")
            return []

        try:
            logger.info(
                "Embedding query for retrieval: '%s' (level=%s, year_from=%s, limit=%d)",
                clean_query[:80],
                level,
                year_from,
                limit,
            )
            # Embed query text into dense vector
            query_vector = self.embeddings.embed_query(clean_query)

            # Build optional metadata filters
            q_filter = self._build_filter(level=level, year_from=year_from)

            # Execute similarity search against Qdrant collection
            search_results = self.client.search(
                collection_name=self.settings.qdrant_collection,
                query_vector=query_vector,
                query_filter=q_filter,
                limit=limit,  # retrieve top-k closest statutory chunks
            )

            results: List[Dict[str, Any]] = []
            for hit in search_results:
                payload = hit.payload or {}
                results.append(
                    {
                        "score": hit.score,
                        "text": payload.get("text", ""),
                        "context_text": payload.get("context_text", ""),
                        "title": payload.get("title") or payload.get("document_title", "Unknown Law"),
                        "year": payload.get("year", "N/A"),
                        "level": payload.get("level", "N/A"),
                        "source_url": payload.get("source_url", ""),
                        "file": payload.get("file", ""),
                        "section_number": payload.get("section_number", "N/A"),
                        "section_title": payload.get("section_title", "N/A"),
                        "subsection_number": payload.get("subsection_number"),
                        "clause_number": payload.get("clause_number"),
                        "page": payload.get("page", 1),
                        "page_start": payload.get("page_start", 1),
                        "page_end": payload.get("page_end", 1),
                    }
                )

            logger.info("Found %d matching chunks in Qdrant for query.", len(results))
            return results

        except Exception as exc:
            logger.error("Failed to query Qdrant collection '%s': %s", self.settings.qdrant_collection, exc, exc_info=True)
            raise RuntimeError(f"Error during vector database retrieval: {exc}") from exc

    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a structured context string for LLM prompting.

        Args:
            results: List of retrieved chunk dictionaries.

        Returns:
            str: Human-readable, structured context block with statutory citations.
        """
        if not results:
            return "No matching statutory provisions found in the Khyber Pakhtunkhwa education laws database."

        formatted_parts: List[str] = []
        for idx, res in enumerate(results, start=1):
            title = res.get("title", "Khyber Pakhtunkhwa Education Law")
            year = res.get("year", "N/A")
            sec_num = res.get("section_number", "N/A")
            sec_title = res.get("section_title", "")
            sub_num = res.get("subsection_number")
            page = res.get("page", 1)
            score = res.get("score", 0.0)

            sec_display = f"Section {sec_num}"
            if sec_title and sec_title != "N/A":
                sec_display += f" ({sec_title})"
            if sub_num:
                sec_display += f", Subsection {sub_num}"

            citation_header = (
                f"[Source {idx}]: {title} ({year}) | {sec_display} | Page {page} | Similarity: {score:.3f}"
            )
            content_body = res.get("text", "").strip()

            formatted_parts.append(f"{citation_header}\n{content_body}")

        return "\n\n---\n\n".join(formatted_parts)


# Module-level singleton instance for shared tool usage
_retriever_instance: Optional[LawRetriever] = None


def get_retriever() -> LawRetriever:
    """Get or create singleton instance of LawRetriever.

    Returns:
        LawRetriever: Initialized retriever instance.
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = LawRetriever()
    return _retriever_instance


@tool
def search_laws(query: str, level: Optional[str] = None, year_from: Optional[int] = None) -> str:
    """Search Khyber Pakhtunkhwa education laws (1897-2020) for relevant statutory provisions.

    Always invoke this tool before answering any question regarding education laws, policies,
    rules, institutions, penalties, authorities, or student regulations in KP.

    Args:
        query: Specific search phrase, topic, or question concerning KP education laws.
        level: Optional educational tier filter ('primary', 'elementary', 'secondary', 'higher', 'technical').
        year_from: Optional minimum enactment year to restrict historical laws (e.g., 2010).

    Returns:
        Structured statutory excerpts with full citations including title, year, section, and page number.
    """
    retriever = get_retriever()
    # Default to 5 relevant chunks per query to ensure comprehensive statutory context
    chunks = retriever.search_chunks(query=query, level=level, year_from=year_from, limit=5)
    return retriever.format_search_results(chunks)

