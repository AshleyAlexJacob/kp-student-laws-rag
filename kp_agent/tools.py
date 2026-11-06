"""Search vector database tools for KP Education Laws Agent.

Provides the LawRetriever class for querying indexed statutory legal chunks
in Qdrant with semantic vector search, metadata filtering (level, year_from),
LLM-based query rewriting (max 3 variants), and Maximal Marginal Relevance (MMR)
reranking for diverse, comprehensive legal context.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from kp_agent.config import Settings, get_settings
from kp_agent.logger import get_logger
from kp_agent.models import ModelFactory

logger = get_logger(__name__)


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculate cosine similarity between two numeric vectors.

    Args:
        vec_a: First vector of floats.
        vec_b: Second vector of floats.

    Returns:
        float: Cosine similarity in range [-1.0, 1.0].
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def compute_mmr(
    query_vector: List[float],
    candidate_items: List[Dict[str, Any]],
    lambda_mult: float = 0.7,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Rank candidate items using Maximal Marginal Relevance (MMR) for relevance and diversity.

    Args:
        query_vector: Dense embedding vector for the user search query.
        candidate_items: List of dictionary candidates each having a 'vector' key.
        lambda_mult: Diversity weight factor in [0, 1]. 1.0 is purely relevance, 0.0 is purely diversity.
        top_k: Maximum number of diverse items to select.

    Returns:
        List[Dict[str, Any]]: Selected items in order of MMR selection.
    """
    if not candidate_items:
        return []

    # If items do not exceed top_k, return sorted by similarity score
    if len(candidate_items) <= top_k:
        return sorted(candidate_items, key=lambda x: x.get("score", 0.0), reverse=True)

    selected: List[Dict[str, Any]] = []
    unselected = list(candidate_items)

    # Pre-extract vector representations
    query_sims = [cosine_similarity(query_vector, item["vector"]) if item.get("vector") else item.get("score", 0.0) for item in unselected]

    while len(selected) < top_k and unselected:
        best_score = -float("inf")
        best_idx = -1

        for i, item in enumerate(unselected):
            sim_to_query = query_sims[i]

            # Compute maximal similarity to any already selected candidate
            if not selected:
                sim_to_selected = 0.0
            else:
                sim_to_selected = max(
                    cosine_similarity(item["vector"], sel["vector"])
                    if item.get("vector") and sel.get("vector")
                    else 0.0
                    for sel in selected
                )

            # MMR formula: lambda * sim(doc, query) - (1 - lambda) * max_sim(doc, selected)
            mmr_score = (lambda_mult * sim_to_query) - ((1.0 - lambda_mult) * sim_to_selected)

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        if best_idx != -1:
            chosen = unselected.pop(best_idx)
            query_sims.pop(best_idx)
            selected.append(chosen)
        else:
            break

    return selected


class QueryRewriter:
    """Generates statutory and legal query variants for dense vector search."""

    REWRITE_PROMPT = """You are a legal search optimizer for Khyber Pakhtunkhwa (KP) education laws (1897–2020).
Given a user query, generate up to {max_variants} distinct search queries optimized for vector database retrieval.

Guidelines:
1. Variant 1: Must be the original user query verbatim.
2. Variant 2: Use formal statutory language, legal terms of art, and relevant act/section terminology.
3. Variant 3: Expand institutional acronyms, incorporate related bodies (e.g. Higher Education Department, Board of Intermediate & Secondary Education, School Quality Management), and legal duties/penalties.

Respond ONLY with the queries, one per line. Do not output numbers, bullet points, or labels.
"""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize QueryRewriter with the LLM chat model.

        Args:
            settings: Application settings.
        """
        self.settings = settings or get_settings()
        self.llm = ModelFactory.get_chat_model(self.settings, temperature=0.2)

    def rewrite(self, query: str, max_variants: int = 3) -> List[str]:
        """Generate up to max_variants query variations.

        Args:
            query: Original user query.
            max_variants: Maximum number of query variants (default 3).

        Returns:
            List[str]: List of diverse search queries including the original.
        """
        clean_query = query.strip()
        if not clean_query:
            return [clean_query]

        variants: List[str] = [clean_query]

        try:
            prompt = self.REWRITE_PROMPT.format(max_variants=max_variants)
            response = self.llm.invoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(content=clean_query),
                ]
            )
            raw_text = str(response.content).strip()

            # Parse lines and strip bullets/numbers
            for line in raw_text.splitlines():
                cleaned = re.sub(r"^[\d\.\-\*\s]+", "", line).strip()
                # Remove quotes if wrapped
                cleaned = cleaned.strip("\"'").strip()
                if cleaned and cleaned.lower() != clean_query.lower() and cleaned not in variants:
                    variants.append(cleaned)
                if len(variants) >= max_variants:
                    break

            logger.info("Rewrote query into %d variants: %s", len(variants), variants)
        except Exception as exc:
            logger.warning("Query rewriting failed; falling back to original query: %s", exc)

        return variants[:max_variants]


class LawRetriever:
    """Manages vector search retrieval against the Qdrant legal knowledge base with MMR and Query Rewriting."""

    def __init__(self, settings: Optional[Settings] = None, client: Optional[QdrantClient] = None) -> None:
        """Initialize the LawRetriever with database connections, embeddings, and query rewriter.

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

        # Query rewriter sub-component
        self.rewriter = QueryRewriter(self.settings)

        # Track last executed query variants for structured response reporting
        self.last_query_variants: List[str] = []

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
        """Search Qdrant for statutory chunks using Query Rewriting and Maximal Marginal Relevance (MMR).

        Args:
            query: Natural language legal question or topic.
            level: Optional education level filter.
            year_from: Optional minimum enactment year filter.
            limit: Maximum number of diverse relevant chunks to retrieve.

        Returns:
            List[Dict[str, Any]]: Deduplicated, MMR-ranked statutory chunks.

        Raises:
            RuntimeError: If vector generation or Qdrant search encounters an operational error.
        """
        clean_query = query.strip()
        if not clean_query:
            logger.warning("Empty query passed to search_chunks.")
            self.last_query_variants = []
            return []

        # 1. Determine query variants based on feature flag
        if self.settings.features.enable_query_rewriter:
            max_vars = self.settings.features.max_query_variants
            query_variants = self.rewriter.rewrite(clean_query, max_variants=max_vars)
        else:
            query_variants = [clean_query]

        self.last_query_variants = list(query_variants)
        logger.info("Executing retrieval with %d query variants", len(query_variants))

        try:
            # Build optional metadata filters
            q_filter = self._build_filter(level=level, year_from=year_from)

            # Dictionary to deduplicate chunks across query variants: key -> chunk dict
            candidates_by_id: Dict[str, Dict[str, Any]] = {}

            # Keep primary query vector for MMR calculations
            primary_query_vector: Optional[List[float]] = None

            for idx, q_var in enumerate(query_variants):
                # Embed query variant into dense vector
                var_vector = self.embeddings.embed_query(q_var)
                if idx == 0:
                    primary_query_vector = var_vector

                # Execute vector search against Qdrant collection requesting vectors for MMR
                if hasattr(self.client, "query_points"):
                    query_response = self.client.query_points(
                        collection_name=self.settings.qdrant_collection,
                        query=var_vector,
                        query_filter=q_filter,
                        limit=limit * 2,  # retrieve candidate pool for MMR reranking
                        with_vectors=True,
                        with_payload=True,
                    )
                    search_results = query_response.points
                else:
                    search_results = self.client.search(
                        collection_name=self.settings.qdrant_collection,
                        query_vector=var_vector,
                        query_filter=q_filter,
                        limit=limit * 2,
                        with_vectors=True,
                    )

                for hit in search_results:
                    payload = hit.payload or {}
                    # Build unique key based on statutory location to deduplicate identical provisions
                    unique_key = (
                        f"{payload.get('file', '')}_"
                        f"{payload.get('section_number', '')}_"
                        f"{payload.get('page', '')}_"
                        f"{str(payload.get('text', ''))[:40]}"
                    )

                    score = float(hit.score)
                    if unique_key not in candidates_by_id or score > candidates_by_id[unique_key]["score"]:
                        candidates_by_id[unique_key] = {
                            "score": score,
                            "vector": hit.vector if isinstance(hit.vector, list) else None,
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

            candidate_pool = list(candidates_by_id.values())
            logger.info("Retrieved %d unique candidate chunks across all query variants.", len(candidate_pool))

            if not candidate_pool:
                return []

            # 2. Apply Maximal Marginal Relevance (MMR) reranking if enabled
            if (
                self.settings.features.enable_mmr_retrieval
                and primary_query_vector is not None
                and len(candidate_pool) > limit
            ):
                logger.info(
                    "Applying MMR reranking (lambda=%.2f, target=%d) on %d candidates",
                    self.settings.features.mmr_diversity_lambda,
                    limit,
                    len(candidate_pool),
                )
                final_results = compute_mmr(
                    query_vector=primary_query_vector,
                    candidate_items=candidate_pool,
                    lambda_mult=self.settings.features.mmr_diversity_lambda,
                    top_k=limit,
                )
            else:
                # Rank strictly by highest similarity score
                candidate_pool.sort(key=lambda x: x.get("score", 0.0), reverse=True)
                final_results = candidate_pool[:limit]

            logger.info("Selected %d statutory chunks after MMR / reranking.", len(final_results))
            return final_results

        except Exception as exc:
            logger.error(
                "Failed to query Qdrant collection '%s': %s",
                self.settings.qdrant_collection,
                exc,
                exc_info=True,
            )
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

    Employs Query Rewriting (up to 3 legal variants) and Maximal Marginal Relevance (MMR)
    to retrieve diverse, non-redundant statutory provisions.

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
    limit = retriever.settings.features.retrieval_limit
    chunks = retriever.search_chunks(query=query, level=level, year_from=year_from, limit=limit)
    return retriever.format_search_results(chunks)
