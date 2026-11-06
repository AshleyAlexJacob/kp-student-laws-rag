"""Structured data schemas for Agentic RAG responses, subagents, and verification."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryClassification(BaseModel):
    """Subagent 1: Request Evaluator & Query Classifier output schema."""

    intent: str = Field(
        ...,
        description="Intent category: 'statutory_query', 'greeting_chitchat', 'out_of_scope', or 'clarification_needed'.",
    )
    needs_retrieval: bool = Field(
        ...,
        description="Whether vector database retrieval is required to answer the query.",
    )
    needs_llm: bool = Field(
        ...,
        description="Whether LLM generation is required (False if direct template refusal).",
    )
    refined_intent: str = Field(
        ...,
        description="Summary of the user's specific legal research intent or topic.",
    )
    direct_response: Optional[str] = Field(
        default=None,
        description="Direct response text if retrieval is bypassed (e.g. greeting or out-of-scope refusal).",
    )


class VerificationResult(BaseModel):
    """Subagent 2: Response Verifier output schema."""

    passed: bool = Field(
        ...,
        description="True if the response strictly adheres to retrieved facts and cites statutory provisions.",
    )
    faithfulness_score: float = Field(
        default=1.0,
        description="Confidence score (0.0 to 1.0) indicating factual grounding without hallucination.",
    )
    citation_score: float = Field(
        default=1.0,
        description="Confidence score (0.0 to 1.0) for presence of law title, year, section, and page citations.",
    )
    critique: str = Field(
        default="",
        description="Explanation of strengths or specific deficiencies found during verification.",
    )
    refinement_instructions: Optional[str] = Field(
        default=None,
        description="Specific instructions for candidate answer refinement if verification failed.",
    )


class AgentResponse(BaseModel):
    """Structured response returned by the Agentic RAG orchestrator."""

    answer: str = Field(..., description="Final statutory answer or refusal message.")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of retrieved legal provisions and citation details.",
    )
    verification_count: int = Field(
        default=0,
        description="Number of verification passes performed on this answer.",
    )
    verification_status: str = Field(
        default="verified",
        description="Verification outcome: 'verified', 'refined', 'unverified', 'out_of_scope', or 'chitchat'.",
    )
    verification_critique: Optional[str] = Field(
        default=None,
        description="Summary critique from the Response Verifier subagent.",
    )
    query_variants: List[str] = Field(
        default_factory=list,
        description="Query rewrites generated for vector database search (max 3 variants).",
    )
    cached: bool = Field(
        default=False,
        description="Whether the answer was retrieved from semantic cache.",
    )
    intent: str = Field(
        default="statutory_query",
        description="Classification intent determined by the Request Evaluator.",
    )

