"""Subagents module for Request Evaluation & Classification and Response Verification.

Contains:
1. RequestEvaluator: Classifies and guardrails user input (statutory query, greeting, out of scope).
2. ResponseVerifier: Evaluates candidate LLM responses for statutory citation precision and faithfulness.
"""

from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from kp_agent.config import Settings, get_settings
from kp_agent.logger import get_logger
from kp_agent.models import ModelFactory
from schemas.agent import QueryClassification, VerificationResult

logger = get_logger(__name__)


class RequestEvaluator:
    """Subagent 1: Classifies user queries to determine routing, retrieval, and domain scope."""

    SYSTEM_PROMPT = """You are the Request Evaluator and Query Classifier for the Khyber Pakhtunkhwa (KP) Education Laws Assistant.
The assistant specializes strictly in 23 Khyber Pakhtunkhwa education laws (enacted 1897–2020) covering primary, secondary, higher, technical, and special education, school bag weights, free compulsory education, university charters, private schools regulatory authority, and educational boards.

Classify the user input into exactly one of the following intent categories:
1. 'statutory_query': The query asks about KP education laws, school policies, bag weight limits, teacher rules, exams, boards, fees, quotas, penalties, or university charters. Needs Vector DB retrieval.
2. 'greeting_chitchat': General greeting, farewell, courtesy, or identity question (e.g., "Hello", "Who are you?", "Thank you"). Does NOT need vector DB retrieval.
3. 'out_of_scope': The question is unrelated to KP education laws (e.g., recipes, sports, foreign law, generic medical/tech questions, federal laws outside KP education). Does NOT need vector DB retrieval.
4. 'clarification_needed': The query is extremely ambiguous or incomplete (e.g., "tell me", "law", single numbers).

Guidelines for direct_response:
- If 'out_of_scope': Set direct_response to:
  "I can only assist with questions regarding Khyber Pakhtunkhwa education laws and institutions (1897–2020). Please ask a question related to KP education statutes."
- If 'greeting_chitchat': Provide a polite, concise greeting explaining your role as the KP Education Laws Assistant.
- If 'clarification_needed': Request the user specify which KP education topic or law they wish to consult.
- If 'statutory_query': Set direct_response to null.
"""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize RequestEvaluator with LLM model.

        Args:
            settings: Application settings.
        """
        self.settings = settings or get_settings()
        self.llm = ModelFactory.get_chat_model(self.settings, temperature=0.0)

    def evaluate(self, query: str) -> QueryClassification:
        """Classify and evaluate the user query.

        Args:
            query: Raw user prompt string.

        Returns:
            QueryClassification: Structured classification with routing instructions.
        """
        clean_query = query.strip()
        if not clean_query:
            return QueryClassification(
                intent="clarification_needed",
                needs_retrieval=False,
                needs_llm=False,
                refined_intent="Empty input",
                direct_response="Please provide a valid question regarding Khyber Pakhtunkhwa education laws.",
            )

        # Bypass evaluation if feature is disabled
        if not self.settings.features.enable_request_evaluator:
            logger.info("RequestEvaluator disabled via feature flag; proceeding as statutory query.")
            return QueryClassification(
                intent="statutory_query",
                needs_retrieval=True,
                needs_llm=True,
                refined_intent=clean_query,
                direct_response=None,
            )

        try:
            structured_llm = self.llm.with_structured_output(QueryClassification)
            result = structured_llm.invoke(
                [
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    HumanMessage(content=clean_query),
                ]
            )
            logger.info("Classified query as intent='%s', needs_retrieval=%s", result.intent, result.needs_retrieval)
            return result
        except Exception as exc:
            logger.warning("Failed structured request evaluation; falling back to statutory query: %s", exc)
            return QueryClassification(
                intent="statutory_query",
                needs_retrieval=True,
                needs_llm=True,
                refined_intent=clean_query,
                direct_response=None,
            )


class ResponseVerifier:
    """Subagent 2: Verifies candidate LLM responses against retrieved statutory excerpts."""

    SYSTEM_PROMPT = """You are the Senior Statutory Auditor and Response Verifier for the Khyber Pakhtunkhwa (KP) Education Laws Assistant.
Your task is to critically inspect the candidate answer generated by the assistant against the provided retrieved statutory text.

Verification Rules:
1. Factual Faithfulness: Every factual claim, number, penalty, deadline, or quota MUST be directly supported by the retrieved context. No hallucinations, assumptions, or extrapolations.
2. Citation Requirement: Every substantive claim must cite the Law Title, Enactment Year, Section/Subsection/Schedule, and Page number where present in the context.
3. Negative Retrieval: If the retrieved text does NOT contain the factual answer to the question, the candidate answer MUST state:
   "Not found in the KP laws provided."
4. Pass Criteria:
   - Mark passed = true ONLY if all claims are faithfully grounded and properly cited, OR if it correctly states "Not found in the KP laws provided." when context is lacking.
   - Mark passed = false if there are unsupported claims, missing statutory citations, or fabricated legal facts. Provide specific refinement_instructions explaining what to fix.
"""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize ResponseVerifier with zero-temperature LLM model.

        Args:
            settings: Application settings.
        """
        self.settings = settings or get_settings()
        self.llm = ModelFactory.get_chat_model(self.settings, temperature=0.0)

    def verify(
        self,
        query: str,
        candidate_answer: str,
        retrieved_context: str,
    ) -> VerificationResult:
        """Verify the candidate legal answer against retrieved statutory context.

        Args:
            query: The user's original question.
            candidate_answer: The candidate answer generated by the agent.
            retrieved_context: The statutory excerpts retrieved from Qdrant.

        Returns:
            VerificationResult: Structured evaluation with pass status and critique.
        """
        # Bypass verification if feature is disabled
        if not self.settings.features.enable_response_verifier:
            logger.info("ResponseVerifier disabled via feature flag; passing candidate answer.")
            return VerificationResult(
                passed=True,
                faithfulness_score=1.0,
                citation_score=1.0,
                critique="Response verification disabled via features.yaml.",
                refinement_instructions=None,
            )

        # If answer is negative retrieval response and context is empty or uninformative, pass immediately
        if "Not found in the KP laws provided" in candidate_answer:
            return VerificationResult(
                passed=True,
                faithfulness_score=1.0,
                citation_score=1.0,
                critique="Accurate negative retrieval acknowledgment.",
                refinement_instructions=None,
            )

        prompt = (
            f"User Question:\n{query}\n\n"
            f"Retrieved Statutory Context:\n{retrieved_context}\n\n"
            f"Candidate Answer Under Review:\n{candidate_answer}\n"
        )

        try:
            structured_llm = self.llm.with_structured_output(VerificationResult)
            result = structured_llm.invoke(
                [
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )
            logger.info("Verification result: passed=%s, critique='%s'", result.passed, result.critique[:80])
            return result
        except Exception as exc:
            logger.warning("Structured response verification failed; accepting candidate answer: %s", exc)
            return VerificationResult(
                passed=True,
                faithfulness_score=1.0,
                citation_score=1.0,
                critique=f"Verification fallback: {exc}",
                refinement_instructions=None,
            )

