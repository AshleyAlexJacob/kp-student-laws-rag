"""LawAgent module for KP Education Laws Assistant.

Constructs the conversational legal agent using LangChain/LangGraph create_agent,
integrating:
1. Subagent 1: Request Evaluator & Query Classifier (routing and scope guardrails).
2. Qdrant vector retrieval tool with Query Rewriting (max 3 variants) and MMR.
3. Subagent 2: Response Verifier for citation verification and anti-hallucination auditing.
4. LangGraph MemorySaver checkpointer for thread-safe multi-turn session persistence.
5. Structured response tracking verification cycles, statuses, and query rewrites.
"""

from typing import Any, Dict, List, Optional, Tuple
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

try:
    from langchain.agents import create_agent
except ImportError:
    from langgraph.prebuilt import create_react_agent as create_agent

from kp_agent.cache import SemanticCache
from kp_agent.config import Settings, get_settings
from kp_agent.logger import get_logger
from kp_agent.models import ModelFactory
from kp_agent.subagents import RequestEvaluator, ResponseVerifier
from kp_agent.tools import LawRetriever, get_retriever, search_laws
from schemas.agent import AgentResponse, QueryClassification, VerificationResult

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an authoritative legal research assistant specialized strictly in Khyber Pakhtunkhwa (KP) education laws (1897–2020).

Strict Behavioral Guidelines:
1. Retrieval First: Always invoke the `search_laws` tool before answering any substantive question. Never answer from internal knowledge alone.
2. Citation Requirement: Every single factual claim, quota, penalty, quorum, power, deadline, or requirement MUST cite:
   - Full law title (e.g. "Khyber Pakhtunkhwa School Bags (Limitation of Weight) Act, 2020")
   - Year of enactment (e.g. "2020")
   - Statutory section, subsection, or schedule (e.g. "Section 3(2)", "Schedule", "Section 22(1)")
   - Page number where available.
3. Negative Retrieval: If the retrieved provisions do not contain the answer, or if the question is unanswerable from the provided KP laws, reply exactly:
   "Not found in the KP laws Database."
4. Scope Restriction: Politely decline any question that is unrelated to Khyber Pakhtunkhwa education laws and institutions.
5. Accuracy & Precision: Do not assume or extrapolate legal facts beyond what the cited statutory text specifies.
"""


class LawAgent:
    """Agentic RAG orchestrator for Khyber Pakhtunkhwa education laws Q&A."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        retriever: Optional[LawRetriever] = None,
        enable_cache: Optional[bool] = None,
    ) -> None:
        """Initialize the legal agent, subagents, models, checkpointer, and tools.

        Args:
            settings: Application settings. Defaults to singleton.
            retriever: Optional LawRetriever instance.
            enable_cache: Whether to enable semantic query caching. Defaults to settings.features flag.
        """
        self.settings = settings or get_settings()
        self.retriever = retriever or get_retriever()

        # Cache enablement respects feature flag and explicit constructor parameter
        if enable_cache is not None:
            self.enable_cache = enable_cache
        else:
            self.enable_cache = self.settings.features.enable_semantic_cache

        # Memory checkpointer for multi-turn conversational state persistence
        self.checkpointer = MemorySaver()

        # Subagent 1: Request Evaluator & Query Classifier
        self.request_evaluator = RequestEvaluator(self.settings)

        # Subagent 2: Response Verifier & Statutory Auditor
        self.response_verifier = ResponseVerifier(self.settings)

        # Semantic cache on secondary collection
        self.cache: Optional[SemanticCache] = None
        if self.enable_cache:
            try:
                self.cache = SemanticCache(
                    settings=self.settings,
                    client=self.retriever.client,  # reuse existing Qdrant client connection
                )
            except Exception as exc:
                logger.warning("Failed to initialize semantic cache, running without cache: %s", exc)

        # Initialize LLM with zero temperature for factual legal fidelity
        self.chat_model = ModelFactory.get_chat_model(self.settings, temperature=0.0)

        # List of tools available to the agent
        self.tools = [search_laws]

        # Compile agent graph
        logger.info("Compiling LawAgent with tools: %s", [t.name for t in self.tools])
        try:
            # LangChain v1 create_agent uses system_prompt and checkpointer
            self.agent_executor = create_agent(
                model=self.chat_model,
                tools=self.tools,
                checkpointer=self.checkpointer,
                system_prompt=SYSTEM_PROMPT,
            )
        except TypeError:
            try:
                self.agent_executor = create_agent(
                    model=self.chat_model,
                    tools=self.tools,
                    checkpointer=self.checkpointer,
                    prompt=SYSTEM_PROMPT,
                )
            except TypeError:
                self.agent_executor = create_agent(
                    self.chat_model,
                    self.tools,
                    checkpointer=self.checkpointer,
                    state_modifier=SYSTEM_PROMPT,
                )

    def _extract_sources_and_context(self, messages: List[BaseMessage]) -> Tuple[List[Dict[str, Any]], str]:
        """Extract citations and raw context from ToolMessages in execution history.

        Args:
            messages: Full message trajectory from agent invocation.

        Returns:
            Tuple[List[Dict[str, Any]], str]: Extracted statutory sources and concatenated context.
        """
        sources: List[Dict[str, Any]] = []
        context_parts: List[str] = []

        for msg in messages:
            if isinstance(msg, ToolMessage) or getattr(msg, "type", "") == "tool":
                content = str(msg.content)
                context_parts.append(content)
                # Parse structured blocks: [Source N]: Title (Year) | Section X | Page Y | Similarity: S
                blocks = content.split("\n\n---\n\n")
                for block in blocks:
                    first_line = block.splitlines()[0] if block.splitlines() else ""
                    if first_line.startswith("[Source"):
                        sources.append(
                            {
                                "citation_header": first_line,
                                "snippet": block.replace(first_line, "").strip(),
                            }
                        )

        return sources, "\n\n".join(context_parts)

    def ask(
        self,
        query: str,
        thread_id: str = "default",
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Ask a legal question through the complete Agentic RAG workflow.

        Workflow steps:
        1. Subagent 1 evaluates & classifies query (statutory, chitchat, out-of-scope).
        2. Semantic cache check (if enabled and applicable).
        3. LangGraph Agent execution with MMR Vector Search tool (max 3 query variants).
        4. Subagent 2 verifies candidate answer (faithfulness, citations, negative check).
        5. Refinement loop if verification fails and retries remain.
        6. Structured response generation with verification metrics.

        Args:
            query: User's question regarding KP education laws.
            thread_id: Conversation session identifier for checkpointer memory.
            use_cache: Whether to check semantic cache before LLM call.

        Returns:
            Dict[str, Any]: Structured dictionary corresponding to AgentResponse schema.

        Raises:
            RuntimeError: If agent invocation fails unexpectedly.
        """
        clean_query = query.strip()
        if not clean_query:
            return AgentResponse(
                answer="Please provide a valid question regarding KP education laws.",
                sources=[],
                verification_count=0,
                verification_status="unverified",
                intent="clarification_needed",
                cached=False,
            ).model_dump()

        # Step 1: Subagent 1 - Request Evaluation & Query Classification
        classification: QueryClassification = self.request_evaluator.evaluate(clean_query)
        if not classification.needs_retrieval and classification.direct_response:
            logger.info(
                "RequestEvaluator short-circuited query '%s' with intent '%s'",
                clean_query[:60],
                classification.intent,
            )
            return AgentResponse(
                answer=classification.direct_response,
                sources=[],
                verification_count=0,
                verification_status=classification.intent,
                intent=classification.intent,
                query_variants=[clean_query],
                cached=False,
            ).model_dump()

        # Step 2: Semantic Cache Lookup
        if use_cache and self.cache is not None:
            cached_result = self.cache.get(clean_query)
            if cached_result:
                logger.info("Serving answer from semantic cache for thread %s", thread_id)
                return AgentResponse(
                    answer=cached_result["answer"],
                    sources=cached_result.get("sources", []),
                    verification_count=1,
                    verification_status="verified",
                    verification_critique="Served directly from verified semantic cache.",
                    query_variants=[clean_query],
                    cached=True,
                    intent=classification.intent,
                ).model_dump()

        # Step 3: Agent Execution via LangGraph Checkpointer
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": [HumanMessage(content=clean_query)]}

        try:
            logger.info("Invoking LawAgent for query in thread '%s': '%s'", thread_id, clean_query[:80])
            result = self.agent_executor.invoke(inputs, config=config)

            messages: List[BaseMessage] = result.get("messages", [])
            if not messages:
                logger.error("No messages returned from agent execution.")
                return AgentResponse(
                    answer="Not found in the KP laws provided.",
                    sources=[],
                    verification_count=0,
                    verification_status="unverified",
                    intent=classification.intent,
                    cached=False,
                ).model_dump()

            # Assistant candidate response is the last message
            candidate_answer = str(messages[-1].content).strip()
            sources, context_str = self._extract_sources_and_context(messages)
            query_variants = list(self.retriever.last_query_variants) or [clean_query]

            # Step 4: Subagent 2 - Response Verification & Refinement Loop
            verification_count = 0
            verification_status = "unverified"
            verification_critique: Optional[str] = None
            max_retries = self.settings.features.max_verification_retries

            if self.settings.features.enable_response_verifier:
                verification: VerificationResult = self.response_verifier.verify(
                    query=clean_query,
                    candidate_answer=candidate_answer,
                    retrieved_context=context_str,
                )
                verification_count = 1
                verification_critique = verification.critique

                # Refinement loop if verification failed and retries remain
                while not verification.passed and verification_count < (max_retries + 1):
                    logger.warning(
                        "Response failed verification on attempt %d: %s. Initiating refinement.",
                        verification_count,
                        verification.critique,
                    )
                    refine_prompt = (
                        f"CRITICAL STATUTORY AUDIT FEEDBACK (Attempt {verification_count}):\n"
                        f"Critique: {verification.critique}\n"
                        f"Specific Instructions: {verification.refinement_instructions or 'Ensure every claim has full statutory citation (Law Title, Year, Section, Page) and remove any ungrounded assertions.'}\n\n"
                        f"Please refine your answer to fully address these deficiencies while remaining strictly faithful to the retrieved statutes."
                    )
                    # Invoke agent again in same thread for stateful refinement
                    refine_inputs = {"messages": [HumanMessage(content=refine_prompt)]}
                    refine_result = self.agent_executor.invoke(refine_inputs, config=config)
                    refine_messages: List[BaseMessage] = refine_result.get("messages", [])
                    if refine_messages:
                        candidate_answer = str(refine_messages[-1].content).strip()
                        sources, context_str = self._extract_sources_and_context(refine_messages)

                    verification_count += 1
                    verification = self.response_verifier.verify(
                        query=clean_query,
                        candidate_answer=candidate_answer,
                        retrieved_context=context_str,
                    )
                    verification_critique = verification.critique

                verification_status = "verified" if verification.passed else "refined"
            else:
                verification_status = "unverified"
                verification_count = 0

            # Step 5: Store verified substantive answers in Semantic Cache
            if (
                self.cache is not None
                and candidate_answer
                and "Not found in the KP laws provided" not in candidate_answer
                and verification_status in ("verified", "refined")
            ):
                self.cache.set(query=clean_query, answer=candidate_answer, sources=sources)

            structured_response = AgentResponse(
                answer=candidate_answer,
                sources=sources,
                verification_count=verification_count,
                verification_status=verification_status,
                verification_critique=verification_critique,
                query_variants=query_variants,
                cached=False,
                intent=classification.intent,
            )

            return structured_response.model_dump()

        except Exception as exc:
            logger.error("Error executing LawAgent for query '%s': %s", clean_query, exc, exc_info=True)
            raise RuntimeError(f"LawAgent execution error: {exc}") from exc
