"""LawAgent module for KP Education Laws Assistant.

Constructs the conversational legal agent using LangChain/LangGraph create_agent,
integrating the Qdrant retrieval tool, memory checkpointer, semantic caching,
and legal citation enforcement.
"""

from typing import Any, Dict, List, Optional
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
from kp_agent.tools import LawRetriever, get_retriever, search_laws

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an authoritative legal research assistant specialized in Khyber Pakhtunkhwa (KP) education laws (1897–2020).

Strict Behavioral Guidelines:
1. Retrieval First: Always invoke the `search_laws` tool before answering any substantive question. Never answer from internal knowledge alone.
2. Citation Requirement: Every single factual claim, quota, penalty, quorum, power, deadline, or requirement MUST cite:
   - Full law title (e.g. "Khyber Pakhtunkhwa School Bags (Limitation of Weight) Act, 2020")
   - Year of enactment (e.g. "2020")
   - Statutory section, subsection, or schedule (e.g. "Section 3(2)", "Schedule", "Section 22(1)")
   - Page number where available.
3. Negative Retrieval: If the retrieved provisions do not contain the answer, or if the question is unanswerable from the provided KP laws, reply exactly:
   "Not found in the KP laws provided."
4. Scope Restriction: Politely decline any question that is unrelated to Khyber Pakhtunkhwa education laws and institutions.
5. Accuracy & Precision: Do not assume or extrapolate legal facts beyond what the cited statutory text specifies.
"""


class LawAgent:
    """Agent orchestrator for Khyber Pakhtunkhwa education laws Q&A."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        retriever: Optional[LawRetriever] = None,
        enable_cache: bool = True,
    ) -> None:
        """Initialize the legal agent, models, checkpointer, and tools.

        Args:
            settings: Application settings. Defaults to singleton.
            retriever: Optional LawRetriever instance.
            enable_cache: Whether to enable semantic query caching.
        """
        self.settings = settings or get_settings()
        self.retriever = retriever or get_retriever()
        self.enable_cache = enable_cache

        # Memory checkpointer for multi-turn conversational state
        self.checkpointer = MemorySaver()

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
            # LangGraph create_react_agent supports checkpointer and prompt
            self.agent_executor = create_agent(
                model=self.chat_model,
                tools=self.tools,
                checkpointer=self.checkpointer,
                prompt=SYSTEM_PROMPT,
            )
        except TypeError:
            # Fallback signature if prompt keyword varies across versions
            self.agent_executor = create_agent(
                self.chat_model,
                self.tools,
                checkpointer=self.checkpointer,
                state_modifier=SYSTEM_PROMPT,
            )

    def _extract_sources_from_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Extract citations and retrieved chunk metadata from ToolMessages in execution history.

        Args:
            messages: Full message trajectory from agent invocation.

        Returns:
            List[Dict[str, Any]]: Extracted statutory source items.
        """
        sources: List[Dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg, ToolMessage) or getattr(msg, "type", "") == "tool":
                content = str(msg.content)
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

        return sources

    def ask(
        self,
        query: str,
        thread_id: str = "default",
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Ask a legal question and receive a cited response with sources.

        Args:
            query: User's question regarding KP education laws.
            thread_id: Conversation session identifier for memory state.
            use_cache: Whether to check semantic cache before LLM call.

        Returns:
            Dict[str, Any]: Response containing 'answer', 'sources', and 'cached' flag.

        Raises:
            RuntimeError: If agent invocation fails unexpectedly.
        """
        clean_query = query.strip()
        if not clean_query:
            return {
                "answer": "Please provide a valid question regarding KP education laws.",
                "sources": [],
                "cached": False,
            }

        # Check semantic cache if enabled
        if use_cache and self.cache is not None:
            cached_result = self.cache.get(clean_query)
            if cached_result:
                logger.info("Serving answer from semantic cache for thread %s", thread_id)
                return {
                    "answer": cached_result["answer"],
                    "sources": cached_result.get("sources", []),
                    "cached": True,
                }

        # Prepare invocation inputs and checkpoint configuration
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": [HumanMessage(content=clean_query)]}

        try:
            logger.info("Invoking LawAgent for query in thread '%s': '%s'", thread_id, clean_query[:80])
            result = self.agent_executor.invoke(inputs, config=config)

            messages: List[BaseMessage] = result.get("messages", [])
            if not messages:
                logger.error("No messages returned from agent execution.")
                return {
                    "answer": "Not found in the KP laws provided.",
                    "sources": [],
                    "cached": False,
                }

            # Assistant response is the last message
            last_message = messages[-1]
            answer_text = str(last_message.content).strip()

            # Extract source citations from tool calls during this turn
            sources = self._extract_sources_from_messages(messages)

            # Store verified answer in semantic cache if meaningful
            if self.cache is not None and answer_text and "Not found" not in answer_text:
                self.cache.set(query=clean_query, answer=answer_text, sources=sources)

            return {
                "answer": answer_text,
                "sources": sources,
                "cached": False,
            }

        except Exception as exc:
            logger.error("Error executing LawAgent for query '%s': %s", clean_query, exc, exc_info=True)
            raise RuntimeError(f"LawAgent execution error: {exc}") from exc

