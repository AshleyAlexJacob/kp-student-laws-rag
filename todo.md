# Tasks - UI: Dark Mode, Sessions, Chat Screen
- [x] Create project environment and Makefile targets (`make dev-setup`, `make app`)
- [x] Configure minimal dark mode theme with Teal accent (`.streamlit/config.toml`)
- [x] Set up project structure and logging (`pyproject.toml`, `kp_agent/logger.py`)
- [x] Implement Session management and Chat screen UI in `app.py`
- [x] Run `make dev-setup` to configure virtual environment and dependencies
- [x] Modularize `schemas/` (`Message` dataclass in `schemas/chat.py`)
- [x] Modularize `ui/` (`SessionStore` in `ui/session.py`, `inject_styles` and CSS in `ui/css.py`)
- [x] Verify execution with `make app` and test syntax

# Tasks - Vector Database Ingestion Phase
- [x] Configure dependencies in `pyproject.toml` (Qdrant, OpenAI, pypdf, python-dotenv, LangChain)
- [x] Configure `.env.example` with Qdrant Docker (`http://localhost:6333`, key `smit-peshawar-123`) and OpenAI `text-embedding-3-large`
- [x] Setup canonical `data/` layout per AGENTS.md (`data/laws/`, `data/metadata.csv`, `data/test_set.json`)
- [x] Implement `Settings` class in `kp_agent/config.py`
- [x] Implement `ModelFactory` in `kp_agent/models.py` (embeddings & chat models for providers)
- [x] Implement `LegalChunk`, `LegalDocParser`, and `LawIngestor` in `kp_agent/ingest.py`:
  - Hierarchical structure preservation (Metadata, Preamble, Sections, Subsections, Clauses, Schedules, Tables)
  - Definitions association retention
  - Schedule tables structured formatting
  - Filter TOC & headers/footers
  - Scanned page detection (< 50 chars flagged as WARNING for OCR)
  - Context header prepending
  - Qdrant Docker indexing with vector size 3072 and payload indexes
- [x] Add `make ingest` target to `Makefile`

# Tasks - Phase 3 Agent: Vector Search Tool, Agent Loop / Graph, and RAG Evaluator
- [x] Fix `pyproject.toml` wheel build target configuration for `kp_agent` package
- [x] Implement `LawRetriever` class and `search_laws` tool (`kp_agent/tools.py`):
  - Qdrant similarity search with query dense embeddings
  - Metadata filtering by `level` and `year_from`
  - Statutory citation formatting (law title, year, section, subsection, page)
- [x] Implement `SemanticCache` class (`kp_agent/cache.py`):
  - Dedicated Qdrant collection `kp_laws_cache` with cosine similarity
  - Fast semantic lookup with threshold and caching of verified statutory answers
- [x] Implement `LawAgent` class (`kp_agent/agent.py`):
  - Agent creation via LangChain / LangGraph with checkpointer memory (`MemorySaver`)
  - Strict system prompt enforcing statutory citation and negative retrieval ("Not found in the KP laws provided.")
  - Multi-turn thread state and citation extraction
- [x] Implement `RagEvaluator` class (`kp_agent/evaluator.py`):
  - Evaluation of faithfulness, answer relevance, context recall, and statutory correctness
  - Benchmark execution against 37 canonical questions in `data/test_set.json`
  - Aggregate and per-question-type reporting, saving to `logs/eval_results.json`
- [x] Connect `LawAgent` into Streamlit `app.py`:
  - Cached agent via `@st.cache_resource`
  - Source expander under assistant responses displaying statutory citations
  - Friendly error handling via `st.error`
- [x] Verify `make dev-setup`, `uv run python -m kp_agent.evaluator`, and app execution

# Tasks - Phase 4: Agentic RAG with Subagents, Vector Search MMR & Query Rewriter, Checkpointers
- [x] Create `features.yaml` with subagent stages switchable via YAML
- [x] Update `kp_agent/config.py` to parse `features.yaml` and support feature toggles
- [x] Create `schemas/agent.py` for structured responses and update `schemas/chat.py`
- [x] Upgrade `kp_agent/tools.py` with `QueryRewriter` (max 3 variants) and client-side MMR reranking
- [x] Create `kp_agent/subagents.py` with `RequestEvaluator` (query classifier & router) and `ResponseVerifier` (faithfulness & citation verifier)
- [x] Upgrade `kp_agent/agent.py` with LangGraph Agentic RAG workflow, verification loop, checkpointer, and structured response
- [x] Update Streamlit `app.py` and `ui/session.py` to display verification badges, query rewrites expander, and multi-turn state
- [x] Test and verify all components and YAML toggles

# Tasks - Phase 5: Ragas Evaluation Framework Integration
- [x] Create `evaluation.yaml` with metric feature flags, model selection, timeouts, and concurrency tweaks
- [x] Update `kp_agent/config.py` to parse `evaluation.yaml` (`EvaluationConfig`)
- [x] Update `kp_agent/models.py` (`ModelFactory`) to support explicit model and provider overrides
- [x] Update `kp_agent/evaluator.py` (`RagEvaluator`) to evaluate benchmark test set using native Ragas dataset and metrics
- [x] Add `make evaluate` target to `Makefile` with `eval` alias
- [x] Integrate Ragas evaluation pipeline with test set, configuration, and Makefile target
