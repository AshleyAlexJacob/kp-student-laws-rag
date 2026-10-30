# AGENTS.md - KP Education Laws RAG Agent

## Project
A chat agent that answers questions about 23 Khyber Pakhtunkhwa education laws (1897-2020).
Every answer cites the law title, year and section or page. If the laws do not cover a question, the agent says so.

## Stack (do not change without asking)
- Python 3.11+, package management with **uv** only (no pip, no requirements.txt)
- **LangChain v1 `create_agent`** for the agent; **LangGraph** only when a fixed multi-step flow is needed
- **Qdrant local mode**: `QdrantClient(path="qdrant_db")`, collection `kp_laws`
- **Streamlit** UI that imports the agent directly (no backend, no API server)
- `pypdf` for PDF text, `python-dotenv` for keys
- Provider switch in `config.py`: `openai` (default: `gpt-4o`, `text-embedding-3-small`) | `gemini` | `ollama`

## Folder structure
```
kp-laws-agent/
├── AGENTS.md
├── pyproject.toml          # managed by uv
├── .env.example            # OPENAI_API_KEY=, GOOGLE_API_KEY=
├── app.py                  # Streamlit chat UI
├── data/
│   ├── laws/               # the 23 law PDFs
│   ├── metadata.csv        # file, title, year, level, source_url
│   └── test_set.json       # evaluation questions + ground truths
├── kp_agent/
│   ├── __init__.py
│   ├── config.py           # Settings class: provider, model names, paths
│   ├── logger.py           # get_logger(name) -> configured logger
│   ├── models.py           # ModelFactory: chat model + embeddings for the provider
│   ├── ingest.py           # LawIngestor: load PDFs + metadata, flag scanned pages, split, index
│   ├── tools.py            # search_laws tool (query, level, year_from filters)
│   ├── agent.py            # LawAgent: create_agent + checkpointer, ask()
│   ├── cache.py            # SemanticCache on a second Qdrant collection
│   └── evaluator.py        # RagEvaluator: faithfulness, relevance, context recall, correctness
├── qdrant_db/              # generated, git-ignored
└── logs/                   # generated, git-ignored
```
Do not add new folders or files outside this structure without asking.

## Commands
```bash
uv sync                                   # install dependencies
uv add <package>                          # add a dependency
uv run python -m kp_agent.ingest          # build or reuse the Qdrant index
uv run python -m kp_agent.evaluator       # run the test set and print scores
uv run streamlit run app.py               # start the UI
```

## Coding rules
- **Class-based** design for reuse: one main class per module, small focused methods.
- **PEP 8**, type hints on every function, and a **docstring** (Google style) on every module, class and public method.
- **One-line comment** above or beside every non-obvious line, especially where a parameter is set (e.g. `k=4  # chunks returned per query`).
- **Logging**: use `kp_agent.logger.get_logger(__name__)`; never `print`. Log to console and `logs/app.log`. INFO for steps, WARNING for scanned pages and cache misses, ERROR with `exc_info=True` for failures.
- **Exception handling**: catch specific exceptions (no bare `except`), log them, and raise a clear message. The UI shows a friendly error with `st.error`, never a traceback.
- Read keys from `.env`; never hard-code or log API keys.
- Keep code simple and readable for students; no extra frameworks (no FastAPI, Docker, ORMs).

## Agent behaviour
- Always call `search_laws` before answering; never answer from model memory.
- Cite law title, year and section or page for every claim.
- If nothing relevant is retrieved, answer: "Not found in the KP laws provided."
- Politely decline questions unrelated to KP education laws.

## Data rules
- Attach `title`, `year`, `level`, `source_url` and `page` to every chunk.
- Pages with fewer than 50 characters of text are scanned: log them as WARNING and list them for OCR.
- Skip re-indexing when the `kp_laws` collection already exists.

## Streamlit rules
- Create `LawAgent` once with `@st.cache_resource` (Qdrant local mode allows only one client per folder).
- Keep chat history and `thread_id` in `st.session_state`.
- Show sources under each answer in an `st.expander`.

## Done means
- Code runs with `uv run`, has docstrings, comments, logging and error handling.
- After changing retrieval, prompts or models, run `uv run python -m kp_agent.evaluator` and report the scores.