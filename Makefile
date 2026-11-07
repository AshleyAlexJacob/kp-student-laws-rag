.PHONY: help dev-setup app ingest eval evaluate clean

# Default python and virtual environment path
VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
STREAMLIT = $(VENV_DIR)/bin/streamlit

help:
	@echo "Available commands:"
	@echo "  make dev-setup  : Create virtual environment and install dependencies"
	@echo "  make app        : Run the Streamlit application"
	@echo "  make ingest     : Parse laws and index embeddings into Qdrant"
	@echo "  make eval       : Run the RAG test set evaluation benchmark"
	@echo "  make evaluate   : Run the RAG test set evaluation benchmark with Ragas"
	@echo "  make clean      : Remove virtual environment and cached files"

dev-setup:
	@echo "Setting up development environment..."
	@if command -v uv >/dev/null 2>&1; then \
		echo "Detected 'uv'. Syncing dependencies with uv..."; \
		uv sync; \
	else \
		echo "'uv' not found. Creating standard Python virtual environment in $(VENV_DIR)..."; \
		python3 -m venv $(VENV_DIR); \
		$(VENV_DIR)/bin/pip install --upgrade pip; \
		$(VENV_DIR)/bin/pip install -e .; \
	fi
	@echo "Development setup complete. Virtual environment ready in $(VENV_DIR)."

app:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment not found. Running make dev-setup first..."; \
		$(MAKE) dev-setup; \
	fi
	@echo "Starting Streamlit app..."
	@$(STREAMLIT) run app.py

ingest:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment not found. Running make dev-setup first..."; \
		$(MAKE) dev-setup; \
	fi
	@echo "Running document ingestion and vector indexing into Qdrant..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run python -m kp_agent.ingest; \
	else \
		$(PYTHON) -m kp_agent.ingest; \
	fi

evaluate:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment not found. Running make dev-setup first..."; \
		$(MAKE) dev-setup; \
	fi
	@echo "Running RAG benchmark evaluation with Ragas against test set..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run python -m kp_agent.evaluator; \
	else \
		$(PYTHON) -m kp_agent.evaluator; \
	fi

eval: evaluate

clean:
	@echo "Cleaning environment and cache..."
	@rm -rf $(VENV_DIR)
	@rm -rf __pycache__ kp_agent/__pycache__ schemas/__pycache__ ui/__pycache__
	@rm -rf logs/*.log
	@echo "Clean completed."
