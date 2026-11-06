"""Configuration module for KP Laws Agent.

Defines the Settings class which loads configuration from environment variables
and `.env` files, supporting provider switches (openai, gemini, ollama),
Qdrant vector database parameters, and filesystem paths.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


class Settings:
    """Application settings and configuration parameters."""

    def __init__(self) -> None:
        """Initialize settings from environment variables with sensible defaults."""
        # Active model provider ('openai', 'gemini', or 'ollama')
        self.provider: str = os.getenv("PROVIDER", "openai").lower()

        # API Keys
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
        self.ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Qdrant Vector Database configurations
        self.qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY", "dev@4321")
        self.qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "kp_laws")

        # Model names
        self.chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o")
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

        # Embedding dimension (3072 for OpenAI text-embedding-3-large)
        self.embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "3072"))

        # Base filesystem paths
        self.base_dir: Path = Path(__file__).resolve().parent.parent
        self.data_dir: Path = self.base_dir / "data"
        self.laws_dir: Path = self.data_dir / "laws"
        self.metadata_file: Path = self.data_dir / "metadata.csv"
        self.test_set_file: Path = self.data_dir / "test_set.json"
        self.logs_dir: Path = self.base_dir / "logs"


# Cached singleton instance
_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """Retrieve the cached application settings instance.

    Returns:
        Settings: Configured application settings.
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance

