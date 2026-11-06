"""Configuration module for KP Laws Agent.

Defines the Settings class which loads configuration from environment variables
and `.env` files, supporting provider switches (openai, gemini, ollama),
Qdrant vector database parameters, and filesystem paths.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
import yaml

# Load environment variables from .env file if present
load_dotenv()


@dataclass
class FeatureFlags:
    """Feature controls for subagents, retrieval pipeline, and caching.

    Values are loaded from features.yaml with fallbacks to defaults.
    """

    enable_request_evaluator: bool = True
    enable_query_rewriter: bool = True
    enable_mmr_retrieval: bool = True
    enable_response_verifier: bool = True
    enable_semantic_cache: bool = True
    max_query_variants: int = 3
    max_verification_retries: int = 2
    mmr_diversity_lambda: float = 0.7
    retrieval_limit: int = 5

    @classmethod
    def load_from_yaml(cls, yaml_path: Path) -> "FeatureFlags":
        """Load feature flags from a YAML configuration file.

        Args:
            yaml_path: Filesystem path to features.yaml.

        Returns:
            FeatureFlags: Configured feature flags instance.
        """
        if not yaml_path.exists():
            return cls()

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data: Dict[str, Any] = yaml.safe_load(f) or {}

            raw_features: Dict[str, Any] = data.get("features", {})
            return cls(
                enable_request_evaluator=bool(raw_features.get("enable_request_evaluator", True)),
                enable_query_rewriter=bool(raw_features.get("enable_query_rewriter", True)),
                enable_mmr_retrieval=bool(raw_features.get("enable_mmr_retrieval", True)),
                enable_response_verifier=bool(raw_features.get("enable_response_verifier", True)),
                enable_semantic_cache=bool(raw_features.get("enable_semantic_cache", True)),
                max_query_variants=int(raw_features.get("max_query_variants", 3)),
                max_verification_retries=int(raw_features.get("max_verification_retries", 2)),
                mmr_diversity_lambda=float(raw_features.get("mmr_diversity_lambda", 0.7)),
                retrieval_limit=int(raw_features.get("retrieval_limit", 5)),
            )
        except Exception:
            return cls()


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
        self.features_file: Path = self.base_dir / "features.yaml"

        # Load feature controls from YAML
        self.features: FeatureFlags = FeatureFlags.load_from_yaml(self.features_file)


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

