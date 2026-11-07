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


@dataclass
class EvaluationModelsConfig:
    """Model selection for the evaluation judge and embeddings."""

    provider: Optional[str] = None
    judge_model: Optional[str] = None
    embedding_model: Optional[str] = None
    temperature: float = 0.0


@dataclass
class EvaluationFeatureFlags:
    """Metric feature flags for Ragas."""

    enable_faithfulness: bool = True
    enable_answer_relevance: bool = True
    enable_context_recall: bool = True
    enable_answer_correctness: bool = True
    enable_context_precision: bool = False
    enable_semantic_similarity: bool = False


@dataclass
class RagasRunConfigSettings:
    """Runtime execution settings for Ragas evaluation."""

    max_workers: int = 2
    timeout: int = 90
    max_retries: int = 3
    batch_size: int = 5
    raise_exceptions: bool = False
    show_progress: bool = True


@dataclass
class EvaluationBenchmarkConfig:
    """Benchmark dataset and output file configuration."""

    test_set_path: str = "data/test_set.json"
    output_dir: str = "logs"
    output_json: str = "logs/ragas_eval_results.json"
    output_csv: str = "logs/ragas_eval_results.csv"
    default_sample_limit: Optional[int] = None
    save_detailed_reports: bool = True


@dataclass
class EvaluationConfig:
    """Top-level evaluation configuration loaded from evaluation.yaml."""

    models: EvaluationModelsConfig = field(default_factory=EvaluationModelsConfig)
    feature_flags: EvaluationFeatureFlags = field(default_factory=EvaluationFeatureFlags)
    ragas_config: RagasRunConfigSettings = field(default_factory=RagasRunConfigSettings)
    benchmark: EvaluationBenchmarkConfig = field(default_factory=EvaluationBenchmarkConfig)

    @classmethod
    def load_from_yaml(cls, yaml_path: Path) -> "EvaluationConfig":
        """Load evaluation configuration from evaluation.yaml.

        Args:
            yaml_path: Filesystem path to evaluation.yaml.

        Returns:
            EvaluationConfig: Parsed evaluation configuration.
        """
        if not yaml_path.exists():
            return cls()

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data: Dict[str, Any] = yaml.safe_load(f) or {}

            eval_data = data.get("evaluation", {})
            models_data = eval_data.get("models", {})
            flags_data = eval_data.get("feature_flags", {})
            ragas_data = eval_data.get("ragas_config", {})
            bench_data = eval_data.get("benchmark", {})

            return cls(
                models=EvaluationModelsConfig(
                    provider=models_data.get("provider"),
                    judge_model=models_data.get("judge_model"),
                    embedding_model=models_data.get("embedding_model"),
                    temperature=float(models_data.get("temperature", 0.0)),
                ),
                feature_flags=EvaluationFeatureFlags(
                    enable_faithfulness=bool(flags_data.get("enable_faithfulness", True)),
                    enable_answer_relevance=bool(flags_data.get("enable_answer_relevance", True)),
                    enable_context_recall=bool(flags_data.get("enable_context_recall", True)),
                    enable_answer_correctness=bool(flags_data.get("enable_answer_correctness", True)),
                    enable_context_precision=bool(flags_data.get("enable_context_precision", False)),
                    enable_semantic_similarity=bool(flags_data.get("enable_semantic_similarity", False)),
                ),
                ragas_config=RagasRunConfigSettings(
                    max_workers=int(ragas_data.get("max_workers", 2)),
                    timeout=int(ragas_data.get("timeout", 90)),
                    max_retries=int(ragas_data.get("max_retries", 3)),
                    batch_size=int(ragas_data.get("batch_size", 5)),
                    raise_exceptions=bool(ragas_data.get("raise_exceptions", False)),
                    show_progress=bool(ragas_data.get("show_progress", True)),
                ),
                benchmark=EvaluationBenchmarkConfig(
                    test_set_path=str(bench_data.get("test_set_path", "data/test_set.json")),
                    output_dir=str(bench_data.get("output_dir", "logs")),
                    output_json=str(bench_data.get("output_json", "logs/ragas_eval_results.json")),
                    output_csv=str(bench_data.get("output_csv", "logs/ragas_eval_results.csv")),
                    default_sample_limit=bench_data.get("default_sample_limit"),
                    save_detailed_reports=bool(bench_data.get("save_detailed_reports", True)),
                ),
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
        self.evaluation_file: Path = self.base_dir / "evaluation.yaml"

        # Load feature controls from YAML
        self.features: FeatureFlags = FeatureFlags.load_from_yaml(self.features_file)

        # Load evaluation configuration from YAML
        self.evaluation: EvaluationConfig = EvaluationConfig.load_from_yaml(self.evaluation_file)


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

