"""Model factory for KP Laws Agent.

Provides initialized chat and embedding models based on the configured provider
(OpenAI, Gemini, or Ollama).
"""

from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from kp_agent.config import Settings, get_settings
from kp_agent.logger import get_logger

logger = get_logger(__name__)


class ModelFactory:
    """Factory class to instantiate chat models and embeddings."""

    @classmethod
    def get_embeddings(
        cls,
        settings: Optional[Settings] = None,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Embeddings:
        """Create and return an embeddings instance for the active provider.

        Args:
            settings: Optional application settings. Defaults to singleton.
            model_name: Optional explicit embedding model name override.
            provider: Optional explicit provider override ('openai', 'gemini', 'ollama').

        Returns:
            Embeddings: Initialized LangChain embeddings instance.

        Raises:
            ValueError: If an unsupported provider is configured or API key is missing.
        """
        cfg = settings or get_settings()
        active_provider = (provider or cfg.provider).lower()

        if active_provider == "openai":
            if not cfg.openai_api_key:
                logger.error("OpenAI API key is missing in environment variables.")
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider.")
            from langchain_openai import OpenAIEmbeddings

            embed_model = model_name or cfg.embedding_model
            logger.info("Initializing OpenAI embeddings with model: %s", embed_model)
            return OpenAIEmbeddings(
                model=embed_model,  # text-embedding-3-large
                api_key=cfg.openai_api_key,
            )

        elif active_provider == "gemini":
            if not cfg.google_api_key:
                logger.error("Google API key is missing in environment variables.")
                raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini provider.")
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            embed_model = model_name or "models/text-embedding-004"
            logger.info("Initializing Gemini embeddings with model: %s", embed_model)
            return GoogleGenerativeAIEmbeddings(
                model=embed_model,  # standard Gemini embedding model
                google_api_key=cfg.google_api_key,
            )

        elif active_provider == "ollama":
            from langchain_community.embeddings import OllamaEmbeddings

            embed_model = model_name or "nomic-embed-text"
            logger.info("Initializing Ollama embeddings at: %s with model: %s", cfg.ollama_base_url, embed_model)
            return OllamaEmbeddings(
                base_url=cfg.ollama_base_url,
                model=embed_model,  # default local embedding model
            )

        else:
            logger.error("Unsupported provider configured: %s", active_provider)
            raise ValueError(f"Unsupported provider '{active_provider}'. Expected 'openai', 'gemini', or 'ollama'.")

    @classmethod
    def get_chat_model(
        cls,
        settings: Optional[Settings] = None,
        temperature: float = 0.0,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> BaseChatModel:
        """Create and return a chat model instance for the active provider.

        Args:
            settings: Optional application settings. Defaults to singleton.
            temperature: Sampling temperature for model generation (0.0 for deterministic legal QA).
            model_name: Optional explicit chat model name override.
            provider: Optional explicit provider override ('openai', 'gemini', 'ollama').

        Returns:
            BaseChatModel: Initialized LangChain chat model instance.

        Raises:
            ValueError: If an unsupported provider is configured or API key is missing.
        """
        cfg = settings or get_settings()
        active_provider = (provider or cfg.provider).lower()

        if active_provider == "openai":
            if not cfg.openai_api_key:
                logger.error("OpenAI API key is missing in environment variables.")
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider.")
            from langchain_openai import ChatOpenAI

            chat_m = model_name or cfg.chat_model
            logger.info("Initializing OpenAI Chat Model: %s (temp=%s)", chat_m, temperature)
            return ChatOpenAI(
                model=chat_m,  # gpt-4o
                temperature=temperature,
                api_key=cfg.openai_api_key,
            )

        elif active_provider == "gemini":
            if not cfg.google_api_key:
                logger.error("Google API key is missing in environment variables.")
                raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini provider.")
            from langchain_google_genai import ChatGoogleGenerativeAI

            chat_m = model_name or "gemini-1.5-pro"
            logger.info("Initializing Gemini Chat Model with model: %s (temp=%s)", chat_m, temperature)
            return ChatGoogleGenerativeAI(
                model=chat_m,
                temperature=temperature,
                google_api_key=cfg.google_api_key,
            )

        elif active_provider == "ollama":
            from langchain_community.chat_models import ChatOllama

            chat_m = model_name or "llama3"
            logger.info("Initializing Ollama Chat Model at: %s with model: %s (temp=%s)", cfg.ollama_base_url, chat_m, temperature)
            return ChatOllama(
                base_url=cfg.ollama_base_url,
                model=chat_m,
                temperature=temperature,
            )

        else:
            logger.error("Unsupported provider configured: %s", active_provider)
            raise ValueError(f"Unsupported provider '{active_provider}'. Expected 'openai', 'gemini', or 'ollama'.")

