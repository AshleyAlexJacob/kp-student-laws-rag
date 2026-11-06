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
    def get_embeddings(cls, settings: Optional[Settings] = None) -> Embeddings:
        """Create and return an embeddings instance for the active provider.

        Args:
            settings: Optional application settings. Defaults to singleton.

        Returns:
            Embeddings: Initialized LangChain embeddings instance.

        Raises:
            ValueError: If an unsupported provider is configured or API key is missing.
        """
        cfg = settings or get_settings()
        provider = cfg.provider

        if provider == "openai":
            if not cfg.openai_api_key:
                logger.error("OpenAI API key is missing in environment variables.")
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider.")
            from langchain_openai import OpenAIEmbeddings

            logger.info("Initializing OpenAI embeddings with model: %s", cfg.embedding_model)
            return OpenAIEmbeddings(
                model=cfg.embedding_model,  # text-embedding-3-large
                api_key=cfg.openai_api_key,
            )

        elif provider == "gemini":
            if not cfg.google_api_key:
                logger.error("Google API key is missing in environment variables.")
                raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini provider.")
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            logger.info("Initializing Gemini embeddings.")
            return GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",  # standard Gemini embedding model
                google_api_key=cfg.google_api_key,
            )

        elif provider == "ollama":
            from langchain_community.embeddings import OllamaEmbeddings

            logger.info("Initializing Ollama embeddings at: %s", cfg.ollama_base_url)
            return OllamaEmbeddings(
                base_url=cfg.ollama_base_url,
                model="nomic-embed-text",  # default local embedding model
            )

        else:
            logger.error("Unsupported provider configured: %s", provider)
            raise ValueError(f"Unsupported provider '{provider}'. Expected 'openai', 'gemini', or 'ollama'.")

    @classmethod
    def get_chat_model(cls, settings: Optional[Settings] = None, temperature: float = 0.0) -> BaseChatModel:
        """Create and return a chat model instance for the active provider.

        Args:
            settings: Optional application settings. Defaults to singleton.
            temperature: Sampling temperature for model generation (0.0 for deterministic legal QA).

        Returns:
            BaseChatModel: Initialized LangChain chat model instance.

        Raises:
            ValueError: If an unsupported provider is configured or API key is missing.
        """
        cfg = settings or get_settings()
        provider = cfg.provider

        if provider == "openai":
            if not cfg.openai_api_key:
                logger.error("OpenAI API key is missing in environment variables.")
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider.")
            from langchain_openai import ChatOpenAI

            logger.info("Initializing OpenAI Chat Model: %s (temp=%s)", cfg.chat_model, temperature)
            return ChatOpenAI(
                model=cfg.chat_model,  # gpt-4o
                temperature=temperature,
                api_key=cfg.openai_api_key,
            )

        elif provider == "gemini":
            if not cfg.google_api_key:
                logger.error("Google API key is missing in environment variables.")
                raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini provider.")
            from langchain_google_genai import ChatGoogleGenerativeAI

            logger.info("Initializing Gemini Chat Model.")
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=temperature,
                google_api_key=cfg.google_api_key,
            )

        elif provider == "ollama":
            from langchain_community.chat_models import ChatOllama

            logger.info("Initializing Ollama Chat Model at: %s", cfg.ollama_base_url)
            return ChatOllama(
                base_url=cfg.ollama_base_url,
                model="llama3",
                temperature=temperature,
            )

        else:
            logger.error("Unsupported provider configured: %s", provider)
            raise ValueError(f"Unsupported provider '{provider}'. Expected 'openai', 'gemini', or 'ollama'.")

