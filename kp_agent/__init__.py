"""KP Laws Agent package initialization."""

import sys
import types

# Compatibility shim for ragas 0.3.1 importing deprecated langchain_community vertexai
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _v_chat_mod = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:
        """Compatibility dummy for removed langchain_community ChatVertexAI."""
        pass

    _v_chat_mod.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _v_chat_mod

from kp_agent.agent import LawAgent
from kp_agent.config import FeatureFlags, Settings, get_settings
from kp_agent.logger import get_logger
from kp_agent.models import ModelFactory
from kp_agent.subagents import RequestEvaluator, ResponseVerifier
from kp_agent.tools import LawRetriever, QueryRewriter, compute_mmr, get_retriever, search_laws

__all__ = [
    "LawAgent",
    "LawRetriever",
    "QueryRewriter",
    "RequestEvaluator",
    "ResponseVerifier",
    "FeatureFlags",
    "Settings",
    "ModelFactory",
    "compute_mmr",
    "get_logger",
    "get_retriever",
    "get_settings",
    "search_laws",
]
