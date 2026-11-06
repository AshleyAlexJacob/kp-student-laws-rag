"""KP Laws Agent package initialization."""

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
