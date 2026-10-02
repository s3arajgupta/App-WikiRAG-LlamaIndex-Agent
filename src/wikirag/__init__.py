"""WikiRAG: Autonomous ReAct Agent & Multi-Topic Wikipedia RAG System."""

from .config import WikiRAGConfig, SUPPORTED_MODELS, get_api_key
from .indexer import parse_wikipage_query, create_index
from .agent import create_react_agent, run_agent
from .mock import MockIndex, MockReActAgent

__version__ = "2.0.0"
__all__ = [
    "WikiRAGConfig",
    "SUPPORTED_MODELS",
    "get_api_key",
    "parse_wikipage_query",
    "create_index",
    "create_react_agent",
    "run_agent",
    "MockIndex",
    "MockReActAgent",
]
