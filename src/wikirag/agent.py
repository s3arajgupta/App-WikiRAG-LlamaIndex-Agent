"""Autonomous ReAct Agent for Multi-Hop Wikipedia RAG."""

from typing import Any, Optional
from .config import WikiRAGConfig, get_api_key
from .mock import MockIndex, MockReActAgent

def create_react_agent(model_name: str, index: Any, config: Optional[WikiRAGConfig] = None) -> Any:
    """Create a ReAct Agent connected to the Wikipedia QueryEngine tool."""
    cfg = config or WikiRAGConfig.load_from_env(model_name=model_name)

    # If mock index or mock-mode requested or missing key, return MockReActAgent
    if isinstance(index, MockIndex) or model_name == "mock-mode" or not get_api_key():
        return MockReActAgent(index=index if isinstance(index, MockIndex) else MockIndex([], []))

    try:
        # Modern llama_index (v0.10+)
        try:
            from llama_index.core.tools import QueryEngineTool, ToolMetadata
            from llama_index.core.agent import ReActAgent
            from llama_index.llms.openai import OpenAI

            query_engine = index.as_query_engine(
                similarity_top_k=cfg.similarity_top_k,
                verbose=True
            )
            tools = [
                QueryEngineTool(
                    query_engine=query_engine,
                    metadata=ToolMetadata(
                        name="wikipedia_search",
                        description="Useful for querying indexed Wikipedia pages for verified factual knowledge.",
                    ),
                )
            ]
            llm = OpenAI(model=model_name)
            agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True)
            return agent
        except ImportError:
            # Legacy llama_index (v0.8-v0.9)
            from llama_index.tools import QueryEngineTool, ToolMetadata
            from llama_index.agent import ReActAgent
            from llama_index.llms import OpenAI as LlamaOpenAI

            query_engine = index.as_query_engine(
                similarity_top_k=cfg.similarity_top_k,
                verbose=True
            )
            tools = [
                QueryEngineTool(
                    query_engine=query_engine,
                    metadata=ToolMetadata(
                        name="wikipedia_search",
                        description="Useful for querying indexed Wikipedia pages for verified factual knowledge.",
                    ),
                )
            ]
            llm = LlamaOpenAI(model=model_name)
            agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True)
            return agent
    except Exception as e:
        print(f"Warning: Live ReActAgent initialization failed ({e}). Falling back to MockReActAgent.")
        return MockReActAgent(index=index if isinstance(index, MockIndex) else MockIndex([], []))

def run_agent(agent: Any, message: str) -> str:
    """Execute a query through the ReAct agent and return its answer."""
    if not agent:
        raise ValueError("Agent has not been initialized.")

    try:
        response = agent.chat(message)
        return str(response)
    except Exception as e:
        return f"Error executing agent query: {e}"
