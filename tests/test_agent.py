"""Unit tests for ReAct agent creation and execution."""

import pytest
from src.wikirag.indexer import create_index
from src.wikirag.agent import create_react_agent, run_agent
from src.wikirag.mock import MockReActAgent

def test_create_react_agent_mock():
    index = create_index("Batman, Paris", model_name="mock-mode")
    agent = create_react_agent("mock-mode", index)
    assert isinstance(agent, MockReActAgent)

def test_run_agent_execution():
    index = create_index("Batman, Paris", model_name="mock-mode")
    agent = create_react_agent("mock-mode", index)
    response = run_agent(agent, "Tell me about Batman")
    assert "[ReAct Reasoning Trace]" in response
    assert "Thought:" in response
    assert "Action: Wikipedia Search" in response
    assert "Final Answer:" in response

def test_run_agent_none_error():
    with pytest.raises(ValueError):
        run_agent(None, "hello")
