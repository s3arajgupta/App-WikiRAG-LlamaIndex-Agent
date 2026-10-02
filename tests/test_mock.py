"""Unit tests for offline mock provider."""

import pytest
from src.wikirag.mock import (
    get_mock_documents,
    MockDocument,
    MockQueryEngine,
    MockIndex,
    MockReActAgent
)

def test_get_mock_documents():
    docs = get_mock_documents(["Paris", "Batman"])
    assert len(docs) == 2
    assert "capital" in docs[0].get_content().lower()
    assert "superhero" in docs[1].get_content().lower()

def test_mock_query_engine_keyword_search():
    docs = [MockDocument(text="Paris has the Eiffel Tower and the Louvre Museum.")]
    engine = MockQueryEngine(docs)
    res = engine.query("Where is the Eiffel Tower?")
    assert "Eiffel Tower" in str(res)

def test_mock_react_agent_reasoning():
    docs = [MockDocument(text="Python was created by Guido van Rossum.")]
    index = MockIndex(["Python"], docs)
    agent = MockReActAgent(index)
    output = agent.chat("Who created Python?")
    assert "Guido van Rossum" in output
    assert "ReAct Reasoning Trace" in output
