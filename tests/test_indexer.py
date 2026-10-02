"""Unit tests for WikiRAG query parsing and indexing."""

import pytest
from src.wikirag.indexer import parse_wikipage_query, create_index
from src.wikirag.mock import MockIndex

def test_parse_wikipage_query_comma_separated():
    pages = parse_wikipage_query("Paris, Batman, Python")
    assert pages == ["Paris", "Batman", "Python"]

def test_parse_wikipage_query_with_prefix():
    pages1 = parse_wikipage_query("please index: Tokyo, Kyoto")
    assert pages1 == ["Tokyo", "Kyoto"]

    pages2 = parse_wikipage_query("/get wikipages: Berlin, Munich")
    assert pages2 == ["Berlin", "Munich"]

def test_parse_wikipage_query_empty():
    assert parse_wikipage_query("") == []
    assert parse_wikipage_query("   ") == []

def test_create_index_mock_mode():
    index = create_index("Paris, Batman", model_name="mock-mode")
    assert isinstance(index, MockIndex)
    assert "Paris" in index.pages
    assert "Batman" in index.pages
    assert len(index.documents) == 2

def test_create_index_empty_error():
    with pytest.raises(ValueError):
        create_index("", model_name="mock-mode")
