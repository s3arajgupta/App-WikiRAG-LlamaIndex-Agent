"""Unit tests for WikiRAG configuration."""

import pytest
from src.wikirag.config import WikiRAGConfig, SUPPORTED_MODELS, get_api_key

def test_default_config():
    config = WikiRAGConfig()
    assert config.model_name == "gpt-4o-mini"
    assert config.chunk_size == 256
    assert config.chunk_overlap == 32
    assert config.similarity_top_k == 4

def test_supported_models():
    assert "gpt-4o-mini" in SUPPORTED_MODELS
    assert "gpt-4o" in SUPPORTED_MODELS
    assert "mock-mode" in SUPPORTED_MODELS

def test_load_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-12345")
    config = WikiRAGConfig.load_from_env(model_name="mock-mode")
    assert config.model_name == "mock-mode"
    assert config.openai_api_key == "sk-test-key-12345"
    assert get_api_key() == "sk-test-key-12345"
