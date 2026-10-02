"""Configuration settings for WikiRAG."""

import os
from typing import List
from pydantic import BaseModel, Field

SUPPORTED_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-3.5-turbo",
    "mock-mode"
]

class WikiRAGConfig(BaseModel):
    """WikiRAG Application Configuration."""
    model_name: str = Field(default="gpt-4o-mini", description="LLM model identifier")
    chunk_size: int = Field(default=256, description="Text chunk size in tokens/words")
    chunk_overlap: int = Field(default=32, description="Overlap between consecutive chunks")
    similarity_top_k: int = Field(default=4, description="Number of top similar passages to retrieve")
    openai_api_key: str = Field(default="", description="OpenAI API key")

    @classmethod
    def load_from_env(cls, model_name: str = "gpt-4o-mini") -> "WikiRAGConfig":
        """Load configuration from environment variables."""
        api_key = os.getenv("OPENAI_API_KEY", "")
        # If no API key is present and model is not specified, default to mock-mode if requested
        return cls(
            model_name=model_name,
            openai_api_key=api_key
        )

def get_api_key() -> str:
    """Retrieve OpenAI API key from environment."""
    return os.getenv("OPENAI_API_KEY", "")
