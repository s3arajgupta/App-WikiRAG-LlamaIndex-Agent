"""Wikipedia Content Fetching and Vector Indexing for WikiRAG."""

import re
from typing import List, Optional, Any
from .config import WikiRAGConfig, get_api_key
from .mock import MockIndex, get_mock_documents

def parse_wikipage_query(query: str) -> List[str]:
    """Parse comma-separated or command-style Wikipedia page requests.
    
    Examples:
        - "Paris, Batman, Python" -> ["Paris", "Batman", "Python"]
        - "please index: Paris, Tokyo" -> ["Paris", "Tokyo"]
        - "/get wikipages: Paris, Lagos" -> ["Paris", "Lagos"]
    """
    if not query:
        return []

    # Strip command prefixes if present
    cleaned = query
    for prefix in ["/get wikipages:", "please index:", "index:"]:
        if prefix.lower() in cleaned.lower():
            idx = cleaned.lower().index(prefix.lower())
            cleaned = cleaned[idx + len(prefix):]

    # Split by comma or semicolon
    items = re.split(r"[,;]+", cleaned)
    pages = [item.strip() for item in items if item.strip()]
    return pages

def fetch_wikipedia_documents(pages: List[str]) -> List[Any]:
    """Fetch Wikipedia articles using the python `wikipedia` package or fallback."""
    try:
        import wikipedia
        docs = []
        for page_title in pages:
            try:
                page = wikipedia.page(page_title, auto_suggest=False)
                # Simple document structure compatible with LlamaIndex Document
                try:
                    from llama_index.core import Document
                    doc = Document(text=page.content, metadata={"title": page.title, "url": page.url})
                except ImportError:
                    try:
                        from llama_index import Document
                        doc = Document(text=page.content, extra_info={"title": page.title, "url": page.url})
                    except ImportError:
                        from .mock import MockDocument
                        doc = MockDocument(text=page.content, extra_info={"title": page.title, "url": page.url})
                docs.append(doc)
            except Exception as e:
                print(f"Warning: Could not fetch Wikipedia page '{page_title}': {e}")
        if docs:
            return docs
    except ImportError:
        pass

    # Fallback to mock documents
    return get_mock_documents(pages)

def create_index(query: str, model_name: str = "mock-mode", config: Optional[WikiRAGConfig] = None) -> Any:
    """Create a vector index over requested Wikipedia pages."""
    cfg = config or WikiRAGConfig.load_from_env(model_name=model_name)
    pages = parse_wikipage_query(query)
    if not pages:
        raise ValueError("No Wikipedia pages specified in the query.")

    # If mock-mode or no OpenAI API key is available, use MockIndex
    if model_name == "mock-mode" or not get_api_key():
        mock_docs = get_mock_documents(pages)
        return MockIndex(pages=pages, documents=mock_docs)

    # Try live LlamaIndex indexing
    try:
        import openai
        openai.api_key = get_api_key()

        # Support both modern llama-index (v0.10+) and legacy llama-index (v0.8-v0.9)
        try:
            from llama_index.core import VectorStoreIndex, Settings
            from llama_index.core.node_parser import SentenceSplitter
            from llama_index.llms.openai import OpenAI
            from llama_index.embeddings.openai import OpenAIEmbedding

            Settings.llm = OpenAI(model=model_name)
            Settings.embed_model = OpenAIEmbedding()
            Settings.node_parser = SentenceSplitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)

            documents = fetch_wikipedia_documents(pages)
            index = VectorStoreIndex.from_documents(documents)
            return index
        except ImportError:
            from llama_index import VectorStoreIndex, ServiceContext
            from llama_index.node_parser import SimpleNodeParser
            from llama_index.text_splitter import get_default_text_splitter
            from llama_index.llms import OpenAI as LlamaOpenAI

            text_splits = get_default_text_splitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
            parser = SimpleNodeParser.from_defaults(text_splitter=text_splits)
            llm = LlamaOpenAI(model=model_name)
            service_context = ServiceContext.from_defaults(node_parser=parser, llm=llm)

            documents = fetch_wikipedia_documents(pages)
            index = VectorStoreIndex.from_documents(documents, service_context=service_context)
            return index
    except Exception as e:
        print(f"Warning: Live LlamaIndex indexing failed ({e}). Falling back to MockIndex.")
        mock_docs = get_mock_documents(pages)
        return MockIndex(pages=pages, documents=mock_docs)
