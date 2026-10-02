"""Offline Mock Provider for WikiRAG.
Enables full local testing, development, and portfolio inspection without an OpenAI API key.
"""

from typing import List, Dict, Any, Optional

MOCK_WIKI_PAGES: Dict[str, str] = {
    "paris": (
        "Paris is the capital and most populous city of France. Since the 17th century, "
        "Paris has been one of the world's major centres of finance, diplomacy, commerce, "
        "culture, fashion, and gastronomy. The City of Paris is the centre of the Île-de-France "
        "region. Famous landmarks include the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, "
        "and the Champs-Élysées. The Seine River flows through the city."
    ),
    "batman": (
        "Batman is a superhero appearing in American comic books published by DC Comics. "
        "The character was created by artist Bob Kane and writer Bill Finger, and debuted in "
        "Detective Comics #27 in 1939. In the DC Universe, Batman is the secret identity of "
        "Bruce Wayne, a wealthy American industrialist and philanthropist living in Gotham City. "
        "Unlike most superheroes, Batman does not possess any superhuman powers, instead relying "
        "on his intellect, martial arts prowess, and high-tech gadgets."
    ),
    "python": (
        "Python is a high-level, general-purpose programming language. Its design philosophy "
        "emphasizes code readability with the use of significant indentation. Python is dynamically "
        "typed and garbage-collected. It supports multiple programming paradigms, including structured, "
        "object-oriented and functional programming. It was created by Guido van Rossum and released in 1991."
    ),
    "star wars": (
        "Star Wars is an American epic space opera multimedia franchise created by George Lucas, "
        "which began with the eponymous 1977 film and quickly became a worldwide pop-culture phenomenon. "
        "The franchise has been expanded into various films and other media, including television series, "
        "video games, novels, and comic books."
    )
}

class MockDocument:
    """Mock document representation."""
    def __init__(self, text: str, extra_info: Optional[Dict[str, Any]] = None):
        self.text = text
        self.extra_info = extra_info or {}

    def get_content(self) -> str:
        return self.text

class MockResponse:
    """Mock query response object."""
    def __init__(self, response: str, source_nodes: Optional[List[Any]] = None):
        self.response = response
        self.source_nodes = source_nodes or []

    def __str__(self) -> str:
        return self.response

class MockQueryEngine:
    """Simulates a VectorStoreIndex QueryEngine."""
    def __init__(self, documents: List[MockDocument]):
        self.documents = documents

    def query(self, query_str: str) -> MockResponse:
        terms = [t.lower() for t in query_str.split() if len(t) > 2]
        matched_sentences = []

        for doc in self.documents:
            sentences = doc.text.split(". ")
            for s in sentences:
                if any(t in s.lower() for t in terms):
                    matched_sentences.append(s.strip())

        if matched_sentences:
            content = ". ".join(matched_sentences[:3]) + "."
            return MockResponse(f"Based on Wikipedia: {content}")
        
        fallback = self.documents[0].text[:200] if self.documents else "No information available."
        return MockResponse(f"Summary excerpt: {fallback}...")

class MockIndex:
    """Mock LlamaIndex VectorStoreIndex."""
    def __init__(self, pages: List[str], documents: List[MockDocument]):
        self.pages = pages
        self.documents = documents

    def as_query_engine(self, **kwargs) -> MockQueryEngine:
        return MockQueryEngine(self.documents)

class MockReActAgent:
    """Simulates LlamaIndex ReActAgent reasoning loop."""
    def __init__(self, index: MockIndex):
        self.index = index
        self.query_engine = index.as_query_engine()

    def chat(self, message: str) -> str:
        retrieval = self.query_engine.query(message)
        thought = (
            f"[ReAct Reasoning Trace]\n"
            f"> Thought: The user is asking about '{message}'. I need to search the indexed Wikipedia knowledge base.\n"
            f"> Action: Wikipedia Search\n"
            f"> Action Input: {message}\n"
            f"> Observation: {retrieval.response}\n"
            f"> Final Answer: {retrieval.response}"
        )
        return thought

def get_mock_documents(pages: List[str]) -> List[MockDocument]:
    """Retrieve mock documents for specified page titles."""
    docs = []
    for p in pages:
        clean = p.strip().lower()
        content = MOCK_WIKI_PAGES.get(clean)
        if not content:
            # Generate a realistic mock fallback
            content = f"{p.strip().title()} is a subject documented in Wikipedia. It encompasses historical, cultural, and technical aspects pertinent to its domain."
        docs.append(MockDocument(text=content, extra_info={"title": p.strip()}))
    return docs
