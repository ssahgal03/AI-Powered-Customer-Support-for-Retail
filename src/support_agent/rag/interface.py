from ..config import llm_enabled
from ..types import Passage
from .index import VectorIndex, load_documents


def retrieve(query: str, product_id: str | None = None, k: int = 4) -> list[Passage]:
    """Return up to k passages from applicable product manuals and policies."""
    query = query.strip()
    if not query or k <= 0 or not llm_enabled():
        return []

    documents = load_documents()
    if not documents:
        return []

    from openai import OpenAIError

    try:
        return VectorIndex.for_documents(documents).search(query, product_id, k)
    except OpenAIError:
        # Let the existing empty-context response handle embedding outages.
        return []
