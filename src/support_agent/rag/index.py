import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from math import fsum, hypot
from pathlib import Path

from ..types import Passage

DOCS_DIR = Path(__file__).resolve().parents[3] / "data" / "docs"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_WORDS = 250
OVERLAP_WORDS = 40
EMBEDDING_BATCH_SIZE = 64

Vector = tuple[float, ...]
Documents = tuple[tuple[str, str], ...]


def load_documents() -> Documents:
    if not DOCS_DIR.is_dir():
        return ()
    return tuple(
        (path.name, path.read_text(encoding="utf-8"))
        for path in sorted(DOCS_DIR.iterdir())
        if path.is_file()
        and path.suffix in {".md", ".txt"}
        and path.stem.endswith(("-manual", "-policy"))
    )


def chunk_text(text: str) -> Iterator[str]:
    """Yield bounded word windows while retaining the source's line breaks."""
    words = list(re.finditer(r"\S+", text))
    for start in range(0, len(words), CHUNK_WORDS - OVERLAP_WORDS):
        end = min(start + CHUNK_WORDS, len(words))
        yield text[words[start].start():words[end - 1].end()]
        if end == len(words):
            break


def embed(texts: list[str]) -> list[Vector]:
    from openai import OpenAI

    vectors = []
    with OpenAI() as client:
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts[start:start + EMBEDDING_BATCH_SIZE],
            )
            for item in sorted(response.data, key=lambda item: item.index):
                norm = hypot(*item.embedding)
                if norm:
                    vectors.append(tuple(value / norm for value in item.embedding))
                else:
                    vectors.append(tuple(item.embedding))
    return vectors


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    product_id: str | None


@dataclass(frozen=True)
class VectorIndex:
    entries: tuple[tuple[Chunk, Vector], ...]

    @staticmethod
    @lru_cache(maxsize=1)
    def for_documents(documents: Documents) -> "VectorIndex":
        chunks = []
        for source, document in documents:
            stem = Path(source).stem
            product_id = stem.removesuffix("-manual") if stem.endswith("-manual") else None
            chunks.extend(Chunk(text, source, product_id) for text in chunk_text(document))

        if not chunks:
            return VectorIndex(())
        vectors = embed([f"{chunk.source}\n{chunk.text}" for chunk in chunks])
        return VectorIndex(tuple(zip(chunks, vectors, strict=True)))

    def search(self, query: str, product_id: str | None, k: int) -> list[Passage]:
        candidates = [
            (chunk, vector)
            for chunk, vector in self.entries
            if product_id is None or chunk.product_id in (None, product_id)
        ]
        if not candidates:
            return []

        query_vector, = embed([query])
        passages = [
            Passage(chunk.text, chunk.source, fsum(a * b for a, b in zip(query_vector, vector, strict=True)))
            for chunk, vector in candidates
        ]
        return sorted(passages, key=lambda passage: passage.score, reverse=True)[:k]
