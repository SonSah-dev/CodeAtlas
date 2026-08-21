from pathlib import Path

from codeatlas.chunking.models import CodeChunk
from codeatlas.embeddings.models import EmbeddedChunk
from codeatlas.embeddings.provider import EmbeddingProvider
from codeatlas.search.models import SearchResult
from codeatlas.search.service import SearchService
from codeatlas.vectorstore.store import VectorStore


class FakeEmbeddingProvider(EmbeddingProvider):
    def embed(self, chunk: CodeChunk) -> list[float]:
        return [0.1, 0.2, 0.3]

    def embed_text(self, text: str) -> list[float]:
        assert text == "authentication"
        return [1.0, 0.0, 0.0]


class FakeVectorStore(VectorStore):
    def __init__(self):
        self.received_vector = None
        self.received_limit = None

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        pass

    def search(
        self,
        vector: list[float],
        limit: int = 5,
        repository_id: str | None = None,
    ) -> list[SearchResult]:
        self.received_vector = vector
        self.received_limit = limit

        return [
            SearchResult(
                file_path=Path("auth.py"),
                language="python",
                chunk_index=0,
                content="def authenticate(): pass",
                score=0.95,
            )
        ]


def test_search_service():
    provider = FakeEmbeddingProvider()
    store = FakeVectorStore()

    service = SearchService(
        embedding_provider=provider,
        vector_store=store,
    )

    results = service.search(
        query="authentication",
        repository_id="test-repo",
        limit=1,
    )

    assert store.received_vector == [1.0, 0.0, 0.0]
    assert store.received_limit == 1

    assert len(results) == 1
    assert results[0].file_path == Path("auth.py")
    assert results[0].score == 0.95