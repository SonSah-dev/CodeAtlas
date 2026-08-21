from pathlib import Path

from codeatlas.chunking.models import CodeChunk
from codeatlas.embeddings.models import EmbeddedChunk
from codeatlas.embeddings.provider import EmbeddingProvider
from codeatlas.indexing.service import IndexingService
from codeatlas.vectorstore.store import VectorStore


class FakeEmbeddingProvider(EmbeddingProvider):
    def embed(self, chunk: CodeChunk) -> list[float]:
        return [0.1, 0.2, 0.3]

    def embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeVectorStore(VectorStore):
    def __init__(self):
        self.received_chunks = []

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        self.received_chunks = chunks

    def search(
        self,
        vector: list[float],
        limit: int = 5,
    ):
        return []


def test_indexing_service():
    provider = FakeEmbeddingProvider()
    store = FakeVectorStore()

    service = IndexingService(
        embedding_provider=provider,
        vector_store=store,
    )

    count = service.index(
        ".", 
        repository_id="test-repo",
        )

    assert count > 0
    assert len(store.received_chunks) == count

    for embedded_chunk in store.received_chunks:
        assert embedded_chunk.vector == [0.1, 0.2, 0.3]
        assert isinstance(embedded_chunk.chunk.file_path, Path)