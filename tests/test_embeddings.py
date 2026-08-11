from pathlib import Path

from codeatlas.chunking.models import CodeChunk
from codeatlas.embeddings.provider import EmbeddingProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    def embed(self, chunk: CodeChunk) -> list[float]:
        return [0.1, 0.2, 0.3]


def test_embedding_provider_returns_vector():
    chunk = CodeChunk(
        file_path=Path("main.py"),
        language="python",
        chunk_index=0,
        content="print('hello')",
    )

    provider = FakeEmbeddingProvider()

    vector = provider.embed(chunk)

    assert vector == [0.1, 0.2, 0.3]