from abc import ABC, abstractmethod

from codeatlas.chunking.models import CodeChunk


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, chunk: CodeChunk) -> list[float]:
        """Convert a code chunk into an embedding vector."""
        raise NotImplementedError

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Convert text into an embedding vector."""
        raise NotImplementedError