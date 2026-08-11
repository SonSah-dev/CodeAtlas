from abc import ABC, abstractmethod

from codeatlas.chunking.models import CodeChunk


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, chunk: CodeChunk) -> list[float]:
        """Convert a code chunk into an embedding vector."""
        raise NotImplementedError