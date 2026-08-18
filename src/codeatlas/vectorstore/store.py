from abc import ABC, abstractmethod

from codeatlas.embeddings.models import EmbeddedChunk
from codeatlas.search.models import SearchResult


class VectorStore(ABC):

    @abstractmethod
    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        """Store embedded code chunks."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        vector: list[float],
        limit: int = 5,
    ) -> list[SearchResult]:
        """Find the most similar code chunks."""
        raise NotImplementedError