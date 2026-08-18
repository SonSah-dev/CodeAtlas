from codeatlas.embeddings.provider import EmbeddingProvider
from codeatlas.search.models import SearchResult
from codeatlas.vectorstore.store import VectorStore


class SearchService:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[SearchResult]:
        vector = self.embedding_provider.embed_text(query)

        return self.vector_store.search(
            vector=vector,
            limit=limit,
        )