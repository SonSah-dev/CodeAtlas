from codeatlas.chunking.chunker import chunk_file
from codeatlas.embeddings.models import EmbeddedChunk
from codeatlas.embeddings.provider import EmbeddingProvider
from codeatlas.scanner.repository import scan_repository
from codeatlas.vectorstore.store import VectorStore


class IndexingService:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def index(
        self,
        repository_path: str,
        repository_id: str,
    ) -> int:
        files = scan_repository(repository_path)

        chunks = []

        for file in files:
            chunks.extend(chunk_file(file))

        embedded_chunks = []

        for chunk in chunks:
            vector = self.embedding_provider.embed(chunk)

            embedded_chunks.append(
                EmbeddedChunk(
                    chunk=chunk,
                    vector=vector,
                    repository_id=repository_id,
                )
            )

        self.vector_store.upsert(embedded_chunks)

        return len(embedded_chunks)