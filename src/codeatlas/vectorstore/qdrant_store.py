from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from codeatlas.chunking.models import CodeChunk
from codeatlas.embeddings.models import EmbeddedChunk
from codeatlas.vectorstore.store import VectorStore
from codeatlas.search.models import SearchResult


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        vector_size: int,
    ):
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size

    def create_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        points = []

        for index, embedded_chunk in enumerate(chunks):
            points.append(
                PointStruct(
                    id=index,
                    vector=embedded_chunk.vector,
                    payload={
                        "file_path": str(
                            embedded_chunk.chunk.file_path
                        ),
                        "language": embedded_chunk.chunk.language,
                        "chunk_index": embedded_chunk.chunk.chunk_index,
                        "content": embedded_chunk.chunk.content,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        vector: list[float],
        limit: int = 5,
    ) -> list[SearchResult]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            with_vectors=False,
        ).points

        search_results = []

        for result in results:
            payload = result.payload

            search_results.append(
                SearchResult(
                    file_path=Path(payload["file_path"]),
                    language=payload["language"],
                    chunk_index=payload["chunk_index"],
                    content=payload["content"],
                    score=result.score,
                )
            )

        return search_results