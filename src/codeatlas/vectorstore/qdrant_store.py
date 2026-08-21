from pathlib import Path
from uuid import UUID, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from codeatlas.embeddings.models import EmbeddedChunk
from codeatlas.search.models import SearchResult
from codeatlas.vectorstore.store import VectorStore


POINT_NAMESPACE = UUID("12345678-1234-5678-1234-567812345678")


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

        for embedded_chunk in chunks:
            chunk = embedded_chunk.chunk

            point_key = (
                f"{embedded_chunk.repository_id}:"
                f"{chunk.file_path}:"
                f"{chunk.chunk_index}"
            )

            point_id = uuid5(POINT_NAMESPACE, point_key)

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedded_chunk.vector,
                    payload={
                        "repository_id": embedded_chunk.repository_id,
                        "file_path": str(chunk.file_path),
                        "language": chunk.language,
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                    },
                )
            )

        if not points:
            return

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        vector: list[float],
        repository_id: str | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        query_filter = None

        if repository_id is not None:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="repository_id",
                        match=MatchValue(value=repository_id),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
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