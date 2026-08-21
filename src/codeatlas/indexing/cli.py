import sys

from google import genai
from qdrant_client import QdrantClient

from codeatlas.embeddings.gemini_provider import GeminiEmbeddingProvider
from codeatlas.indexing.service import IndexingService
from codeatlas.vectorstore.qdrant_store import QdrantVectorStore


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "Usage: python -m codeatlas.indexing.cli "
            "<repository_path> <repository_id>"
        )
        sys.exit(1)

    repository_path = sys.argv[1]
    repository_id = sys.argv[2]

    client = genai.Client()

    embedding_provider = GeminiEmbeddingProvider(client)

    qdrant_client = QdrantClient(
        host="localhost",
        port=6333,
    )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name="codeatlas",
        vector_size=3072,
    )

    vector_store.create_collection()

    service = IndexingService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    count = service.index(
        repository_path=repository_path,
        repository_id=repository_id,
    )

    print(
        f"Indexed {count} chunks "
        f"from repository '{repository_id}'"
    )


if __name__ == "__main__":
    main()