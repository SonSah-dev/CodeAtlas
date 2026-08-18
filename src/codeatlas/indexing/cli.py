import sys

from google import genai
from qdrant_client import QdrantClient

from codeatlas.embeddings.gemini_provider import GeminiEmbeddingProvider
from codeatlas.indexing.service import IndexingService
from codeatlas.vectorstore.qdrant_store import QdrantVectorStore

def main():
    if len(sys.argv) != 2:
        print("Usage: python -m codeatlas.indexing.cli <repository_path>")
        raise SystemExit(1)

    repository_path = sys.argv[1]

    gemini_client = genai.Client()

    embedding_provider = GeminiEmbeddingProvider(
        client=gemini_client,
    )

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

    indexing_service = IndexingService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    count = indexing_service.index(repository_path)

    print(f"Indexed {count} chunks from {repository_path}")


if __name__ == "__main__":
    main()