from google import genai
from qdrant_client import QdrantClient

from codeatlas.embeddings.gemini_provider import GeminiEmbeddingProvider
from codeatlas.indexing.indexer import (
    embed_and_store_chunks,
    index_repository,
)
from codeatlas.vectorstore.qdrant_store import QdrantVectorStore


def main():
    repository_path = "."

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

    chunks = index_repository(repository_path)

    print(f"Found {len(chunks)} code chunks.")

    embed_and_store_chunks(
        chunks=chunks,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    print("Indexing complete.")


if __name__ == "__main__":
    main()