import os

from google import genai
from qdrant_client import QdrantClient

from codeatlas.context.builder import ContextBuilder
from codeatlas.embeddings.gemini_provider import GeminiEmbeddingProvider
from codeatlas.llm.gemini import GeminiLLMProvider
from codeatlas.rag import RAGService
from codeatlas.search.service import SearchService
from codeatlas.vectorstore.qdrant_store import QdrantVectorStore


def create_rag_service() -> RAGService:
    # Gemini client
    gemini_client = genai.Client()

    # Embeddings
    embedding_provider = GeminiEmbeddingProvider(
        client=gemini_client,
    )

    # Qdrant
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

    # Search
    search_service = SearchService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    # Context
    context_builder = ContextBuilder()

    # LLM
    llm_provider = GeminiLLMProvider()

    # RAG
    return RAGService(
        search_service=search_service,
        context_builder=context_builder,
        llm_provider=llm_provider,
    )