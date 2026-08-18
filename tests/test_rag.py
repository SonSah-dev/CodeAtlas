from pathlib import Path

from codeatlas.context.builder import ContextBuilder
from codeatlas.embeddings.models import EmbeddedChunk
from codeatlas.embeddings.provider import EmbeddingProvider
from codeatlas.chunking.models import CodeChunk
from codeatlas.llm.provider import LLMProvider
from codeatlas.rag import RAGService
from codeatlas.search.models import SearchResult
from codeatlas.search.service import SearchService
from codeatlas.vectorstore.store import VectorStore


class FakeEmbeddingProvider(EmbeddingProvider):

    def embed(self, chunk: CodeChunk) -> list[float]:
        return [0.1, 0.2, 0.3]

    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class FakeVectorStore(VectorStore):

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        pass

    def search(
        self,
        vector: list[float],
        limit: int = 5,
    ) -> list[SearchResult]:
        return [
            SearchResult(
                file_path=Path("auth.py"),
                language="python",
                chunk_index=0,
                content="def authenticate(): pass",
                score=0.95,
            )
        ]


class FakeLLMProvider(LLMProvider):

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        assert question == "Where is authentication handled?"
        assert "auth.py" in context
        assert "def authenticate()" in context

        return "Authentication is handled in auth.py."


def test_rag_service():
    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()

    search_service = SearchService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    context_builder = ContextBuilder()
    llm_provider = FakeLLMProvider()

    rag_service = RAGService(
        search_service=search_service,
        context_builder=context_builder,
        llm_provider=llm_provider,
    )

    answer = rag_service.answer(
        "Where is authentication handled?"
    )

    assert answer == "Authentication is handled in auth.py."