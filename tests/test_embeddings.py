from pathlib import Path

from google import genai

from codeatlas.chunking.models import CodeChunk
from codeatlas.embeddings.gemini_provider import GeminiEmbeddingProvider


def test_gemini_embedding_provider():
    chunk = CodeChunk(
        file_path=Path("main.py"),
        language="python",
        chunk_index=0,
        content="print('hello')",
    )

    client = genai.Client()
    provider = GeminiEmbeddingProvider(client)

    vector = provider.embed(chunk)

    assert isinstance(vector, list)
    assert len(vector) == 3072