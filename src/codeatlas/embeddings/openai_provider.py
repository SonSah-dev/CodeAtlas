from openai import OpenAI

from codeatlas.chunking.models import CodeChunk
from codeatlas.embeddings.provider import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, client: OpenAI):
        self.client = client

    def embed(self, chunk: CodeChunk) -> list[float]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk.content,
        )

        return response.data[0].embedding