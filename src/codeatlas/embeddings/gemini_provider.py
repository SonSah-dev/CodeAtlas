from dotenv import load_dotenv
from google import genai

from codeatlas.chunking.models import CodeChunk
from codeatlas.embeddings.provider import EmbeddingProvider


load_dotenv()


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        client: genai.Client,
        model: str = "gemini-embedding-2",
    ):
        self.client = client
        self.model = model

    def embed(self, chunk: CodeChunk) -> list[float]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=chunk.content,
        )

        return response.embeddings[0].values

    def embed_text(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
        )

        return response.embeddings[0].values