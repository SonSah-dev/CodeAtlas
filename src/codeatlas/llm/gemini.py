from dotenv import load_dotenv
from google import genai

from codeatlas.llm.provider import LLMProvider


load_dotenv()


class GeminiLLMProvider(LLMProvider):
    def __init__(
        self,
        client: genai.Client | None = None,
        model: str = "gemini-3.5-flash",
    ):
        self.client = client or genai.Client()
        self.model = model

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        prompt = f"""
You are CodeAtlas, an AI assistant that answers questions about a software repository.

Use only the provided repository context to answer the question.

If the context does not contain enough information to answer confidently,
say that the repository context does not provide enough information.

Repository context:

{context}

Question:

{question}
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text