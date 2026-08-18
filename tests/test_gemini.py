from codeatlas.llm.gemini import GeminiLLMProvider


class FakeResponse:
    text = "Authentication is handled in auth.py."


class FakeModels:
    def generate_content(self, model, contents):
        assert model == "gemini-3.5-flash"
        assert "Where is authentication handled?" in contents
        assert "def authenticate():" in contents

        return FakeResponse()


class FakeGeminiClient:
    def __init__(self):
        self.models = FakeModels()


def test_gemini_generation():
    provider = GeminiLLMProvider(
        client=FakeGeminiClient(),
    )

    answer = provider.generate(
        question="Where is authentication handled?",
        context="""
        File: auth.py
        Language: python

        def authenticate():
            pass
        """,
    )

    assert answer == "Authentication is handled in auth.py."