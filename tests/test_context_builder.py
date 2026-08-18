from pathlib import Path

from codeatlas.context.builder import ContextBuilder
from codeatlas.search.models import SearchResult


def test_context_builder():
    results = [
        SearchResult(
            file_path=Path("auth.py"),
            language="python",
            chunk_index=0,
            content="def authenticate(): pass",
            score=0.95,
        ),
        SearchResult(
            file_path=Path("users.py"),
            language="python",
            chunk_index=0,
            content="def get_user(): pass",
            score=0.85,
        ),
    ]

    builder = ContextBuilder()

    context = builder.build(results)

    assert "File: auth.py" in context
    assert "def authenticate(): pass" in context
    assert "File: users.py" in context
    assert "def get_user(): pass" in context
    assert "---" in context