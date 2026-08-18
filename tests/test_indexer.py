from pathlib import Path

from codeatlas.indexing.indexer import index_repository


def test_index_repository_returns_chunks(tmp_path: Path):
    source = tmp_path / "main.py"
    source.write_text(
        "\n".join(f"line {i}" for i in range(120))
    )

    chunks = index_repository(str(tmp_path))

    assert len(chunks) == 3
    assert chunks[0].language == "python"
    assert chunks[0].file_path == Path("main.py")