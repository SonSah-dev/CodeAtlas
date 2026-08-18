from pathlib import Path

from codeatlas.chunking.chunker import chunk_file
from codeatlas.scanner.models import FileMetadata


def test_chunk_file_splits_large_file():
    content = "\n".join(f"line {i}" for i in range(120))

    file = FileMetadata(
        path=Path("main.py"),
        relative_path=Path("main.py"),
        extension=".py",
        language="python",
        size=len(content),
        content=content,
    )

    chunks = chunk_file(file, max_lines=50)

    assert len(chunks) == 3
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[2].chunk_index == 2