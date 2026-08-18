from codeatlas.chunking.models import CodeChunk
from codeatlas.scanner.models import FileMetadata


def chunk_file(
    file: FileMetadata,
    max_lines: int = 50,
) -> list[CodeChunk]:
    lines = file.content.splitlines()

    chunks = []

    for index in range(0, len(lines), max_lines):
        chunk = "\n".join(lines[index:index + max_lines])

        chunks.append(
            CodeChunk(
                file_path=file.relative_path,
                language=file.language,
                chunk_index=len(chunks),
                content=chunk,
            )
        )

    return chunks