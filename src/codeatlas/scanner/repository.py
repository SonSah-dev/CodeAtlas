from pathlib import Path

from codeatlas.scanner.languages import EXTENSION_TO_LANGUAGE
from codeatlas.scanner.models import FileMetadata


IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
}


def scan_repository(repository_path: str) -> list[FileMetadata]:
    root = Path(repository_path).resolve()

    if not root.exists():
        raise ValueError(f"Repository does not exist: {root}")

    if not root.is_dir():
        raise ValueError(f"Repository path is not a directory: {root}")

    files = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if any(part in IGNORED_DIRECTORIES for part in path.parts):
            continue

        extension = path.suffix.lower()
        language = EXTENSION_TO_LANGUAGE.get(extension)

        if language is None:
            continue

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        files.append(
            FileMetadata(
                path=path,
                relative_path=path.relative_to(root),
                extension=extension,
                language=language,
                size=path.stat().st_size,
                content=content,
            )
        )

    return files