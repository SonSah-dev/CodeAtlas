from pathlib import Path


IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
}


def scan_repository(repository_path: str) -> list[Path]:
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

        files.append(path)

    return files