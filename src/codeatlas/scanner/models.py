from dataclasses import dataclass
from pathlib import Path


@dataclass
class FileMetadata:
    path: Path
    relative_path: Path
    extension: str
    language: str | None
    size: int
    content: str