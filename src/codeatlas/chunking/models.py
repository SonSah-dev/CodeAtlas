from dataclasses import dataclass
from pathlib import Path


@dataclass
class CodeChunk:
    file_path: Path
    language: str
    chunk_index: int
    content: str