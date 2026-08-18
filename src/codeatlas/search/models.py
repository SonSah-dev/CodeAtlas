from pathlib import Path

from pydantic import BaseModel


class SearchResult(BaseModel):
    file_path: Path
    language: str
    chunk_index: int
    content: str
    score: float