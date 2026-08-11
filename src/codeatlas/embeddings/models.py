from dataclasses import dataclass

from codeatlas.chunking.models import CodeChunk


@dataclass
class EmbeddedChunk:
    chunk: CodeChunk
    vector: list[float]