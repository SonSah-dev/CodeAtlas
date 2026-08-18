from codeatlas.chunking.models import CodeChunk
from codeatlas.embeddings.models import EmbeddedChunk
from codeatlas.embeddings.provider import EmbeddingProvider
from codeatlas.vectorstore.store import VectorStore
from codeatlas.chunking.chunker import chunk_file
from codeatlas.scanner.repository import scan_repository


def index_repository(repository_path: str) -> list[CodeChunk]:
    """
    Scan a repository and convert every supported source file into code chunks.
    """

    all_chunks: list[CodeChunk] = []

    files = scan_repository(repository_path)

    for file in files:
        all_chunks.extend(chunk_file(file))

    return all_chunks


def embed_and_store_chunks(
    chunks: list[CodeChunk],
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
) -> None:
    """
    Convert code chunks into embeddings and store them in the vector store.
    """

    embedded_chunks: list[EmbeddedChunk] = []

    for chunk in chunks:
        vector = embedding_provider.embed(chunk)

        embedded_chunks.append(
            EmbeddedChunk(
                chunk=chunk,
                vector=vector,
            )
        )

    vector_store.upsert(embedded_chunks)