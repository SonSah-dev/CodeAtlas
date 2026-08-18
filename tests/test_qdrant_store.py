from pathlib import Path

from qdrant_client import QdrantClient

from codeatlas.chunking.models import CodeChunk
from codeatlas.embeddings.models import EmbeddedChunk
from codeatlas.vectorstore.qdrant_store import QdrantVectorStore


def test_qdrant_connection():
    client = QdrantClient(
        host="localhost",
        port=6333,
    )

    collection_name = "test_codeatlas"

    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        vector_size=3,
    )

    store.create_collection()

    assert client.collection_exists(collection_name)

    client.delete_collection(collection_name)


def test_qdrant_upsert():
    client = QdrantClient(
        host="localhost",
        port=6333,
    )

    collection_name = "test_codeatlas_upsert"

    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        vector_size=3,
    )

    store.create_collection()

    chunk = CodeChunk(
        file_path=Path("main.py"),
        language="python",
        chunk_index=0,
        content="print('hello')",
    )

    embedded_chunk = EmbeddedChunk(
        chunk=chunk,
        vector=[0.1, 0.2, 0.3],
    )

    store.upsert([embedded_chunk])

    result = client.retrieve(
        collection_name=collection_name,
        ids=[0],
    )

    assert len(result) == 1
    assert result[0].payload["file_path"] == "main.py"
    assert result[0].payload["language"] == "python"
    assert result[0].payload["content"] == "print('hello')"

    client.delete_collection(collection_name)


def test_qdrant_search():
    client = QdrantClient(
        host="localhost",
        port=6333,
    )

    collection_name = "test_codeatlas_search"

    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        vector_size=3,
    )

    store.create_collection()

    chunks = [
        EmbeddedChunk(
            chunk=CodeChunk(
                file_path=Path("auth.py"),
                language="python",
                chunk_index=0,
                content="def authenticate_user(): pass",
            ),
            vector=[1.0, 0.0, 0.0],
        ),
        EmbeddedChunk(
            chunk=CodeChunk(
                file_path=Path("payments.py"),
                language="python",
                chunk_index=0,
                content="def process_payment(): pass",
            ),
            vector=[0.0, 1.0, 0.0],
        ),
    ]

    store.upsert(chunks)

    results = store.search(
        vector=[1.0, 0.0, 0.0],
        limit=1,
    )

    assert len(results) == 1
    assert results[0].file_path == Path("auth.py")
    client.delete_collection(collection_name)