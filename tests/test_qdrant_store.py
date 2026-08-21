from pathlib import Path
from uuid import uuid5

from qdrant_client import QdrantClient

from codeatlas.chunking.models import CodeChunk
from codeatlas.embeddings.models import EmbeddedChunk
from codeatlas.vectorstore.qdrant_store import (
    POINT_NAMESPACE,
    QdrantVectorStore,
)


def test_qdrant_collection_creation():
    client = QdrantClient(
        host="localhost",
        port=6333,
    )

    collection_name = "test_codeatlas_collection"

    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        vector_size=3,
    )

    store.create_collection()

    assert client.collection_exists(collection_name)


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
        repository_id="test-repo",
    )

    store.upsert([embedded_chunk])

    point_id = uuid5(
        POINT_NAMESPACE,
        "test-repo:main.py:0",
    )

    result = client.retrieve(
        collection_name=collection_name,
        ids=[point_id],
    )

    assert len(result) == 1
    assert result[0].payload["repository_id"] == "test-repo"
    assert result[0].payload["file_path"] == "main.py"
    assert result[0].payload["content"] == "print('hello')"


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
            repository_id="test-repo",
        ),
        EmbeddedChunk(
            chunk=CodeChunk(
                file_path=Path("payments.py"),
                language="python",
                chunk_index=0,
                content="def process_payment(): pass",
            ),
            vector=[0.0, 1.0, 0.0],
            repository_id="test-repo",
        ),
    ]

    store.upsert(chunks)

    results = store.search(
        vector=[1.0, 0.0, 0.0],
        limit=1,
        repository_id="test-repo",
    )

    assert len(results) == 1
    assert results[0].file_path == Path("auth.py")
    assert results[0].content == "def authenticate_user(): pass"


def test_qdrant_repository_isolation():
    client = QdrantClient(
        host="localhost",
        port=6333,
    )

    collection_name = "test_codeatlas_isolation"

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
            repository_id="repo-a",
        ),
        EmbeddedChunk(
            chunk=CodeChunk(
                file_path=Path("payments.py"),
                language="python",
                chunk_index=0,
                content="def process_payment(): pass",
            ),
            vector=[1.0, 0.0, 0.0],
            repository_id="repo-b",
        ),
    ]

    store.upsert(chunks)

    repo_a_results = store.search(
        vector=[1.0, 0.0, 0.0],
        limit=5,
        repository_id="repo-a",
    )

    repo_b_results = store.search(
        vector=[1.0, 0.0, 0.0],
        limit=5,
        repository_id="repo-b",
    )

    assert len(repo_a_results) == 1
    assert repo_a_results[0].file_path == Path("auth.py")
    assert repo_a_results[0].content == "def authenticate_user(): pass"

    assert len(repo_b_results) == 1
    assert repo_b_results[0].file_path == Path("payments.py")
    assert repo_b_results[0].content == "def process_payment(): pass"