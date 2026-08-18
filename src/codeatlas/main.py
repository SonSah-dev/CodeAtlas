from fastapi import FastAPI

from codeatlas.api.routes import create_router
from codeatlas.app.factory import create_rag_service


def create_app() -> FastAPI:
    app = FastAPI(
        title="CodeAtlas",
        description="AI-powered codebase search and question answering",
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    rag_service = create_rag_service()

    app.include_router(
        create_router(rag_service),
    )

    return app


app = create_app()