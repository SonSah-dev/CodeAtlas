from fastapi import APIRouter

from codeatlas.api.models import AskRequest, AskResponse
from codeatlas.rag import RAGService


router = APIRouter()


def create_router(rag_service: RAGService) -> APIRouter:
    @router.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        answer = rag_service.answer(
            question=request.question,
            repository_id=request.repository_id,
            limit=request.limit,
        )

        return AskResponse(answer=answer)

    return router