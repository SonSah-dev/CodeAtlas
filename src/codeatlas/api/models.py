from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    repository_id: str
    limit: int = 5


class AskResponse(BaseModel):
    answer: str