from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    limit: int = 5


class AskResponse(BaseModel):
    answer: str