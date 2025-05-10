from pydantic import BaseModel
from typing import List

class RAGRequest(BaseModel):
    question: str
    
class Answer(BaseModel):
    excerpt: str
    score: float

class RAGResponse(BaseModel):
    answers: List[Answer]
