from fastapi import APIRouter
from src.retriever import retriever
from src.models.query import RAGRequest, RAGResponse

router = APIRouter()

#added include_in_schema in query file
@router.post("/similar_responses", response_model=RAGResponse,include_in_schema=False)
def get_similar_responses(request: RAGRequest):
    results = retriever.get_similar_responses(request.question)
    return RAGResponse(answers=results)
