from fastapi import FastAPI
from src.api import query
from fastapi.responses import RedirectResponse
from src.utils.helpers import load_data
from pydantic import BaseModel
from typing import List, Dict
from src.retriever.retriever import get_similar_responses
import numpy as np


# TODO: Pre-load the dataset
# pre-load the approx nearest neighbour
# === Preload dataset and HNSW index ===
#df = get_data()  
#embeddings = np.vstack(df['prompt_embedding'].values)
#index = build_hnsw_index(embeddings)

# app = FastAPI(
#     title="ML API",
#     description="API for ML Model Inference",
#     version="1.0.0",
# )

# @app.get("/")
# async def redirect_to_docs():
#     return RedirectResponse(url="/docs")

# app.include_router(query.router)

# === FastAPI App Setup ===
app = FastAPI(
    title="ML API",
    description="API for ML Model Inference",
    version="1.0.0",
)

@app.get("/")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

app.include_router(query.router)

# === Optional: Add a prompt similarity endpoint here ===
# class PromptRequest(BaseModel):
#     prompt: str
#     top_k: int = 5

# @app.post("/similar", response_model=List[Dict])
# async def similar_prompts(request: PromptRequest):
#     return get_similar_responses(request.prompt, df, index, request.top_k)


# class ExcerptRequest(BaseModel):
#     excerpt: str
#     top_k: int = 5
    
# # === Response Model ===
# class SimilarResponse(BaseModel):
#     excerpt: str
#     #excerpt: str
#     #response: str
#     score: float

# @app.post("/similar", response_model=List[SimilarResponse])
# def similar_excerpt(request: ExcerptRequest):
#     print(request)
    
#     # return request.prompt
#     responses = get_similar_responses(request.excerpt, request.top_k)
#     return [SimilarResponse(**resp) for resp in responses]
        


