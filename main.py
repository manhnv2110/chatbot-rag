from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field 
from app.retrieval.retrieve import retrieve 
from typing import Optional
from app.generator.rag_pipeline import run_rag_pipeline
from fastapi.middleware.cors import CORSMiddleware
from app.chat.chat_api import router as chat_router 

app = FastAPI()

app.include_router(chat_router, prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# class RetrieveRequest(BaseModel):
#     query: str

# @app.post("/api/ask")
# def retrieve_answer(request: RetrieveRequest):
#     query_text = request.query.strip()
#     if not query_text:
#         raise HTTPException(status_code=400, 
#                             detail="Query text cannot be empty")
    
#     result = run_rag_pipeline(query_text)
#     return {
#         "query": result["query"],
#         "answer": result["answer"]
#     }
