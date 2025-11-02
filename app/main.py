from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field 
from app.retrieval.retrieve import retrieve 
from typing import Optional

app = FastAPI()

class RetrieveRequest(BaseModel):
    query: str = Field(..., description="User query")
    fetch_n: Optional[int] = Field(20, gt=0, le=1000, description="Số lượng results dư để lọc")
    max_price: Optional[float] = None 
    type: Optional[str] = None 

@app.post("/api/retrieve")
def api_retrieve(request: RetrieveRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, 
                            detail="Query text cannot be empty")
    
    filters = {}
    if request.max_price is not None:
        filters["max_price"] = request.max_price 
    if request.type is not None: 
        filters["type"] = request.type
    
    try:
        response = retrieve(request.query, 
                            request.fetch_n,
                            filters)
    except Exception as e: 
        raise HTTPException(status_code=500,
                            detail=f"Error during retrieval: {e}")
    
    return response 

