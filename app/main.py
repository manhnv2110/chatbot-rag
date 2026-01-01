from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.chat.chat_api import router as chat_router

# Initialize FastAPI app
app = FastAPI(
    title="E-commerce RAG Chatbot API",
    description="RAG-powered chatbot cho cửa hàng thời trang",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên chỉ định domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "E-commerce RAG Chatbot API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8088)
