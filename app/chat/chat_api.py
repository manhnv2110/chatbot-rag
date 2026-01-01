# from fastapi import APIRouter, Request, HTTPException
# from app.chat.conversation_manager import ConversationManager
# from app.generator.rag_pipeline import run_rag_pipeline

# router = APIRouter()
# conversation_manager = ConversationManager()

# @router.post("/chat")
# async def chat_endpoint(request: Request):
#     data = await request.json()
#     user_query = data.get("query", "").strip()
#     session_id = data.get("session_id", "default") 

#     if not user_query:
#         raise HTTPException(status_code=400,
#                             detail="Truy vấn không được để trống.")
    
#     # Lấy lịch sử chat gần nhất 
#     context = conversation_manager.get_context(session_id)

#     # Tạo ngữ cảnh từ lịch sử chat  
#     context_text = "\n".join(
#         [f"{msg['role'].capitalize()}: {msg['content']}" for msg in context]
#     )

#     # Thực thi RAG pipeline 
#     result = run_rag_pipeline(user_query)

#     # Lưu lại lịch sử chat 
#     conversation_manager.add_message(session_id, "user", user_query)
#     conversation_manager.add_message(session_id, "bot", result["answer"])

#     return {
#         "session_id": session_id,
#         "query": user_query,
#         "context": context_text,
#         "answer": result["answer"]
#     }

# @router.post("/reset")
# async def reset_chat(request: Request):
#     data = await request.json()
#     session_id = data.get("session_id", "default")
#     conversation_manager.clear_history(session_id)
#     return {
#         "message": f"Lịch sử hội thoại {session_id} đã được xóa."
#     }

"""
FastAPI endpoints cho chatbot
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import json

from app.services.rag_service import RAGService
from app.models.conversation import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])

# Initialize RAG service
rag_service = RAGService()

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ProductSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    category: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    limit: int = Field(default=10, ge=1, le=50)

class ConversationListRequest(BaseModel):
    user_id: int
    limit: int = Field(default=20, ge=1, le=100)

# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/chat-bot", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - xử lý tin nhắn và trả về response
    
    Request body:
    {
        "message": "Tôi muốn mua áo thun",
        "user_id": 1,
        "conversation_id": null  // null = tạo conversation mới
    }
    
    Response:
    {
        "conversation_id": 123,
        "message": {
            "id": 456,
            "role": "assistant",
            "content": "Chào bạn! Tôi có thể giới thiệu...",
            "created_at": "2024-01-01T10:00:00"
        },
        "retrieved_context": [...]  // Top 3 documents được retrieve
    }
    """
    try:
        result = rag_service.chat(
            user_id=request.user_id,
            message=request.message,
            conversation_id=request.conversation_id,
            n_results=5
        )
        
        return ChatResponse(
            conversation_id=result["conversation_id"],
            message=result["assistant_message"],
            retrieved_context=result["retrieved_context"]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/chat-bot/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint - trả về response theo chunks (real-time)
    
    Response format (Server-Sent Events):
    data: {"type": "metadata", "conversation_id": 123, "intent": "product_search"}
    data: {"type": "content", "chunk": "Chào"}
    data: {"type": "content", "chunk": " bạn"}
    data: {"type": "complete", "message_id": 456}
    """
    try:
        async def event_generator():
            for chunk in rag_service.stream_chat(
                user_id=request.user_id,
                message=request.message,
                conversation_id=request.conversation_id,
                n_results=5
            ):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        print(f"❌ Error in stream endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: int, user_id: int):
    """
    Lấy chi tiết conversation kèm messages
    
    Query params:
    - user_id: ID của user (để verify ownership)
    
    Response:
    {
        "conversation": {
            "id": 123,
            "title": "Hỏi về áo thun",
            "created_at": "...",
            "updated_at": "..."
        },
        "messages": [
            {"id": 1, "role": "user", "content": "...", "created_at": "..."},
            {"id": 2, "role": "assistant", "content": "...", "created_at": "..."}
        ]
    }
    """
    try:
        result = rag_service.get_conversation_history(
            conversation_id=conversation_id,
            user_id=user_id
        )
        
        if result is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found or access denied"
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting conversation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/conversations/list")
async def list_conversations(request: ConversationListRequest):
    """
    Lấy danh sách conversations của user
    
    Request body:
    {
        "user_id": 1,
        "limit": 20
    }
    
    Response:
    [
        {
            "id": 123,
            "title": "Hỏi về áo thun",
            "created_at": "...",
            "updated_at": "..."
        },
        ...
    ]
    """
    try:
        conversations = rag_service.get_user_conversations(
            user_id=request.user_id,
            limit=request.limit
        )
        
        return conversations
        
    except Exception as e:
        print(f"❌ Error listing conversations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/search/products")
async def search_products(request: ProductSearchRequest):
    """
    Search sản phẩm với filters
    
    Request body:
    {
        "query": "áo thun nam",
        "category": "Áo thun",  // optional
        "min_price": 100000,    // optional
        "max_price": 500000,    // optional
        "limit": 10
    }
    
    Response:
    [
        {
            "id": "product_123",
            "text": "Sản phẩm: Áo thun basic...",
            "metadata": {
                "product_id": 123,
                "product_name": "Áo thun basic",
                "price": 250000,
                "category_name": "Áo thun",
                ...
            },
            "weighted_score": 0.95
        },
        ...
    ]
    """
    try:
        results = rag_service.search_products(
            query=request.query,
            category=request.category,
            min_price=request.min_price,
            max_price=request.max_price,
            n_results=request.limit
        )
        
        return results
        
    except Exception as e:
        print(f"❌ Error searching products: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "RAG Chatbot API"
    }

# ============================================================================
# ADMIN ENDPOINTS (Optional - for testing/debugging)
# ============================================================================

@router.get("/admin/collections/stats")
async def get_collection_stats():
    """
    Lấy thống kê về các ChromaDB collections
    (Chỉ dùng cho admin/testing)
    """
    try:
        stats = rag_service.search_service.collections_config
        
        collection_stats = {}
        for key, config in stats.items():
            try:
                collection = rag_service.search_service.collections.get(key)
                if collection:
                    collection_stats[key] = {
                        "name": config["name"],
                        "description": config["description"],
                        "weight": config["weight"],
                        "count": collection.count()
                    }
            except:
                collection_stats[key] = {
                    "error": "Collection not loaded"
                }
        
        return collection_stats
        
    except Exception as e:
        print(f"❌ Error getting collection stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")