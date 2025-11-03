from fastapi import APIRouter, Request
from app.chat.conversation_manager import ConversationManager
from app.generator.rag_pipeline import run_rag_pipeline

router = APIRouter()
conversation_manager = ConversationManager()

@router.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    user_query = data.get("query")
    session_id = data.get("session_id", "default") 

    # Lấy lịch sử chat gần nhất 
    context = conversation_manager.get_context(session_id)

    # Tạo ngữ cảnh từ lịch sử chat  
    context_text = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in context]
    )

    # Tạo prompt có ngữ cảnh 
    prompt = f"""
        Ngữ cảnh hội thoại trước đây: 
        {context_text}

        Người dùng vừa hỏi: 
        {user_query}

        Hãy trả lời chính xác và tự nhiên dựa vào thông tin có sẵn.
        Nếu bạn không biết thì hãy trả lời không biết. 
        Không được bịa câu trả lời. 
    """

    # Thực thi RAG pipeline 
    result = run_rag_pipeline(prompt)

    # Lưu lại lịch sử chat 
    conversation_manager.add_message(session_id, "user", user_query)
    conversation_manager.add_message(session_id, "bot", result["answer"])

    return {
        "session_id": session_id,
        "query": user_query,
        "answer": result["answer"]
    }

@router.post("/reset")
async def reset_chat(request: Request):
    data = await request.json()
    session_id = data.get("session_id", "default")
    conversation_manager.clear_history(session_id)
    return {
        "message": f"Lịch sử hội thoại {session_id} đã được xóa."
    }

