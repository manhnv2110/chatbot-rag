from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class MessageCreate(BaseModel):
    role: str = Field(..., description="'user' hoặc 'assistant'")
    content: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = None

class Message(MessageCreate):
    id: int
    conversation_id: int
    created_at: datetime

    class Config:
        from_attributes = True

class ConversationCreate(BaseModel):
    user_id: int
    title: Optional[str] = "Hội thoại mới"

class Conversation(BaseModel):
    id: int
    user_id: int
    title: str
    created_at: datetime
    updated_at: datetime
    is_active: bool = True

    class Config:
        from_attributes = True

class ConversationWithMessages(Conversation):
    messages: List[Message] = []

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[int] = None
    user_id: int

class ChatResponse(BaseModel):
    conversation_id: int
    message: Message
    retrieved_context: Optional[List[Dict]] = None