from typing import Dict, Optional, List
from app.services.search_service import SearchService
from app.services.conversation_service import ConversationService
from app.core.llm_client import GroqClient

class RAGService:
    """
    Main service để xử lý RAG pipeline
    Flow: User query -> Search context -> Generate response -> Save to DB
    """
    
    def __init__(self):
        self.search_service = SearchService()
        self.conversation_service = ConversationService()
        self.llm_client = GroqClient()
    
    def chat(
        self,
        user_id: int,
        message: str,
        conversation_id: Optional[int] = None,
        n_results: int = 5
    ) -> Dict:
        """
        Main chat function - xử lý toàn bộ flow
        
        Args:
            user_id: ID của user
            message: Tin nhắn từ user
            conversation_id: ID conversation (None = tạo mới)
            n_results: Số lượng documents để retrieve
        
        Returns:
            {
                "conversation_id": int,
                "user_message": Message,
                "assistant_message": Message,
                "retrieved_context": List[Dict],
                "intent": str
            }
        """
        
        try:
            # Step 1: Lấy hoặc tạo conversation
            if conversation_id is None:
                # Tạo conversation mới
                conversation = self.conversation_service.create_conversation(
                    user_id=user_id,
                    title=self._generate_conversation_title(message)
                )
                conversation_id = conversation.id
            else:
                # Kiểm tra conversation có tồn tại không
                conversation = self.conversation_service.get_conversation(conversation_id)
                if not conversation:
                    raise ValueError(f"Conversation {conversation_id} không tồn tại")
            
            # Step 2: Lưu user message
            user_msg = self.conversation_service.add_message(
                conversation_id=conversation_id,
                role="user",
                content=message
            )
            
            # Step 3: Smart search để lấy context
            search_result = self.search_service.smart_search(
                query=message,
                n_results=n_results
            )
            
            retrieved_context = search_result["results"]
            intent = search_result["intent"]
            
            # Step 4: Lấy conversation history
            conversation_history = self.conversation_service.get_conversation_history_for_llm(
                conversation_id=conversation_id,
                limit=10
            )
            
            # Step 5: Generate response từ LLM
            assistant_response = self.llm_client.generate_response(
                user_message=message,
                context=retrieved_context,
                conversation_history=conversation_history[:-1]  # Exclude current message
            )
            
            # Step 6: Lưu assistant response
            assistant_msg = self.conversation_service.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=assistant_response,
                metadata={
                    "intent": intent,
                    "retrieved_docs_count": len(retrieved_context),
                    "top_collection": retrieved_context[0]["collection"] if retrieved_context else None
                }
            )
            
            # Step 7: Return response
            return {
                "conversation_id": conversation_id,
                "user_message": user_msg,
                "assistant_message": assistant_msg,
                "retrieved_context": retrieved_context[:3],  # Chỉ trả về top 3 cho frontend
                "intent": intent
            }
            
        except Exception as e:
            print(f"❌ Error in RAG chat: {e}")
            raise
    
    def stream_chat(
        self,
        user_id: int,
        message: str,
        conversation_id: Optional[int] = None,
        n_results: int = 5
    ):
        """
        Streaming chat response (cho real-time chat UI)
        
        Yields:
            Chunks of assistant response
        """
        
        try:
            # Setup conversation (tương tự như chat)
            if conversation_id is None:
                conversation = self.conversation_service.create_conversation(
                    user_id=user_id,
                    title=self._generate_conversation_title(message)
                )
                conversation_id = conversation.id
            
            # Save user message
            user_msg = self.conversation_service.add_message(
                conversation_id=conversation_id,
                role="user",
                content=message
            )
            
            # Search context
            search_result = self.search_service.smart_search(
                query=message,
                n_results=n_results
            )
            
            retrieved_context = search_result["results"]
            intent = search_result["intent"]
            
            # Get conversation history
            conversation_history = self.conversation_service.get_conversation_history_for_llm(
                conversation_id=conversation_id,
                limit=10
            )
            
            # Yield metadata first
            yield {
                "type": "metadata",
                "conversation_id": conversation_id,
                "intent": intent,
                "context_count": len(retrieved_context)
            }
            
            # Stream response
            full_response = ""
            for chunk in self.llm_client.generate_stream_response(
                user_message=message,
                context=retrieved_context,
                conversation_history=conversation_history[:-1]
            ):
                full_response += chunk
                yield {
                    "type": "content",
                    "chunk": chunk
                }
            
            # Save complete response
            assistant_msg = self.conversation_service.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=full_response,
                metadata={
                    "intent": intent,
                    "retrieved_docs_count": len(retrieved_context)
                }
            )
            
            # Yield completion
            yield {
                "type": "complete",
                "message_id": assistant_msg.id
            }
            
        except Exception as e:
            print(f"❌ Error in stream chat: {e}")
            yield {
                "type": "error",
                "error": str(e)
            }
    
    def get_conversation_history(
        self,
        conversation_id: int,
        user_id: int
    ) -> Optional[Dict]:
        """
        Lấy lịch sử hội thoại
        """
        try:
            conversation = self.conversation_service.get_conversation_with_messages(
                conversation_id
            )
            
            if not conversation or conversation.user_id != user_id:
                return None
            
            return {
                "conversation": {
                    "id": conversation.id,
                    "title": conversation.title,
                    "created_at": conversation.created_at.isoformat(),
                    "updated_at": conversation.updated_at.isoformat()
                },
                "messages": [
                    {
                        "id": msg.id,
                        "role": msg.role,
                        "content": msg.content,
                        "created_at": msg.created_at.isoformat()
                    }
                    for msg in conversation.messages
                ]
            }
            
        except Exception as e:
            print(f"❌ Error getting conversation history: {e}")
            return None
    
    def get_user_conversations(self, user_id: int, limit: int = 20) -> List[Dict]:
        """Lấy danh sách conversations của user"""
        try:
            conversations = self.conversation_service.get_user_conversations(
                user_id, limit
            )
            
            return [
                {
                    "id": conv.id,
                    "title": conv.title,
                    "created_at": conv.created_at.isoformat(),
                    "updated_at": conv.updated_at.isoformat()
                }
                for conv in conversations
            ]
            
        except Exception as e:
            print(f"❌ Error getting user conversations: {e}")
            return []
    
    def _generate_conversation_title(self, first_message: str) -> str:
        """
        Generate title cho conversation dựa trên first message
        """
        # Simple implementation - lấy 50 ký tự đầu
        title = first_message[:50]
        if len(first_message) > 50:
            title += "..."
        return title
    
    def search_products(
        self,
        query: str,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        n_results: int = 10
    ) -> List[Dict]:
        """
        Search sản phẩm với filters
        """
        try:
            # Build filter
            filter_dict = {}
            if category:
                filter_dict["category_name"] = category
            
            # Search
            results = self.search_service.search(
                query=query,
                n_results=n_results * 2,  # Get more for filtering
                collections=["products"],
                filter_metadata=filter_dict if filter_dict else None
            )
            
            # Filter by price range
            if min_price is not None or max_price is not None:
                filtered_results = []
                for result in results:
                    price = result['metadata'].get('price', 0)
                    if min_price is not None and price < min_price:
                        continue
                    if max_price is not None and price > max_price:
                        continue
                    filtered_results.append(result)
                results = filtered_results
            
            # Return top N
            return results[:n_results]
            
        except Exception as e:
            print(f"❌ Error searching products: {e}")
            return []