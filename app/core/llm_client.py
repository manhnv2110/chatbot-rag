from typing import List, Dict, Optional
from groq import Groq
from app.core.config import settings


class GroqClient:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.MODEL_GENERATE 

    def generate_response(
        self,
        user_message: str,
        context: List[Dict],
        conversation_history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate response từ Groq với RAG context
        """

        context_text = self._build_context_text(context)

        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        messages = []

        # System prompt (Groq/OpenAI-style)
        messages.append({
            "role": "system",
            "content": system_prompt
        })

        # Conversation history
        if conversation_history:
            for msg in conversation_history[-10:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # User message with RAG context
        user_message_with_context = f"""<context>
{context_text}
</context>

<user_question>
{user_message}
</user_question>

Hãy trả lời câu hỏi của khách hàng dựa trên context được cung cấp. 
Nếu context không chứa thông tin cần thiết, hãy nói rõ và đưa ra câu trả lời chung.
"""

        messages.append({
            "role": "user",
            "content": user_message_with_context
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"❌ Error calling Groq API: {e}")
            return "Xin lỗi, tôi đang gặp sự cố kỹ thuật. Vui lòng thử lại sau."

    def generate_stream_response(
        self,
        user_message: str,
        context: List[Dict],
        conversation_history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Generate streaming response (real-time chat)
        """

        context_text = self._build_context_text(context)

        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        messages = []

        messages.append({
            "role": "system",
            "content": system_prompt
        })

        if conversation_history:
            for msg in conversation_history[-10:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        user_message_with_context = f"""<context>
{context_text}
</context>

<user_question>
{user_message}
</user_question>

Hãy trả lời câu hỏi của khách hàng dựa trên context được cung cấp.
"""

        messages.append({
            "role": "user",
            "content": user_message_with_context
        })

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            print(f"❌ Error streaming Groq response: {e}")
            yield "Xin lỗi, tôi đang gặp sự cố kỹ thuật."

    def _build_context_text(self, context: List[Dict]) -> str:
        if not context:
            return "Không có thông tin liên quan."

        context_parts = []
        for i, doc in enumerate(context, 1):
            collection = doc.get("collection", "unknown")
            text = doc.get("text", "")
            score = doc.get("weighted_score", 0)

            context_parts.append(
                f"""Document {i} [Source: {collection}, Relevance: {score:.2f}]:
{text}
"""
            )

        return "\n".join(context_parts)

    def _get_default_system_prompt(self) -> str:
        return """Bạn là trợ lý AI của một cửa hàng thời trang trực tuyến.
Mục tiêu chính:
- Trả lời đúng trọng tâm câu hỏi của người dùng
- Ngắn gọn, súc tích, không lan man

Nguyên tắc bắt buộc:
1. Trả lời trực tiếp vào kết luận trước
2. Chỉ dùng thông tin có trong context
3. Không giải thích dài dòng, không nhắc lại context
4. Không suy đoán, không tự mở rộng câu trả lời
5. Nếu thiếu dữ liệu → trả lời ngắn gọn: “Hiện chưa có thông tin …”
6. Không đưa ra lời mời, không gợi ý thêm trừ khi được hỏi
7. Giữ giọng thân thiện, chuyên nghiệp
8. Trả lời bằng tiếng Việt

Định dạng câu trả lời:
- Ưu tiên 1–2 câu
- Nếu cần liệt kê: dùng gạch đầu dòng
- Không quá 3 câu cho mỗi câu trả lời

"""
