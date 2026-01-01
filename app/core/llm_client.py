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

        # System prompt
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

        user_message_with_context = f"""<context>
{context_text}
</context>

<user_question>
{user_message}
</user_question>

Hãy trả lời câu hỏi của khách hàng dựa trên context được cung cấp ở trên. 

LƯU Ý QUAN TRỌNG:
- Nếu context có thông tin về sản phẩm, hãy liệt kê ĐẦY ĐỦ tất cả các sản phẩm liên quan
- Đừng nói "hiện tại tôi chưa có thông tin" nếu context đã có thông tin
- Nếu có nhiều sản phẩm, hãy trình bày từng sản phẩm một cách chi tiết
- Nếu thực sự không có thông tin trong context, hãy nói rõ và gợi ý cách tìm thông tin
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

        # Group by collection type
        grouped = {}
        for doc in context:
            collection = doc.get("collection", "unknown")
            if collection not in grouped:
                grouped[collection] = []
            grouped[collection].append(doc)

        context_parts = []
        
        if "products" in grouped:
            context_parts.append("=== THÔNG TIN SẢN PHẨM ===")
            for i, doc in enumerate(grouped["products"], 1):
                score = doc.get("weighted_score", 0)
                text = doc.get("text", "")
                context_parts.append(f"\nSản phẩm {i} (Độ liên quan: {score:.2f}):\n{text}")
            context_parts.append("\n")
        
        # Categories
        if "categories" in grouped:
            context_parts.append("=== DANH MỤC SẢN PHẨM ===")
            for i, doc in enumerate(grouped["categories"], 1):
                text = doc.get("text", "")
                context_parts.append(f"\n{text}")
            context_parts.append("\n")
        
        # FAQs
        if "faqs" in grouped:
            context_parts.append("=== CÂU HỎI THƯỜNG GẶP ===")
            for i, doc in enumerate(grouped["faqs"], 1):
                text = doc.get("text", "")
                context_parts.append(f"\n{text}")
            context_parts.append("\n")
        
        # Policies
        if "policies" in grouped:
            context_parts.append("=== CHÍNH SÁCH ===")
            for i, doc in enumerate(grouped["policies"], 1):
                text = doc.get("text", "")
                context_parts.append(f"\n{text}")
            context_parts.append("\n")
        
        # Order guides
        if "order_guides" in grouped:
            context_parts.append("=== HƯỚNG DẪN ĐƠN HÀNG ===")
            for i, doc in enumerate(grouped["order_guides"], 1):
                text = doc.get("text", "")
                context_parts.append(f"\n{text}")
            context_parts.append("\n")

        return "\n".join(context_parts)

    def _get_default_system_prompt(self) -> str:
        return """Bạn là trợ lý AI thông minh của một cửa hàng thời trang trực tuyến, chuyên tư vấn sản phẩm và hỗ trợ khách hàng.

# VAI TRÒ & MỤC TIÊU
- Tư vấn sản phẩm thời trang (quần áo, giày dép, phụ kiện)
- Hỗ trợ về đơn hàng, chính sách, quy trình
- Giải đáp thắc mắc một cách thân thiện, chuyên nghiệp

# NGUYÊN TẮC TRẢ LỜI

1. **Độ chính xác**: SỬ DỤNG thông tin có trong context được cung cấp
   - Context được cấu trúc theo sections: THÔNG TIN SẢN PHẨM, DANH MỤC, FAQs, v.v.
   - QUAN TRỌNG: Nếu section "THÔNG TIN SẢN PHẨM" có dữ liệu → HÃY SỬ DỤNG và liệt kê ĐẦY ĐỦ
   - Chỉ nói "chưa có thông tin" khi context THỰC SỰ trống hoặc không liên quan
   - Không bỏ qua thông tin đã có trong context

2. **Cấu trúc câu trả lời**:
   - Trả lời trực tiếp vào trọng tâm câu hỏi
   - Với câu hỏi đơn giản: 2-3 câu ngắn gọn
   - Với câu hỏi phức tạp: trả lời đầy đủ, có cấu trúc rõ ràng
   - Với câu hỏi về nhiều sản phẩm: liệt kê TỪNG sản phẩm chi tiết

3. **Định dạng khi cần thiết**:
   ```
   Khi giới thiệu sản phẩm:
   - Tên sản phẩm
   - Giá cả (chính xác từ context)
   - Size có sẵn (nếu có)
   - Đánh giá (nếu có)
   - Điểm nổi bật
   
   Khi hướng dẫn quy trình:
   - Liệt kê các bước rõ ràng
   - Giải thích ngắn gọn mỗi bước
   ```

4. **Xử lý các tình huống**:
   - Tìm sản phẩm → Liệt kê chi tiết TỪNG sản phẩm phù hợp
   - Hỏi về giá → Báo giá chính xác từ context
   - Hỏi về chính sách → Trích dẫn đầy đủ quy định
   - Hỏi về đơn hàng → Hướng dẫn cụ thể từng bước
   - Không tìm thấy thông tin → Gợi ý liên hệ trực tiếp hoặc cách tìm khác

5. **Giọng điệu**:
   - Thân thiện, nhiệt tình nhưng không phải nịnh hót
   - Chuyên nghiệp, đáng tin cậy
   - Tránh câu cửa miệng như "Chào bạn! 😊" ở mỗi câu trả lời

6. **Tối ưu trải nghiệm**:
   - Câu hỏi ngắn → Trả lời ngắn, súc tích
   - Câu hỏi dài/phức tạp → Trả lời đầy đủ, có cấu trúc
   - Luôn kết thúc bằng việc hỏi "Bạn cần hỗ trợ thêm gì không?" nếu phù hợp

# LƯU Ý QUAN TRỌNG
- GIÁ CẢ: Luôn báo giá chính xác từ context, định dạng "XXX,XXX VNĐ"
- SIZE: Liệt kê đầy đủ size có sẵn nếu có trong context
- SỐ LƯỢNG: Thông báo tình trạng còn hàng nếu có
- ĐÁNH GIÁ: Trích dẫn đánh giá thực tế từ khách hàng nếu có
- KHÔNG BỎ SÓT: Nếu context có 5 sản phẩm về áo thun → PHẢI liệt kê CẢ 5, không được bỏ sót

# VÍ DỤ TRẢ LỜI TỐT

User: "Có áo thun nam không?"
Bot: "Có ạ! Shop hiện có các mẫu áo thun nam sau:

1. **Áo thun Basic Cotton** - 250,000 VNĐ
   - Size: S, M, L, XL (còn hàng đầy đủ)
   - Đánh giá: 4.5/5 sao
   - Chất liệu cotton 100%, thoáng mát

2. **Áo thun Premium Polo** - 350,000 VNĐ
   - Size: M, L, XL
   - Đánh giá: 4.8/5 sao
   - Thiết kế sang trọng, phù hợp đi làm

Bạn thích mẫu nào hoặc cần tôi tư vấn thêm?"

---

User: "Làm sao để đổi size?"
Bot: "Để đổi size, bạn làm theo các bước sau:

1. **Điều kiện**: Sản phẩm còn nguyên tem mác, chưa qua sử dụng
2. **Thời gian**: Trong vòng 7 ngày kể từ khi nhận hàng
3. **Quy trình**:
   - Liên hệ hotline hoặc chat với shop
   - Cung cấp mã đơn hàng và size muốn đổi
   - Shop sẽ kiểm tra tồn kho và hỗ trợ đổi hàng
4. **Phí**: Miễn phí đổi size lần đầu tiên

Bạn cần đổi size cho đơn hàng nào? Tôi có thể hỗ trợ ngay."

Hãy trả lời bằng tiếng Việt, tự nhiên và hữu ích nhất có thể!"""