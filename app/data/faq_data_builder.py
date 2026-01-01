import json
import chromadb
from sentence_transformers import SentenceTransformer
from app.core.config import settings

def get_faq_data():
    faqs = [
        {
            "id": "faq_1",
            "question": "Làm sao để kiểm tra đơn hàng của tôi?",
            "answer": """
            Bạn có thể kiểm tra đơn hàng bằng cách:
            1. Đăng nhập vào tài khoản trên website
            2. Vào mục "Đơn hàng của tôi"
            3. Xem chi tiết trạng thái đơn hàng và mã vận đơn
            Hoặc liên hệ với chúng tôi qua hotline với mã đơn hàng để được hỗ trợ.
            """,
            "category": "order_tracking",
            "keywords": ["kiểm tra đơn hàng", "tracking", "tra cứu", "đơn hàng"]
        },
        {
            "id": "faq_2",
            "question": "Tôi muốn hủy đơn hàng thì làm thế nào?",
            "answer": """
            Để hủy đơn hàng:
            - Nếu đơn hàng đang ở trạng thái "Chờ xác nhận": Bạn có thể hủy trực tiếp trên website
            - Nếu đơn hàng đã được xác nhận: Vui lòng liên hệ hotline ngay để được hỗ trợ
            - Đơn hàng đang giao không thể hủy, nhưng bạn có thể từ chối nhận hàng
            Lưu ý: Nếu đã thanh toán online, tiền sẽ được hoàn lại trong 5-7 ngày làm việc.
            """,
            "category": "order_cancellation",
            "keywords": ["hủy đơn", "cancel order", "không muốn mua"]
        },
        {
            "id": "faq_3",
            "question": "Sản phẩm tôi nhận được bị lỗi, tôi phải làm gì?",
            "answer": """
            Nếu sản phẩm bị lỗi:
            1. Chụp ảnh/video sản phẩm lỗi làm bằng chứng
            2. Liên hệ với chúng tôi qua hotline hoặc chat trong vòng 48h kể từ khi nhận hàng
            3. Cung cấp mã đơn hàng và hình ảnh sản phẩm lỗi
            Chúng tôi sẽ:
            - Đổi sản phẩm mới miễn phí (bao gồm phí vận chuyển)
            - Hoặc hoàn tiền 100% nếu không còn hàng thay thế
            """,
            "category": "product_issue",
            "keywords": ["sản phẩm lỗi", "hàng bị hỏng", "defect", "warranty"]
        },
        {
            "id": "faq_4",
            "question": "Tôi có thể đổi size/màu sau khi đã đặt hàng không?",
            "answer": """
            Đổi size/màu sau khi đặt hàng:
            - Nếu đơn hàng chưa được giao: Liên hệ ngay với chúng tôi để thay đổi
            - Nếu đơn hàng đã giao: Áp dụng chính sách đổi trả trong 7 ngày
            - Miễn phí đổi size/màu lần đầu tiên
            - Sản phẩm cần còn nguyên tem mác, chưa qua sử dụng
            """,
            "category": "exchange",
            "keywords": ["đổi size", "đổi màu", "change size", "exchange"]
        },
        {
            "id": "faq_5",
            "question": "Mất bao lâu để nhận được hàng?",
            "answer": """
            Thời gian giao hàng:
            - Nội thành Hà Nội/TP.HCM: 1-2 ngày
            - Các tỉnh thành khác: 3-5 ngày làm việc
            - Vùng sâu, vùng xa: 5-7 ngày làm việc
            - Dịch vụ giao nhanh 24h: chỉ áp dụng tại Hà Nội và TP.HCM (phụ thu phí)
            Thời gian tính từ khi đơn hàng được xác nhận và đóng gói.
            """,
            "category": "shipping_time",
            "keywords": ["thời gian giao hàng", "bao lâu", "delivery time"]
        },
        {
            "id": "faq_6",
            "question": "Tôi có bắt buộc phải có tài khoản để đặt hàng không?",
            "answer": """
            Có. Hiện tại, bạn cần đăng ký và đăng nhập tài khoản để có thể đặt hàng trên hệ thống của chúng tôi.

            Việc yêu cầu tài khoản giúp chúng tôi:
            - Quản lý và theo dõi trạng thái đơn hàng chính xác
            - Lưu thông tin giao hàng để đặt hàng nhanh hơn cho lần sau
            - Hỗ trợ chăm sóc khách hàng tốt hơn khi có vấn đề phát sinh
            - Cung cấp ưu đãi, khuyến mãi và tích điểm thành viên

            Việc đăng ký tài khoản hoàn toàn miễn phí và chỉ mất khoảng 30 giây.
            """,
            "category": "account",
            "keywords": ["bắt buộc tài khoản", "đăng nhập", "đăng ký", "đặt hàng"]
        },
        {
            "id": "faq_7",
            "question": "Shop có chương trình khuyến mãi nào không?",
            "answer": """
            Shop thường xuyên có các chương trình khuyến mãi:
            - Sale cuối tuần: giảm 20-50% các sản phẩm
            - Flash sale hàng ngày: giảm sốc trong thời gian giới hạn
            - Mã giảm giá cho khách hàng mới: giảm 10% đơn đầu tiên
            - Ưu đãi sinh nhật: giảm 15% trong tháng sinh nhật
            - Freeship cho đơn từ 500k
            Theo dõi fanpage và website để không bỏ lỡ ưu đãi nào nhé!
            """,
            "category": "promotion",
            "keywords": ["khuyến mãi", "giảm giá", "sale", "promotion", "voucher"]
        },
        {
            "id": "faq_8",
            "question": "Làm sao để liên hệ với shop?",
            "answer": """
            Bạn có thể liên hệ với chúng tôi bằng các cách sau:

            - Chat trực tiếp với admin ngay trên website để được hỗ trợ nhanh chóng
            - Gọi hotline: 19001111 (8h – 22h hàng ngày)

            Đội ngũ hỗ trợ của chúng tôi luôn sẵn sàng giải đáp mọi thắc mắc của bạn.
            """,
            "category": "contact",
            "keywords": ["liên hệ", "chat với admin", "hotline", "hỗ trợ"]
        }
    ]
    
    return faqs

def build_faq_document(faq: dict) -> dict:
    """Tạo document từ FAQ"""
    text = f"Câu hỏi: {faq['question']} Trả lời: {faq['answer'].strip()}"
    
    return {
        "id": faq["id"],
        "text": " ".join(text.split()),  # Clean whitespace
        "metadata": {
            "type": "faq",
            "category": faq["category"],
            "question": faq["question"],
            "keywords": ",".join(faq["keywords"])
        }
    }

def save_faqs_json(faqs, output_path="app/data/json/faqs_chunks.json"):
    """Lưu FAQs dạng JSON"""
    try:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        documents = [build_faq_document(f) for f in faqs]
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Đã lưu {len(documents)} FAQ documents vào {output_path}")
        return documents
        
    except Exception as e:
        print(f"❌ Lỗi khi lưu JSON: {e}")
        return []

def embed_faqs_to_chroma(documents):
    """Tạo embeddings và lưu vào ChromaDB"""
    if not documents:
        print("⚠️ Không có documents để embed")
        return
    
    try:
        print("📦 Đang load embedding model...")
        model = SentenceTransformer(settings.MODEL_ENCODE)
        
        print("🔗 Đang kết nối ChromaDB...")
        client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        
        collection_name = f"{settings.CHROMA_COLLECTION}_faqs"
        
        if collection_name in [c.name for c in client.list_collections()]:
            client.delete_collection(collection_name)
            print(f"🗑️ Đã xóa collection cũ: {collection_name}")
        
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "FAQ embeddings for e-commerce chatbot"}
        )
        
        batch_texts = [doc["text"] for doc in documents]
        batch_ids = [doc["id"] for doc in documents]
        batch_metadata = [doc["metadata"] for doc in documents]
        
        print("🔄 Đang tạo embeddings...")
        embeddings = model.encode(
            batch_texts, 
            convert_to_numpy=True, 
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        collection.upsert(
            ids=batch_ids,
            embeddings=embeddings.tolist(),
            metadatas=batch_metadata,
            documents=batch_texts
        )
        
        print(f"✅ Đã embed {len(documents)} FAQs vào ChromaDB")
        
    except Exception as e:
        print(f"❌ Lỗi khi embed vào ChromaDB: {e}")

def build_faq_embeddings():
    faqs = get_faq_data()
    documents = save_faqs_json(faqs)
    
    if not documents:
        return
    
    embed_faqs_to_chroma(documents)

if __name__ == "__main__":
    build_faq_embeddings()