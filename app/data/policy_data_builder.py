import json
import chromadb
from sentence_transformers import SentenceTransformer
from app.core.config import settings

def get_policy_data():
    """
    Dữ liệu chính sách của shop
    Bạn nên tùy chỉnh theo chính sách thực tế của shop
    """
    policies = [
        {
            "id": "policy_shipping_1",
            "type": "shipping",
            "title": "Chính sách giao hàng",
            "content": """
            Chính sách giao hàng của shop:
            - Giao hàng toàn quốc trong 2-5 ngày làm việc
            - Miễn phí giao hàng cho đơn hàng từ 500.000 VNĐ trở lên
            - Phí vận chuyển tiêu chuẩn: 30.000 VNĐ (nội thành), 50.000 VNĐ (ngoại thành)
            - Giao hàng nhanh trong 24h: phụ thu 50.000 VNĐ (chỉ áp dụng tại Hà Nội và TP.HCM)
            - Kiểm tra hàng trước khi thanh toán
            """,
            "keywords": ["giao hàng", "vận chuyển", "ship", "delivery", "phí ship"]
        },
        {
            "id": "policy_return_1",
            "type": "return",
            "title": "Chính sách đổi trả",
            "content": """
            Chính sách đổi trả hàng:
            - Đổi trả trong vòng 7 ngày kể từ ngày nhận hàng
            - Sản phẩm còn nguyên tem mác, chưa qua sử dụng
            - Miễn phí đổi size/màu lần đầu tiên
            - Hoàn tiền 100% nếu sản phẩm lỗi từ nhà sản xuất
            - Không áp dụng đổi trả với sản phẩm sale trên 50%
            - Liên hệ hotline hoặc chat để được hỗ trợ đổi trả
            """,
            "keywords": ["đổi trả", "hoàn tiền", "return", "refund", "đổi hàng"]
        },
        {
            "id": "policy_payment_1",
            "type": "payment",
            "title": "Phương thức thanh toán",
            "content": """
            Có 2 phương thức thanh toán được chấp nhận:
            - COD (Thanh toán khi nhận hàng): áp dụng toàn quốc
            - Thanh toán qua VNPay: an toàn, bảo mật
            Tất cả giao dịch đều được mã hóa và bảo mật
            """,
            "keywords": ["thanh toán", "payment", "COD", "VNPay"]
        },
        {
            "id": "policy_warranty_1",
            "type": "warranty",
            "title": "Chính sách bảo hành",
            "content": """
            Chính sách bảo hành sản phẩm:
            - Bảo hành 12 tháng với lỗi từ nhà sản xuất
            - Miễn phí sửa chữa, thay thế linh kiện bị hỏng
            - Không áp dụng bảo hành với hư hỏng do người dùng
            - Vui lòng giữ hóa đơn để được bảo hành
            - Liên hệ trung tâm bảo hành để được hỗ trợ
            """,
            "keywords": ["bảo hành", "warranty", "sửa chữa", "lỗi sản phẩm"]
        },
        {
            "id": "policy_size_guide_1",
            "type": "size_guide",
            "title": "Hướng dẫn chọn size",
            "content": """
            Hướng dẫn chọn size phù hợp:
            - Size S: Chiều cao 1m50-1m60, cân nặng 45-52kg
            - Size M: Chiều cao 1m60-1m68, cân nặng 52-60kg
            - Size L: Chiều cao 1m68-1m75, cân nặng 60-70kg
            - Size XL: Chiều cao 1m75-1m80, cân nặng 70-80kg
            Nếu bạn nằm giữa 2 size, nên chọn size lớn hơn để thoải mái
            Miễn phí đổi size lần đầu nếu không vừa
            """,
            "keywords": ["size", "chọn size", "số đo", "chiều cao", "cân nặng"]
        },
        {
            "id": "policy_privacy_1",
            "type": "privacy",
            "title": "Chính sách bảo mật",
            "content": """
            Cam kết bảo mật thông tin khách hàng:
            - Thông tin cá nhân được mã hóa và bảo mật tuyệt đối
            - Không chia sẻ thông tin cho bên thứ ba
            - Chỉ sử dụng thông tin cho mục đích giao hàng và chăm sóc khách hàng
            - Khách hàng có quyền yêu cầu xóa dữ liệu cá nhân
            - Tuân thủ nghiêm ngặt luật bảo vệ dữ liệu cá nhân
            """,
            "keywords": ["bảo mật", "privacy", "thông tin cá nhân", "dữ liệu"]
        }
    ]
    
    return policies

def build_policy_document(policy: dict) -> dict:
    """Tạo document từ policy"""
    text = f"{policy['title']}. {policy['content'].strip()}"
    
    return {
        "id": policy["id"],
        "text": " ".join(text.split()),  # Clean whitespace
        "metadata": {
            "type": policy["type"],
            "title": policy["title"],
            "keywords": ",".join(policy["keywords"])
        }
    }

def save_policies_json(policies, output_path="app/data/json/policies_chunks.json"):
    """Lưu policies dạng JSON"""
    try:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        documents = [build_policy_document(p) for p in policies]
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Đã lưu {len(documents)} policy documents vào {output_path}")
        return documents
        
    except Exception as e:
        print(f"❌ Lỗi khi lưu JSON: {e}")
        return []

def embed_policies_to_chroma(documents):
    """Tạo embeddings và lưu vào ChromaDB"""
    if not documents:
        print("⚠️ Không có documents để embed")
        return
    
    try:
        print("📦 Đang load embedding model...")
        model = SentenceTransformer(settings.MODEL_ENCODE)
        
        print("🔗 Đang kết nối ChromaDB...")
        client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        
        collection_name = f"{settings.CHROMA_COLLECTION}_policies"
        
        if collection_name in [c.name for c in client.list_collections()]:
            client.delete_collection(collection_name)
            print(f"🗑️ Đã xóa collection cũ: {collection_name}")
        
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "Policy embeddings for e-commerce chatbot"}
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
        
        print(f"✅ Đã embed {len(documents)} chính sách vào ChromaDB")
        
    except Exception as e:
        print(f"❌ Lỗi khi embed vào ChromaDB: {e}")

def build_policy_embeddings():
    policies = get_policy_data()
    documents = save_policies_json(policies)
    
    if not documents:
        return
    
    embed_policies_to_chroma(documents)

if __name__ == "__main__":
    build_policy_embeddings()