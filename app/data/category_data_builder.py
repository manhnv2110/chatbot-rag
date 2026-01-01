from app.core.database import connectDB
import json 
import chromadb
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from typing import List, Dict

def clean_text(text: str) -> str:
    return " ".join(text.split()).strip() if text else ""

def build_category_text(category: dict) -> str:
    parts = [
        f"Danh mục: {category['name']}"
    ]
    
    if category.get('description'):
        parts.append(f"Mô tả: {category['description']}")
    
    if category.get('product_count', 0) > 0:
        parts.append(f"Có {category['product_count']} sản phẩm trong danh mục này")
    
    # Liệt kê các sản phẩm mẫu
    if category.get('sample_products'):
        products_list = ", ".join([p['name'] for p in category['sample_products'][:8]])
        parts.append(f"Các sản phẩm tiêu biểu: {products_list}")
    
    # Thêm thông tin giá trung bình
    if category.get('avg_price'):
        parts.append(f"Giá trung bình: {category['avg_price']:,.0f} VNĐ")
    
    if category.get('price_range'):
        min_price = category['price_range']['min']
        max_price = category['price_range']['max']
        parts.append(f"Khoảng giá: {min_price:,.0f} - {max_price:,.0f} VNĐ")
    
    return clean_text(". ".join(parts) + ".")

def build_category_document(category: dict) -> dict:
    return {
        "id": f"category_{category['id']}",
        "text": build_category_text(category),
        "metadata": {
            "type": "category",
            "category_id": category["id"],
            "category_name": category["name"],
            "product_count": category.get("product_count", 0),
            "avg_price": float(category.get("avg_price", 0)) if category.get("avg_price") else 0,
            "min_price": float(category.get("price_range", {}).get("min", 0)),
            "max_price": float(category.get("price_range", {}).get("max", 0))
        }
    }

def fetch_categories_from_db() -> List[Dict]:
    categories = []
    
    try:
        with connectDB() as conn:
            with conn.cursor() as cur:
                query = """
                    SELECT 
                        c.id,
                        c.name,
                        c.description,
                        COUNT(p.id) as product_count,
                        AVG(p.price) as avg_price,
                        MIN(p.price) as min_price,
                        MAX(p.price) as max_price
                    FROM categories c
                    LEFT JOIN products p ON c.id = p.category_id
                    GROUP BY c.id, c.name, c.description
                    ORDER BY c.name
                """
                cur.execute(query)
                category_records = cur.fetchall()

                for record in category_records:
                    category = {
                        "id": record['id'],
                        "name": record['name'],
                        "description": record['description'],
                        "product_count": record['product_count'],
                        "avg_price": record['avg_price'],
                        "price_range": {
                            "min": record['min_price'] if record['min_price'] else 0,
                            "max": record['max_price'] if record['max_price'] else 0
                        },
                        "sample_products": []
                    }
                    
                    sample_query = """
                        SELECT 
                            p.name, 
                            p.price,
                            COALESCE(AVG(r.rating), 0) as avg_rating
                        FROM products p
                        LEFT JOIN reviews r ON p.id = r.product_id
                        WHERE p.category_id = %s
                        GROUP BY p.id, p.name, p.price
                        ORDER BY avg_rating DESC, p.price DESC
                        LIMIT 8
                    """
                    cur.execute(sample_query, (category["id"],))
                    sample_records = cur.fetchall()

                    for sample in sample_records:
                        category["sample_products"].append({
                            "name": sample['name'],
                            "price": sample['price'],
                            "rating": sample['avg_rating']
                        })
                    
                    categories.append(category)
        
        print(f"✅ Đã lấy {len(categories)} danh mục từ database")
        return categories
        
    except Exception as e:
        print(f"❌ Lỗi khi lấy dữ liệu danh mục: {e}")
        import traceback
        traceback.print_exc()
        return []

def save_categories_json(categories: List[Dict], output_path="app/data/json/categories_chunks.json"):
    try:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        documents = [build_category_document(c) for c in categories]
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Đã lưu {len(documents)} documents vào {output_path}")
        return documents
        
    except Exception as e:
        print(f"❌ Lỗi khi lưu JSON: {e}")
        return []

def embed_categories_to_chroma(documents: List[Dict]):
    if not documents:
        print("⚠️ Không có documents để embed")
        return
    
    try:
        print("📦 Đang load embedding model...")
        model = SentenceTransformer(settings.MODEL_ENCODE)
        
        print("🔗 Đang kết nối ChromaDB...")
        client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        
        collection_name = f"{settings.CHROMA_COLLECTION}_categories"
        
        if collection_name in [c.name for c in client.list_collections()]:
            client.delete_collection(collection_name)
            print(f"🗑️ Đã xóa collection cũ: {collection_name}")
        
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "Category embeddings for e-commerce chatbot"}
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
        
        print(f"✅ Đã embed {len(documents)} danh mục vào ChromaDB")
        
    except Exception as e:
        print(f"❌ Lỗi khi embed vào ChromaDB: {e}")
        import traceback
        traceback.print_exc()

def build_categories_embeddings():
    categories = fetch_categories_from_db()
    if not categories:
        print("⚠️ Không có danh mục nào để xử lý")
        return
    
    documents = save_categories_json(categories)
    if not documents:
        return
    
    embed_categories_to_chroma(documents)

if __name__ == "__main__":
    build_categories_embeddings()