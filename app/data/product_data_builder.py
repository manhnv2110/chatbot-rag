from app.core.database import connectDB
import json 
import chromadb
from tqdm import tqdm
from app.core.config import settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict

def clean_text(text: str) -> str:
    return " ".join(text.split()).strip() if text else ""

def build_product_text(product: dict) -> str:
    parts = [
        f"Sản phẩm: {product['name']}",
        f"Danh mục: {product['category_name']}",
        f"Giá: {product['price']:,.0f} VNĐ"
    ]
    
    if product.get('description'):
        parts.append(f"Mô tả: {product['description']}")
    
    if product.get('variants') and len(product['variants']) > 0:
        sizes_info = []
        total_stock = 0
        for variant in product['variants']:
            size = variant.get('size', 'Standard')
            quantity = variant.get('quantity', 0)
            sizes_info.append(f"{size} (còn {quantity} sản phẩm)")
            total_stock += quantity
        
        if sizes_info:
            parts.append(f"Size có sẵn: {', '.join(sizes_info)}")
            parts.append(f"Tổng tồn kho: {total_stock} sản phẩm")
    
    if product.get('avg_rating') and product.get('review_count'):
        parts.append(
            f"Đánh giá: {product['avg_rating']:.1f}/5 sao "
            f"từ {product['review_count']} người mua"
        )
        
        if product.get('sample_reviews'):
            reviews_text = []
            for review in product['sample_reviews'][:3]:
                reviews_text.append(
                    f"'{review['comment']}' ({review['rating']} sao)"
                )
            if reviews_text:
                parts.append(f"Nhận xét: {'; '.join(reviews_text)}")
    
    if product.get('image_count', 0) > 0:
        parts.append(f"Có {product['image_count']} hình ảnh sản phẩm")
    
    return clean_text(". ".join(parts) + ".")

def build_product_document(product: dict) -> dict:
    return {
        "id": f"product_{product['id']}",
        "text": build_product_text(product),
        "metadata": {
            "type": "product",
            "product_id": product["id"],
            "product_name": product["name"],
            "category_id": product.get("category_id"),
            "category_name": product.get("category_name", ""),
            "price": float(product.get("price", 0)),
            "avg_rating": float(product.get("avg_rating", 0)) if product.get("avg_rating") else 0,
            "review_count": product.get("review_count", 0),
            "total_stock": sum(v.get('quantity', 0) for v in product.get('variants', [])),
            "has_variants": len(product.get("variants", [])) > 0,
            "image_count": product.get("image_count", 0)
        }
    }

def fetch_products_from_db() -> List[Dict]:
    products = []
    
    try:
        with connectDB() as conn:
            with conn.cursor() as cur:
                product_query = """
                    SELECT 
                        p.id,
                        p.name,
                        p.price,
                        p.description,
                        p.category_id,
                        c.name as category_name,
                        AVG(r.rating) as avg_rating,
                        COUNT(DISTINCT r.id) as review_count,
                        COUNT(DISTINCT piu.image_url) as image_count
                    FROM products p
                    LEFT JOIN categories c ON p.category_id = c.id
                    LEFT JOIN reviews r ON p.id = r.product_id
                    LEFT JOIN product_image_urls piu ON p.id = piu.product_id
                    GROUP BY p.id, p.name, p.price, p.description, p.category_id, c.name
                    ORDER BY p.created_at DESC
                """
                cur.execute(product_query)
                product_records = cur.fetchall()

                for record in product_records:
                    product = {
                        "id": record['id'],
                        "name": record['name'],
                        "price": record['price'],
                        "description": record['description'],
                        "category_id": record['category_id'],
                        "category_name": record['category_name'],
                        "avg_rating": record['avg_rating'],
                        "review_count": record['review_count'],
                        "image_count": record['image_count'],
                        "variants": [],
                        "sample_reviews": [],
                        "image_urls": []
                    }
                    
                    variant_query = """
                        SELECT id, size, quantity
                        FROM product_variants
                        WHERE product_id = %s
                        ORDER BY size
                    """
                    cur.execute(variant_query, (product["id"],))
                    variant_records = cur.fetchall()
                    
                    print(variant_records[0])

                    for variant in variant_records:
                        product["variants"].append({
                            "variant_id": variant["id"],
                            "size": variant["size"],
                            "quantity": variant["quantity"]
                        })
                    
                    review_query = """
                        SELECT comment, rating
                        FROM reviews
                        WHERE product_id = %s AND comment IS NOT NULL AND comment != ''
                        ORDER BY rating DESC, created_at DESC
                        LIMIT 3
                    """
                    cur.execute(review_query, (product["id"],))
                    review_records = cur.fetchall()

                    for review in review_records:
                        product["sample_reviews"].append({
                            "comment": review['comment'],
                            "rating": review['rating']
                        })
                    
                    image_query = """
                        SELECT image_url
                        FROM product_image_urls
                        WHERE product_id = %s
                        LIMIT 5
                    """
                    cur.execute(image_query, (product["id"],))
                    image_records = cur.fetchall()

                    product["image_urls"] = [img["image_url"] for img in image_records]
                    products.append(product)
                
        print(f"✅ Đã lấy {len(products)} sản phẩm từ database")
        return products
        
    except Exception as e:
        print(f"❌ Lỗi khi lấy dữ liệu sản phẩm: {e}")
        import traceback
        traceback.print_exc()
        return []

def save_products_json(products: List[Dict], output_path="app/data/json/products_chunks.json"):
    try:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        documents = [build_product_document(p) for p in products]
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Đã lưu {len(documents)} documents vào {output_path}")
        return documents
        
    except Exception as e:
        print(f"❌ Lỗi khi lưu JSON: {e}")
        return []

def embed_products_to_chroma(documents: List[Dict]):
    if not documents:
        print("⚠️ Không có documents để embed")
        return
    
    try:
        model = SentenceTransformer(settings.MODEL_ENCODE)
        
        client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        
        collection_name = f"{settings.CHROMA_COLLECTION}_products"
        
        if collection_name in [c.name for c in client.list_collections()]:
            client.delete_collection(collection_name)
        
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "Product embeddings for e-commerce chatbot"}
        )
        
        batch_size = 16
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(documents), batch_size), 
                     desc="Embedding products", 
                     total=total_batches):
            batch = documents[i:i+batch_size]
            batch_texts = [doc["text"] for doc in batch]
            batch_ids = [doc["id"] for doc in batch]
            batch_metadata = [doc["metadata"] for doc in batch]
            
            try:
                # Tạo embeddings
                batch_embeddings = model.encode(
                    batch_texts, 
                    convert_to_numpy=True, 
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
                
                # Lưu vào ChromaDB
                collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings.tolist(),
                    metadatas=batch_metadata,
                    documents=batch_texts
                )
            except Exception as e:
                print(f"⚠️ Lỗi khi xử lý batch {i // batch_size + 1}: {e}")
        
        print(f"✅ Đã embed {len(documents)} sản phẩm vào ChromaDB")
        
    except Exception as e:
        print(f"❌ Lỗi khi embed vào ChromaDB: {e}")
        import traceback
        traceback.print_exc()

def build_products_embeddings():
    # Bước 1: Lấy dữ liệu từ database
    products = fetch_products_from_db()
    if not products:
        print("⚠️ Không có sản phẩm nào để xử lý")
        return
    
    # Bước 2: Tạo documents và lưu JSON
    documents = save_products_json(products)
    if not documents:
        return
    
    # Bước 3: Tạo embeddings và lưu vào ChromaDB
    embed_products_to_chroma(documents)

if __name__ == "__main__":
    build_products_embeddings()