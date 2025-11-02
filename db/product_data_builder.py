from db.database import connectDB
import logging 
import json 
logging.basicConfig(level=logging.INFO)

def clean_text(text):
    return " ".join(text.split()).strip()

def build_product_text(product):
    return clean_text(f"Sản phẩm: {product['name']}. Loại: {product['type']}. Màu: {product['color']}. Giá: {product['price']} VNĐ")

def build_product_document(product):
    return {
        "id": product["id"],
        "text": build_product_text(product),
        "metadata": {
            "type": product["type"],
            "price": product["price"],
            "color": product["color"]
        }
    }

def fetch_products(limit=None):
    try: 
        products = []
        with connectDB() as conn:
            with conn.cursor() as cur:
                query = "SELECT id, name, price, type, color FROM products"
                if limit: 
                    query += f" LIMIT {limit}"
                cur.execute(query)
                records = cur.fetchall()

                for record in records:
                    products.append({
                        'id': record['id'],
                        'name': record['name'],
                        'price': record['price'],
                        'type': record['type'],
                        'color': record['color']
                    }) 
        logging.info(f"Fetched {len(products)} products from DB")
        return products
    except Exception as e:
        logging.error(f"Error fetching products: {e}")
        return []
        
if __name__ == "__main__":
    products = fetch_products()
    products_docs = [build_product_document(p) for p in products]
    
    with open("products_chunks.json", "w", encoding="utf-8") as f:
        json.dump(products_docs, f, ensure_ascii=False, indent=2)
    
    print(f"Đã tạo file products_chunks.json.")