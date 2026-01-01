import chromadb
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from typing import List, Dict, Optional

class SearchService:
    def __init__(self):
        self.model = SentenceTransformer(settings.MODEL_ENCODE)
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        
        # Định nghĩa các collections và trọng số
        self.collections_config = {
            "products": {
                "name": f"{settings.CHROMA_COLLECTION}_products",
                "weight": 1.0,
                "enabled": True,
                "description": "Thông tin sản phẩm, giá, size, reviews"
            },
            "categories": {
                "name": f"{settings.CHROMA_COLLECTION}_categories",
                "weight": 0.8,
                "enabled": True,
                "description": "Danh mục sản phẩm"
            },
            "faqs": {
                "name": f"{settings.CHROMA_COLLECTION}_faqs",
                "weight": 0.95,
                "enabled": True,
                "description": "Câu hỏi thường gặp"
            },
            "policies": {
                "name": f"{settings.CHROMA_COLLECTION}_policies",
                "weight": 0.9,
                "enabled": True,
                "description": "Chính sách shop"
            },
            "order_guides": {
                "name": f"{settings.CHROMA_COLLECTION}_order_guides",
                "weight": 0.92,
                "enabled": True,
                "description": "Hướng dẫn về đơn hàng"
            }
        }
        
        # Load collections
        self.collections = {}
        for key, config in self.collections_config.items():
            if config["enabled"]:
                try:
                    self.collections[key] = self.client.get_collection(config["name"])
                    print(f"✅ Loaded collection: {config['name']}")
                except Exception as e:
                    print(f"⚠️ Không thể load collection {config['name']}: {e}")
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        collections: Optional[List[str]] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        
        # Encode query
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()
        
        # Nếu không chỉ định collections, search all
        if collections is None:
            collections = list(self.collections.keys())
        
        all_results = []
        
        # Search từng collection
        for coll_key in collections:
            if coll_key not in self.collections:
                continue
            
            collection = self.collections[coll_key]
            weight = self.collections_config[coll_key]["weight"]
            
            try:
                # Query collection
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results * 2,  # Lấy nhiều hơn để có pool lớn
                    where=filter_metadata,
                    include=["documents", "metadatas", "distances"]
                )
                
                # Parse results
                if results and results['ids'] and len(results['ids'][0]) > 0:
                    for i, doc_id in enumerate(results['ids'][0]):
                        all_results.append({
                            "id": doc_id,
                            "text": results['documents'][0][i],
                            "metadata": results['metadatas'][0][i],
                            "distance": results['distances'][0][i],
                            "collection": coll_key,
                            "weighted_score": (1 - results['distances'][0][i]) * weight
                        })
            
            except Exception as e:
                print(f"⚠️ Lỗi khi search collection {coll_key}: {e}")
        
        # Sort theo weighted_score (cao nhất = relevant nhất)
        all_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        # Trả về top n_results
        return all_results[:n_results]
    
    def smart_search(self, query: str, n_results: int = 5) -> Dict:
        query_lower = query.lower()
        
        # Phân loại intent
        intent = self._classify_intent(query_lower)
        
        results = {
            "intent": intent,
            "query": query,
            "results": []
        }
        
        if intent == "product_search":
            results["results"] = self.search(
                query, n_results, collections=["products", "categories"]
            )
        elif intent == "order_inquiry":
            results["results"] = self.search(
                query, n_results, collections=["order_guides", "policies", "faqs"]
            )
        elif intent == "support":
            results["results"] = self.search(
                query, n_results, collections=["faqs", "policies"]
            )
        else:
            # General search
            results["results"] = self.search(query, n_results)
        
        return results
    
    def _classify_intent(self, query: str) -> str:
        # Keywords cho product search
        product_keywords = [
            "sản phẩm", "áo", "quần", "giày", "mua", "giá", "size", "màu",
            "váy", "đầm", "shop", "bán", "có", "tìm", "mẫu"
        ]
        
        # Keywords cho order inquiry
        order_keywords = [
            "đơn hàng", "order", "giao hàng", "ship", "tracking", "trạng thái",
            "hủy đơn", "đặt hàng", "mua hàng", "thanh toán"
        ]
        
        # Keywords cho support
        support_keywords = [
            "làm sao", "như thế nào", "cách", "đổi trả", "hoàn tiền",
            "chính sách", "bảo hành", "liên hệ", "hotline"
        ]
        
        if any(keyword in query for keyword in product_keywords):
            return "product_search"
        elif any(keyword in query for keyword in order_keywords):
            return "order_inquiry"
        elif any(keyword in query for keyword in support_keywords):
            return "support"
        else:
            return "general"