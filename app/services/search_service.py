import chromadb
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from typing import List, Dict, Optional
import numpy as np

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
                # Query collection với n_results lớn hơn để có pool tốt hơn
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results * 3,  # ✅ TĂNG GẤP 3 để có nhiều candidates
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
                            "weighted_score": (1 - results['distances'][0][i]) * weight,
                            "raw_score": 1 - results['distances'][0][i]  # ✅ LƯU raw score để debug
                        })
            
            except Exception as e:
                print(f"⚠️ Lỗi khi search collection {coll_key}: {e}")
        
        # Sort theo weighted_score
        all_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        return all_results[:n_results]
    
    def smart_search(self, query: str, n_results: int = 20) -> Dict:  # ✅ TĂNG default từ 50 → 20
        """
        Smart search với intent classification và adaptive retrieval
        """
        query_lower = query.lower()
        
        # Phân loại intent
        intent = self._classify_intent(query_lower)
        
        results = {
            "intent": intent,
            "query": query,
            "results": []
        }
        
        # ✅ ADAPTIVE SEARCH: Tùy intent mà điều chỉnh strategy
        if intent == "product_search":
            # Với product search, cần nhiều kết quả hơn
            results["results"] = self._product_focused_search(query, n_results)
            
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
        
        # ✅ RERANKING: Đảm bảo kết quả tốt nhất lên đầu
        results["results"] = self._rerank_results(query, results["results"])
        
        return results
    
    def _product_focused_search(self, query: str, n_results: int) -> List[Dict]:
        """
        ✅ MỚI: Search tập trung vào products với multi-stage retrieval
        """
        all_results = []
        
        # Stage 1: Search products trực tiếp
        product_results = self.search(
            query, 
            n_results=n_results,  # Lấy nhiều products
            collections=["products"]
        )
        all_results.extend(product_results)
        
        # Stage 2: Search categories để có context về nhóm sản phẩm
        category_results = self.search(
            query,
            n_results=max(3, n_results // 4),  # Lấy ít categories hơn
            collections=["categories"]
        )
        all_results.extend(category_results)
        
        # Merge và sort lại
        all_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        return all_results[:n_results]
    
    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        ✅ MỚI: Rerank kết quả dựa trên semantic similarity và relevance
        """
        if len(results) <= 1:
            return results
        
        query_lower = query.lower()
        
        # Extract keywords từ query
        product_keywords = self._extract_product_keywords(query_lower)
        
        # Rerank dựa trên keyword matching
        for result in results:
            boost_score = 0.0
            text_lower = result['text'].lower()
            
            # Boost nếu match exact keywords
            for keyword in product_keywords:
                if keyword in text_lower:
                    boost_score += 0.1
            
            # Boost products cao hơn khi query là product-related
            if result['collection'] == 'products' and product_keywords:
                boost_score += 0.05
            
            # Apply boost
            result['weighted_score'] += boost_score
        
        # Sort lại
        results.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        return results
    
    def _extract_product_keywords(self, query: str) -> List[str]:
        """
        ✅ MỚI: Extract product-related keywords từ query
        """
        product_terms = [
            "áo", "quần", "giày", "váy", "đầm", "thun", "sơ mi", "khoác",
            "jean", "kaki", "polo", "hoodie", "nỉ", "len", "dạ"
        ]
        
        found_keywords = []
        for term in product_terms:
            if term in query:
                found_keywords.append(term)
        
        return found_keywords
    
    def _classify_intent(self, query: str) -> str:
        """
        ✅ CẢI TIẾN: Intent classification chính xác hơn
        """
        # Keywords cho product search - MỞ RỘNG
        product_keywords = [
            "sản phẩm", "áo", "quần", "giày", "mua", "giá", "size", "màu",
            "váy", "đầm", "shop", "bán", "có", "tìm", "mẫu", "loại",
            "thun", "sơ mi", "khoác", "jean", "những", "nào", "gì"  # ✅ THÊM
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
        
        # ✅ SCORING-BASED classification thay vì first-match
        scores = {
            "product_search": 0,
            "order_inquiry": 0,
            "support": 0
        }
        
        for keyword in product_keywords:
            if keyword in query:
                scores["product_search"] += 1
        
        for keyword in order_keywords:
            if keyword in query:
                scores["order_inquiry"] += 1
        
        for keyword in support_keywords:
            if keyword in query:
                scores["support"] += 1
        
        # Return intent với score cao nhất
        max_intent = max(scores, key=scores.get)
        
        if scores[max_intent] > 0:
            return max_intent
        else:
            return "general"