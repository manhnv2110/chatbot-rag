from app.embedding.embedding_model import encode_text
from app.chroma.chroma_client import get_collection 
from app.config import DEFAULT_FETCH_N 

def retrieve(query_text, fetch_n=DEFAULT_FETCH_N, filters=None):
    if not query_text.strip():
        raise ValueError("Query text cannot be empty")

    # Encode query 
    query_embedding = encode_text([query_text])

    # Query Chroma 
    collection = get_collection()
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=fetch_n,
        include=["documents", "metadatas", "distances"]
    )

    # Process results
    retrieved = []
    for id, doc, metadata, dist in zip(results["ids"][0], 
                                       results["documents"][0], 
                                       results["metadatas"][0], 
                                       results["distances"][0]):
        # similarity score (cosine similarity)
        score = 1 - dist
        
        keep = True 
        if filters: 
            if "max_price" in filters and metadata.get("price") is not None:
                keep = keep and (metadata["price"] <= filters["max_price"])
            if "type" in filters and metadata.get("type") is not None:
                keep = keep and (metadata["type"].lower() == filters["type"].lower())

        if keep:
            retrieved.append({
                "id": id,
                "document_text": doc,
                "metadata": metadata,
                "score": round(score, 4)
            })
    
    # Sort by score descending 
    retrieved.sort(key=lambda x: x["score"], reverse=True)

    return { "results": retrieved }
