import chromadb 
from app.config import CHROMA_PATH, COLLECTION_NAME

def get_collection():
    client = chromadb.PersistentClient(CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={ "description": "Product description embeddings" }
    )
    return collection 
