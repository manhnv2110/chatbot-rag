import json 
import chromadb
from tqdm import tqdm
import logging 
from sentence_transformers import SentenceTransformer

# Model name
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Load model 
model = SentenceTransformer(model_name)

# Connect and Initialize Chroma 
client = chromadb.PersistentClient(path="embeddings/chroma_data")

collection = client.get_or_create_collection(
    name="products",
    metadata={"description": "Product embeddings for chatbot search"}
)

# Batch encode and Upsert 
batch_size = 4

with open("./db/products_chunks.json", "r", encoding="utf-8") as f:
    products = json.load(f)
    products_ids = [p['id'] for p in products]
    products_texts = [p['text'] for p in products]
    products_metadata = [p['metadata'] for p in products]
    
    for i in tqdm(range(0, len(products), batch_size), desc="Embedding batches"):
        batch_texts = products_texts[i:i+batch_size]
        batch_ids = [str(id) for id in products_ids[i:i+batch_size]]
        batch_metadata = products_metadata[i:i+batch_size]

        try:
            batch_embeddings = model.encode(batch_texts, 
                                            convert_to_numpy=True, 
                                            show_progress_bar=False)
            collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadata,
                documents=batch_texts
            )
        except Exception as e:
            print(f"Lỗi khi xử lý batch {i // batch_size}: {e}")

    logging.basicConfig(filename="embeddings/embedding_log.txt", level=logging.INFO)
    logging.info(f"Upserted {len(products)} products into Chroma.")