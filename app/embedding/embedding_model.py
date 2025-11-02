from sentence_transformers import SentenceTransformer
from app.config import MODEL_NAME 

# Load model 
model = SentenceTransformer(MODEL_NAME)

def encode_text(texts, batch_size=32):
    if isinstance(texts, str):
        texts = [texts]
    return model.encode(texts, 
                        batch_size=batch_size,
                        convert_to_numpy=True)