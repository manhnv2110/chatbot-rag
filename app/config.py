import os 
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION")
MODEL_NAME = os.getenv("EMBED_MODEL")

DEFAULT_FETCH_N = int(os.getenv("DEFAULT_FETCH_N"))