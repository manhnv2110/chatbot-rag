import os 
from dotenv import load_dotenv

load_dotenv()

class Settings: 
    DB_HOST=os.getenv("DB_HOST", "localhost")
    DB_PORT=int(os.getenv("DB_PORT", 3306))
    DB_USER=os.getenv("DB_USER", "root")
    DB_PASSWORD=os.getenv("DB_PASSWORD", "")
    DB_NAME=os.getenv("DB_NAME", "")
    DB_CHARSET=os.getenv("DB_CHARSET", "utf8mb4")

    GROQ_API_KEY=os.getenv("GROQ_API_KEY", "")

    CHROMA_PATH = os.getenv("CHROMA_PATH", "")
    CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "")
    MODEL_ENCODE = os.getenv("MODEL_ENCODE", "")
    MODEL_GENERATE = os.getenv("MODEL_GENERATE", "")

    DEFAULT_FETCH_N = int(os.getenv("DEFAULT_FETCH_N", 5))

settings = Settings()