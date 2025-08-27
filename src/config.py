import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    ES_URL = os.getenv("ES_URL", "http://localhost:9200")
    ES_USERNAME = os.getenv("ES_USERNAME", "elastic")
    ES_PASSWORD = os.getenv("ES_PASSWORD", "elastic")
    ES_INDEX = os.getenv("ES_INDEX", "docs_rag")

    CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "300"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))
    DENSE_MODEL = os.getenv("DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    ELSER_ENDPOINT_ID = os.getenv("ELSER_ENDPOINT_ID", "my-elser-endpoint-01")

settings = Settings()
