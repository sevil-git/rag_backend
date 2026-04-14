import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "is456-slab-design")

    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")

    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBEDDING_DIMENSION: int = 384  # all-MiniLM-L6-v2 outputs 384-d vectors

    PDF_PATH: str = os.getenv("PDF_PATH", "../is.456.2000 (1).pdf")
    PDF_SOURCE_LABEL: str = os.getenv("PDF_SOURCE_LABEL", "")

    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Retrieval
    TOP_K: int = 5


settings = Settings()
