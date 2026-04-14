import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "is456-slab-design")

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").lower()
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    FRONTEND_ORIGINS: str = os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,http://127.0.0.1:3001",
    )
    FRONTEND_ORIGIN_REGEX: str = os.getenv("FRONTEND_ORIGIN_REGEX", "")

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
