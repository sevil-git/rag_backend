"""
Embedding service using HuggingFace sentence-transformers.
Loads the model once and reuses it across requests.
"""

from sentence_transformers import SentenceTransformer
from config import settings


class EmbeddingService:
    _instance: "EmbeddingService | None" = None
    _model: SentenceTransformer | None = None

    def __new__(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
            print("Embedding model loaded.")
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string and return the vector."""
        model = self._load_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings and return the vectors."""
        model = self._load_model()
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        return embeddings.tolist()


# Singleton instance
embedding_service = EmbeddingService()
