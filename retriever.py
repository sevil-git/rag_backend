"""
Retriever: searches Pinecone for relevant IS 456 chunks
given a user query.
"""

from pinecone import Pinecone
from config import settings
from embeddings import embedding_service


class Retriever:
    def __init__(self):
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = pc.Index(settings.PINECONE_INDEX_NAME)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Embed the query, search Pinecone, and return the top-k
        matching chunks with their metadata and scores.
        """
        k = top_k or settings.TOP_K

        # Embed the query
        query_embedding = embedding_service.embed_text(query)

        # Search Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
        )

        # Format results
        retrieved = []
        for match in results.matches:
            retrieved.append(
                {
                    "text": match.metadata.get("text", ""),
                    "source": match.metadata.get("source", ""),
                    "page": match.metadata.get("page", 0),
                    "score": match.score,
                }
            )

        return retrieved


# Singleton
retriever = Retriever()
