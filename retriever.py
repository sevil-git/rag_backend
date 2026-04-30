"""
Retriever: searches Pinecone for relevant chunks given a user query.
Uses intent detection to allocate retrieval slots per source so that
IS 456 clause questions get IS 456 chunks, while history/dataset
questions get DOCX chunks — and general questions get a fair mix.
"""

import re
from pinecone import Pinecone
from config import settings
from embeddings import embedding_service


# Keywords that strongly signal an IS 456 / design-code question.
_IS456_PATTERNS = re.compile(
    r"\b(is\s*456|clause\s*\d|section\s*\d|\bclause\b|span.to.depth|"
    r"reinforcement\s+detail|clear\s+cover|nominal\s+cover|"
    r"effective\s+depth|bending\s+moment|shear\s+force|deflection|"
    r"one.way\s+slab|two.way\s+slab|flat\s+slab|design\s+a\s+slab|"
    r"rcc\s+slab|m\d{2}\s+concrete|fe\s*\d{3}|limit\s+state)\b",
    re.IGNORECASE,
)

# Keywords that signal history / research / dataset questions.
_DOCX_PATTERNS = re.compile(
    r"\b(history|historical|evolution|research|dataset|data\s+set|"
    r"origin|background|development|timeline|study|survey|"
    r"literature|review|past|ancient|traditional|introduced|"
    r"who\s+(invented|developed|created)|when\s+was|"
    r"case\s+study|example\s+data|reference\s+data)\b",
    re.IGNORECASE,
)


def _detect_intent(query: str) -> str:
    """Return 'is456', 'docx', or 'balanced' based on query keywords."""
    is456_hits = len(_IS456_PATTERNS.findall(query))
    docx_hits = len(_DOCX_PATTERNS.findall(query))
    if is456_hits > docx_hits:
        return "is456"
    if docx_hits > is456_hits:
        return "docx"
    return "balanced"


# Domain expansion terms appended to the query before embedding.
# This moves the embedding vector toward structural-design space and away
# from accidental matches (e.g. "minimum thickness" matching fire tables).
_IS456_EXPANSION = (
    "IS 456:2000 RCC structural design clause concrete slab "
    "span depth reinforcement deflection"
)
_DOCX_EXPANSION = "civil engineering history research background development"


def _expand_query(query: str, intent: str) -> str:
    """Append domain keywords to improve retrieval precision."""
    if intent == "is456":
        return f"{query} {_IS456_EXPANSION}"
    if intent == "docx":
        return f"{query} {_DOCX_EXPANSION}"
    return query


# Slot allocations per intent: {source_label: top_k_slots}
_ALLOCATIONS = {
    "is456": {
        "IS 456:2000": 10,
        "Dataset": 1,
        "Research and History Main": 1,
    },
    "docx": {
        "IS 456:2000": 1,
        "Dataset": 4,
        "Research and History Main": 4,
    },
    "balanced": {
        "IS 456:2000": 5,
        "Dataset": 3,
        "Research and History Main": 3,
    },
}


class Retriever:
    def __init__(self):
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = pc.Index(settings.PINECONE_INDEX_NAME)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Plain top-K retrieval without source balancing."""
        k = top_k or settings.TOP_K
        query_embedding = embedding_service.embed_text(query)
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
        )
        return self._format(results.matches)

    def retrieve_balanced(self, query: str) -> list[dict]:
        """
        Intent-aware retrieval: detects what the query is about and
        allocates Pinecone slots per source accordingly.

        - IS 456 / clause / design questions → 5 IS 456 + 1+1 DOCX
        - History / research / dataset questions → 1 IS 456 + 3+3 DOCX
        - Ambiguous → 3 IS 456 + 2+2 DOCX
        """
        intent = _detect_intent(query)
        allocation = _ALLOCATIONS[intent]
        expanded = _expand_query(query, intent)
        query_embedding = embedding_service.embed_text(expanded)

        is456_chunks: list[dict] = []
        docx_chunks: list[dict] = []

        for source, slots in allocation.items():
            if slots == 0:
                continue
            try:
                results = self.index.query(
                    vector=query_embedding,
                    top_k=slots,
                    include_metadata=True,
                    filter={"source": {"$eq": source}},
                )
                chunks = self._format(results.matches)
                if source == "IS 456:2000":
                    is456_chunks.extend(chunks)
                else:
                    docx_chunks.extend(chunks)
            except Exception:
                pass

        # For docx-intent queries put DOCX chunks first; otherwise IS 456 first.
        if intent == "docx":
            return docx_chunks + is456_chunks
        return is456_chunks + docx_chunks

    def _format(self, matches) -> list[dict]:
        return [
            {
                "text": m.metadata.get("text", ""),
                "source": m.metadata.get("source", ""),
                "page": m.metadata.get("page", 0),
                "score": m.score,
            }
            for m in matches
        ]


_retriever_instance: Retriever | None = None


def get_retriever() -> Retriever:
    """Lazily initialize retriever to avoid startup-time failures."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance
