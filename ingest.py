"""
PDF ingestion pipeline:
    1. Extract text from a PDF using PyMuPDF
  2. Split into overlapping chunks
  3. Generate embeddings using HuggingFace
  4. Upsert into Pinecone vector store
"""

import os
import uuid
from pathlib import Path

import fitz  # PyMuPDF
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from embeddings import embedding_service


def resolve_source_label(pdf_path: str) -> str:
        """Resolve a human-readable source label for the ingested PDF."""
        if settings.PDF_SOURCE_LABEL.strip():
                return settings.PDF_SOURCE_LABEL.strip()

        file_stem = Path(pdf_path).stem
        normalized = file_stem.lower().replace(" ", "")
        if normalized.startswith("is.456.2000"):
                return "IS 456:2000"

        return file_stem


def extract_text_from_pdf(pdf_path: str, source_label: str) -> list[dict]:
    """Extract text from each page of the PDF with metadata."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append(
                {
                    "text": text,
                    "metadata": {
                        "source": source_label,
                        "file_name": Path(pdf_path).name,
                        "page": page_num + 1,
                    },
                }
            )
    doc.close()
    print(f"Extracted text from {len(pages)} pages.")
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Split page texts into overlapping chunks with metadata preserved."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for i, chunk_text in enumerate(page_chunks):
            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        **page["metadata"],
                        "chunk_index": i,
                    },
                }
            )

    print(f"Created {len(chunks)} chunks from {len(pages)} pages.")
    return chunks


def init_pinecone_index() -> object:
    """Initialize Pinecone and create/get the index."""
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)

    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if settings.PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index: {settings.PINECONE_INDEX_NAME}")
        pc.create_index(
            name=settings.PINECONE_INDEX_NAME,
            dimension=settings.EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Index created.")
    else:
        print(f"Index '{settings.PINECONE_INDEX_NAME}' already exists.")

    return pc.Index(settings.PINECONE_INDEX_NAME)


def upsert_to_pinecone(index, chunks: list[dict], batch_size: int = 100):
    """Generate embeddings and upsert chunks into Pinecone."""
    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")
    all_embeddings = embedding_service.embed_texts(texts)

    # Upsert in batches
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
        vectors.append(
            {
                "id": str(uuid.uuid4()),
                "values": embedding,
                "metadata": {
                    "text": chunk["text"],
                    "source": chunk["metadata"]["source"],
                    "file_name": chunk["metadata"]["file_name"],
                    "page": chunk["metadata"]["page"],
                    "chunk_index": chunk["metadata"]["chunk_index"],
                },
            }
        )

    # Batch upsert
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        print(f"  Upserted batch {i // batch_size + 1}/{(len(vectors) - 1) // batch_size + 1}")

    print(f"Successfully upserted {len(vectors)} vectors to Pinecone.")


def ingest():
    """Main ingestion pipeline."""
    # Resolve PDF path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, settings.PDF_PATH)
    pdf_path = os.path.normpath(pdf_path)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    source_label = resolve_source_label(pdf_path)

    print(f"Starting ingestion of: {pdf_path}")
    print(f"Source label: {source_label}")
    print("=" * 60)

    # Step 1: Extract text
    pages = extract_text_from_pdf(pdf_path, source_label)

    # Step 2: Chunk
    chunks = chunk_pages(pages)

    # Step 3: Init Pinecone
    index = init_pinecone_index()

    # Step 4: Embed & upsert
    upsert_to_pinecone(index, chunks)

    print("=" * 60)
    print("Ingestion complete!")


if __name__ == "__main__":
    ingest()
