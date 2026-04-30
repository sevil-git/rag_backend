"""
Document ingestion pipeline:
    1. Extract text from PDF (PyMuPDF) or DOCX (python-docx)
    2. Split into overlapping chunks
    3. Generate embeddings using HuggingFace
    4. Upsert into Pinecone vector store
"""

import os
import uuid
from pathlib import Path

import fitz  # PyMuPDF
from docx import Document as DocxDocument
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from embeddings import embedding_service


def resolve_source_label(file_path: str) -> str:
    """Resolve a human-readable source label for the ingested document."""
    file_stem = Path(file_path).stem
    normalized = file_stem.lower().replace(" ", "")
    if normalized.startswith("is.456.2000"):
        return "IS 456:2000"
    return file_stem


def extract_text_from_pdf(pdf_path: str, source_label: str) -> list[dict]:
    """Extract text from each page of a PDF with metadata."""
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
    print(f"  Extracted text from {len(pages)} pages.")
    return pages


def extract_text_from_docx(docx_path: str, source_label: str) -> list[dict]:
    """Extract text from a DOCX file, treating each paragraph as a unit."""
    doc = DocxDocument(docx_path)
    full_text = "\n".join(
        para.text for para in doc.paragraphs if para.text.strip()
    )
    if not full_text.strip():
        return []
    return [
        {
            "text": full_text,
            "metadata": {
                "source": source_label,
                "file_name": Path(docx_path).name,
                "page": 1,
            },
        }
    ]


def extract_text(file_path: str, source_label: str) -> list[dict]:
    """Dispatch to the correct extractor based on file extension."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path, source_label)
    elif ext == ".docx":
        return extract_text_from_docx(file_path, source_label)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only .pdf and .docx are supported.")


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Split page/document texts into overlapping chunks with metadata preserved."""
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

    print(f"  Created {len(chunks)} chunks.")
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

    print("  Generating embeddings...")
    all_embeddings = embedding_service.embed_texts(texts)

    vectors = []
    for chunk, embedding in zip(chunks, all_embeddings):
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

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        print(f"  Upserted batch {i // batch_size + 1}/{(len(vectors) - 1) // batch_size + 1}")

    print(f"  Successfully upserted {len(vectors)} vectors to Pinecone.")


def resolve_paths(script_dir: str) -> list[str]:
    """Collect all document paths to ingest."""
    paths = []

    # Primary document (PDF_PATH from config/env)
    primary = os.path.normpath(os.path.join(script_dir, settings.PDF_PATH))
    if os.path.exists(primary):
        paths.append(primary)
    else:
        print(f"Warning: Primary document not found: {primary}")

    # Extra documents (EXTRA_DOCUMENT_PATHS from env, comma-separated)
    if settings.EXTRA_DOCUMENT_PATHS.strip():
        for raw in settings.EXTRA_DOCUMENT_PATHS.split(","):
            p = raw.strip()
            if not p:
                continue
            resolved = os.path.normpath(os.path.join(script_dir, p))
            if os.path.exists(resolved):
                paths.append(resolved)
            else:
                print(f"Warning: Extra document not found: {resolved}")

    return paths


def ingest():
    """Main ingestion pipeline — processes all configured documents."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paths = resolve_paths(script_dir)

    if not paths:
        raise FileNotFoundError("No documents found to ingest. Check PDF_PATH and EXTRA_DOCUMENT_PATHS.")

    index = init_pinecone_index()

    for file_path in paths:
        source_label = resolve_source_label(file_path)
        print(f"\nIngesting: {file_path}")
        print(f"Source label: {source_label}")
        print("-" * 60)

        pages = extract_text(file_path, source_label)
        if not pages:
            print("  No text extracted, skipping.")
            continue

        chunks = chunk_pages(pages)
        upsert_to_pinecone(index, chunks)

    print("\n" + "=" * 60)
    print(f"Ingestion complete! Processed {len(paths)} document(s).")


if __name__ == "__main__":
    ingest()
