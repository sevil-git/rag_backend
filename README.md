# Smart Civilian — RAG Backend

AI-powered slab design assistant using IS 456:2000, built with FastAPI + Ollama + HuggingFace + Pinecone.

## Architecture

```
┌─────────────────────────┐      ┌──────────────────────────────┐
│  Next.js Frontend       │      │  FastAPI Backend              │
│  (localhost:3000)       │ ───→ │  (localhost:8000)             │
│                         │      │                              │
│  Chat UI with SSE       │ ←─── │  /api/chat (streaming)       │
│  streaming              │      │                              │
└─────────────────────────┘      │  ┌────────────────────────┐  │
                                 │  │ 1. Embed query (HF)    │  │
                                 │  │ 2. Search Pinecone     │  │
                                 │  │ 3. Build prompt        │  │
                                 │  │ 4. Stream from Ollama  │  │
                                 │  └────────────────────────┘  │
                                 └──────────────────────────────┘
```

## Prerequisites

1. **Python 3.11+**
2. **Ollama** — local LLM runtime
3. **Pinecone account** — free tier at [pinecone.io](https://www.pinecone.io/)

## Setup

### 1. Install Ollama

```bash
# macOS
brew install ollama

# or download from https://ollama.ai
```

Pull a model (Mistral recommended for balanced speed/quality):

```bash
ollama pull mistral
# or for better quality:
ollama pull llama3
```

Start Ollama:

```bash
ollama serve
```

### 2. Create Pinecone Index

1. Sign up at [pinecone.io](https://www.pinecone.io/) (free tier)
2. Create an API key from the dashboard
3. No need to manually create an index — the ingestion script does it automatically

### 3. Setup Python Environment

```bash
cd rag_backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set your Pinecone API key:

```
PINECONE_API_KEY=your-actual-pinecone-api-key
PINECONE_INDEX_NAME=is456-slab-design
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
PDF_PATH=../is.456.2000 (1).pdf
```

### 5. Ingest the PDF

This extracts text from IS 456:2000, chunks it, generates embeddings, and stores them in Pinecone:

```bash
python ingest.py
```

This takes ~2-5 minutes depending on your machine. You only need to run this once.

### 6. Start the Backend

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

### 7. Start the Frontend

In a separate terminal:

```bash
cd ../smart_civilian
pnpm dev
```

Open `http://localhost:3000` and start chatting!

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/chat` | Chat with streaming (SSE) or sync response |
| POST | `/api/chat/sync` | Always returns full JSON response |

### Chat Request Body

```json
{
  "message": "What is the minimum thickness of a slab as per IS 456?",
  "stream": true
}
```

### Example Queries

- "What are the span-to-depth ratios for one-way slabs?"
- "Explain the reinforcement detailing for two-way slabs as per IS 456"
- "What is the minimum cover requirement for slabs?"
- "How to calculate deflection for a simply supported slab?"
- "What are the load combinations as per IS 456 clause 18?"

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Next.js 15, Tailwind CSS, Framer Motion |
| Backend | Python, FastAPI |
| LLM | Ollama (Mistral / Llama 3) |
| Embeddings | HuggingFace sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | Pinecone (cloud, free tier) |
| PDF Parsing | PyMuPDF |
| Chunking | LangChain RecursiveCharacterTextSplitter |
