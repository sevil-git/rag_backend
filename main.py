"""
Smart Civilian — FastAPI Backend
RAG-powered API for IS 456:2000 slab design queries.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_chain import query_rag, query_rag_stream

app = FastAPI(
    title="Smart Civilian RAG API",
    description="AI assistant for civil engineering slab design based on IS 456:2000",
    version="1.0.0",
)

# CORS — allow Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    stream: bool = True


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    visualization: dict | None = None


@app.get("/health")
async def health():
    return {"status": "ok", "service": "smart-civilian-rag"}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Main chat endpoint.
    - If stream=True (default): returns Server-Sent Events
    - If stream=False: returns full JSON response
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if req.stream:
        return StreamingResponse(
            query_rag_stream(req.message),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        result = await query_rag(req.message)
        return ChatResponse(**result)


@app.post("/api/chat/sync")
async def chat_sync(req: ChatRequest):
    """Non-streaming chat endpoint — always returns full JSON."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    result = await query_rag(req.message)
    return ChatResponse(**result)
