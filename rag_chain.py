import httpx
import json
import re
from typing import AsyncGenerator

from config import settings
from retriever import get_retriever


SYSTEM_PROMPT = """You are **Smart Civilian**, an expert AI assistant specializing in civil engineering slab design based on IS 456:2000 (Indian Standard — Plain and Reinforced Concrete: Code of Practice).

Your role:
- Answer questions about RCC slab design: one-way slabs, two-way slabs, flat slabs, span-to-depth ratios, reinforcement detailing, load calculations, deflection control, durability, etc.
- Always base your answers on the IS 456:2000 code provisions provided in the context below.
- Cite specific clause numbers, tables, or figures from IS 456 when applicable.
- If the context doesn't contain enough information, say so honestly — don't fabricate clauses.
- Use clear, structured formatting with headings, bullet points, and formulas where helpful.
- When giving numerical values or formulas, include units and explain the variables.
- When the user asks for visualization, DO NOT refuse. Assume a companion 3D viewer exists and provide structured numeric slab design details that can be visualized.
- Never say you are unable to provide 3D visualization. Instead, provide the slab parameters and assumptions used for visualization.

Context from IS 456:2000:
{context}
"""


def _extract_value(text: str, patterns: list[str]) -> float | None:
    """Extract first numeric value from regex patterns."""
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            continue
    return None


def build_visualization_payload(user_query: str, answer: str) -> dict | None:
    """Generate slab visualization payload with geometry + extracted design data."""
    merged_text = f"{user_query} {answer}"
    text = merged_text.lower()
    slab_keywords = ["slab", "one-way", "two-way", "rcc", "reinforcement", "span"]
    if not any(keyword in text for keyword in slab_keywords):
        return None

    slab_type = "two_way" if "two-way" in text or "two way" in text else "one_way"

    # Basic heuristic extraction of dimensions from query/answer text.
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(m|meter|metre|mm)", text)
    dims_m = []
    for raw_value, unit in matches:
        value = float(raw_value)
        if unit == "mm":
            value = value / 1000.0
        dims_m.append(value)

    length_m = dims_m[0] if len(dims_m) > 0 else 5.0
    width_m = dims_m[1] if len(dims_m) > 1 else 3.5

    # Prefer explicit slab thickness values from the answer/query.
    thickness_mm = _extract_value(
        merged_text,
        [
            r"thickness\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*mm",
            r"overall\s*depth\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*mm",
            r"d\s*=\s*(\d+(?:\.\d+)?)\s*mm",
        ],
    )

    if thickness_mm is not None:
        thickness_m = thickness_mm / 1000.0
    elif len(dims_m) > 2 and dims_m[2] < 0.6:
        thickness_m = dims_m[2]
    else:
        thickness_m = 0.15

    # Extract additional engineering values for complete visualization metadata.
    effective_depth_mm = _extract_value(
        merged_text,
        [
            r"effective\s*depth\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*mm",
            r"\bd\s*=\s*(\d+(?:\.\d+)?)\s*mm",
        ],
    )
    clear_cover_mm = _extract_value(
        merged_text,
        [r"clear\s*cover\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*mm"],
    )
    factored_load_kn_m2 = _extract_value(
        merged_text,
        [
            r"factored\s*load\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*k[n]?/?m\^?2",
            r"wu\s*=\s*(\d+(?:\.\d+)?)\s*k[n]?/?m\^?2",
        ],
    )
    bending_moment_knm = _extract_value(
        merged_text,
        [
            r"(?:design\s*)?moment\s*(?:mu\s*)?(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*k[n]?\s*m",
            r"mu\s*=\s*(\d+(?:\.\d+)?)\s*k[n]?\s*m",
        ],
    )
    ast_mm2 = _extract_value(
        merged_text,
        [
            r"(?:required\s*)?steel\s*area\s*(?:ast\s*)?(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*mm2",
            r"ast\s*=\s*(\d+(?:\.\d+)?)\s*mm2",
        ],
    )
    bar_dia_mm = _extract_value(
        merged_text,
        [
            r"(\d+(?:\.\d+)?)\s*mm\s*(?:dia|diameter)\s*bars",
            r"bars?\s*of\s*(\d+(?:\.\d+)?)\s*mm",
        ],
    )
    spacing_mm = _extract_value(
        merged_text,
        [
            r"@\s*(\d+(?:\.\d+)?)\s*mm",
            r"spacing\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*mm",
        ],
    )

    concrete_grade_match = re.search(r"\b(M\d{2,3})\b", merged_text, flags=re.IGNORECASE)
    steel_grade_match = re.search(r"\b(Fe\s*\d{3,4})\b", merged_text, flags=re.IGNORECASE)

    aspect_ratio = round(length_m / width_m, 3) if width_m > 0 else None
    span_ratio = max(length_m, width_m) / max(min(length_m, width_m), 0.001)
    support_condition = "simply_supported"
    if "continuous" in text:
        support_condition = "continuous"
    if "cantilever" in text:
        support_condition = "cantilever"

    return {
        "schema_version": "1.0",
        "type": "slab",
        "slab": {
            "slab_type": slab_type,
            "length_m": round(length_m, 3),
            "width_m": round(width_m, 3),
            "thickness_m": round(thickness_m, 3),
            "supports": support_condition,
            "reinforcement": {
                "main_bar_direction": "short_span",
                "distribution_bar_direction": "long_span",
            },
        },
        "design_data": {
            "overall_depth_mm": round(thickness_m * 1000.0, 1),
            "effective_depth_mm": round(effective_depth_mm, 1)
            if effective_depth_mm is not None
            else None,
            "clear_cover_mm": round(clear_cover_mm, 1)
            if clear_cover_mm is not None
            else None,
            "factored_load_kn_m2": round(factored_load_kn_m2, 3)
            if factored_load_kn_m2 is not None
            else None,
            "bending_moment_knm": round(bending_moment_knm, 3)
            if bending_moment_knm is not None
            else None,
            "required_steel_area_mm2": round(ast_mm2, 2) if ast_mm2 is not None else None,
            "main_bar_dia_mm": round(bar_dia_mm, 1) if bar_dia_mm is not None else None,
            "main_bar_spacing_mm": round(spacing_mm, 1) if spacing_mm is not None else None,
            "concrete_grade": concrete_grade_match.group(1).upper()
            if concrete_grade_match
            else None,
            "steel_grade": steel_grade_match.group(1).upper().replace(" ", "")
            if steel_grade_match
            else None,
            "aspect_ratio_l_by_w": aspect_ratio,
            "span_ratio_long_by_short": round(span_ratio, 3),
        },
        "confidence": 0.7,
        "assumptions": [
            "Dimensions and design values inferred from prompt/answer text when available.",
            "Values not present explicitly in text are left null in design_data.",
        ],
    }


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"--- Chunk {i} (Page {chunk['page']}, Relevance: {chunk['score']:.3f}) ---\n"
            f"{chunk['text']}\n"
        )
    return "\n".join(context_parts)


def _resolve_gemini_api_key() -> str:
    """Read Gemini key from GEMINI_API_KEY or legacy OLLAMA_BASE_URL usage."""
    if settings.GEMINI_API_KEY:
        return settings.GEMINI_API_KEY
    # Backward compatibility: user may have pasted Gemini key into OLLAMA_BASE_URL.
    if settings.OLLAMA_BASE_URL.startswith("AIza"):
        return settings.OLLAMA_BASE_URL
    return ""


def _build_full_prompt(user_query: str, context: str) -> str:
    return (
        f"{SYSTEM_PROMPT.format(context=context)}\n\n"
        f"User question:\n{user_query}\n\n"
        "Provide the final answer only."
    )


async def _query_with_ollama(messages: list[dict]) -> str:
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": settings.OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
    return data["message"]["content"]


async def _query_with_gemini(prompt: str) -> str:
    api_key = _resolve_gemini_api_key()
    if not api_key:
        raise RuntimeError("Gemini provider selected but GEMINI_API_KEY is missing")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{settings.GEMINI_MODEL}:generateContent"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, params={"key": api_key}, json=payload)
        response.raise_for_status()
        data = response.json()

    texts = []
    for candidate in data.get("candidates", []):
        parts = candidate.get("content", {}).get("parts", [])
        for part in parts:
            text = part.get("text")
            if text:
                texts.append(text)

    return "\n".join(texts).strip() or "No answer generated."


async def query_rag(user_query: str) -> dict:
    """
    Non-streaming RAG query:
    1. Retrieve relevant chunks
    2. Build prompt with context
    3. Call configured LLM and return full response
    """
    # Retrieve
    chunks = get_retriever().retrieve(user_query)
    context = build_context(chunks)

    # Build messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": user_query},
    ]

    # Call selected LLM provider.
    if settings.LLM_PROVIDER == "gemini":
        answer = await _query_with_gemini(_build_full_prompt(user_query, context))
    else:
        answer = await _query_with_ollama(messages)
    visualization = build_visualization_payload(user_query, answer)

    return {
        "answer": answer,
        "sources": [
            {"page": c["page"], "score": c["score"], "preview": c["text"][:150]}
            for c in chunks
        ],
        "visualization": visualization,
    }


async def query_rag_stream(user_query: str) -> AsyncGenerator[str, None]:
    """
    Streaming RAG query:
    1. Retrieve relevant chunks
    2. Build prompt with context
    3. Stream response token-by-token
    Yields Server-Sent Events (SSE) formatted strings.
    """
    # Retrieve
    chunks = get_retriever().retrieve(user_query)
    context = build_context(chunks)

    # Build messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": user_query},
    ]

    accumulated_answer = ""

    if settings.LLM_PROVIDER == "gemini":
        full_answer = await _query_with_gemini(_build_full_prompt(user_query, context))
        for i in range(0, len(full_answer), 32):
            token = full_answer[i : i + 32]
            accumulated_answer += token
            yield f"data: {json.dumps({'token': token})}\n\n"
    else:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{settings.OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": settings.OLLAMA_MODEL,
                    "messages": messages,
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            accumulated_answer += token
                            yield f"data: {json.dumps({'token': token})}\n\n"
                    except json.JSONDecodeError:
                        continue

    # Send sources and visualization at the end for both providers.
    sources = [
        {
            "page": c["page"],
            "score": round(c["score"], 3),
            "preview": c["text"][:150],
        }
        for c in chunks
    ]
    visualization = build_visualization_payload(user_query, accumulated_answer)
    yield f"data: {json.dumps({'done': True, 'sources': sources, 'visualization': visualization})}\n\n"
