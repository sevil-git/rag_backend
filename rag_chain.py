import httpx
import json
import re
import asyncio
from typing import AsyncGenerator

from config import settings
from retriever import get_retriever


SYSTEM_PROMPT = """You are **Smart Civilian**, a civil engineering assistant with access to three knowledge sources:

1. **IS 456:2000** — Indian Standard for Plain and Reinforced Concrete (clauses, design rules, detailing, durability)
2. **Research and History Main** — historical development of RCC, academic research, code evolution, background context
3. **Dataset** — engineering case studies, numerical examples, and reference data

## Instructions:
- Each context chunk below is labelled **[Source: ...]**. Read that label and answer from whichever source is most relevant to the question.
- For questions about specific IS 456 clauses, span-depth ratios, reinforcement rules, or design calculations → use IS 456:2000 chunks.
- For questions about history, research background, evolution of codes, or dataset examples → use Research or Dataset chunks.
- Do NOT force every answer through IS 456. If the question is about history or data, answer from those sources.
- Base your answers strictly on what the context contains. Do not fabricate clauses or data not present in the context.
- If the relevant source chunk isn't in the context, say so honestly.
- Use clear, structured formatting with headings and bullet points where helpful.
- When the user asks for 3D visualization, always provide slab design parameters — never refuse.

Retrieved context:
{context}
"""

_GEMINI_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


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

    # Extract slab span dimensions — try explicit "LxW" patterns first,
    # then fall back to scanning for metre-scale values (1m–30m).
    # NOTE: mm must come before m in the alternation to avoid "125 mm"
    # being read as "125 m".
    length_m, width_m = 5.0, 3.5  # safe defaults

    # Pattern 1: "4m x 6m", "4 m × 6 m", "4mx6m"
    span_pair = re.search(
        r"(\d+(?:\.\d+)?)\s*m\s*[x×by]\s*(\d+(?:\.\d+)?)\s*m",
        text,
        re.IGNORECASE,
    )
    if span_pair:
        length_m = float(span_pair.group(1))
        width_m = float(span_pair.group(2))
    else:
        # Pattern 2: "span = 4 m", "length = 6 m" etc.
        explicit_span = re.search(
            r"(?:span|length|l)\s*(?:=|of|is)?\s*(\d+(?:\.\d+)?)\s*m(?!m)",
            text,
            re.IGNORECASE,
        )
        explicit_width = re.search(
            r"(?:width|breadth|b)\s*(?:=|of|is)?\s*(\d+(?:\.\d+)?)\s*m(?!m)",
            text,
            re.IGNORECASE,
        )
        if explicit_span:
            length_m = float(explicit_span.group(1))
        if explicit_width:
            width_m = float(explicit_width.group(1))
        elif explicit_span:
            # single span mentioned — use as both (square slab approximation)
            width_m = length_m

        # Pattern 3: fallback — pick metre-scale numbers (1–30 m), skip mm values
        if length_m == 5.0 and width_m == 3.5:
            metre_vals = [
                float(v)
                for v, _ in re.findall(r"(\d+(?:\.\d+)?)\s*(m(?!m))", text)
                if 1.0 <= float(v) <= 30.0
            ]
            if len(metre_vals) >= 2:
                length_m, width_m = metre_vals[0], metre_vals[1]
            elif len(metre_vals) == 1:
                length_m = width_m = metre_vals[0]

    # Prefer explicit slab thickness values from the answer/query.
    thickness_mm = _extract_value(
        merged_text,
        [
            r"thickness\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*mm",
            r"overall\s*depth\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*mm",
            r"d\s*=\s*(\d+(?:\.\d+)?)\s*mm",
        ],
    )

    thickness_m = (thickness_mm / 1000.0) if thickness_mm is not None else 0.15

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
        source = chunk.get("source", "Unknown")
        context_parts.append(
            f"--- Chunk {i} [Source: {source}, Page {chunk['page']}, Relevance: {chunk['score']:.3f}] ---\n"
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

    candidate_models = [
        settings.GEMINI_MODEL,
        "gemini-2.0-flash",
        "gemini-flash-latest",
    ]
    # Keep order and remove duplicates/empties.
    unique_models = [m for i, m in enumerate(candidate_models) if m and m not in candidate_models[:i]]

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }

    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=120.0) as client:
        for model in unique_models:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/"
                f"models/{model}:generateContent"
            )

            for attempt in range(3):
                try:
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

                    joined = "\n".join(texts).strip()
                    if joined:
                        return joined
                    return "No answer generated."
                except httpx.HTTPStatusError as exc:
                    last_error = exc
                    status = exc.response.status_code

                    # Try next model immediately if current one is unavailable/not found.
                    if status == 404:
                        break

                    # Retry only transient statuses.
                    if status in _GEMINI_RETRYABLE_STATUS_CODES and attempt < 2:
                        await asyncio.sleep(1.2 * (attempt + 1))
                        continue
                    break
                except httpx.HTTPError as exc:
                    last_error = exc
                    if attempt < 2:
                        await asyncio.sleep(1.2 * (attempt + 1))
                        continue
                    break

    raise RuntimeError(f"Gemini request failed: {last_error}")


async def query_rag(user_query: str) -> dict:
    """
    Non-streaming RAG query:
    1. Retrieve relevant chunks
    2. Build prompt with context
    3. Call configured LLM and return full response
    """
    # Retrieve — balanced across all sources so DOCX docs aren't drowned out
    chunks = get_retriever().retrieve_balanced(user_query)
    context = build_context(chunks)

    # Build messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": user_query},
    ]

    # Call selected LLM provider.
    if settings.LLM_PROVIDER == "gemini":
        try:
            answer = await _query_with_gemini(_build_full_prompt(user_query, context))
        except Exception:
            answer = (
                "The AI service is temporarily unavailable (Gemini upstream issue). "
                "Please try again in a few seconds."
            )
    else:
        answer = await _query_with_ollama(messages)
    visualization = build_visualization_payload(user_query, answer)

    return {
        "answer": answer,
        "sources": [
            {"source": c["source"], "page": c["page"], "score": c["score"], "preview": c["text"][:150]}
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
    # Retrieve — balanced across all sources so DOCX docs aren't drowned out
    chunks = get_retriever().retrieve_balanced(user_query)
    context = build_context(chunks)

    # Build messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": user_query},
    ]

    accumulated_answer = ""

    if settings.LLM_PROVIDER == "gemini":
        try:
            full_answer = await _query_with_gemini(_build_full_prompt(user_query, context))
        except Exception:
            full_answer = (
                "The AI service is temporarily unavailable (Gemini upstream issue). "
                "Please try again in a few seconds."
            )
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
            "source": c["source"],
            "page": c["page"],
            "score": round(c["score"], 3),
            "preview": c["text"][:150],
        }
        for c in chunks
    ]
    visualization = build_visualization_payload(user_query, accumulated_answer)
    yield f"data: {json.dumps({'done': True, 'sources': sources, 'visualization': visualization})}\n\n"
