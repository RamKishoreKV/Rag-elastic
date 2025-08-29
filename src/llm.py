# src/llm.py
import os, requests
from typing import List, Dict, Optional

PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()   # "ollama"
OLLAMA  = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL   = os.getenv("OLLAMA_MODEL", "llama3.2")

SYS_PROMPT = """You are a precise assistant for a RAG system.
Use ONLY the provided context to answer. If the answer is not clearly supported, reply exactly: "I don’t know."
Keep answers concise and add inline citations like [Title p.Page] after sentences that use that source.
Refuse unsafe or harmful requests.
If the conversation history includes prior turns, you may use them ONLY to resolve coreference (what "it/they/that" refer to). Do not invent facts not present in the current context.
"""

UNSAFE_KEYWORDS = [
    "build a bomb", "make a bomb", "malware", "ransomware", "suicide", "self harm",
    "harm someone", "how to hack", "credit card dump", "child sexual", "terrorism"
]

def is_unsafe(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in UNSAFE_KEYWORDS)

def _format_context(blocks: List[Dict]) -> str:
    # Expect: [{title, page, snippet}]
    lines = []
    for b in blocks:
        title = b.get("title", "?")
        page  = b.get("page", "?")
        snippet = b.get("snippet") or b.get("content") or ""
        lines.append(f"[{title} p.{page}] {snippet}")
    return "\n\n".join(lines) if lines else "NO CONTEXT"

def _format_history(history: Optional[List[Dict]]) -> str:
    """history like: [{"user":"...", "answer":"..."}, ...] most-recent last."""
    if not history:
        return ""
    # take last 4 turns, trim long content
    turns = history[-4:]
    safe = []
    for t in turns:
        uq = (t.get("user") or "").strip()
        aa = (t.get("answer") or "").strip()
        if len(uq) > 400: uq = uq[:400] + "…"
        if len(aa) > 400: aa = aa[:400] + "…"
        safe.append(f"User: {uq}\nAssistant: {aa}")
    return "\n\n".join(safe)

def answer_with_llm(question: str, context_blocks: List[Dict], history: Optional[List[Dict]] = None) -> str:
    if is_unsafe(question):
        return "I can’t help with that request."

    ctx = _format_context(context_blocks)
    hist = _format_history(history)

    user_prompt = f"""Question:
{question}

Conversation history (use only for reference resolution):
{hist if hist else "(none)"}

Context:
{ctx}

Answer (with citations):"""

    try:
        if PROVIDER == "ollama":
            payload = {
                "model": MODEL,
                "prompt": f"{SYS_PROMPT}\n\n{user_prompt}",
                "stream": False,
                "options": {"temperature": 0.2}
            }
            r = requests.post(f"{OLLAMA}/api/generate", json=payload, timeout=180)
            r.raise_for_status()
            return (r.json().get("response") or "").strip() or "I don’t know."
    except Exception as e:
        return f"I don’t know. (LLM error: {e})"

    return "I don’t know."
