import os, requests

# Ollama config
OLLAMA = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:7b")  # default to Llama 3.2:7B

SYS_PROMPT = """You are a precise assistant for a RAG system.
Answer using ONLY the provided context. If the answer is not clearly supported by the context, reply: "I don’t know."
Keep answers concise and include inline citations like [Title p.Page] at the end of sentences that use that source.
Refuse unsafe or harmful requests."""

UNSAFE_KEYWORDS = [
    "build a bomb", "make a bomb", "malware", "ransomware", "suicide", "self harm",
    "harm someone", "how to hack", "credit card dump", "child sexual", "terrorism"
]

def is_unsafe(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in UNSAFE_KEYWORDS)

def _format_context(blocks):
    lines = []
    for b in blocks:
        title = b.get("title", "?")
        page = b.get("page", "?")
        snippet = b.get("snippet") or b.get("content") or ""
        lines.append(f"[{title} p.{page}] {snippet}")
    return "\n\n".join(lines) if lines else "NO CONTEXT"

def answer_with_llm(question: str, context_blocks: list[dict]) -> str:
    if is_unsafe(question):
        return "I can’t help with that request."

    ctx = _format_context(context_blocks)
    user = f"Question:\n{question}\n\nContext:\n{ctx}\n\nAnswer (with citations):"

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"{SYS_PROMPT}\n\n{user}",
            "stream": False,
            "options": {"temperature": 0.2}
        }
        r = requests.post(f"{OLLAMA}/api/generate", json=payload, timeout=180)
        r.raise_for_status()
        return r.json().get("response", "").strip() or "I don’t know."
    except Exception as e:
        return f"I don’t know. (LLM error: {e})"
