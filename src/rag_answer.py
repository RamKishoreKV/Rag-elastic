# src/rag_answer.py
import os, json, requests
from collections import defaultdict
from typing import List, Dict, Optional
from requests.auth import HTTPBasicAuth

from sentence_transformers import SentenceTransformer
from .llm import answer_with_llm

ES_URL   = os.getenv("ES_URL", "http://localhost:9200")
ES_USER  = os.getenv("ES_USERNAME", "elastic")
ES_PASS  = os.getenv("ES_PASSWORD", "elastic")
INDEX    = os.getenv("ES_INDEX", "docs_rag")
ELSER_ID = os.getenv("ELSER_ENDPOINT_ID", "elser-v2-rk-02")
DENSE_MODEL = os.getenv("DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

UNSAFE_KEYWORDS = [
    "build a bomb", "make a bomb", "malware", "ransomware", "suicide", "self harm",
    "harm someone", "how to hack", "credit card dump", "child sexual", "terrorism"
]
def is_unsafe(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in UNSAFE_KEYWORDS)

auth = HTTPBasicAuth(ES_USER, ES_PASS)
_session = requests.Session()
HEADERS = {"Content-Type": "application/json"}

_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(DENSE_MODEL)
    return _model

def q_bm25(query: str, size: int = 10):
    body = {
        "size": size,
        "_source": ["title","source","page","content","drive_url"],
        "query": {"multi_match": {"query": query, "fields": ["title^2","content"]}}
    }
    r = _session.post(f"{ES_URL}/{INDEX}/_search", auth=auth, headers=HEADERS, data=json.dumps(body), timeout=30)
    r.raise_for_status()
    return r.json()["hits"]["hits"]

def q_elser(query: str, size: int = 10):
    body = {
        "size": size,
        "_source": ["title","source","page","content","drive_url"],
        "query": {"text_expansion": {"ml.tokens": {"model_id": ELSER_ID, "model_text": query}}}
    }
    r = _session.post(f"{ES_URL}/{INDEX}/_search", auth=auth, headers=HEADERS, data=json.dumps(body), timeout=30)
    r.raise_for_status()
    return r.json()["hits"]["hits"]

def q_dense(query: str, size: int = 10, k: int = 50, num_candidates: int = 750):
    vec = get_model().encode([query], normalize_embeddings=True)[0].tolist()
    body = {
        "size": size,
        "_source": ["title","source","page","content","drive_url"],
        "knn": {"field": "dense_vec", "query_vector": vec, "k": k, "num_candidates": num_candidates}
    }
    r = _session.post(f"{ES_URL}/{INDEX}/_search", auth=auth, headers=HEADERS, data=json.dumps(body), timeout=30)
    r.raise_for_status()
    return r.json()["hits"]["hits"]

def rrf_merge(*rankings, k: int = 60):
    scores = defaultdict(float); id2hit = {}
    for ranklist in rankings:
        for rank, hit in enumerate(ranklist, start=1):
            _id = hit["_id"]
            id2hit[_id] = hit
            scores[_id] += 1.0 / (k + rank)
    merged = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [id2hit[_id] for _id, _ in merged]

def _snippet(text: str, limit: int = 280) -> str:
    if not text: return ""
    text = text.strip().replace("\n", " ")
    return text if len(text) <= limit else (text[:limit] + "…")

def pack_for_llm(hits: List[Dict], top: int = 5) -> List[Dict]:
    blocks = []
    for h in hits[:top]:
        s = h.get("_source", {})
        blocks.append({
            "id": h.get("_id"),
            "title": s.get("title"),
            "page": s.get("page"),
            "source": s.get("source"),
            "drive_url": s.get("drive_url"),
            "snippet": s.get("content",""),
        })
    return blocks

def pack_for_ui(hits: List[Dict], top: int = 5) -> List[Dict]:
    out = []
    for h in hits[:top]:
        s = h.get("_source", {})
        out.append({
            "id": h.get("_id"),
            "score": h.get("_score", 0.0),
            "title": s.get("title"),
            "page":  s.get("page"),
            "source": s.get("source"),
            "drive_url": s.get("drive_url"),
            "snippet": _snippet(s.get("content","")),
        })
    return out

def is_idk_or_refusal(text: str) -> bool:
    if not text: return True
    t = text.strip().lower()
    return (
        t == "i don’t know."
        or t == "i don't know."
        or t.startswith("i can’t help")
        or t.startswith("i can't help")
    )

def answer(query: str, mode: str = "hybrid", size: int = 5, history: Optional[List[Dict]] = None) -> dict:
    # Early guardrail
    if is_unsafe(query):
        return {"mode": mode, "query": query, "answer": "I can’t help with that request.", "results": [], "citations": []}

    # retrieval
    if mode == "bm25":
        hits = q_bm25(query, size=size)
    elif mode == "elser":
        hits = q_elser(query, size=size)
    elif mode == "dense":
        hits = q_dense(query, size=size)
    else:
        h1 = q_elser(query, size=min(10, max(5, size)))
        h2 = q_bm25(query, size=min(10, max(5, size)))
        h3 = q_dense(query, size=min(10, max(5, size)))
        hits = rrf_merge(h1, h2, h3, k=60)[:size]

    ctx_blocks = pack_for_llm(hits, top=min(5, size))
    ui_blocks  = pack_for_ui(hits,  top=size)

    text = answer_with_llm(query, ctx_blocks, history=history)

    # dedupe citations by (title,page)
    raw_citations = [
        {"title": b.get("title"), "page": b.get("page"),
         "source": b.get("source"), "link": b.get("drive_url") or b.get("source")}
        for b in ctx_blocks
    ]
    seen, citations = set(), []
    for c in raw_citations:
        key = (c.get("title"), c.get("page"))
        if key not in seen:
            seen.add(key); citations.append(c)

    if is_idk_or_refusal(text):
        citations = []
        ui_blocks = []

    return {"mode": mode, "query": query, "answer": text or "I don’t know.", "results": ui_blocks, "citations": citations}
