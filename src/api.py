import os, json, requests
from typing import Literal
from collections import defaultdict
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from requests.auth import HTTPBasicAuth
from fastapi import Body
from sentence_transformers import SentenceTransformer

ES_URL   = os.getenv("ES_URL", "http://localhost:9200")
ES_USER  = os.getenv("ES_USERNAME", "elastic")
ES_PASS  = os.getenv("ES_PASSWORD", "elastic")
INDEX    = os.getenv("ES_INDEX", "docs_rag")
ELSER_ID = os.getenv("ELSER_ENDPOINT_ID", "elser-v2-rk-02")
DENSE_MODEL = os.getenv("DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

auth = HTTPBasicAuth(ES_USER, ES_PASS)
HEADERS = {"Content-Type": "application/json"}

app = FastAPI(title="RAG-Elastic MDP API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(DENSE_MODEL)
    return _model

def q_bm25(q: str, size: int = 10):
    body = {
        "size": size,
        "_source": ["title","source","page","content"],
        "query": {"multi_match": {"query": q, "fields": ["title^2","content"]}}
    }
    r = requests.post(f"{ES_URL}/{INDEX}/_search", auth=auth, headers=HEADERS, data=json.dumps(body))
    r.raise_for_status()
    return r.json()["hits"]["hits"]

def q_elser(q: str, size: int = 10):
    body = {
        "size": size,
        "_source": ["title","source","page","content"],
        "query": {
            "text_expansion": {
                "ml.tokens": {
                    "model_id": ELSER_ID,
                    "model_text": q
                }
            }
        }
    }
    r = requests.post(f"{ES_URL}/{INDEX}/_search", auth=auth, headers=HEADERS, data=json.dumps(body))
    r.raise_for_status()
    return r.json()["hits"]["hits"]

def q_dense(q: str, size: int = 10, k: int = 50, num_candidates: int = 1000):
    vec = get_model().encode([q], normalize_embeddings=True)[0].tolist()
    body = {
        "size": size,
        "_source": ["title","source","page","content"],
        "knn": {"field": "dense_vec", "query_vector": vec, "k": k, "num_candidates": num_candidates}
    }
    r = requests.post(f"{ES_URL}/{INDEX}/_search", auth=auth, headers=HEADERS, data=json.dumps(body))
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

def pack(hits):
    out = []
    for h in hits:
        s = h.get("_source", {})
        out.append({
            "id": h.get("_id"),
            "score": h.get("_score", 0.0),
            "title": s.get("title"),
            "page":  s.get("page"),
            "source": s.get("source"),
            "snippet": (s.get("content","")[:300] + "…") if len(s.get("content","")) > 300 else s.get("content","")
        })
    return out

@app.get("/health")
def health():
    try:
        r = requests.get(f"{ES_URL}", auth=auth, timeout=5)
        ok = r.status_code == 200
    except Exception:
        ok = False
    return {"ok": ok}

from fastapi import Body
from .rag_answer import answer as rag_answer


@app.get("/healthz")
def healthz():
    return health()

@app.post("/query")
def query_answer(payload: dict = Body(...)):
    q = payload.get("q", "").strip()
    mode = payload.get("mode", "hybrid")
    size = int(payload.get("size", 10))

    if not q:
        return {"mode": mode, "query": q, "answer": "I don’t know."}

    return rag_answer(q, mode=mode, size=size)
