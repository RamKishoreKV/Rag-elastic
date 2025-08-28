import os, json, requests
from collections import defaultdict
from requests.auth import HTTPBasicAuth
from sentence_transformers import SentenceTransformer
from .llm import answer_with_llm

ES_URL   = os.getenv("ES_URL", "http://localhost:9200")
ES_USER  = os.getenv("ES_USERNAME", "elastic")
ES_PASS  = os.getenv("ES_PASSWORD", "elastic")
INDEX    = os.getenv("ES_INDEX", "docs_rag")
ELSER_ID = os.getenv("ELSER_ENDPOINT_ID", "elser-v2-rk-02")
DENSE_MODEL = os.getenv("DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

auth = HTTPBasicAuth(ES_USER, ES_PASS)
headers = {"Content-Type": "application/json"}

_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(DENSE_MODEL)
    return _model

def q_bm25(query, size=20):
    body = {"size": size, "_source": ["title","source","page","content","drive_url"],
            "query": {"multi_match": {"query": query, "fields": ["title^2","content"]}}}
    r = requests.post(f"{ES_URL}/{INDEX}/_search", auth=auth, headers=headers, data=json.dumps(body))
    r.raise_for_status(); return r.json()["hits"]["hits"]

def q_elser(query, size=20):
    body = {"size": size, "_source": ["title","source","page","content","drive_url"],
            "query": {"text_expansion": {"ml.tokens": {"model_id": ELSER_ID, "model_text": query}}}}
    r = requests.post(f"{ES_URL}/{INDEX}/_search", auth=auth, headers=headers, data=json.dumps(body))
    r.raise_for_status(); return r.json()["hits"]["hits"]

def q_dense(query, size=20, k=50, num_candidates=1000):
    vec = get_model().encode([query], normalize_embeddings=True)[0].tolist()
    body = {"size": size, "_source": ["title","source","page","content","drive_url"],
            "knn": {"field": "dense_vec", "query_vector": vec, "k": k, "num_candidates": num_candidates}}
    r = requests.post(f"{ES_URL}/{INDEX}/_search", auth=auth, headers=headers, data=json.dumps(body))
    r.raise_for_status(); return r.json()["hits"]["hits"]

def rrf_merge(*rankings, k=60):
    scores = defaultdict(float); seen = {}
    for ranklist in rankings:
        for rank, hit in enumerate(ranklist, start=1):
            _id = hit["_id"]; seen[_id] = hit
            scores[_id] += 1.0 / (k + rank)
    return [seen[_id] for _id, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]

def pack_for_llm(hits, top=6):
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

def answer(query: str, mode: str = "hybrid", size: int = 10) -> dict:
    if mode == "bm25":   hits = q_bm25(query, size=size)
    elif mode == "elser": hits = q_elser(query, size=size)
    elif mode == "dense": hits = q_dense(query, size=size)
    else:
        h1, h2, h3 = q_elser(query, size=20), q_bm25(query, size=20), q_dense(query, size=20)
        hits = rrf_merge(h1, h2, h3, k=60)[:size]
    ctx = pack_for_llm(hits, top=6)
    text = answer_with_llm(query, ctx)
    citations = [
        {
            "title": b.get("title"),
            "page": b.get("page"),
            "source": b.get("source"),
            "link": b.get("drive_url") or b.get("source"),
        } for b in ctx
    ]
    return {"mode": mode, "query": query, "answer": text, "results": ctx, "citations": citations}
