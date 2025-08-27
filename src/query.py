import os, sys, json, requests
from collections import defaultdict
from requests.auth import HTTPBasicAuth
from sentence_transformers import SentenceTransformer

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
    body = {
        "size": size,
        "_source": ["title","source","page","content"],
        "query": {"multi_match": {"query": query, "fields": ["title^2","content"]}}
    }
    r = requests.post(f"{ES_URL}/{INDEX}/_search", auth=auth, headers=headers, data=json.dumps(body))
    r.raise_for_status()
    return r.json()["hits"]["hits"]

def q_elser(query, size=20):
    body = {
        "size": size,
        "_source": ["title","source","page","content"],
        "query": {"text_expansion": {"ml.tokens": {"model_id": ELSER_ID, "model_text": query}}}
    }
    r = requests.post(f"{ES_URL}/{INDEX}/_search", auth=auth, headers=headers, data=json.dumps(body))
    r.raise_for_status()
    return r.json()["hits"]["hits"]

def q_dense(query, size=20, k=50, num_candidates=1000):
    vec = get_model().encode([query], normalize_embeddings=True)[0].tolist()
    body = {
        "size": size,
        "_source": ["title","source","page","content"],
        "knn": {"field": "dense_vec", "query_vector": vec, "k": k, "num_candidates": num_candidates}
    }
    r = requests.post(f"{ES_URL}/{INDEX}/_search", auth=auth, headers=headers, data=json.dumps(body))
    r.raise_for_status()
    return r.json()["hits"]["hits"]

def rrf_merge(*rankings, k=60):
    scores = defaultdict(float); seen = {}
    for ranklist in rankings:
        for rank, hit in enumerate(ranklist, start=1):
            _id = hit["_id"]; seen[_id] = hit
            scores[_id] += 1.0 / (k + rank)
    return [seen[_id] for _id, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]

def print_hits(hits, top=10):
    for i, h in enumerate(hits[:top], 1):
        s = h.get("_source", {})
        print(f"{i}. ({h.get('_score',0):.3f}) {s.get('title')} [p.{s.get('page')}] - {s.get('source')}")
        text = s.get("content","")
        print("   ", (text[:200]+'…') if len(text)>200 else text)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python src/query.py <mode> "<query text>"')
        sys.exit(1)

    mode  = sys.argv[1].lower()
    query = " ".join(sys.argv[2:])

    if mode == "bm25":
        hits = q_bm25(query, size=10)
    elif mode == "elser":
        hits = q_elser(query, size=10)
    elif mode == "dense":
        hits = q_dense(query, size=10)
    elif mode == "hybrid":
        h1 = q_elser(query, size=20)
        h2 = q_bm25(query, size=20)
        h3 = q_dense(query, size=20)
        hits = rrf_merge(h1, h2, h3, k=60)[:10]
    else:
        print("Unknown mode. Use: bm25 | elser | dense | hybrid")
        sys.exit(1)

    print_hits(hits, top=10)
