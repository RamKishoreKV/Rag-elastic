# src/setup_es.py
import os
import json
import requests
from requests.auth import HTTPBasicAuth

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------- ENV / Defaults --------
ES_URL   = os.getenv("ES_URL", "http://localhost:9200").rstrip("/")
ES_USER  = os.getenv("ES_USERNAME", "elastic")
ES_PASS  = os.getenv("ES_PASSWORD", "elastic")
INDEX    = os.getenv("ES_INDEX", "docs_rag")
PIPELINE_ID = os.getenv("INGEST_PIPELINE_ID", "elser_enrich")

# IMPORTANT: this must match your code (rag_answer.py/api.py)
ELSER_ID = os.getenv("ELSER_ENDPOINT_ID", "elser-v2-rk-02")
# Built-in model id for ELSER v2 on Elastic 8.x:
ELSER_MODEL = os.getenv("ELSER_MODEL_ID", ".elser_model_2")

auth = HTTPBasicAuth(ES_USER, ES_PASS)
HJSON = {"Content-Type": "application/json"}


def _ok(r, *codes):
    return r is not None and r.status_code in (codes or (200,))


def _get(path):
    return requests.get(f"{ES_URL}{path}", auth=auth, timeout=30)


def _head(path):
    return requests.head(f"{ES_URL}{path}", auth=auth, timeout=30)


def _put(path, body):
    return requests.put(f"{ES_URL}{path}", auth=auth, headers=HJSON, data=json.dumps(body))


def _post(path, body):
    return requests.post(f"{ES_URL}{path}", auth=auth, headers=HJSON, data=json.dumps(body))


# ---------- ELSER endpoint ----------
def ensure_elser_endpoint(endpoint_id=ELSER_ID, model_id=ELSER_MODEL, allocations=1, threads=2):
    # Check if exists
    r = _get(f"/_inference/sparse_embedding/{endpoint_id}")
    if _ok(r, 200):
        print(f"ELSER endpoint already exists: {endpoint_id}")
        return

    body = {
        "service": "elser",
        "service_settings": {
            "model_id": model_id,
            "num_allocations": allocations,
            "num_threads": threads,
        }
    }
    r = _put(f"/_inference/sparse_embedding/{endpoint_id}", body)
    if _ok(r, 200):
        print(f"Create ELSER endpoint: {endpoint_id} -> 200")
    else:
        print(f"Create ELSER endpoint FAILED: {r.status_code} {r.text}")


# ---------- Index (mappings + settings) ----------
def ensure_index(index_name=INDEX):
    # HEAD to check existence
    r = _head(f"/{index_name}")
    if _ok(r, 200):
        print(f"Index already exists: {index_name}")
        return

    body = {
        "settings": {
            # Optional search-time HNSW tweak; you can raise ef_search later per-query too.
            "index": {
                "refresh_interval": "1s"
            }
        },
        "mappings": {
            "properties": {
                "title":     {"type": "keyword"},
                "source":    {"type": "keyword"},
                "page":      {"type": "integer"},
                "content":   {"type": "text"},
                "drive_url": {"type": "keyword"},   # so UI can link out
                "chunk_id":  {"type": "keyword"},   # optional future use

                # ELSER sparse expansion target (rank_features)
                "ml": {
                    "properties": {
                        "tokens": {"type": "rank_features"}
                    }
                },

                # Dense embedding (MiniLM-L6-v2 dims=384)
                "dense_vec": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine",
                    "index_options": {"type": "hnsw", "m": 16, "ef_construction": 100}
                }
            }
        }
    }
    r = _put(f"/{index_name}", body)
    if _ok(r, 200):
        print(f"Create index: {index_name} -> 200")
    else:
        print(f"Create index FAILED: {r.status_code} {r.text}")


def ensure_dense_vec_mapping(index_name=INDEX):
    """
    Safe to call repeatedly; will upsert dense_vec mapping if missing.
    """
    body = {
        "properties": {
            "dense_vec": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine",
                "index_options": {"type": "hnsw", "m": 16, "ef_construction": 100}
            }
        }
    }
    r = _put(f"/{index_name}/_mapping", body)
    if _ok(r, 200):
        print("Add/ensure dense_vec mapping: 200")
    else:
        print(f"Add dense_vec mapping FAILED: {r.status_code} {r.text}")


# ---------- Pipeline ----------
def ensure_ingest_pipeline(pipeline_id=PIPELINE_ID, endpoint_id=ELSER_ID):
    body = {
        "processors": [
            {
                "inference": {
                    "model_id": endpoint_id,
                    "input_output": {
                        "input_field": "content",
                        "output_field": "ml.tokens"
                    }
                }
            }
        ]
    }
    r = _put(f"/_ingest/pipeline/{pipeline_id}", body)
    if _ok(r, 200):
        print(f"Create/ensure pipeline '{pipeline_id}': 200")
    else:
        print(f"Create pipeline FAILED: {r.status_code} {r.text}")


# ---------- Smoke tests ----------
def smoke_test_sparse(endpoint_id=ELSER_ID):
    r = _post(f"/_inference/{endpoint_id}", {"input": "Find content about a submission deadline"})
    print("ELSER inference smoke test:", r.status_code)
    try:
        print(r.json())
    except Exception:
        print(r.text)


def smoke_test_index(index_name=INDEX):
    # tiny test search (BM25)
    body = {"size": 0, "query": {"match_all": {}}}
    r = _post(f"/{index_name}/_search", body)
    print("Index search smoke test:", r.status_code)


if __name__ == "__main__":
    ensure_elser_endpoint(endpoint_id=ELSER_ID, model_id=ELSER_MODEL)
    ensure_index(index_name=INDEX)
    ensure_dense_vec_mapping(index_name=INDEX)
    ensure_ingest_pipeline(pipeline_id=PIPELINE_ID, endpoint_id=ELSER_ID)

    smoke_test_sparse(endpoint_id=ELSER_ID)
    smoke_test_index(index_name=INDEX)
