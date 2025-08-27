import requests
from requests.auth import HTTPBasicAuth
import json

ES_URL = "http://localhost:9200"
ES_USER = "elastic"
ES_PASS = "elastic"

auth = HTTPBasicAuth(ES_USER, ES_PASS)

def create_elser_endpoint(endpoint_id="elser-v2-rk-02", allocations=1, threads=2):
    url = f"{ES_URL}/_inference/sparse_embedding/{endpoint_id}"
    payload = {
        "service": "elser",
        "service_settings": {
            "model_id": ".elser_model_2",
            "num_allocations": allocations,  
            "num_threads": threads           
        }
    }
    r = requests.put(url, auth=auth, json=payload)
    print("Create ELSER endpoint:", r.status_code, r.text)


def create_index(index_name="docs_rag"):
    url = f"{ES_URL}/{index_name}"
    payload = {
        "mappings": {
            "properties": {
                "title":   { "type": "keyword" },
                "source":  { "type": "keyword" },
                "page":    { "type": "integer" },
                "content": { "type": "text" },
                "ml": {
                    "properties": {
                        "tokens": { "type": "rank_features" }
                    }
                },
                "dense_vec": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine",
                    "index_options": { "type": "hnsw", "m": 16, "ef_construction": 100 }
                }
            }
        }
    }
    r = requests.put(url, auth=auth, json=payload)
    print("Create index:", r.status_code, r.text)

def create_pipeline(pipeline_id="elser_enrich", endpoint_id="elser-v2-rk-02"):
    url = f"{ES_URL}/_ingest/pipeline/{pipeline_id}"
    payload = {
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
    r = requests.put(url, auth=auth, json=payload)
    print("Create pipeline:", r.status_code, r.text)

def smoke_test(endpoint_id="elser-v2-rk-02"):
    url = f"{ES_URL}/_inference/{endpoint_id}"
    payload = {"input": "Find content about the MDP submission deadline"}
    r = requests.post(url, auth=auth, json=payload)
    print("Smoke test:", r.status_code)
    try:
        print(r.json())
    except Exception:
        print(r.text)

if __name__ == "__main__":
    create_elser_endpoint()
    create_index()
    create_pipeline()
    smoke_test()
