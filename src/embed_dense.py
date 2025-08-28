import os, json, requests
from requests.auth import HTTPBasicAuth
from sentence_transformers import SentenceTransformer

ES_URL  = os.getenv("ES_URL", "http://localhost:9200")
ES_USER = os.getenv("ES_USERNAME", "elastic")
ES_PASS = os.getenv("ES_PASSWORD", "elastic")
INDEX   = os.getenv("ES_INDEX", "docs_rag")
MODEL_NAME = os.getenv("DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

auth = HTTPBasicAuth(ES_USER, ES_PASS)
headers_json   = {"Content-Type": "application/json"}
headers_ndjson = {"Content-Type": "application/x-ndjson"}

_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def scan(batch_size=256):
    """
    Stream ONLY docs that do NOT yet have dense_vec.
    """
    url = f"{ES_URL}/{INDEX}/_search"
    body = {
        "size": batch_size,
        "sort": [{"_doc": "asc"}],
        "_source": ["content"],
        "query": {
            "bool": {
                "must_not": {"exists": {"field": "dense_vec"}}
            }
        }
    }
    search_after = None
    while True:
        if search_after:
            body["search_after"] = search_after
        r = requests.post(url, auth=auth, headers=headers_json, data=json.dumps(body))
        r.raise_for_status()
        hits = r.json()["hits"]["hits"]
        if not hits:
            break
        yield hits
        search_after = hits[-1]["sort"]

def bulk_update(pairs):
    """
    pairs: list of tuples (_id, vector:list[float])
    """
    if not pairs:
        return
    lines = []
    for _id, vec in pairs:
        lines.append(json.dumps({"update": {"_index": INDEX, "_id": _id}}))
        lines.append(json.dumps({"doc": {"dense_vec": vec}}))
    ndjson = "\n".join(lines) + "\n"
    r = requests.post(f"{ES_URL}/_bulk?refresh=false", auth=auth,
                      headers=headers_ndjson, data=ndjson.encode("utf-8"))
    if r.status_code != 200:
        print("Bulk HTTP error:", r.status_code, r.text)
        return
    resp = r.json()
    if resp.get("errors"):
        for it in resp.get("items", []):
            err = it.get("update", {}).get("error")
            if err:
                print("Bulk error (first):", json.dumps(err, indent=2))
                break
    else:
        print(f"Updated {len(pairs)} docs")

def main():
    total = 0
    model = get_model()
    for hits in scan():
        texts = []
        ids   = []
        for h in hits:
            t = (h["_source"] or {}).get("content") or ""
            if t and t.strip():
                texts.append(t)
                ids.append(h["_id"])
        if not ids:
            continue
        vecs = model.encode(texts, normalize_embeddings=True).tolist()
        bulk_update(list(zip(ids, vecs)))
        total += len(ids)
        print(f"Progress: {total} vectors")
    print(f"Done. Total vectors written: {total}")
    return total

if __name__ == "__main__":
    main()
