import os
import glob
import json
from typing import List, Dict
import fitz  
import requests
from requests.auth import HTTPBasicAuth


ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_USER = os.getenv("ES_USERNAME", "elastic")
ES_PASS = os.getenv("ES_PASSWORD", "elastic")
INDEX   = os.getenv("ES_INDEX", "docs_rag")
PIPELINE_ID = "elser_enrich"

DATA_DIR = os.getenv("DATA_DIR", "data/pdfs")  
CHUNK_TOKENS  = int(os.getenv("CHUNK_TOKENS", "300"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))

auth = HTTPBasicAuth(ES_USER, ES_PASS)


def tokenize(text: str) -> List[str]:
    return text.split()

def chunk_text(text: str, max_tokens: int, overlap: int) -> List[str]:
    toks = tokenize(text)
    if not toks:
        return []
    chunks = []
    step = max(1, max_tokens - overlap)
    i = 0
    while i < len(toks):
        chunk = toks[i:i + max_tokens]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += step
    return chunks

def extract_pdf(path: str) -> List[Dict]:
    doc = fitz.open(path)
    out = []
    base = os.path.basename(path)
    title = os.path.splitext(base)[0]
    for page_no in range(len(doc)):
        page = doc.load_page(page_no)
        text = page.get_text("text")
        if not text or not text.strip():
            continue
        for chunk in chunk_text(text, CHUNK_TOKENS, CHUNK_OVERLAP):
            out.append({
                "title": title,
                "source": base,
                "page": page_no + 1,
                "content": chunk
            })
    doc.close()
    return out

def bulk_index(docs: List[Dict]) -> None:
    if not docs:
        return
    lines = []
    for d in docs:
        lines.append(json.dumps({"index": {"_index": INDEX}}))
        lines.append(json.dumps(d, ensure_ascii=False))
    ndjson = "\n".join(lines) + "\n"

    url = f"{ES_URL}/_bulk?pipeline={PIPELINE_ID}&refresh=true"
    headers = {"Content-Type": "application/x-ndjson"}
    r = requests.post(url, data=ndjson.encode("utf-8"), headers=headers, auth=auth)
    try:
        resp = r.json()
    except Exception:
        print("Bulk index error:", r.status_code, r.text)
        return

    if r.status_code != 200 or resp.get("errors"):
        first_err = None
        for item in resp.get("items", []):
            err = item.get("index", {}).get("error")
            if err:
                first_err = err
                break
        print("Bulk index completed with errors:", json.dumps(first_err, indent=2))
    else:
        took = resp.get("took")
        print(f"Bulk index OK: {len(docs)} docs (took {took} ms)")

def main(return_count: bool = False) -> int:
    pdf_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    if not pdf_paths:
        print(f"No PDFs found in {DATA_DIR}. Put files there and rerun.")
        return 0

    total = 0
    for p in pdf_paths:
        print(f"Processing: {p}")
        docs = extract_pdf(p)
        print(f"  -> {len(docs)} chunks")
        total += len(docs)
        BATCH = 500
        for i in range(0, len(docs), BATCH):
            bulk_index(docs[i:i+BATCH])

    print(f"Done. Total chunks indexed: {total}")
    return total if return_count else 0

if __name__ == "__main__":
    main()
