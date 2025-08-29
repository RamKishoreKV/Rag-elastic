import os
import argparse
import subprocess
import glob
import fitz
import json
import requests
from requests.auth import HTTPBasicAuth

# Elastic config
ES_URL   = os.getenv("ES_URL", "http://localhost:9200")
ES_USER  = os.getenv("ES_USERNAME", "elastic")
ES_PASS  = os.getenv("ES_PASSWORD", "elastic")
INDEX    = os.getenv("ES_INDEX", "docs_rag")
PIPELINE_ID = "elser_enrich"

auth = HTTPBasicAuth(ES_USER, ES_PASS)

# Local download path
DATA_DIR = "data/pdfs/_drive_sync"
os.makedirs(DATA_DIR, exist_ok=True)

CHUNK_TOKENS  = int(os.getenv("CHUNK_TOKENS", "300"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))

def download_drive_folder(url: str, out_dir: str):
    """Use gdown to download all files from a public Drive folder."""
    print(f"Downloading from Drive folder: {url}")
    try:
        subprocess.run(
            ["gdown", "--folder", "--continue", "--output", out_dir, url],
            check=True
        )
    except Exception as e:
        print("Download failed:", e)
        return False
    return True

def tokenize(text: str):
    return text.split()

def chunk_text(text: str, max_tokens: int, overlap: int):
    toks = tokenize(text)
    chunks, step, i = [], max(1, max_tokens - overlap), 0
    while i < len(toks):
        chunk = toks[i:i+max_tokens]
        if not chunk: break
        chunks.append(" ".join(chunk))
        i += step
    return chunks

def extract_pdf(path: str, drive_url: str):
    doc = fitz.open(path)
    out, base, title = [], os.path.basename(path), os.path.splitext(os.path.basename(path))[0]
    for page_no in range(len(doc)):
        page = doc.load_page(page_no)
        text = page.get_text("text")
        if not text.strip(): continue
        for chunk in chunk_text(text, CHUNK_TOKENS, CHUNK_OVERLAP):
            out.append({
                "title": title,
                "source": base,
                "page": page_no+1,
                "content": chunk,
                "drive_url": drive_url
            })
    doc.close()
    return out

def bulk_index(docs):
    if not docs: return
    lines = []
    for d in docs:
        lines.append(json.dumps({"index": {"_index": INDEX}}))
        lines.append(json.dumps(d, ensure_ascii=False))
    ndjson = "\n".join(lines) + "\n"

    url = f"{ES_URL}/_bulk?pipeline={PIPELINE_ID}&refresh=true"
    r = requests.post(url, data=ndjson.encode("utf-8"),
                      headers={"Content-Type":"application/x-ndjson"}, auth=auth)
    try:
        resp = r.json()
    except:
        print("Bulk index error:", r.status_code, r.text)
        return

    if r.status_code != 200 or resp.get("errors"):
        print("Bulk index completed with errors.")
    else:
        print(f"Bulk index OK: {len(docs)} docs")

def main(url: str):
    if not download_drive_folder(url, DATA_DIR):
        print("No files downloaded.")
        return
    pdf_paths = sorted(glob.glob(os.path.join(DATA_DIR, "**/*.pdf"), recursive=True))
    if not pdf_paths:
        print("No PDFs found after download.")
        return
    total = 0
    for p in pdf_paths:
        print(f"Processing {p}")
        docs = extract_pdf(p, url)
        total += len(docs)
        for i in range(0, len(docs), 200):
            bulk_index(docs[i:i+200])
    print(f"Done. Total chunks indexed: {total}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Google Drive folder URL")
    args = parser.parse_args()
    main(args.url)
