# src/api.py
import os
import json
import requests
import traceback
import shutil
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

from .rag_answer import answer as rag_answer
from .ingest_pdfs import main as ingest_local_main
from .embed_dense import main as embed_dense_main

# Optional: use gdown if available for Drive folder downloads
try:
    import gdown
except Exception:
    gdown = None

# ---------------- env ----------------
load_dotenv()

ES_URL   = os.getenv("ES_URL", "http://localhost:9200")
ES_USER  = os.getenv("ES_USERNAME", "elastic")
ES_PASS  = os.getenv("ES_PASSWORD", "elastic")
INDEX    = os.getenv("ES_INDEX", "docs_rag")

# Where ingestion reads PDFs from
DATA_DIR = Path(os.getenv("DATA_DIR", "data/pdfs/_drive_sync")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

auth = HTTPBasicAuth(ES_USER, ES_PASS)
HEADERS = {"Content-Type": "application/json"}

app = FastAPI(title="RAG-Elastic MDP API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------------- health ----------------
@app.get("/health")
def health():
    try:
        r = requests.get(f"{ES_URL}", auth=auth, timeout=5)
        ok = r.status_code == 200
    except Exception:
        ok = False
    return {"ok": ok}

@app.get("/healthz")
def healthz():
    return health()

# ---------------- /query ----------------
@app.post("/query")
def query_answer(payload: dict = Body(...)):
    """
    Body:
      {
        "q": "...",
        "mode": "hybrid|elser|dense|bm25",
        "size": 5,
        "history": [{"user":"...", "answer":"..."}, ...]   # optional
      }
    """
    q = payload.get("q", "")
    mode = payload.get("mode", "hybrid")
    size = int(payload.get("size", 5))
    history = payload.get("history") or None
    try:
        return rag_answer(q, mode=mode, size=size, history=history)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"Query failed: {e}")

# ---------------- helpers ----------------
def _download_drive_folder(folder_url: str, out_dir: Path) -> int:
    """
    Download a *public* Google Drive folder to out_dir using gdown.
    Returns number of files it reported.
    """
    if gdown is None:
        raise RuntimeError("gdown is not installed. Add to requirements.txt and pip install.")
    # Clean-replace for each new folder download
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = gdown.download_folder(url=folder_url, output=str(out_dir), quiet=False, use_cookies=False)
    return len(saved) if isinstance(saved, list) else 0

# ---------------- /ingest ----------------
@app.post("/ingest")
def ingest_from_drive_or_disk(payload: Optional[dict] = Body(default=None)):
    """
    Body (optional): { "folder_url": "https://drive.google.com/drive/folders/<id>" }
      - If folder_url is provided (or DRIVE_FOLDER_URL is set), we CLEAN the target dir,
        download PDFs into DATA_DIR, then ingest + embed.
      - If nothing is provided, we ingest whatever PDFs are already in DATA_DIR.

    Returns: { status, downloaded, ingested_chunks, embedded_vectors, data_dir }
    """
    folder_url = None
    if payload and isinstance(payload, dict):
        folder_url = payload.get("folder_url")
    if not folder_url:
        folder_url = os.getenv("DRIVE_FOLDER_URL")  # optional fallback

    downloaded = 0
    try:
        if folder_url:
            downloaded = _download_drive_folder(folder_url, DATA_DIR)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Drive download failed: {e}")

    # Ensure ingest scripts read the same folder
    os.environ["DATA_DIR"] = str(DATA_DIR)

    try:
        ingested = ingest_local_main(return_count=True) or 0
        embedded = embed_dense_main() or 0
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ingestion/embedding failed: {e}")

    return {
        "status": "ok",
        "downloaded": downloaded,
        "ingested_chunks": ingested,
        "embedded_vectors": embedded,
        "data_dir": str(DATA_DIR),
    }
