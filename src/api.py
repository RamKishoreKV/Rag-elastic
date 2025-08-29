# src/api.py
import os, json, requests, traceback
from typing import Optional, List, Dict
from pathlib import Path
from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

from .rag_answer import answer as rag_answer
from .ingest_pdfs import main as ingest_local_main
from .embed_dense import main as embed_dense_main

load_dotenv()

ES_URL   = os.getenv("ES_URL", "http://localhost:9200")
ES_USER  = os.getenv("ES_USERNAME", "elastic")
ES_PASS  = os.getenv("ES_PASSWORD", "elastic")
INDEX    = os.getenv("ES_INDEX", "docs_rag")

auth = HTTPBasicAuth(ES_USER, ES_PASS)
HEADERS = {"Content-Type": "application/json"}

app = FastAPI(title="RAG-Elastic MDP API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

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

@app.post("/ingest")
def ingest_local():
    ingested = ingest_local_main(return_count=True) or 0
    embedded = embed_dense_main() or 0
    return {"status": "ok", "ingested_chunks": ingested, "embedded_vectors": embedded}
