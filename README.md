
# RAG-Elastic MDP 🚀

Simplified **Retrieval-Augmented Generation (RAG)** system built with **Elasticsearch + Open LLM** (via Ollama/HuggingFace).  

## 🎥 Demo
[![Demo Video](https://img.shields.io/badge/Demo-Video-green?logo=google-drive&logoColor=white)](https://drive.google.com/file/d/13FA1lljjs4p9g8G4r0N9I59sVVICxyCo/view?usp=drive_link)
*(Click the badge above to watch the demo video on Google Drive)*

## 📌 Features

- **Elastic Retrieval**:
  - BM25 keyword search
  - ELSER sparse embeddings (`.elser_model_2`)
  - Dense embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
  - Hybrid mode (RRF merge of ELSER + Dense + BM25)
- **Ingestion**:
  - PDFs pulled directly from a shared Google Drive folder (`gdown`)
  - Automatic text extraction → chunking (~300 tokens, 60 overlap)
  - Indexed with metadata (title, filename, page, drive_url)
- **Answer Generation**:
  - Uses open LLM (Ollama by default, e.g., `llama3.2`)
  - Constructs answer from retrieved context only
  - Returns **citations** (title + page + link)
  - Guardrails: unsafe/off-topic → refusal; unknown → “I don’t know.”
- **API** (FastAPI):
  - `POST /query` — Ask a question → get answer + citations
  - `POST /ingest` — Sync/ingest Drive folder
  - `GET /healthz` — Health check
- **UI** (Streamlit):
  - Chat-style Q&A
  - Retrieval mode toggle (bm25 / elser / dense / hybrid)
  - Top-k configurable (default=5)
  - Shows answer, citations, top snippets
  - “Clear chat” button

---

## ⚡ Quickstart

### 1. Clone + Install
```bash
git clone <this-repo>
cd rag-elastic-mdp
python -m venv .venv
.venv/Scripts/activate   # (Windows PowerShell)
pip install -r requirements.txt
```

### 2. Start Elasticsearch
Requires **Docker Desktop** running.
```bash
docker compose up -d
```

### 3. Run All-in-One

You can run the system in **two ways**:

#### Option A: Pass Drive folder at startup (recommended)
```bash
python main.py "https://drive.google.com/drive/folders/<your-drive-folder-id>"
```
➡ This will: ensure deps → bring up ES → create index & ELSER → pull Ollama model → download Drive PDFs → ingest → embed → launch API+UI.

#### Option B: Start system first, then ingest separately
```bash
python main.py
```
Then ingest manually:
```bash
curl -X POST "http://127.0.0.1:8000/ingest" `
  -H "Content-Type: application/json" `
  -d "{"folder_url": "https://drive.google.com/drive/folders/<your-drive-folder-id>"}"
```

When complete you’ll see:
- API: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/healthz
- UI: http://127.0.0.1:8501

---

## 🧪 Testing

Run unit tests:
```bash
pytest -q
```

- `test_chunking.py` → validates 300+overlap chunking
- `test_rrf.py` → validates RRF merge correctness
- `test_api_smoke.py` → sanity check API up

---

## 📂 Repo Structure

```
rag-elastic-mdp/
├── src/
│   ├── api.py            # FastAPI endpoints
│   ├── ingest_pdfs.py    # PDF extraction + chunking + indexing
│   ├── embed_dense.py    # Dense embeddings → ES
│   ├── rag_answer.py     # Retrieval + answer generation
│   ├── llm.py            # LLM wrapper (Ollama/HF)
│   ├── ui.py             # Streamlit chat UI
│   ├── setup_es.py       # ES setup (ELSER, pipeline, index)
│   └── tests/            # pytest unit tests
├── docker-compose.yml    # Elasticsearch container
├── requirements.txt      # reproducible deps
├── README.md             # this file
└── main.py               # orchestrates end-to-end run
```

---

## 🎥 Demo Expectations

In your submission video (≤5 min), show:

1. **Run `python main.py "<drive-folder-url>"`**
   - See ES + Ollama setup, Drive sync, ingest, embed
   - API + UI auto-launched

2. **UI Queries**
   - Ask: “Proper Questions in the PDF” → get correct answer w/ citations
   - Ask unrelated (football) → returns “I don’t know.”
   - Ask unsafe (“how to make a bomb”) → refusal

3. **Mode Toggle**
   - Switch between `bm25`, `elser`, `dense`, `hybrid`

4. **API Swagger**
   - POST `/query` with payload `{"q":"...", "mode":"hybrid","size":5}`

---

## ⚠ Notes

- Latency target: small dataset queries now ~2–4s (after warmup).  
  - Default `top_k=5`, reduced `num_candidates`, simple cache enabled.  
- Cost: free, open-source only.  
- Guardrails: refusal for unsafe, “I don’t know.” for unsupported topics.  

---

## ✅  Coverage

-  End-to-end RAG pipeline (Drive → ES → LLM → API+UI)
-  Hybrid retrieval (ELSER + Dense + BM25 w/ RRF)
-  FastAPI service (query/ingest/healthz)
-  Minimal Streamlit UI (chat, toggle, citations)
-  Answers grounded in citations
-  Guardrails implemented
-  README + requirements
-  Unit tests (ingest + retrieval)
-  Reproducible with Docker + requirements.txt

---

## 👤 Author
Ram Kishore KV
