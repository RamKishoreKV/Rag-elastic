# RAG‑Elastic MDP 🚀

A **Retrieval‑Augmented Generation** system that connects **Elasticsearch** retrieval (BM25 + ELSER sparse + dense vectors with RRF hybrid) to an **open LLM** (default: Ollama) with a small **FastAPI** service and a **Streamlit** chat UI.

> **Why this exists**: A compact, reproducible MDP that demonstrates end‑to‑end RAG on PDFs pulled from a shared Google Drive folder, grounded answers with citations, and light guardrails.

---

## ✨ Features

* **Hybrid Retrieval**

  * BM25 keyword search
  * **ELSER** sparse embeddings (`.elser_model_2`)
  * Dense embeddings (`sentence-transformers/all-MiniLM-L6-v2` by default)
  * **RRF merge** (Reciprocal Rank Fusion) across BM25 + ELSER + Dense
* **Ingestion**

  * Pull PDFs from a **Google Drive folder** (via `gdown`)
  * PDF → text extraction → **chunking** (\~300 tokens, 60 overlap)
  * Index with **metadata** (title, filename, page, drive\_url)
* **Answering**

  * Uses an **open LLM** (Ollama by default, e.g., `llama3.2`)
  * Builds answers **only from retrieved context**
  * Returns **citations** (title + page + link)
  * Guardrails: unsafe/off‑topic → refusal; unknown → “I don’t know.”
* **Services**

  * **FastAPI** endpoints: `/query`, `/ingest`, `/healthz`
  * **Streamlit** chat UI with retrieval‑mode toggle and Top‑K control
* **DX**

  * **Docker**‑based Elasticsearch (ML enabled)
  * **pytest** unit tests (chunking, RRF merge, API smoke)

---

## 📦 Tech Stack

* **Python** 3.11
* **Elasticsearch** 8.x (with ML features enabled)
* **Ollama** (local LLM); or optional **Hugging Face Inference**
* **FastAPI** + **Uvicorn**
* **Streamlit**
* **sentence‑transformers** for dense embeddings
* **gdown** for Drive folder sync (public/shared link)

---

## 📁 Repository Structure

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
├── docker-compose.yml    # Elasticsearch container (ML enabled)
├── requirements.txt      # Python deps
├── README.md             # this file
└── main.py               # orchestrator (end‑to‑end run)
```

---

## 🔧 Prerequisites

* **Docker Desktop** (for Elasticsearch)
* **Python 3.11**
* **Windows PowerShell** / macOS Terminal / Linux shell
* **Ollama** installed locally *or* a Hugging Face inference key (optional)

> **Note**: The system can run without a GPU; small models keep the demo light.

---

## ⚡ Quickstart

### 1) Clone & Install

```bash
git clone https://github.com/RamKishoreKV/Rag-elastic.git
cd Rag-elastic
python -m venv .venv
# Windows Powershell
.venv/ScriptS/Activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Start Elasticsearch (Docker)

```bash
docker compose up -d
```

* Wait until ES is healthy. Default endpoint: `http://localhost:9200`

### 3) Run All‑in‑One (Recommended)

```bash
python main.py "https://drive.google.com/drive/folders/<your-drive-folder-id>"
```

This will:

1. Ensure dependencies and connections
2. Initialize ES: create ELSER endpoint & index
3. Pull Ollama model if missing (e.g., `llama3.2`)
4. Download PDFs from the Drive folder
5. Ingest → embed → index
6. Launch **FastAPI** (port **8000**) and **Streamlit UI** (port **8501**)

**URLs**

* API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* Health: [http://127.0.0.1:8000/healthz](http://127.0.0.1:8000/healthz)
* UI: [http://127.0.0.1:8501](http://127.0.0.1:8501)

### Option B: Start then Ingest Manually

```bash
python main.py
```

Then in a second terminal:

```bash
curl -X POST "http://127.0.0.1:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"folder_url":"https://drive.google.com/drive/folders/<your-drive-folder-id>"}'
```

---

## ⚙️ Configuration

Environment variables (optional; defaults are sensible for the MDP):

| Variable              | Default                                  | Description                              |
| --------------------- | ---------------------------------------- | ---------------------------------------- |
| `ES_URL`              | `http://localhost:9200`                  | Elasticsearch base URL                   |
| `ES_INDEX`            | `docs_rag`                               | Index name for chunks                    |
| `ES_USER` / `ES_PASS` | none                                     | Basic auth (if enabled)                  |
| `ELSER_MODEL_ID`      | `.elser_model_2`                         | ELSER model identifier                   |
| `ELSER_ENDPOINT_ID`   | `my-elser-endpoint`                      | Inference endpoint name (must be unique) |
| `DENSE_MODEL`         | `sentence-transformers/all-MiniLM-L6-v2` | Dense embeddings model                   |
| `OLLAMA_MODEL`        | `llama3.2`                               | Local LLM model name                     |
| `HF_API_KEY`          | none                                     | If using HF Inference instead of Ollama  |
| `CHUNK_SIZE`          | `300`                                    | Approx tokens per chunk                  |
| `CHUNK_OVERLAP`       | `60`                                     | Overlap between chunks                   |
| `TOP_K`               | `5`                                      | Top documents per retrieval mode         |
| `NUM_CANDIDATES`      | `50`                                     | Candidate pool size before RRF           |
| `RETRIEVAL_MODE`      | `hybrid`                                 | `bm25` \| `elser` \| `dense` \| `hybrid` |

Create a `.env` file to override, e.g.:

```
ES_INDEX=my_docs
RETRIEVAL_MODE=hybrid
```

---

## 🧠 How Hybrid Retrieval Works

We compute scores/ranks for **BM25**, **ELSER**, and **Dense** searches separately, then combine via **Reciprocal Rank Fusion (RRF)**:

> `score(doc) = Σ ( 1 / (k + rank_source(doc)) )` with a small `k` (e.g., 60)

This keeps the method simple and surprisingly strong in practice for heterogeneous signals.

---

## 🖥️ Streamlit UI

* Chat input for queries
* Retrieval **mode toggle** (bm25 / elser / dense / hybrid)
* Adjustable **Top‑K**
* Shows **final answer**, **citations**, and **top snippets**
* "Clear chat" button

Run is automatic via `main.py`. To run UI only:

```bash
streamlit run src/ui.py
```

---

## 🔌 API Reference (FastAPI)

Base URL: `http://127.0.0.1:8000`

### `GET /healthz`

Health check. Returns `{ "status": "ok" }` when services are ready.

### `POST /ingest`

Trigger ingest of a Google Drive folder.

```json
{
  "folder_url": "https://drive.google.com/drive/folders/<your-drive-folder-id>"
}
```

**Response**: counts and metadata for ingested files/chunks.

### `POST /query`

Ask a question and get an answer + citations.

```json
{
  "q": "What are the proper questions in the PDF?",
  "mode": "hybrid",
  "size": 5
}
```

**Response**:

```json
{
  "answer": "... grounded answer ...",
  "citations": [
    {"title": "Doc A", "page": 3, "url": "https://..."}
  ],
  "snippets": [
    {"chunk": "...", "score": 13.2, "source": "elser"}
  ]
}
```

---

## 🧪 Testing

```bash
pytest -q
```

Included tests:

* `test_chunking.py` — validates 300 + 60 overlap
* `test_rrf.py` — validates RRF merge correctness
* `test_api_smoke.py` — basic API liveness

---

## 🔐 Guardrails

* **Unsafe** prompts (e.g., instructions for harm) → **refusal**
* **Off‑domain** questions (not supported by the ingested docs) → **"I don’t know."**
* Answers are **citation‑grounded**; if context confidence is low, the system abstains.

---

## 🛠️ Troubleshooting

### 1) ELSER endpoint ID conflict

**Error**: `Inference endpoint IDs must be unique ... matches existing trained model ID(s)`

* Use a different `ELSER_ENDPOINT_ID` (e.g., `elser-endpoint-2`), or
* **Delete** the old endpoint/model in ES, or
* Make setup idempotent: the code already checks/creates if missing, but if a stale endpoint exists from prior runs, choose a new ID.

### 2) Elasticsearch not healthy

* Ensure `docker compose up -d` completed and container is healthy
* Give ES 30–60s to start ML components
* Check logs: `docker compose logs -f`

### 3) Ollama model not found

* Install Ollama: [https://ollama.com](https://ollama.com)
* Pull a small model (e.g., `ollama pull llama3.2`)
* Or set `HF_API_KEY` and switch LLM backend in `src/llm.py`

### 4) No PDFs downloaded from Drive

* Ensure the Drive folder link is **public/shared**
* Verify the folder has actual **PDFs** (other types are skipped)

### 5) Slow first query

* Warm‑up time: first embedding/LLM calls can be slower; subsequent queries will be faster.

---

## 📹 Demo (≤ 5 min) — Checklist

1. Run: `python main.py "<drive-folder-url>"`
2. Show ES + Ollama setup, Drive sync, ingest & embeddings
3. Open **API docs** and **UI**
4. Queries:

   * On‑topic → correct answer with citations
   * Unrelated (e.g., football) → “I don’t know.”
   * Unsafe → refusal
5. Toggle modes (bm25 / elser / dense / hybrid)
6. `POST /query` in Swagger with `{ "q": "...", "mode": "hybrid", "size": 5 }`

---

## ⚡ Performance Notes

* Default `TOP_K=5`, smaller `NUM_CANDIDATES` to keep latency \~2–4s on small sets
* Use lighter LLMs for faster responses; consider prompt caching for demos

---

## 🔭 Roadmap (Post‑MDP)

* Add **reranking** (e.g., LLM or cross‑encoder) after hybrid retrieve
* Add **document upload** directly in UI
* Add **auth** for API/UI (e.g., basic token)
* Add **evaluations** (QA pairs; `ragas`/`deepEval`)
* Add **persistent vector store** alternative (FAISS/Chroma) for offline runs

---

## 👤 Author

**Ram Kishore KV**

---

## 📞 Support / Contact

Open an issue on GitHub or reach out via the contact listed in the repo.

