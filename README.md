# RAG System (Elastic + ELSER + MiniLM)

This repo contains a Minimum Demo Product (MDP) that uses:
- **Elasticsearch (ELSER + BM25 + dense vectors)**
- **FastAPI** for API
- **Streamlit** for UI
- **Ollama (llama3)** optional for answer synthesis

## Quick Start
1. `cp .env.sample .env`
2. `pip install -r requirements.txt`
3. Run `docker-compose up` to start Elastic

## Run search (4 modes)
```bash
# BM25 (keyword search)
python src/query.py bm25 "your query text"

# ELSER (sparse semantic search)
python src/query.py elser "your query text"

# Dense (MiniLM embeddings)
python src/query.py dense "your query text"

# Hybrid (BM25 + ELSER + Dense with RRF)
python src/query.py hybrid "your query text"
```

## 3) Sanity checklist for the MDP (so you can report status)
- Docker ES up with ML ✅
- ELSER endpoint + ingest pipeline ✅
- PDFs chunked + indexed ✅
- ELSER tokens written via pipeline ✅
- Dense vectors written to `dense_vec` ✅
- Query CLI for BM25 / ELSER / Dense / Hybrid ✅





