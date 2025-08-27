# RAG System (Elastic + ELSER + MiniLM)

This repo contains a Minimum Demo Product (MDP) that uses:
- **Elasticsearch (ELSER + BM25 + dense vectors)**
- **FastAPI** for API
- **Streamlit** for UI
- **Ollama (llama3)** optional for answer synthesis

## Quick Start
1. `cp .env.sample .env`
2. `pip install -r requirements.txt`
3. Run `docker-compose up` to start Elastic + Kibana.
