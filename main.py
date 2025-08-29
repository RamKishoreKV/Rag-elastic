# main.py  — one-button runner with warmups + multi-worker API
import os
import sys
import time
import subprocess
from pathlib import Path
import requests

# ---------- Repo & env ----------
REPO_ROOT = Path(__file__).parent.resolve()

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

# Paths / env defaults
ES_URL        = os.getenv("ES_URL", "http://localhost:9200")
ES_USER       = os.getenv("ES_USERNAME", "elastic")
ES_PASS       = os.getenv("ES_PASSWORD", "elastic")
ES_INDEX      = os.getenv("ES_INDEX", "docs_rag")
ELSER_ID      = os.getenv("ELSER_ENDPOINT_ID", "elser-v2-rk-02")
DATA_DIR      = Path(os.getenv("DATA_DIR", "data/pdfs/_drive_sync")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

LLM_PROVIDER  = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_HOST   = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "llama3.2")
API_HOST      = os.getenv("API_HOST", "127.0.0.1")
API_PORT      = int(os.getenv("API_PORT", "8000"))
UI_PORT       = int(os.getenv("UI_PORT", "8501"))
UVICORN_WORKERS = int(os.getenv("UVICORN_WORKERS", "2"))  # speed-up for parallel requests

# Make sure the ingestion script picks the Drive sync dir
os.environ["DATA_DIR"] = str(DATA_DIR)

# ---------- utils ----------
def sh(cmd, cwd=None, check=True, env=None, bg=False, name=None):
    if not bg:
        print(f"\n$ {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=cwd, check=check, env=env)
    else:
        print(f"\n$ (background) {' '.join(cmd)}")
        creation = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
        return subprocess.Popen(cmd, cwd=cwd, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                creationflags=creation)

def wait_http(url, expected=200, timeout=120, method="GET", payload=None, auth=None):
    print(f"Waiting for {url} …")
    start = time.time()
    last = None
    while time.time() - start < timeout:
        try:
            r = requests.request(method, url, json=payload, timeout=5, auth=auth)
            if r.status_code == expected:
                print(f"OK: {url} -> {r.status_code}")
                return True
            last = f"{r.status_code} {r.text[:200]}"
        except Exception as e:
            last = str(e)
        time.sleep(2)
    print(f"Timeout waiting for {url}. Last: {last}")
    return False

# ---------- steps ----------
def ensure_python_bits():
    # these are small/light; most are already installed in your venv
    sh([sys.executable, "-m", "pip", "install", "gdown", "python-dotenv", "requests", "streamlit", "uvicorn"])

def up_elastic():
    compose = REPO_ROOT / "docker-compose.yml"
    if not compose.exists():
        sys.exit("docker-compose.yml not found.")
    print("\nBringing up Elasticsearch with Docker Compose…")
    sh(["docker", "compose", "up", "-d"], cwd=str(REPO_ROOT))
    from requests.auth import HTTPBasicAuth
    wait_http(ES_URL, 200, auth=HTTPBasicAuth(ES_USER, ES_PASS))

def setup_es_objects():
    # idempotent: will print 400 if they already exist (fine)
    sh([sys.executable, str(REPO_ROOT / "src" / "setup_es.py")])

def ensure_ollama():
    if LLM_PROVIDER != "ollama":
        print("LLM_PROVIDER != ollama; skipping Ollama.")
        return
    if not wait_http(f"{OLLAMA_HOST}/api/tags", 200, timeout=10):
        # try to start Ollama daemon if CLI is available
        try:
            sh(["ollama", "serve"], bg=True)
            wait_http(f"{OLLAMA_HOST}/api/tags", 200, timeout=30)
        except Exception:
            pass
    # pull model if missing
    try:
        tags = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10).json()
        have = {m.get("name") for m in tags.get("models", [])}
        if not any(OLLAMA_MODEL in (n or "") for n in have):
            sh(["ollama", "pull", OLLAMA_MODEL])
        else:
            print(f"Ollama model already present: {OLLAMA_MODEL}")
    except Exception as e:
        sys.exit(f"Ollama check failed: {e}")

def warm_dense_encoder():
    # warm SentenceTransformer (so first real query is faster)
    try:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        print(f"Warming dense encoder: {model_name}")
        m = SentenceTransformer(model_name)
        _ = m.encode(["warmup"], normalize_embeddings=True)
        print("Dense encoder warmed.")
    except Exception as e:
        print(f"Skip dense warmup (non-fatal): {e}")

def warm_elser():
    # touch the ELSER inference endpoint so it JIT-loads
    from requests.auth import HTTPBasicAuth
    payload = {"input": "warmup test"}
    url = f"{ES_URL}/_inference/{ELSER_ID}"
    try:
        requests.post(url, json=payload, auth=HTTPBasicAuth(ES_USER, ES_PASS), timeout=10)
        print("ELSER warmed.")
    except Exception as e:
        print(f"Skip ELSER warmup (non-fatal): {e}")

def drive_sync(folder_url: str):
    print(f"\nDownloading PDFs from Drive → {DATA_DIR}")
    # gdown returns a list; use --folder to mirror the directory
    sh(["gdown", "--folder", "--continue", "--output", str(DATA_DIR), folder_url])

def ingest_and_embed():
    # 1) chunk+bulk index with ELSER pipeline
    sh([sys.executable, str(REPO_ROOT / "src" / "ingest_pdfs.py")])
    # 2) compute dense vectors for all docs
    sh([sys.executable, str(REPO_ROOT / "src" / "embed_dense.py")])

def start_api_ui():
    # Start FastAPI with multiple workers
    print(f"\nStarting FastAPI on http://{API_HOST}:{API_PORT} (workers={UVICORN_WORKERS}) …")
    api_proc = sh([sys.executable, "-m", "uvicorn", "src.api:app",
                   "--host", API_HOST, "--port", str(API_PORT),
                   "--workers", str(UVICORN_WORKERS)],
                  cwd=str(REPO_ROOT), bg=True)

    # Wait for healthz
    wait_http(f"http://{API_HOST}:{API_PORT}/healthz", 200, timeout=90)

    # Start the Streamlit UI
    print(f"\nStarting Streamlit UI on http://127.0.0.1:{UI_PORT} …")
    ui_proc = sh(["streamlit", "run", str(REPO_ROOT / "src" / "ui.py"),
                  "--server.port", str(UI_PORT),
                  "--server.headless", "true"],
                 cwd=str(REPO_ROOT), bg=True)

    return api_proc, ui_proc

def warm_api():
    # Warm Ollama (tiny prompt) to avoid first-token delay
    if LLM_PROVIDER == "ollama":
        try:
            requests.post(f"{OLLAMA_HOST}/api/generate",
                          json={"model": OLLAMA_MODEL, "prompt": "hi", "stream": False},
                          timeout=15)
            print("Ollama model warmed.")
        except Exception as e:
            print(f"Skip Ollama warmup (non-fatal): {e}")

    # Warm the query stack end-to-end with a tiny BM25 query (size 1)
    try:
        requests.post(f"http://{API_HOST}:{API_PORT}/query",
                      json={"q": "warmup", "mode": "bm25", "size": 1},
                      timeout=20)
        print("API warm query complete.")
    except Exception as e:
        print(f"Skip API warm query (non-fatal): {e}")

def main():
    # You can pass a Drive folder URL as argv[1]; otherwise default to recruiter’s shared folder
    folder_url = None
    if len(sys.argv) >= 2 and sys.argv[1].strip():
        folder_url = sys.argv[1].strip()
    else:
        folder_url = os.getenv("DRIVE_FOLDER_URL",
            "https://drive.google.com/drive/folders/1h6GptTW3DPCdhu7q5tY-83CXrpV8TmY_")

    print("== RAG Elastic: one-command launcher (fast) ==")
    print(f"Elasticsearch: {ES_URL} (index={ES_INDEX})")
    print(f"ELSER ID:      {ELSER_ID}")
    print(f"Ollama:        {OLLAMA_HOST} (model={OLLAMA_MODEL})")
    print(f"Drive folder:  {folder_url}")
    print("----------------------------------------------")

    ensure_python_bits()
    up_elastic()
    setup_es_objects()
    ensure_ollama()
    warm_dense_encoder()
    warm_elser()

    # Sync PDFs from Drive (idempotent: gdown will skip what’s already present)
    drive_sync(folder_url)

    # Ingest + embed
    ingest_and_embed()

    # Run services
    api_proc, ui_proc = start_api_ui()

    # Final warm-ups (LLM + /query path)
    warm_api()

    print("\nAll set! Open:")
    print(f"  • API docs:  http://{API_HOST}:{API_PORT}/docs")
    print(f"  • Health:    http://{API_HOST}:{API_PORT}/healthz")
    print(f"  • UI:        http://127.0.0.1:{UI_PORT}")
    print("\nPress Ctrl+C here to stop (then close spawned consoles if any).")

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nShutting down…")
        for p in (api_proc, ui_proc):
            try:
                p.terminate()
            except Exception:
                pass

if __name__ == "__main__":
    main()
