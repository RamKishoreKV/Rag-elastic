# main.py
import os
import sys
import time
import subprocess
from pathlib import Path
import requests
import shutil

REPO_ROOT = Path(__file__).parent.resolve()
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

ES_URL   = os.getenv("ES_URL", "http://localhost:9200")
ES_USER  = os.getenv("ES_USERNAME", "elastic")
ES_PASS  = os.getenv("ES_PASSWORD", "elastic")
ES_INDEX = os.getenv("ES_INDEX", "docs_rag")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


DATA_DIR = REPO_ROOT / "data" / "pdfs" / "_drive_sync"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable  # current venv

def run(cmd, cwd=None, check=True, env=None):
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check, env=env)

def run_bg(cmd, cwd=None, env=None):
    print(f"\n$ (background) {' '.join(cmd)}")
    creation = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
    return subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        creationflags=creation
    )

def wait_for_http(url, expected=200, timeout=180, auth=None):
    print(f"Waiting for {url} ...")
    start = time.time()
    last_err = None
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=5, auth=auth)
            if r.status_code == expected:
                print(f"OK: {url} -> {r.status_code}")
                return True
            else:
                print(f"HTTP {r.status_code} from {url}, retrying...")
        except Exception as e:
            last_err = e
        time.sleep(2)
    print(f"Timeout waiting for {url}. Last error: {last_err}")
    return False

def ensure_python_packages():
    need = ["gdown", "python-dotenv", "requests", "streamlit", "uvicorn"]
    run([PYTHON, "-m", "pip", "install", *need], check=True)

def ensure_docker_compose_up():
    compose_file = REPO_ROOT / "docker-compose.yml"
    if not compose_file.exists():
        sys.exit("docker-compose.yml not found in repo root.")
    print("\nBringing up Elasticsearch with Docker Compose…")
    run(["docker", "compose", "up", "-d"], cwd=str(REPO_ROOT))

def setup_elastic():
    from requests.auth import HTTPBasicAuth
    auth = HTTPBasicAuth(ES_USER, ES_PASS)
    if not wait_for_http(f"{ES_URL}", 200, auth=auth):
        sys.exit("Elasticsearch not reachable. Is Docker Desktop running?")
    script = REPO_ROOT / "src" / "setup_es.py"
    if not script.exists():
        sys.exit("src/setup_es.py not found.")
    run([PYTHON, str(script)])

def ensure_ollama_and_model():
    if LLM_PROVIDER != "ollama":
        print("LLM_PROVIDER is not 'ollama'; skipping Ollama checks.")
        return
    if not wait_for_http(f"{OLLAMA_HOST}/api/tags", 200, timeout=15):
        print("Ollama API not responding. Trying to start 'ollama serve' …")
        run_bg(["ollama", "serve"])
        time.sleep(3)
        if not wait_for_http(f"{OLLAMA_HOST}/api/tags", 200, timeout=30):
            sys.exit("Could not reach Ollama API. Start Ollama Desktop or run 'ollama serve'.")
    # Pull model if missing
    try:
        tags = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10).json()
        names = {m.get("name") for m in tags.get("models", [])}
        if not any(OLLAMA_MODEL in (n or "") for n in names):
            print(f"Pulling Ollama model: {OLLAMA_MODEL} …")
            run(["ollama", "pull", OLLAMA_MODEL])
        else:
            print(f"Ollama model already present: {OLLAMA_MODEL}")
    except Exception as e:
        sys.exit(f"Failed checking/pulling Ollama model: {e}")

def drive_download(folder_url: str):
    # CLEAN REPLACE: wipe previous _drive_sync so each run is isolated
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR, ignore_errors=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading PDFs from Drive folder -> {DATA_DIR}")
    run(["gdown", "--folder", "--continue", "--output", str(DATA_DIR), folder_url])


def ingest_and_embed():
    os.environ["DATA_DIR"] = str(DATA_DIR)
    ing = REPO_ROOT / "src" / "ingest_pdfs.py"
    emb = REPO_ROOT / "src" / "embed_dense.py"
    if not ing.exists():
        sys.exit("src/ingest_pdfs.py not found.")
    if not emb.exists():
        sys.exit("src/embed_dense.py not found.")
    run([PYTHON, str(ing)])
    run([PYTHON, str(emb)])

def start_api_and_ui():
    api = REPO_ROOT / "src" / "api.py"
    ui  = REPO_ROOT / "src" / "ui.py"
    if not api.exists():
        sys.exit("src/api.py not found.")
    if not ui.exists():
        sys.exit("src/ui.py not found.")
    print("\nStarting FastAPI on http://127.0.0.1:8000 …")
    api_proc = run_bg([PYTHON, "-m", "uvicorn", "src.api:app", "--reload", "--port", "8000"], cwd=str(REPO_ROOT))
    time.sleep(2)
    wait_for_http("http://127.0.0.1:8000/healthz", 200, timeout=60)
    print("\nStarting Streamlit UI on http://127.0.0.1:8501 …")
    ui_proc = run_bg(["streamlit", "run", str(ui)], cwd=str(REPO_ROOT))

    print("\nAll set! Open:")
    print("  • API:       http://127.0.0.1:8000/docs")
    print("  • Health:    http://127.0.0.1:8000/healthz")
    print("  • UI:        http://127.0.0.1:8501")
    print("\nPress Ctrl+C here to stop.")

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

def main():
    folder_url = None
    if len(sys.argv) >= 2 and sys.argv[1].strip():
        folder_url = sys.argv[1].strip()
    else:
        folder_url = os.getenv("DRIVE_FOLDER_URL")

    if not folder_url:
        print("❌ No Google Drive folder URL provided.")
        print("   Set DRIVE_FOLDER_URL in .env or run:  python main.py <drive-folder-url>")
        sys.exit(1)

    # Export so ingest can stamp source metadata if desired
    os.environ["DRIVE_FOLDER_URL"] = folder_url

    print("== RAG Elastic ALL-IN-ONE Runner ==")
    print(f"Elasticsearch: {ES_URL} (index={ES_INDEX})")
    print(f"Ollama:        {OLLAMA_HOST} (model={OLLAMA_MODEL})")
    print(f"Drive folder:  {folder_url}")
    print("------------------------------------")

    ensure_python_packages()
    ensure_docker_compose_up()
    setup_elastic()
    ensure_ollama_and_model()
    drive_download(folder_url)
    ingest_and_embed()
    start_api_and_ui()

if __name__ == "__main__":
    main()
