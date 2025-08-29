# src/ui.py
import os
import time
import requests
import streamlit as st

API = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG-Elastic (Chat)", page_icon="💬", layout="centered")
st.title("RAG over PDFs • Chat 💬")
st.caption("Hybrid retrieval (ELSER + dense + BM25) with grounded answers & citations")

# ----- Sidebar controls -----
with st.sidebar:
    mode = st.radio("Retrieval mode", ["hybrid", "elser", "dense", "bm25"], horizontal=True, index=0)
    size = st.slider("Top K (default 5)", 1, 10, 5)
    if st.button("Clear chat"):
        st.session_state.messages = []

# ----- Chat state -----
if "messages" not in st.session_state:
    st.session_state.messages = []  # each: {"role": "user"|"assistant", "content": str}

def _is_idk_or_refusal(text: str) -> bool:
    if not text:
        return True
    t = text.strip().lower()
    return (
        t == "i don’t know."
        or t == "i don't know."
        or t.startswith("i can’t help")
        or t.startswith("i can't help")
    )

def _render_citations(citations):
    seen = set()
    lines = []
    for c in citations or []:
        key = (c.get("title"), c.get("page"))
        if key in seen:
            continue
        seen.add(key)
        title = c.get("title", "?")
        page = c.get("page", "?")
        link = (c.get("link") or "").strip()
        src  = c.get("source", "")
        if link:
            lines.append(f"- [{title} p.{page}]({link}) — {src}")
        else:
            lines.append(f"- **{title}** p.{page} — {src}")
    return "\n".join(lines)

def _render_results(results, limit=4):
    seen = set()
    out = []
    for r in results or []:
        key = (r.get("title"), r.get("page"))
        if key in seen:
            continue
        seen.add(key)
        title = r.get("title", "?")
        page  = r.get("page", "?")
        src   = r.get("source", "")
        snip  = r.get("snippet") or ""
        snip  = snip[:500] + ("…" if len(snip) > 500 else "")
        out.append(f"**{title}** • p.{page} · *{src}*\n\n{snip}")
        if len(out) >= limit:
            break
    return "\n\n---\n\n".join(out)

# ----- Replay chat history -----
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ----- Input box -----
prompt = st.chat_input("Ask a question")

if prompt:
    # 1) echo user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) call API
    payload = {"q": prompt, "mode": mode, "size": size}
    started = time.perf_counter()
    try:
        r = requests.post(f"{API}/query", json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        err = f"Request failed: {e}"
        st.session_state.messages.append({"role": "assistant", "content": err})
        with st.chat_message("assistant"):
            st.error(err)
    else:
        latency_ms = int((time.perf_counter() - started) * 1000)
        answer = (data.get("answer") or "I don’t know.").strip()

        # 3) render assistant message
        with st.chat_message("assistant"):
            st.markdown(answer)

            # Only show citations & results when it's a grounded answer
            if not _is_idk_or_refusal(answer):
                cits_md = _render_citations(data.get("citations"))
                if cits_md:
                    with st.expander("Citations"):
                        st.markdown(cits_md)
                results_md = _render_results(data.get("results"), limit=4)
                if results_md:
                    with st.expander("Top results"):
                        st.markdown(results_md)

            st.caption(f"Mode: **{data.get('mode', mode)}** • Top K: **{size}** • Latency: **{latency_ms} ms**")

        # append to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
