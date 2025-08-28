import os, requests, streamlit as st

API = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG-Elastic", page_icon="🔎", layout="centered")
st.title("RAG over PDFs 🔎")

mode = st.radio("Retrieval mode", ["hybrid","elser","dense","bm25"], horizontal=True, index=0)
q = st.text_input("Ask a question")
size = st.slider("Top K", 1, 10, 5)

if st.button("Search") and q.strip():
    payload = {"q": q, "mode": mode, "size": size}
    r = requests.post(f"{API}/query", json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    st.subheader("Answer")
    st.write(data.get("answer") or "I don’t know.")
    st.subheader("Citations")
    for c in data.get("citations", []):
        st.markdown(f"- **{c['title']}** p.{c['page']} — {c['source']}")
    st.subheader("Top results")
    for r in data.get("results", []):
        st.markdown(f"**{r['title']}** [p.{r['page']}] — *{r['source']}*")
        st.write((r.get('snippet') or "")[:500] + ("…" if len(r.get('snippet',''))>500 else ""))
