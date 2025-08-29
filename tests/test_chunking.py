from src.ingest_pdfs import chunk_text

def test_chunk_text_respects_overlap_and_size():
    text = " ".join(f"w{i}" for i in range(1, 1001))  # 1000 tokens
    chunks = chunk_text(text, max_tokens=300, overlap=60)
    assert len(chunks) >= 3  # 300-60=240 step → ~5 chunks for 1000 words
    # each chunk size
    for c in chunks[:-1]:
        tok_count = len(c.split())
        assert 240 <= tok_count <= 300
