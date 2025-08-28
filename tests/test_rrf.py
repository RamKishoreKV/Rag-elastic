# tests/test_rrf.py
from src.rrf import rrf_merge

def test_rrf_merge_basic():
    a = [{"_id": "a"}, {"_id": "b"}, {"_id": "c"}]
    b = [{"_id": "b"}, {"_id": "a"}, {"_id": "d"}]
    out = rrf_merge(a, b, k=60)
    ids = [h["_id"] for h in out]
    assert set(ids[:2]) == {"a", "b"}
    assert "c" in ids and "d" in ids
