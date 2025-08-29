from src.rag_answer import rrf_merge

def _hit(i):
    return {"_id": f"id-{i}", "_score": 1.0, "_source": {"title": f"T{i}", "page": 1, "content": "x"}}

def test_rrf_merge_prefers_consensus():
    # id-2 appears near the top in both, should win
    list1 = [_hit(2), _hit(3), _hit(4)]
    list2 = [_hit(2), _hit(5), _hit(6)]
    merged = rrf_merge(list1, list2, k=60)
    assert merged[0]["_id"] == "id-2"
    # No duplicates
    ids = [h["_id"] for h in merged]
    assert len(ids) == len(set(ids))
