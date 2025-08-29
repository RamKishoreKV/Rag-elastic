import os
import pytest


RUN_INTEGRATION = os.getenv("RUN_INTEGRATION") == "1"
pytestmark = pytest.mark.skipif(not RUN_INTEGRATION, reason="set RUN_INTEGRATION=1 to run")

def test_healthz_live():
    import requests
    r = requests.get("http://127.0.0.1:8000/healthz", timeout=3)
    assert r.status_code == 200
    assert r.json().get("ok") in (True, False)
