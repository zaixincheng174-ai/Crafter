"""Check readiness stays responsive while inference is waiting for its lock."""
from concurrent.futures import ThreadPoolExecutor

from serve import inference_lock, native


def inference_request():
    with native.app.test_client() as client:
        return client.post("/segment_bbox").status_code


with ThreadPoolExecutor(max_workers=1) as pool:
    inference_lock.acquire()
    pending = pool.submit(inference_request)
    try:
        with native.app.test_client() as client:
            assert client.get("/ping").status_code == 204
            assert client.get("/health").json["model_loaded"] is False
        assert not pending.done()
    finally:
        inference_lock.release()
    assert pending.result(timeout=5) == 500
print("Health concurrency check passed; no model inference was mocked or claimed.")
