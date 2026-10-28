"""Check readiness stays responsive while inference is waiting for its lock."""
from concurrent.futures import ThreadPoolExecutor

import torch
from sam3.model.vitdet import Block

from serve import inference_lock, native

block = Block(dim=64, num_heads=4, input_size=(4, 4)).eval().float()
with torch.inference_mode():
    result = block(torch.randn(1, 4, 4, 64, dtype=torch.float32))
assert result.dtype == torch.float32
print("Official SAM3 block supports the author's FP32 server.")


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
