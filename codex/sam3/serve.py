"""Run the unchanged author server, with Runpod's readiness route added."""
import importlib.util
import logging
import os
import threading

import torch
from flask import g, request

spec = importlib.util.spec_from_file_location("official_sam3", "/opt/crafter/scripts/sam3_server.py")
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
native.app.add_url_rule("/ping", "runpod_ping", lambda: ("", 200 if native.sam3_processor is not None else 204))
inference_lock = threading.Lock()


@native.app.before_request
def serialize_inference():
    if request.path in ("/segment", "/segment_bbox"):
        inference_lock.acquire()
        g.sam3_lock_held = True


@native.app.teardown_request
def release_inference(_error):
    if g.pop("sam3_lock_held", False):
        inference_lock.release()


def load():
    try:
        assert torch.cuda.is_available(), "The endpoint must have a CUDA GPU."
        native.load_model()
    except Exception:
        logging.exception("Official SAM3 model initialization failed")
        os._exit(1)


if __name__ == "__main__":
    threading.Thread(target=load, daemon=True).start()
    native.app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8765")), threaded=True)
