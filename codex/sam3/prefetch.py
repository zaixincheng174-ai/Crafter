"""Cache the approved official weights once on the endpoint's existing volume."""
import json
import os
from pathlib import Path

REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
cache = Path(os.environ["HF_HOME"]) / "hub"
repo = cache / "models--facebook--sam3"
snapshot = repo / "snapshots" / REVISION
required = ("config.json", "sam3.pt")

if not all((snapshot / name).is_file() for name in required):
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("Initial SAM3 cache needs an approved HF_TOKEN.")
    from huggingface_hub import hf_hub_download

    for name in required:
        hf_hub_download("facebook/sam3", name, revision=REVISION, cache_dir=cache)

assert (snapshot / "sam3.pt").stat().st_size > 1_000_000_000
json.loads((snapshot / "config.json").read_text())
(repo / "refs").mkdir(parents=True, exist_ok=True)
(repo / "refs" / "main").write_text(REVISION)
print("Official SAM3 cache ready:", REVISION, flush=True)
