#!/usr/bin/env bash
set -euo pipefail
python /opt/service/prefetch.py
export HF_HUB_OFFLINE=1
exec python /opt/service/serve.py
