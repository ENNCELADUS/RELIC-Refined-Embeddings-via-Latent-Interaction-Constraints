#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/v3.yaml}"

python -m src.run --config "${CONFIG_PATH}"
