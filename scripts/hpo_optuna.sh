#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate relic

CONFIG_PATH="${1:-configs/v5_hpo.yaml}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

python -m src.optimize.run --config "$CONFIG_PATH"
