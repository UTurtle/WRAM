#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "Usage: bash run.sh MANIFEST CHECKPOINT NEW_OUTPUT_DIR" >&2
  exit 2
fi
EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$EXPERIMENT_DIR/../.." && pwd)"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec "${PYTHON_BIN:-python}" -m wram run --manifest "$1" --checkpoint "$2" --config "$EXPERIMENT_DIR/config.json" --output "$3"
