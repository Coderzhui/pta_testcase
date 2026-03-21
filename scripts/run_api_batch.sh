#!/usr/bin/env bash
set -euo pipefail

INPUT_PATH="${1:-apis.txt}"
shift || true

python -m scripts.pipeline run --input "${INPUT_PATH}" "$@"
