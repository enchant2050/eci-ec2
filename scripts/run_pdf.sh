#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_PREFIX="${MAMBA_ENV_PREFIX:-$APP_DIR/.runtime/envs/eci-ocr}"

cd "$APP_DIR"

PDF_PATH="${1:?Usage: scripts/run_pdf.sh path/to/file.pdf [output.json]}"
shift

if [ "${1:-}" != "" ] && [ "${1#--}" = "$1" ]; then
  OUTPUT_PATH="$1"
  shift
else
  OUTPUT_PATH="outputs/$(basename "${PDF_PATH%.*}")_result.json"
fi

export PATH="$ENV_PREFIX/bin:$PATH"
PYTHONPATH=. python -m src.ec2_runner "$PDF_PATH" --output "$OUTPUT_PATH" "$@"

echo "Wrote $OUTPUT_PATH"
