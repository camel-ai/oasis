#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/examples/experiment/short_video_negative_feedback/output_quick_probe}"

echo "[quick-probe] output dir: ${OUT_DIR}"
mkdir -p "${OUT_DIR}"

python "${ROOT_DIR}/examples/experiment/short_video_negative_feedback/run_experiment.py" \
  --output-dir "${OUT_DIR}" \
  --n-runs 3 \
  --seed-start 4242 \
  --watch-jitter 0.08 \
  --behavior-flip-prob 0.05 \
  --treatment-extra-neg-prob 1.00 \
  --strategy-set baseline_treatment \
  --creator-multiplier 20 \
  --viewer-multiplier 20 \
  --video-multiplier 20

echo "[quick-probe] done."
echo "[quick-probe] key outputs:"
echo "  - ${OUT_DIR}/multirun_metrics.csv"
echo "  - ${OUT_DIR}/multirun_summary.csv"
echo "  - ${OUT_DIR}/MULTIRUN_REPORT.md"
