#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="$(mktemp -d -t oasis-short-video-smoke-XXXXXX)"
trap 'rm -rf "${OUT_DIR}"' EXIT

echo "[short-video] Running experiment pipeline smoke test in ${OUT_DIR} ..."
python examples/experiment/short_video_negative_feedback/run_experiment.py \
  --output-dir "${OUT_DIR}"

python visualization/short_video_simulation/code/generate_report.py \
  "${OUT_DIR}/negative_feedback_baseline.db" \
  --output "${OUT_DIR}/baseline_report.md"

python visualization/short_video_simulation/code/compare_runs.py \
  baseline="${OUT_DIR}/negative_feedback_baseline.db" \
  treatment="${OUT_DIR}/negative_feedback_treatment.db" \
  --output-dir "${OUT_DIR}/comparison"

test -f "${OUT_DIR}/baseline_report.md"
test -f "${OUT_DIR}/comparison/comparison_time_series.csv"
test -f "${OUT_DIR}/comparison/comparison_metrics.png"

echo "[short-video] Smoke test passed."
