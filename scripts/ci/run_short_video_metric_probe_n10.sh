#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/examples/experiment/short_video_negative_feedback/output_metric_probe_n10}"

echo "[metric-probe-n10] output dir: ${OUT_DIR}"
mkdir -p "${OUT_DIR}"

python "${ROOT_DIR}/examples/experiment/short_video_negative_feedback/run_experiment.py" \
  --output-dir "${OUT_DIR}" \
  --n-runs 10 \
  --seed-start 5252 \
  --watch-jitter 0.30 \
  --behavior-flip-prob 0.10 \
  --treatment-extra-neg-prob 1.00 \
  --strategy-set baseline_treatment \
  --recsys-profile negative_sensitive \
  --creator-multiplier 20 \
  --viewer-multiplier 20 \
  --video-multiplier 20

python - <<'PY' "${OUT_DIR}"
from pathlib import Path
import csv
import sys

out = Path(sys.argv[1])
summary = out / "multirun_summary.csv"
rows = list(csv.DictReader(summary.open("r", encoding="utf-8")))
lookup = {r["metric"]: r for r in rows}

def show(metric: str):
    r = lookup[metric]
    print(
        f"{metric}: diff={float(r['paired_diff_mean']):.6f}, "
        f"p={float(r['paired_signflip_pvalue']):.6f}, "
        f"baseline={float(r['baseline_mean']):.6f}, "
        f"treatment={float(r['treatment_mean']):.6f}"
    )

print("[metric-probe-n10] key metrics:")
for metric in [
    "negative_feedback_rate",
    "retention_3s_rate",
    "creator_coverage",
    "avg_watch_ratio",
    "comedy_traffic_pool_level",
]:
    if metric in lookup:
        show(metric)
PY

echo "[metric-probe-n10] done."
echo "[metric-probe-n10] outputs:"
echo "  - ${OUT_DIR}/multirun_metrics.csv"
echo "  - ${OUT_DIR}/multirun_summary.csv"
