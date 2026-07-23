#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

echo "[all] Stage 1/3: deterministic tests"
bash scripts/ci/run_deterministic_tests.sh

echo "[all] Stage 2/3: short-video pipeline smoke"
bash scripts/ci/run_short_video_smoke.sh

echo "[all] Stage 3/3: optional LLM tests"
bash scripts/ci/run_llm_tests.sh

echo "[all] Optional stage: online model tests"
bash scripts/ci/run_online_model_tests.sh

echo "[all] All requested stages completed."
