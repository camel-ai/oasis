#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

echo "[deterministic] Running pytest suite without external model dependencies..."
pytest -q test \
  --ignore test/agent/test_agent_graph.py \
  --ignore test/agent/test_agent_generator.py \
  --ignore test/agent/test_multi_agent_signup_create.py \
  --ignore test/agent/test_twitter_user_agent_all_actions.py \
  --ignore test/agent/test_agent_tools.py \
  --ignore test/agent/test_agent_custom_prompt.py \
  --ignore test/agent/test_interview_action.py \
  --ignore test/infra/recsys/test_recsys.py \
  --ignore test/infra/recsys/test_update_rec_table.py

echo "[deterministic] Done."
