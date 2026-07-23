#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${OPENAI_API_KEY:-}" && -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "[online] No OPENAI_API_KEY or ANTHROPIC_API_KEY found, skipping online/model-dependent tests."
  exit 0
fi

HAS_USABLE_ANTHROPIC_KEY=0
if [[ -n "${ANTHROPIC_API_KEY:-}" && "${ANTHROPIC_API_KEY}" == sk-ant-api* ]]; then
  HAS_USABLE_ANTHROPIC_KEY=1
elif [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "[online] ANTHROPIC_API_KEY does not look like a standard API key (expected sk-ant-api...), skipping Anthropic path."
fi

if [[ ${HAS_USABLE_ANTHROPIC_KEY} -eq 1 ]]; then
  if ! python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('anthropic') else 1)"; then
    echo "[online] Installing missing 'anthropic' SDK..."
    pip install anthropic
  fi
fi

if [[ -z "${OPENAI_API_KEY:-}" && ${HAS_USABLE_ANTHROPIC_KEY} -eq 0 ]]; then
  echo "[online] No usable OpenAI/Anthropic credentials found, skipping online/model-dependent tests."
  exit 0
fi

echo "[online] Running tests that depend on OpenAI/HuggingFace model access..."
pytest -q \
  test/agent/test_agent_graph.py \
  test/agent/test_agent_generator.py \
  test/agent/test_multi_agent_signup_create.py \
  test/agent/test_twitter_user_agent_all_actions.py \
  test/agent/test_agent_tools.py \
  test/agent/test_agent_custom_prompt.py \
  test/agent/test_interview_action.py \
  test/infra/recsys/test_recsys.py \
  test/infra/recsys/test_update_rec_table.py

echo "[online] Done."
