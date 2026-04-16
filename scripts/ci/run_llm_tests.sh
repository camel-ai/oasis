#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${OPENAI_API_KEY:-}" && -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "[llm] No OPENAI_API_KEY or ANTHROPIC_API_KEY found, skipping LLM-dependent tests."
  exit 0
fi

HAS_USABLE_ANTHROPIC_KEY=0
if [[ -n "${ANTHROPIC_API_KEY:-}" && "${ANTHROPIC_API_KEY}" == sk-ant-api* ]]; then
  HAS_USABLE_ANTHROPIC_KEY=1
elif [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "[llm] ANTHROPIC_API_KEY does not look like a standard API key (expected sk-ant-api...), skipping Anthropic path."
fi

if [[ ${HAS_USABLE_ANTHROPIC_KEY} -eq 1 ]]; then
  if ! python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('anthropic') else 1)"; then
    echo "[llm] Installing missing 'anthropic' SDK..."
    pip install anthropic
  fi
fi

if [[ -z "${OPENAI_API_KEY:-}" && ${HAS_USABLE_ANTHROPIC_KEY} -eq 0 ]]; then
  echo "[llm] No usable OpenAI/Anthropic credentials found, skipping LLM-dependent tests."
  exit 0
fi

echo "[llm] Running LLM-dependent pytest suite..."
pytest -q \
  test/agent/test_agent_generator.py \
  test/agent/test_agent_tools.py \
  test/agent/test_agent_custom_prompt.py \
  test/agent/test_interview_action.py

echo "[llm] Done."
