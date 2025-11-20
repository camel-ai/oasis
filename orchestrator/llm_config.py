from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


@dataclass(frozen=True)
class LLMConfig:
    r"""Centralized constants for LLM params and rate limits (no env reads).

    Values can be overridden by providing a YAML file at `configs/llm_config.yaml`
    in the repository root (or by passing a path to `load_llm_config`).
    """

    # Token budgets
    xai_max_tokens: int
    openai_max_tokens: int
    gemini_max_output_tokens: int
    est_prompt_tokens: int

    # Iteration caps
    max_step_iterations: int

    # xAI rate limits
    xai_rpm: int
    xai_tpm: int
    xai_rps: int
    rate_limit_enabled: bool
    xai_retry_attempts: int


# Single source of truth for defaults used across the codebase
_DEFAULTS = LLMConfig(
    # Tokens
    xai_max_tokens=8912,
    openai_max_tokens=8912,
    gemini_max_output_tokens=8912,
    est_prompt_tokens=12000,
    # Iterations
    max_step_iterations=1,
    # xAI rate limits
    xai_rpm=480,
    xai_tpm=4_000_000,
    xai_rps=8,
    rate_limit_enabled=True,
    xai_retry_attempts=5,
)


def _coerce_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true", "1", "yes", "on"):  # pragmatic acceptance
            return True
        if s in ("false", "0", "no", "off"):
            return False
    return bool(val)


def _apply_overrides(base: LLMConfig, raw: Mapping[str, Any]) -> LLMConfig:
    updated = base
    for key, value in raw.items():
        if not hasattr(base, key):
            continue
        if key in ("rate_limit_enabled",):
            coerced = _coerce_bool(value)
        elif isinstance(getattr(base, key), int):
            try:
                coerced = int(value)
            except Exception:
                continue
        else:
            coerced = value
        updated = replace(updated, **{key: coerced})
    return updated


def load_llm_config(config_path: Optional[str] = None) -> LLMConfig:
    """Load LLMConfig from YAML if present; otherwise return defaults.

    Args:
        config_path: Optional path to a YAML file. If not provided, we look for
            `configs/llm_config.yaml` at the repository root.
    """
    # Resolve default path relative to repo root
    default_path = Path(__file__).resolve().parents[1] / "configs" / "llm_config.yaml"
    cfg_path = Path(config_path) if config_path else default_path
    if not cfg_path.exists():
        return _DEFAULTS
    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, Mapping):
            return _DEFAULTS
        return _apply_overrides(_DEFAULTS, data)
    except Exception:
        return _DEFAULTS


# Load-time configuration (can be imported directly as a constant)
LLM_CONFIG = load_llm_config()


