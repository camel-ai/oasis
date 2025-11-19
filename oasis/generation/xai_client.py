from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

try:
    # xAI official SDK (optional)
    # https://docs.x.ai
    from xai_sdk import Client as XAIClient  # type: ignore
    from xai_sdk.chat import system as xai_system  # type: ignore
    from xai_sdk.chat import user as xai_user  # type: ignore
    _HAS_XAI_SDK = True
except Exception:  # pragma: no cover
    XAIClient = None
    xai_user = None
    xai_system = None
    _HAS_XAI_SDK = False

try:
    # OpenAI-compatible SDK (optional)
    # Used with base_url="https://api.x.ai/v1"
    import httpx  # type: ignore
    from openai import OpenAI  # type: ignore
    _HAS_OPENAI_SDK = True
except Exception:  # pragma: no cover
    httpx = None
    OpenAI = None
    _HAS_OPENAI_SDK = False

import requests


@dataclass
class XAIConfig:
    r"""Configuration for xAI Grok chat completions."""
    api_key: str
    model: str = "grok-4-fast-non-reasoning"
    temperature: float = 0.7
    top_p: float = 0.9
    max_output_tokens: int = 256
    base_url: str = "https://api.x.ai/v1"
    timeout_seconds: float = 3600.0


def _get_env(key: str, default: str = "") -> str:
    value = os.getenv(key, default)
    return (value or "").strip()


def generate_text(
    system_instruction: str,
    user_text: str,
    config: Optional[XAIConfig] = None,
) -> str:
    r"""Generate text using xAI Grok, preferring SDK with fallbacks.

    Order of backends:
        1) xAI Python SDK (if installed)
        2) OpenAI SDK pointed at xAI base_url
        3) Raw REST call to /chat/completions

    Args:
        system_instruction (str): System prompt to guide model behavior.
        user_text (str): User prompt content.
        config (Optional[XAIConfig]): Optional configuration. When not
            provided, environment variables are used:
            - XAI_API_KEY (required)
            - XAI_MODEL_NAME (optional; default: "grok-4-fast-non-reasoning")
            - XAI_BASE_URL (optional; default: "https://api.x.ai/v1")
            - XAI_TIMEOUT_SECONDS (optional; default: "3600")

    Returns:
        str: Model response text (may be empty on failure).
    """
    api_key = (config.api_key if config else _get_env("XAI_API_KEY"))
    if not api_key:
        raise RuntimeError("XAI_API_KEY not set")

    model = (config.model if config else _get_env("XAI_MODEL_NAME", "grok-4-fast-non-reasoning"))
    base_url = (config.base_url if config else _get_env("XAI_BASE_URL", "https://api.x.ai/v1"))
    timeout_raw = (str(config.timeout_seconds) if config else _get_env("XAI_TIMEOUT_SECONDS", "3600"))
    try:
        timeout = float(timeout_raw)
    except ValueError as exc:  # pragma: no cover
        raise RuntimeError(f"Invalid XAI_TIMEOUT_SECONDS value '{timeout_raw}'; expected numeric.") from exc

    temperature = config.temperature if config else 0.7
    top_p = config.top_p if config else 0.9
    max_tokens = config.max_output_tokens if config else 256

    # 1) Try xAI SDK
    if _HAS_XAI_SDK:
        try:
            client = XAIClient(api_key=api_key, timeout=timeout)  # type: ignore[misc]
            chat = client.chat.create(model=model)
            chat.append(xai_system(system_instruction))  # type: ignore[misc]
            chat.append(xai_user(user_text))  # type: ignore[misc]
            # The SDK's sampling parameters are mostly model-level; not all are exposed on sample()
            response = chat.sample()
            content = getattr(response, "content", "") or ""
            return str(content).strip()
        except Exception:
            # Fall through to OpenAI SDK route
            pass

    # 2) Try OpenAI SDK against xAI base_url
    if _HAS_OPENAI_SDK:
        try:
            client = OpenAI(  # type: ignore[call-arg]
                api_key=api_key,
                base_url=base_url,
                timeout=httpx.Timeout(timeout) if httpx else None,  # type: ignore[arg-type]
            )
            completion = client.chat.completions.create(  # type: ignore[attr-defined]
                model=model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_text},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            message = completion.choices[0].message  # type: ignore[index]
            content = getattr(message, "content", "") or ""
            return str(content).strip()
        except Exception:
            # Fall through to REST
            pass

    # 3) Raw REST to /chat/completions
    try:
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_text},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        message = (choices[0] or {}).get("message", {})
        content = message.get("content", "")
        return str(content or "").strip()
    except Exception:
        return ""


