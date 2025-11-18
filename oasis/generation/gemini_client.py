from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

try:
    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore
    _HAS_GENAI = True
except Exception:  # pragma: no cover
    genai = None
    genai_types = None
    _HAS_GENAI = False
import requests

load_dotenv()

@dataclass
class GeminiConfig:
    api_key: str
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 64
    max_output_tokens: int = 256
    candidate_count: int = 1
    model_id: str = "gemini-2.5-flash-lite"


def _safety_settings_off_rest() -> List[Dict[str, str]]:
    categories = [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ]
    return [{"category": c, "threshold": "OFF"} for c in categories]


def _safety_settings_off_sdk():
    if not _HAS_GENAI:
        return None
    categories = [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ]
    return [
        genai_types.SafetySetting(category=category, threshold="OFF")
        for category in categories
    ]


def _make_client(api_key: str):
    if not _HAS_GENAI:
        return None
    return genai.Client(api_key=api_key)


def generate_text(
    system_instruction: str,
    user_text: str,
    config: Optional[GeminiConfig] = None,
) -> str:
    api_key = (config.api_key if config else os.getenv("GEMINI_API_KEY", "")).strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    model_id = (config.model_id if config else os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    # Prefer SDK if available
    if _HAS_GENAI:
        client = _make_client(api_key)
        gen_cfg = genai_types.GenerateContentConfig(
            temperature=(config.temperature if config else 0.7),
            top_p=(config.top_p if config else 0.9),
            top_k=(config.top_k if config else 64),
            candidate_count=(config.candidate_count if config else 1),
            max_output_tokens=(config.max_output_tokens if config else 256),
            safety_settings=_safety_settings_off_sdk(),
            system_instruction=system_instruction,
        )
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=user_text,
                config=gen_cfg,
            )
            return (getattr(response, "text", "") or "").strip()
        except Exception as e:  # fall through to REST
            print(f"Gemini SDK error: {e}")

    # REST fallback to Developer API v1beta
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent?key={api_key}"
    )
    headers = {"Content-Type": "application/json"}
    generation_config = {
        "temperature": (config.temperature if config else 0.7),
        "topP": (config.top_p if config else 0.9),
        "topK": (config.top_k if config else 64),
        "candidateCount": (config.candidate_count if config else 1),
        "maxOutputTokens": (config.max_output_tokens if config else 256),
    }
    payload: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_text}],
            }
        ],
        "systemInstruction": {
            "role": "system",
            "parts": [{"text": system_instruction}],
        },
        "generationConfig": generation_config,
        "safetySettings": _safety_settings_off_rest(),
    }
    try:
        resp = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=30)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        return "\n".join([t for t in texts if t]).strip()
    except Exception as e:
        print(f"Gemini REST error: {e}")
        return ""


