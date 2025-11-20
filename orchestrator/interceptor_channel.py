from __future__ import annotations

import asyncio
import re
from typing import Any, Optional, Tuple

from oasis.social_platform.channel import Channel
from oasis.social_platform.typing import ActionType
from orchestrator.expect_registry import ExpectRegistry

_TOKEN_RE = re.compile(r"<LBL:[A-Z_]+>")
_CLOSE_TAG_RE = re.compile(r"</LBL:[A-Z_]+>")
_SPAN_RE = re.compile(r"<LBL:([A-Z_]+)>(.*?)</LBL:\1>", re.DOTALL)
_MALFORMED_OPEN_RE = re.compile(r"<LBL:[^>]*")
_BARE_TOKEN_RE = re.compile(r"\bLBL:([A-Z_]+)\b")

# Canonical class tokens only
_CANONICAL_CLASSES = {"INCEL", "MISINFO", "CONSPIRACY", "ED_RISK", "RECOVERY", "BENIGN"}
_CANONICAL_TOKENS = {f"LBL:{c}" for c in _CANONICAL_CLASSES}

# Legacy child→parent best-effort mapping (ambiguous ones are deliberately omitted and will be dropped)
_CHILD_TO_PARENT = {
    "INCEL_SLANG": "INCEL",
    "MISINFO_CLAIM": "MISINFO",
    "MISINFO_SOURCE": "MISINFO",
    "CONSPIRACY_NARRATIVE": "CONSPIRACY",
    "DEEPSTATE": "CONSPIRACY",
    "ED_METHOD": "ED_RISK",
    "ED_PROMO": "ED_RISK",
    # Ambiguous tokens like SELF_HARM or ANTI_INSTITUTION are intentionally not mapped
    # to avoid introducing wrong classes; they will be dropped during normalization.
}


def _ensure_tokens_in_content(content: str, expected_tokens: list[str]) -> tuple[str, bool]:
    """Append any missing expected tokens to the content, return (new_content, inserted?)."""
    inserted = False
    text = content or ""
    for tok in expected_tokens or []:
        if tok and tok not in text:
            if text and not text.endswith((" ", "\n")):
                text += " "
            text += tok
            inserted = True
    return text, inserted


def _normalize_and_enforce_tokens(text: str, expected_tokens: list[str]) -> tuple[str, bool]:
    """Normalize legacy/malformed tokens and enforce exact expected token set cardinality.

    Rules:
    - Convert any <LBL:TYPE>...</LBL:TYPE> spans to bare ' LBL:TYPE'
    - Drop closing tags and malformed opens
    - Map legacy child tokens to canonical parent when unambiguous; drop others
    - Keep only canonical class tokens {LBL:INCEL,...}
    - Enforce exact cardinality:
      * If expected_tokens is empty -> strip all tokens
      * Else -> remove all tokens and append the expected tokens once (space-separated)
    """
    original = text or ""
    s = str(original)
    changed = False
    # 1) Collapse spans to bare tokens
    def _span_repl(m: re.Match) -> str:
        token = f" LBL:{m.group(1)}"
        return token
    s2 = _SPAN_RE.sub(_span_repl, s)
    if s2 != s:
        changed = True
        s = s2
    # 2) Remove closing tags and malformed opens
    s2 = _CLOSE_TAG_RE.sub("", s)
    if s2 != s:
        changed = True
        s = s2
    s2 = _MALFORMED_OPEN_RE.sub("", s)
    if s2 != s:
        changed = True
        s = s2
    # 3) Extract all bare tokens
    present = [m.group(1) for m in _BARE_TOKEN_RE.finditer(s)]
    # 4) Remove all bare tokens from text (we will re-append as needed)
    if present:
        changed = True
        s = _BARE_TOKEN_RE.sub("", s)
    # 5) Map legacy child tokens to parents and filter to canonical
    normalized_tokens = []
    for t in present:
        parent = _CHILD_TO_PARENT.get(t, t)
        if f"LBL:{parent}" in _CANONICAL_TOKENS:
            normalized_tokens.append(f"LBL:{parent}")
    # 6) Enforce cardinality
    expected = list(expected_tokens or [])
    if not expected:
        # Benign/none: ensure no tokens remain; nothing to append
        return (s, changed or bool(normalized_tokens))
    # For harmful modes: append exactly the expected tokens (discard any others)
    # Trim trailing spaces first
    s = s.rstrip()
    append_str = " " + " ".join(expected)
    s2 = s + append_str
    if s2 != original:
        changed = True
    return (s2, changed)


class InterceptorChannel(Channel):
    """A Channel wrapper that programmatically inserts expected tokens if missing.

    It delegates all operations to an underlying Channel instance while
    intercepting `receive_from()` to adjust content for CREATE_POST/CREATE_COMMENT.
    """

    def __init__(self, base: Channel, registry: ExpectRegistry):
        # Do not call super().__init__ to avoid creating new queues/dicts;
        # we purely delegate to base.
        self._base = base
        self._registry = registry
        # Ensure compatibility with callers that access these directly
        # by exposing proxy properties.
        # Note: properties defined below.

    def __getattr__(self, name: str):
        # Delegate unknown attributes to the base channel (e.g., future fields)
        return getattr(self._base, name)

    async def receive_from(self):
        message = await self._base.receive_from()
        # message is (message_id, (agent_id, message, action_type))
        try:
            message_id, data = message
            agent_id, payload, action = data
            action = ActionType(action)
        except Exception:
            return message

        if action in (ActionType.CREATE_POST, ActionType.CREATE_COMMENT):
            rec = await self._registry.get_expected(int(agent_id))
            if rec and rec.tokens:
                # payload is content (post) or (post_id, content) tuple (comment)
                if action == ActionType.CREATE_POST:
                    # Normalize any legacy/malformed tokens and enforce exact expected set
                    new_content, inserted = _normalize_and_enforce_tokens(str(payload), rec.tokens)
                    if inserted:
                        await self._registry.note_insertion(int(agent_id))
                        data = (agent_id, new_content, action.value)
                        return (message_id, data)
                else:
                    try:
                        post_id, content = payload
                        new_content, inserted = _normalize_and_enforce_tokens(str(content), rec.tokens)
                        if inserted:
                            await self._registry.note_insertion(int(agent_id))
                            data = (agent_id, (post_id, new_content), action.value)
                            return (message_id, data)
                    except Exception:
                        pass
            elif rec and not rec.tokens:
                # Benign/none mode: strip any stray tokens if present
                if action == ActionType.CREATE_POST:
                    cleaned, changed = _normalize_and_enforce_tokens(str(payload), [])
                    if changed:
                        await self._registry.note_insertion(int(agent_id))
                        data = (agent_id, cleaned, action.value)
                        return (message_id, data)
                else:
                    try:
                        post_id, content = payload
                        cleaned, changed = _normalize_and_enforce_tokens(str(content), [])
                        if changed:
                            await self._registry.note_insertion(int(agent_id))
                            data = (agent_id, (post_id, cleaned), action.value)
                            return (message_id, data)
                    except Exception:
                        pass
        return message

    async def send_to(self, message):
        return await self._base.send_to(message)

    async def write_to_receive_queue(self, action_info):
        return await self._base.write_to_receive_queue(action_info)

    async def read_from_send_queue(self, message_id):
        return await self._base.read_from_send_queue(message_id)

    # Proxy properties for direct attribute access
    @property
    def send_dict(self):
        return self._base.send_dict

    @property
    def receive_queue(self):
        return self._base.receive_queue


