from __future__ import annotations

import asyncio
import re
from typing import Any, Optional, Tuple

from oasis.social_platform.channel import Channel
from oasis.social_platform.typing import ActionType
from orchestrator.expect_registry import ExpectRegistry

_TOKEN_RE = re.compile(r"<LBL:[A-Z_]+>")


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
                    new_content, inserted = _ensure_tokens_in_content(str(payload), rec.tokens)
                    if inserted:
                        await self._registry.note_insertion(int(agent_id))
                        data = (agent_id, new_content, action.value)
                        return (message_id, data)
                else:
                    try:
                        post_id, content = payload
                        new_content, inserted = _ensure_tokens_in_content(str(content), rec.tokens)
                        if inserted:
                            await self._registry.note_insertion(int(agent_id))
                            data = (agent_id, (post_id, new_content), action.value)
                            return (message_id, data)
                    except Exception:
                        pass
        return message

    async def send_to(self, message):
        return await self._base.send_to(message)

    async def write_to_receive_queue(self, action_info):
        return await self._base.write_to_receive_queue(action_info)


