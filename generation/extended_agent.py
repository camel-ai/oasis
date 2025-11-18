from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from camel.messages import BaseMessage
from oasis.social_agent.agent import SocialAgent
from oasis.social_platform.typing import ActionType

from generation.emission_policy import EmissionPolicy, PersonaConfig
from generation.labeler import assign_labels
from orchestrator.rng import DeterministicRNG
from orchestrator.sidecar_logger import SidecarLogger
from orchestrator.expect_registry import ExpectRegistry
from orchestrator.scheduler import MultiLabelScheduler

_TOKEN_RE = re.compile(r"<LBL:[A-Z_]+>")


class ExtendedSocialAgent(SocialAgent):
    r"""SocialAgent extension that injects per-step label-token instructions and logs sidecar."""

    def __init__(
        self,
        *args,
        persona_cfg: PersonaConfig,
        emission_policy: EmissionPolicy,
        sidecar_logger: SidecarLogger,
        run_seed: int,
        expect_registry: Optional[ExpectRegistry] = None,
        scheduler: Optional[MultiLabelScheduler] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._persona_cfg: PersonaConfig = persona_cfg
        self._policy: EmissionPolicy = emission_policy
        self._sidecar: SidecarLogger = sidecar_logger
        self._run_seed: int = int(run_seed)
        self._step_index: int = 0
        self._expect_registry: Optional[ExpectRegistry] = expect_registry
        self._scheduler: Optional[MultiLabelScheduler] = scheduler

    async def perform_action_by_llm(self):
        # Build step-scoped RNG and emission decision
        # For lack of direct thread_id here, use agent_id as a stable scope.
        user_id = int(self.social_agent_id)
        thread_scope = f"a_{user_id}"
        override = self._scheduler.get_mode_override(self._persona_cfg) if self._scheduler else None
        decision = self._policy.decide(
            user_id=user_id,
            thread_id=thread_scope,
            step_idx=self._step_index,
            persona=self._persona_cfg,
            context=None,
            override_post_mode_probs=override,
        )

        # Record expectation for programmatic fallback
        if self._expect_registry is not None:
            await self._expect_registry.set_expected(user_id, self._step_index, decision.get("tokens", []))

        # Augment user message with a concise step instruction
        step_hint = self._format_step_hint(decision)
        env_prompt = await self.env.to_text_prompt()
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=(
                f"{step_hint}\n"
                f"Here is your social media environment: {env_prompt}"
            ),
        )
        try:
            response = await self.astep(user_msg)
            # Log each tool call to sidecar with detected tokens and labels.
            tool_calls = response.info.get("tool_calls", [])
            for tool_call in tool_calls:
                action_name = tool_call.tool_name
                args = tool_call.args or {}
                result = getattr(tool_call, "result", None)
                await self._log_sidecar(action_name, args, result, decision)
            self._step_index += 1
            return response
        except Exception as e:
            # Keep parity with base class error handling
            from oasis.social_agent.agent import agent_log
            agent_log.error(f"Agent {self.social_agent_id} error: {e}")
            return e

    def _format_step_hint(self, decision: Dict[str, Any]) -> str:
        mode = decision.get("mode", "none")
        tokens = decision.get("tokens", [])
        if mode == "none":
            return "This is a benign neutral comment. Do not include any label markers."
        if mode == "single" and tokens:
            return f"Use exactly one label marker inline: {tokens[0]}."
        if mode == "double" and len(tokens) >= 2:
            return f"Use exactly two label markers inline: {tokens[0]} and {tokens[1]}."
        # Fallback
        return "If natural, include label markers as instructed; otherwise keep it neutral."

    async def _log_sidecar(
        self,
        action_name: str,
        args: Dict[str, Any],
        result: Any,
        decision: Dict[str, Any],
    ) -> None:
        if action_name not in (ActionType.CREATE_POST.value, ActionType.CREATE_COMMENT.value):
            return
        content: str = ""
        parent_id: Optional[int] = None
        if action_name == ActionType.CREATE_POST.value:
            content = str(args.get("content", "")) if isinstance(args, dict) else str(args)
        else:
            # comment args typically include {"post_id": int, "content": str}
            parent_id = int(args.get("post_id", 0)) if isinstance(args, dict) else None
            content = str(args.get("content", "")) if isinstance(args, dict) else str(args)

        detected = _TOKEN_RE.findall(content or "")
        labels = assign_labels(detected, self._persona_cfg.allowed_labels)

        # Extract IDs from platform result if available
        rid: Dict[str, Any] = result if isinstance(result, dict) else {}
        post_id = rid.get("post_id")
        comment_id = rid.get("comment_id")

        insertion_fallback = False
        # Consume registry state to include fallback flag once per step
        if self._expect_registry is not None:
            consumed = await self._expect_registry.consume(int(self.social_agent_id))
            if consumed:
                insertion_fallback = bool(consumed.insertion_fallback)

        record = {
            "agent_id": int(self.social_agent_id),
            "action": action_name,
            "step_idx": int(self._step_index),
            "expected_mode": decision.get("mode"),
            "expected_tokens": decision.get("tokens", []),
            "detected_tokens": detected,
            "category_labels": labels,
            "insertion_fallback": insertion_fallback,
            "post_id": post_id,
            "comment_id": comment_id,
            "parent_post_id": parent_id,
            "persona_id": self._persona_cfg.persona_id,
        }
        self._sidecar.write(record)
        # Update scheduler with observed labels
        if self._scheduler:
            try:
                self._scheduler.observe(labels)
            except Exception:
                pass


