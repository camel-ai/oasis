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
        harm_priors: Optional[Dict[str, float]] = None,
        guidance_config: Optional[Dict[str, Any]] = None,
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
        self._harm_priors: Dict[str, float] = dict(harm_priors or {})
        self._guidance_config: Dict[str, Any] = dict(guidance_config or {})

    async def perform_action_by_llm(self):
        # Build step-scoped RNG and emission decision
        # For lack of direct thread_id here, use agent_id as a stable scope.
        user_id = int(self.social_agent_id)
        thread_scope = f"a_{user_id}"
        override = self._scheduler.get_mode_override(self._persona_cfg) if self._scheduler else None
        # Optional dynamic token weighting context derived from harm priors
        context: Optional[Dict[str, Any]] = None
        try:
            if bool(self._guidance_config.get("token_weighting", False)):
                intensity = float(self._guidance_config.get("intensity", 1.0))
                pri = self._harm_priors or {}
                toxicity = float(pri.get("toxicity", 0.0))
                insult = float(pri.get("insult", 0.0))
                profanity = float(pri.get("profanity", 0.0))
                # Token biases (small, additive; label-agnostic)
                dyn: Dict[str, float] = {}
                # Incel space
                dyn["LBL:HARASSMENT"] = max(0.0, 0.6 * (insult + profanity + toxicity) / 3.0 * intensity)
                dyn["LBL:INCEL_SLANG"] = 0.0
                # Misinfo/conspiracy
                dyn["LBL:MISINFO_CLAIM"] = max(0.0, 0.4 * toxicity * intensity)
                dyn["LBL:MISINFO_SOURCE"] = max(0.0, 0.2 * toxicity * intensity)
                # Recovery/benign unchanged
                context = {"dynamic_token_probs": dyn}
        except Exception:
            context = None
        decision = self._policy.decide(
            user_id=user_id,
            thread_id=thread_scope,
            step_idx=self._step_index,
            persona=self._persona_cfg,
            context=context,
            override_post_mode_probs=override,
        )

        # Record expectation for programmatic fallback
        if self._expect_registry is not None:
            await self._expect_registry.set_expected(user_id, self._step_index, decision.get("tokens", []))

        # Augment user message with a concise step instruction
        style_hint = self._maybe_style_hint(decision)
        step_hint = self._format_step_hint(decision)
        env_prompt = await self.env.to_text_prompt()
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=(
                f"{(style_hint + '\n') if style_hint else ''}{step_hint}\n"
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

    def _maybe_style_hint(self, decision: Dict[str, Any]) -> str:
        """Optionally return a short style hint string for harmful posts."""
        try:
            enabled = bool(self._guidance_config.get("enable", False))
            if not enabled:
                return ""
            mode = decision.get("mode", "none")
            if mode == "none":
                return ""
            # Only for personas that can emit harmful labels
            harmful_allowed = [lab for lab in (self._persona_cfg.allowed_labels or []) if lab not in ("benign", "recovery")]
            if not harmful_allowed:
                return ""
            priors = self._harm_priors or {}
            intensity = float(self._guidance_config.get("intensity", 1.0))
            if intensity <= 0.0:
                return ""
            user_id = int(self.social_agent_id)
            rng = DeterministicRNG(self._run_seed).fork("harm_guidance", f"user:{user_id}", f"step:{self._step_index}")
            toxicity = float(priors.get("toxicity", 0.0))
            insult = float(priors.get("insult", 0.0))
            profanity = float(priors.get("profanity", 0.0))
            identity_attack = float(priors.get("identity_attack", 0.0))
            threat = float(priors.get("threat", 0.0))
            # Overall chance to show a hint
            p_use = max(0.0, min(1.0, 0.2 + 0.6 * toxicity * intensity))
            if not rng.fork("use").bernoulli(p_use):
                return ""
            bits: List[str] = []
            if insult > 0.4 and rng.fork("insult").bernoulli(min(0.5, insult * intensity)):
                bits.append("a more confrontational tone")
                bits.append("slightly insulting phrasing")
            if profanity > 0.4 and rng.fork("profanity").bernoulli(min(0.4, profanity * intensity)):
                bits.append("mild profanity (no slurs)")
            # Include extreme styles when intensity is high
            if intensity >= 0.8 and identity_attack > 0.4 and rng.fork("identity").bernoulli(min(0.3, identity_attack * (intensity - 0.2))):
                bits.append("identity-based denunciation (avoid slurs)")
            if intensity >= 0.9 and threat > 0.3 and rng.fork("threat").bernoulli(min(0.2, threat * (intensity - 0.3))):
                bits.append("veiled threatening language (no illegal incitement)")
            # Always keep safety guard if any confrontational element is suggested
            if bits:
                return f"Style hint: use {', '.join(bits)}; avoid slurs."
            # Fallback generic hint if nothing specific
            return "Style hint: be more confrontational; avoid slurs."
        except Exception:
            return ""

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
            "guidance_enabled": bool(self._guidance_config.get("enable", False)),
        }
        self._sidecar.write(record)
        # Update scheduler with observed labels
        if self._scheduler:
            try:
                self._scheduler.observe(labels)
            except Exception:
                pass


