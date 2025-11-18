from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from orchestrator.rng import DeterministicRNG

DEFAULT_LABEL_TO_TOKENS: Dict[str, List[str]] = {
    # Minimal default token inventory (extendable later).
    "incel": ["LBL:INCEL_SLANG", "LBL:HARASSMENT"],
    "misinfo": ["LBL:MISINFO_CLAIM", "LBL:MISINFO_SOURCE"],
    "conspiracy": ["LBL:MISINFO_CLAIM", "LBL:MISINFO_SOURCE"],
    "recovery": ["LBL:SUPPORTIVE"],
    "benign": ["LBL:SUPPORTIVE"],
}


@dataclass(frozen=True)
class PersonaConfig:
    persona_id: str
    primary_label: str
    allowed_labels: List[str]
    label_mode_cap: str  # "single" | "double"
    benign_on_none_prob: float = 0.6
    max_labels_per_post: int = 2
    emission_probs: Optional[Dict[str, float]] = None  # token -> prob
    pair_probs: Optional[Dict[str, float]] = None      # "L_i↔L_j" -> prob


class EmissionPolicy:
    r"""Compute deterministic per-step emission decisions (none/single/double).

    Decision is driven by:
    - Global post_label_mode_probs (none/single/double)
    - Persona caps and preferences (allowed_labels, pair_probs, emission_probs)
    - A DeterministicRNG keyed by (run_seed, user_id, thread_id, step_idx)
    """

    def __init__(
        self,
        run_seed: int,
        post_label_mode_probs: Dict[str, float],
        label_to_tokens: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self._run_seed = int(run_seed)
        self._post_mode = dict(post_label_mode_probs or {})
        self._label_to_tokens = dict(label_to_tokens or DEFAULT_LABEL_TO_TOKENS)

    def decide(
        self,
        user_id: int,
        thread_id: str | int,
        step_idx: int,
        persona: PersonaConfig,
        context: Optional[Dict] = None,
        override_post_mode_probs: Optional[Dict[str, float]] = None,
    ) -> Dict:
        rng_root = DeterministicRNG(self._run_seed).fork(
            f"user:{user_id}", f"thread:{thread_id}", f"step:{step_idx}"
        )
        mode = self._sample_mode(rng_root, persona, override_post_mode_probs)
        if mode == "none":
            return {"mode": "none", "tokens": []}

        # Determine labels for single/double and select tokens.
        if mode == "single":
            label = self._sample_single_label(rng_root, persona)
            token = self._sample_token_for_label(rng_root, persona, label)
            return {"mode": "single", "tokens": [token]}
        else:
            l1, l2 = self._sample_label_pair(rng_root, persona)
            t1 = self._sample_token_for_label(rng_root.fork("l1"), persona, l1)
            t2 = self._sample_token_for_label(rng_root.fork("l2"), persona, l2)
            # Respect max cap even if configured to 1.
            tokens = [t1]
            if persona.max_labels_per_post >= 2 and l2 != l1:
                tokens.append(t2)
            return {"mode": "double" if len(tokens) == 2 else "single", "tokens": tokens}

    def _sample_mode(self, rng: DeterministicRNG, persona: PersonaConfig, override: Optional[Dict[str, float]]) -> str:
        probs = override if override else self._post_mode
        mode = rng.fork("mode").categorical(probs)
        if persona.label_mode_cap == "single" and mode == "double":
            return "single"
        if persona.label_mode_cap == "double":
            return mode
        # default guard
        return "single" if mode not in ("none", "double") else mode

    def _sample_single_label(self, rng: DeterministicRNG, persona: PersonaConfig) -> str:
        # Uniform over allowed_labels as a simple default policy.
        if not persona.allowed_labels:
            return persona.primary_label
        probs = {lab: 1.0 for lab in persona.allowed_labels}
        return rng.fork("single_label").categorical(probs)

    def _sample_label_pair(self, rng: DeterministicRNG, persona: PersonaConfig) -> Tuple[str, str]:
        allowed = persona.allowed_labels or [persona.primary_label]
        if len(allowed) == 1:
            return allowed[0], allowed[0]
        # If pair_probs exists, use it; otherwise sample uniform distinct pair.
        if persona.pair_probs:
            pair = rng.fork("pair").categorical(persona.pair_probs)
            if "↔" in pair:
                a, b = pair.split("↔", 1)
                if a in allowed and b in allowed:
                    return a, b
        # Fallback uniform distinct
        a = rng.fork("pair_a").categorical({lab: 1.0 for lab in allowed})
        remaining = [lab for lab in allowed if lab != a] or [a]
        b = rng.fork("pair_b").categorical({lab: 1.0 for lab in remaining})
        return a, b

    def _sample_token_for_label(self, rng: DeterministicRNG, persona: PersonaConfig, label: str) -> str:
        candidates = self._label_to_tokens.get(label, [])
        if persona.emission_probs:
            # persona.emission_probs defined over tokens (e.g., "LBL:MISINFO_CLAIM")
            filtered = {tok: w for tok, w in persona.emission_probs.items() if tok in candidates}
            if filtered:
                return rng.fork("token_probs").categorical(filtered)
        if candidates:
            return rng.fork("token_uniform").categorical({t: 1.0 for t in candidates})
        # Default: token name mirrors the class label, e.g., incel -> LBL:INCEL
        return f"LBL:{label.upper()}"


