from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from generation.emission_policy import PersonaConfig


@dataclass
class MultiLabelTargets:
    r"""Target band for harmful multi-label posts."""
    target_rate: float = 0.20
    min_rate: float = 0.18
    max_rate: float = 0.22


class MultiLabelScheduler:
    r"""Runtime controller to steer multi-label frequency into a target band.

    Tracks observed harmful posts and harmful multi-label posts and returns
    per-step overrides to the post_label_mode_probs to nudge the rate.
    """

    def __init__(self, targets: MultiLabelTargets) -> None:
        self.targets = targets
        self.harmful_total: int = 0
        self.harmful_ml: int = 0

    def get_mode_override(self, persona: PersonaConfig) -> Optional[Dict[str, float]]:
        # Only override for personas that can emit harmful classes (i.e., not benign-only)
        harmful_allowed = [lab for lab in persona.allowed_labels if lab not in ("benign", "recovery")]
        if not harmful_allowed:
            return None
        # Compute current rate
        rate = (self.harmful_ml / self.harmful_total) if self.harmful_total > 0 else None
        # Default neutral override (no change)
        if rate is None:
            return None
        # If rate below band, increase 'double' weight; if above, decrease
        base = {"none": 0.5, "single": 0.4, "double": 0.1}
        if rate < self.targets.min_rate:
            # nudge up double
            base["double"] = min(0.3, base["double"] + 0.1)
            base["single"] = max(0.2, base["single"] - 0.05)
            base["none"] = max(0.3, base["none"] - 0.05)
        elif rate > self.targets.max_rate:
            # nudge down double
            base["double"] = max(0.05, base["double"] - 0.05)
            base["single"] = min(0.5, base["single"] + 0.05)
            base["none"] = min(0.6, base["none"] + 0.0)
        # Normalize
        s = sum(base.values()) or 1.0
        return {k: v / s for k, v in base.items()}

    def observe(self, category_labels: Sequence[str]) -> None:
        """Update counters based on final labels of a content item."""
        if not category_labels:
            return
        labs = set(category_labels)
        # Harmful if any label not benign
        is_harmful = any(l for l in labs if l != "benign")
        if not is_harmful:
            return
        self.harmful_total += 1
        if len(labs - {"benign"}) >= 2:
            self.harmful_ml += 1


