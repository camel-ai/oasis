from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from yaml import safe_load


@dataclass(frozen=True)
class Manifest:
    data: Dict[str, Any]

    @property
    def run_seed(self) -> int:
        return int(self.data.get("rng_seed", 314159))

    @property
    def post_label_mode_probs(self) -> Dict[str, float]:
        return dict(self.data.get("post_label_mode_probs", {"none": 0.5, "single": 0.4, "double": 0.1}))

    @property
    def multi_label_targets(self) -> Dict[str, float]:
        targets = self.data.get("multi_label_targets", {})
        # Defaults to 20% with 18–22% band
        return {
            "target_rate": float(targets.get("target_rate", 0.20)),
            "min_rate": float(targets.get("min_rate", 0.18)),
            "max_rate": float(targets.get("max_rate", 0.22)),
        }


def load_manifest(path: Path) -> Manifest:
    with path.open("r", encoding="utf-8") as f:
        data = safe_load(f) or {}
    return Manifest(data=data)


