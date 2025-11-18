from __future__ import annotations

import hashlib
import math
import random
from typing import Dict, Iterable, List, Tuple


class DeterministicRNG:
    r"""Deterministic RNG with hierarchical forking keyed by strings.

    This class avoids global state and produces stable pseudo-random
    draws for a given seed and key path. All public draws are pure and
    reproducible given the same `(seed, key)` inputs.
    """

    def __init__(self, seed: int, key_path: Tuple[str, ...] | None = None) -> None:
        self._seed: int = int(seed)
        self._key_path: Tuple[str, ...] = tuple(key_path or ())

    def fork(self, *keys: str) -> "DeterministicRNG":
        r"""Create a child RNG with the current seed and an extended key path."""
        return DeterministicRNG(self._seed, self._key_path + tuple(keys))

    def _derive_int(self) -> int:
        r"""Derive a stable 32-bit integer from (seed, key_path)."""
        h = hashlib.sha256()
        h.update(str(self._seed).encode("utf-8"))
        for k in self._key_path:
            h.update(b":")
            h.update(k.encode("utf-8"))
        # Use lower 32 bits
        return int.from_bytes(h.digest()[:4], "big", signed=False)

    def _python_random(self) -> random.Random:
        r"""Create a Python Random instance seeded from the derived int."""
        return random.Random(self._derive_int())

    def bernoulli(self, p: float) -> bool:
        r"""Return True with probability p."""
        pr = self._python_random()
        return pr.random() < max(0.0, min(1.0, p))

    def categorical(self, probs: Dict[str, float]) -> str:
        r"""Sample one key from an unnormalized probability dict."""
        items = list(probs.items())
        weights = [max(0.0, float(w)) for _, w in items]
        total = sum(weights)
        if total <= 0.0:
            # deterministic fallback to smallest key for stability
            return sorted(probs.keys())[0]
        pr = self._python_random()
        r = pr.random() * total
        acc = 0.0
        for key, w in items:
            acc += w
            if r <= acc:
                return key
        return items[-1][0]


