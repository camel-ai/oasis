from __future__ import annotations

from typing import Dict, Iterable, List, Optional

DEFAULT_TOKEN_TO_CATEGORIES: Dict[str, List[str]] = {
    "LBL:INCEL_SLANG": ["incel"],
    "LBL:MISINFO_CLAIM": ["misinfo", "conspiracy"],
    "LBL:SUPPORTIVE": ["recovery", "benign"],
}


def assign_labels(
    emitted_tokens: List[str],
    allowed_labels: Optional[Iterable[str]] = None,
) -> List[str]:
    r"""Map emitted label tokens to final category labels (no confidence).

    Args:
        emitted_tokens: tokens detected in text (e.g., ["LBL:MISINFO_CLAIM"])
        allowed_labels: optional allowed categories for this persona

    Returns:
        sorted unique category_labels
    """
    categories: List[str] = []
    for tok in emitted_tokens:
        categories.extend(DEFAULT_TOKEN_TO_CATEGORIES.get(tok, []))
    if allowed_labels is not None:
        allowed_set = set(allowed_labels)
        categories = [c for c in categories if c in allowed_set]
        if not categories and allowed_set:
            # Fallback to a single allowed category if mapping produced none.
            categories = [sorted(allowed_set)[0]]
    categories = sorted(set(categories))
    return categories


