from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schema import PersonaOntology, PersonaVariantSpec


@dataclass(frozen=True)
class PersonaBuildRequest:
    """Desired number of instances for a persona variant."""

    variant_id: str
    count: int

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("persona build request count must be non-negative")


class PersonaGenerator:
    """Generate persona rows from an ontology definition."""

    def __init__(self, ontology: PersonaOntology, seed: int = 0) -> None:
        self.ontology = ontology
        self.seed = seed
        self._rng = random.Random(seed)

    # --------------------------------------------------------------------- plan
    def expand_spec(
        self,
        spec: Mapping[str, object],
    ) -> List[PersonaBuildRequest]:
        """Convert a high-level spec into concrete build requests."""

        requests: List[PersonaBuildRequest] = []
        for key, value in spec.items():
            if key in {"seed", "personas_csv", "ontology"}:
                continue
            if key in self.ontology.archetypes:
                requests.extend(self._expand_archetype(key, value))
            else:
                requests.append(self._expand_variant_entry(key, value))
        return [req for req in requests if req.count > 0]

    def _expand_variant_entry(self, key: str, value: object) -> PersonaBuildRequest:
        if "." not in key:
            raise ValueError(
                f"Unknown persona key '{key}'. Use '<archetype>' or 'archetype.variant'."
            )
        variant = self.ontology.get_variant(key)
        count = self._coerce_count(value, allow_zero=True)
        return PersonaBuildRequest(variant_id=variant.id, count=count)

    def _expand_archetype(
        self,
        archetype_id: str,
        raw_value: object,
    ) -> List[PersonaBuildRequest]:
        archetype = self.ontology.archetypes[archetype_id]

        if isinstance(raw_value, Mapping):
            variants_spec = dict(raw_value.get("variants") or {})
            # Allow shorthand where variant names are provided directly at this level.
            for key, val in raw_value.items():
                if key in archetype.variants and key not in variants_spec:
                    variants_spec[key] = val
            total = raw_value.get("total")
            if total is None:
                total = raw_value.get("count")
            if total is None:
                # If no total supplied, infer from explicit counts if possible.
                if variants_spec and all(
                    self._is_count_like(val) for val in variants_spec.values()
                ):
                    total = sum(int(val) for val in variants_spec.values())
                else:
                    raise ValueError(
                        f"Persona archetype '{archetype_id}' requires 'total' or explicit counts."
                    )
            total = int(total)
        else:
            variants_spec = {}
            total = self._coerce_count(raw_value, allow_zero=True)

        if total <= 0:
            return []

        if not variants_spec:
            counts = self._distribute_uniform(total, len(archetype.variants))
            variant_ids = list(archetype.variants.keys())
            return [
                PersonaBuildRequest(
                    variant_id=archetype.variants[variant_ids[i]].id, count=count
                )
                for i, count in enumerate(counts)
                if count > 0
            ]

        if all(self._is_count_like(val) for val in variants_spec.values()):
            counts = {
                slug: int(self._coerce_count(val))
                for slug, val in variants_spec.items()
            }
            residual = total - sum(counts.values())
            if residual < 0:
                raise ValueError(
                    f"Persona counts for '{archetype_id}' exceed requested total."
                )
            if residual > 0:
                spreads = self._distribute_uniform(residual, len(counts))
                for add, slug in zip(spreads, counts.keys()):
                    counts[slug] += add
        else:
            # Treat inputs as weights.
            weights = {
                slug: float(val) for slug, val in variants_spec.items()
            }
            counts = self._distribute_with_weights(total, weights)

        requests: List[PersonaBuildRequest] = []
        for slug, count in counts.items():
            if slug not in archetype.variants:
                raise KeyError(
                    f"Persona archetype '{archetype_id}' has no variant '{slug}'"
                )
            if count <= 0:
                continue
            requests.append(
                PersonaBuildRequest(
                    variant_id=archetype.variants[slug].id,
                    count=count,
                )
            )
        return requests

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _is_count_like(value: object) -> bool:
        return isinstance(value, int) or (
            isinstance(value, float) and float(value).is_integer()
        )

    @staticmethod
    def _coerce_count(value: object, allow_zero: bool = False) -> int:
        if isinstance(value, bool):
            raise TypeError("Boolean is not a valid count")
        if isinstance(value, (int, float)):
            count = int(value)
        elif isinstance(value, Mapping) and "count" in value:
            count = int(value["count"])
        elif isinstance(value, Mapping) and "total" in value:
            count = int(value["total"])
        else:
            raise TypeError(f"Cannot interpret count from value: {value!r}")
        if count < 0 or (count == 0 and not allow_zero):
            raise ValueError("Persona count must be positive")
        return count

    @staticmethod
    def _distribute_uniform(total: int, buckets: int) -> List[int]:
        if buckets <= 0:
            raise ValueError("Bucket count must be positive")
        base, remainder = divmod(total, buckets)
        counts = [base] * buckets
        for idx in range(remainder):
            counts[idx] += 1
        return counts

    @staticmethod
    def _distribute_with_weights(
        total: int,
        weights: Mapping[str, float],
    ) -> Dict[str, int]:
        positive = {k: v for k, v in weights.items() if v > 0}
        if not positive:
            raise ValueError("At least one positive weight is required")
        weight_sum = sum(positive.values())
        allocations: Dict[str, int] = {}
        fractions: List[Tuple[str, float]] = []
        running_total = 0
        for key, weight in positive.items():
            exact = total * weight / weight_sum
            count = int(exact)
            allocations[key] = count
            running_total += count
            fractions.append((key, exact - count))
        residual = total - running_total
        fractions.sort(key=lambda item: item[1], reverse=True)
        for idx in range(residual):
            key = fractions[idx % len(fractions)][0]
            allocations[key] += 1
        return allocations

    # ---------------------------------------------------------------- generation
    def generate(self, requests: Sequence[PersonaBuildRequest]) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        local_counters: Dict[str, int] = {}
        for req in requests:
            variant = self.ontology.get_variant(req.variant_id)
            for _ in range(req.count):
                local_index = local_counters.get(variant.id, 0)
                local_counters[variant.id] = local_index + 1
                username = self._build_username(variant, local_index)
                row = self._build_row(variant, username)
                rows.append(row)

        self._rng.shuffle(rows)
        return rows

    def _build_username(self, variant: PersonaVariantSpec, index: int) -> str:
        archetype = self.ontology.archetypes[variant.archetype]
        prefix = archetype.username_prefix or variant.slug
        digest = hashlib.sha1(
            f"{prefix}:{variant.id}:{self.seed}:{index}".encode("utf-8")
        ).hexdigest()[:6]
        return f"{prefix}_{index:04d}_{digest}"

    def _build_row(
        self,
        variant: PersonaVariantSpec,
        username: str,
    ) -> Dict[str, str]:
        prompt = self._format_prompt(variant.prompt_sections)
        archetype = self.ontology.archetypes[variant.archetype]
        style = variant.style

        metadata: Dict[str, str] = {
            "persona_group": variant.archetype,
            "persona_variant": variant.slug,
            "persona_summary": variant.summary,
            "persona_topics": ";".join(variant.topics),
            "persona_tone": style.tone,
            "persona_register": style.register,
            "persona_sentiment": style.sentiment or "",
            "persona_lexical_required": ";".join(style.lexical_required),
            "persona_lexical_optional": ";".join(style.lexical_optional),
            "persona_writing_style": style.writing_style or "",
            "persona_discourse": style.discourse or "",
            "persona_style_emoji_usage": style.emoji_usage or "",
            "persona_style_dialect": style.dialect or "",
            "persona_style_pacing": style.pacing or "",
            "persona_style_typo": style.typo_tendency or "",
            "persona_style_quirks": ";".join(style.quirks),
            "label_primary": variant.label_emission.primary,
            "label_secondary": ";".join(variant.label_emission.secondary),
            "label_instructions_text": variant.label_emission.instructions,
            "label_randomness": json.dumps(variant.label_emission.randomness or {}),
            "action_bias": json.dumps(variant.action_bias.normalized_weights()),
            "persona_username_prefix": archetype.username_prefix or "",
        }
        user_prompts = variant.user_prompts or {}
        metadata["persona_prompt_post"] = user_prompts.get("post", "")
        metadata["persona_prompt_comment"] = user_prompts.get("comment", "")
        metadata["persona_prompt_reflection"] = user_prompts.get("reflection", "")

        goal = variant.metadata.get("goal")
        if goal:
            metadata.setdefault("persona_goal", str(goal))
        traits = variant.metadata.get("persona_traits")
        if traits:
            metadata.setdefault("persona_personality", str(traits))
        fallback_post = variant.metadata.get("fallback_post")
        if fallback_post:
            metadata.setdefault("persona_fallback_post", str(fallback_post))
        fallback_comment = variant.metadata.get("fallback_comment")
        if fallback_comment:
            metadata.setdefault("persona_fallback_comment", str(fallback_comment))

        # Optional metadata passthrough.
        for key, value in variant.dataset_tags.items():
            metadata.setdefault(key, value)
        for key, value in variant.metadata.items():
            metadata.setdefault(f"variant_{key}", str(value))
        for key, value in archetype.metadata.items():
            metadata.setdefault(f"archetype_{key}", str(value))

        row: Dict[str, str] = {
            "username": username,
            "name": username,
            "description": variant.description,
            "user_char": prompt,
        }
        row.update(metadata)
        return {key: value for key, value in row.items() if value != ""}

    @staticmethod
    def _format_prompt(sections: Mapping[str, str]) -> str:
        order = ["identity", "style", "topics", "label_instructions"]
        blocks: List[str] = []
        for key in order:
            block = sections.get(key)
            if block:
                blocks.append(str(block).strip())
        for key, block in sections.items():
            if key not in order and block:
                blocks.append(str(block).strip())
        return "\n\n".join(blocks)


def build_requests_from_spec(
    generator: PersonaGenerator,
    spec: Mapping[str, object],
) -> List[PersonaBuildRequest]:
    """Convenience wrapper used by scripts/tests."""

    return generator.expand_spec(spec)
