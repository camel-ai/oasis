from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml


@dataclass(frozen=True)
class LabelSpec:
    """Definition of a controllable label token."""

    id: str
    token: str
    description: str
    guidelines: List[str] = field(default_factory=list)
    dataset_field: Optional[str] = None
    default_frequency: Optional[Dict[str, float]] = None

    @classmethod
    def from_dict(cls, label_id: str, raw: Mapping[str, Any]) -> "LabelSpec":
        if "token" not in raw:
            raise ValueError(f"Label '{label_id}' is missing required 'token'")
        return cls(
            id=label_id,
            token=str(raw["token"]),
            description=str(raw.get("description", "")).strip(),
            guidelines=[str(item) for item in raw.get("guidelines", [])],
            dataset_field=(
                str(raw["dataset_field"])
                if raw.get("dataset_field") is not None
                else None
            ),
            default_frequency=(
                dict(raw["default_frequency"])
                if raw.get("default_frequency") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class LabelEmissionSpec:
    """Controls how a persona emits label tokens."""

    primary: str
    instructions: str
    secondary: List[str] = field(default_factory=list)
    randomness: Optional[Dict[str, float]] = None
    retries: int = 0

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LabelEmissionSpec":
        if "primary" not in raw:
            raise ValueError("persona label emission requires 'primary'")
        instructions = raw.get("instructions", "")
        if not instructions:
            raise ValueError(
                "persona label emission requires human-readable 'instructions'"
            )
        return cls(
            primary=str(raw["primary"]),
            instructions=str(instructions).strip(),
            secondary=[str(val) for val in raw.get("secondary", [])],
            randomness=(
                dict(raw["randomness"]) if raw.get("randomness") else None
            ),
            retries=int(raw.get("retries", 0)),
        )


@dataclass(frozen=True)
class ContentStyleSpec:
    """Captures tone, lexical patterns, and writing dynamics for a persona."""

    tone: str
    register: str
    sentiment: Optional[str] = None
    lexical_required: List[str] = field(default_factory=list)
    lexical_optional: List[str] = field(default_factory=list)
    writing_style: Optional[str] = None
    discourse: Optional[str] = None
    emoji_usage: Optional[str] = None
    dialect: Optional[str] = None
    pacing: Optional[str] = None
    typo_tendency: Optional[str] = None
    quirks: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ContentStyleSpec":
        if "tone" not in raw or "register" not in raw:
            raise ValueError("persona style requires 'tone' and 'register'")
        lexical_raw = raw.get("lexical") or {}
        if not isinstance(lexical_raw, Mapping):
            lexical_raw = {}
        return cls(
            tone=str(raw["tone"]),
            register=str(raw["register"]),
            sentiment=str(raw.get("sentiment")) if raw.get("sentiment") else None,
            lexical_required=[str(val) for val in lexical_raw.get("required", [])],
            lexical_optional=[str(val) for val in lexical_raw.get("optional", [])],
            writing_style=str(raw.get("writing_style")) if raw.get("writing_style") else None,
            discourse=str(raw.get("discourse")) if raw.get("discourse") else None,
            emoji_usage=str(raw.get("emoji_usage")) if raw.get("emoji_usage") else None,
            dialect=str(raw.get("dialect")) if raw.get("dialect") else None,
            pacing=str(raw.get("pacing")) if raw.get("pacing") else None,
            typo_tendency=str(raw.get("typo_rate")) if raw.get("typo_rate") else None,
            quirks=[str(val) for val in raw.get("quirks", [])],
        )


@dataclass(frozen=True)
class ActionBiasSpec:
    """Relative action weights and simulation controls."""

    weights: Dict[str, float]
    cooldown_steps: Optional[int] = None
    burstiness: Optional[str] = None

    def normalized_weights(self) -> Dict[str, float]:
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("action weights must sum to a positive value")
        return {action: weight / total for action, weight in self.weights.items()}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ActionBiasSpec":
        weights = raw.get("weights")
        if not isinstance(weights, Mapping) or not weights:
            raise ValueError("persona action bias requires a non-empty 'weights' map")
        weights_map = {str(k): float(v) for k, v in weights.items()}
        return cls(
            weights=weights_map,
            cooldown_steps=(
                int(raw["cooldown_steps"]) if raw.get("cooldown_steps") is not None else None
            ),
            burstiness=str(raw.get("burstiness")) if raw.get("burstiness") else None,
        )


@dataclass(frozen=True)
class PersonaVariantSpec:
    """Concrete persona variant definition."""

    id: str
    archetype: str
    slug: str
    display_name: str
    summary: str
    description: str
    style: ContentStyleSpec
    topics: List[str]
    prompt_sections: Dict[str, str]
    label_emission: LabelEmissionSpec
    action_bias: ActionBiasSpec
    user_prompts: Dict[str, str] = field(default_factory=dict)
    dataset_tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        archetype_id: str,
        slug: str,
        raw: Mapping[str, Any],
    ) -> "PersonaVariantSpec":
        display_name = raw.get("display_name") or slug.replace("_", " ").title()
        summary = raw.get("summary") or ""
        description = raw.get("description") or summary or display_name
        style = ContentStyleSpec.from_dict(raw.get("style", {}))
        topics = [str(val) for val in raw.get("topics", [])]
        prompt_sections_raw = raw.get("prompt_sections") or {}
        if not prompt_sections_raw:
            raise ValueError(
                f"persona variant '{archetype_id}.{slug}' requires prompt_sections"
            )
        prompt_sections = {
            str(section): str(text).strip()
            for section, text in prompt_sections_raw.items()
        }
        user_prompts_raw = raw.get("user_prompts") or {}
        if not isinstance(user_prompts_raw, Mapping):
            raise TypeError(
                f"persona variant '{archetype_id}.{slug}' expected user_prompts mapping"
            )
        user_prompts = {
            str(key): str(value).strip()
            for key, value in user_prompts_raw.items()
        } if user_prompts_raw else {}
        label_spec = LabelEmissionSpec.from_dict(raw.get("label_emission", {}))
        action_bias = ActionBiasSpec.from_dict(raw.get("action_bias", {}))

        dataset_tags = {
            str(key): str(value) for key, value in raw.get("dataset_tags", {}).items()
        }
        metadata = dict(raw.get("metadata", {}))
        return cls(
            id=f"{archetype_id}.{slug}",
            archetype=archetype_id,
            slug=slug,
            display_name=str(display_name),
            summary=str(summary).strip(),
            description=str(description).strip(),
            style=style,
            topics=topics,
            prompt_sections=prompt_sections,
            user_prompts=user_prompts,
            label_emission=label_spec,
            action_bias=action_bias,
            dataset_tags=dataset_tags,
            metadata=metadata,
        )


@dataclass(frozen=True)
class PersonaArchetypeSpec:
    """Grouping for related persona variants."""

    id: str
    display_name: str
    description: str
    username_prefix: Optional[str]
    variants: Dict[str, PersonaVariantSpec]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, archetype_id: str, raw: Mapping[str, Any]) -> "PersonaArchetypeSpec":
        display_name = raw.get("display_name") or archetype_id.title()
        description = raw.get("description") or ""
        username_prefix = raw.get("username_prefix")
        variants_raw = raw.get("variants") or {}
        if not variants_raw:
            raise ValueError(f"persona archetype '{archetype_id}' has no variants defined")
        variants: Dict[str, PersonaVariantSpec] = {}
        for slug, variant_raw in variants_raw.items():
            variants[slug] = PersonaVariantSpec.from_dict(archetype_id, slug, variant_raw)
        return cls(
            id=archetype_id,
            display_name=str(display_name),
            description=str(description).strip(),
            username_prefix=str(username_prefix) if username_prefix else None,
            variants=variants,
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass(frozen=True)
class PersonaOntology:
    """Top-level ontology definition."""

    version: str
    dataset_schema: Dict[str, Any]
    labels: Dict[str, LabelSpec]
    archetypes: Dict[str, PersonaArchetypeSpec]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def iter_variants(self) -> Iterable[PersonaVariantSpec]:
        for archetype in self.archetypes.values():
            for variant in archetype.variants.values():
                yield variant

    def get_variant(self, variant_id: str) -> PersonaVariantSpec:
        archetype_id, _, slug = variant_id.partition(".")
        archetype = self.archetypes.get(archetype_id)
        if archetype is None:
            raise KeyError(f"unknown persona archetype '{archetype_id}'")
        variant = archetype.variants.get(slug)
        if variant is None:
            raise KeyError(f"unknown persona variant '{variant_id}'")
        return variant

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PersonaOntology":
        version = str(raw.get("version", "0"))
        dataset_schema = dict(raw.get("dataset_schema", {}))

        labels_raw = raw.get("labels") or {}
        labels = {
            label_id: LabelSpec.from_dict(label_id, label_raw)
            for label_id, label_raw in labels_raw.items()
        }
        archetypes_raw = raw.get("archetypes") or {}
        if not archetypes_raw:
            raise ValueError("persona ontology requires at least one archetype")
        archetypes = {
            archetype_id: PersonaArchetypeSpec.from_dict(archetype_id, archetype_raw)
            for archetype_id, archetype_raw in archetypes_raw.items()
        }
        return cls(
            version=version,
            dataset_schema=dataset_schema,
            labels=labels,
            archetypes=archetypes,
            metadata=dict(raw.get("metadata", {})),
        )


def load_ontology(path: str | Path) -> PersonaOntology:
    """Load a PersonaOntology definition from YAML."""

    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"persona ontology file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return PersonaOntology.from_dict(data)
