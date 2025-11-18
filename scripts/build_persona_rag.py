#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAG_DIR = DATA_DIR / "rag_corpus"

LABEL_TOKEN_MAP = {
    "benign": "<LBL:BENIGN>",
    "recovery_support": "<LBL:SUPPORTIVE>",
    "eating_disorder_risk": "<LBL:ED_RISK>",
    "incel_misogyny": "<LBL:INCEL_SLANG>",
    "misinformation": "<LBL:MISINFO_CLAIM>",
    "conspiracy": "<LBL:CONSPIRACY>",
}
HARMFUL_LABELS = {
    "eating_disorder_risk",
    "incel_misogyny",
    "misinformation",
    "conspiracy",
}


@dataclass(frozen=True)
class LabelDefinition:
    name: str
    description: str
    goal_template: str
    lexical_required: List[str]
    lexical_optional: List[str]
    style_quirks: List[str]
    fallback_templates: List[str]
    intensity: str
    tone: str
    is_harmful: bool


LABEL_DEFINITIONS: Dict[str, LabelDefinition] = {
    "benign": LabelDefinition(
        name="benign",
        description="sharing slice-of-life observations, local updates, and practical advice",
        goal_template="Share grounded daily-life observation",
        lexical_required=["neighborly", "routine", "practical"],
        lexical_optional=["cozy", "midweek", "commute", "coffee run"],
        style_quirks=["talks like a detail-oriented neighbor", "mentions time-of-day"],
        fallback_templates=["Observing the platform today", "Quick check-in from my block"],
        intensity="low",
        tone="neutral-positive",
        is_harmful=False,
    ),
    "recovery_support": LabelDefinition(
        name="recovery_support",
        description="modeling peer-support, validation, and recovery motivation",
        goal_template="Offer grounded recovery encouragement",
        lexical_required=["recovery", "check-in", "holding space"],
        lexical_optional=["hotline", "therapist tip", "daily coping"],
        style_quirks=["uses gentle validation", "names specific coping tactics"],
        fallback_templates=["Sending steadiness to anyone pushing through", "Sharing a grounding exercise"],
        intensity="low",
        tone="warm",
        is_harmful=False,
    ),
    "eating_disorder_risk": LabelDefinition(
        name="eating_disorder_risk",
        description="obsessing over weight control rituals and caloric tactics",
        goal_template="Normalise restrictive rituals",
        lexical_required=["calorie cap", "fasting window", "scale drop"],
        lexical_optional=["liquid dinner", "mirror check", "bodycheck"],
        style_quirks=["matter-of-fact about restriction", "quotes numbers precisely"],
        fallback_templates=["Locking into another fast", "Clocking macros tighter"],
        intensity="high",
        tone="clinical-cold",
        is_harmful=True,
    ),
    "incel_misogyny": LabelDefinition(
        name="incel_misogyny",
        description="resentful toward women and social hierarchies",
        goal_template="Vent status resentment",
        lexical_required=["beta", "smv", "stacy"],
        lexical_optional=["hypergamy", "looksmax", "blackpill"],
        style_quirks=["mixes slang with pseudo-logic", "calls out social scoreboards"],
        fallback_templates=["Watching the same rigged dating market", "Logging another reminder that status rules"],
        intensity="high",
        tone="aggrieved",
        is_harmful=True,
    ),
    "misinformation": LabelDefinition(
        name="misinformation",
        description="spreading pseudo-scientific or distorted claims",
        goal_template="Seed distrust in mainstream sources",
        lexical_required=["wake up", "evidence leak", "shadow report"],
        lexical_optional=["forbidden study", "buried data", "ratio proof"],
        style_quirks=["cites dubious stats", "questions experts"],
        fallback_templates=["Dropping another buried stat", "Connecting dots nobody wants posted"],
        intensity="medium",
        tone="alarmist",
        is_harmful=True,
    ),
    "conspiracy": LabelDefinition(
        name="conspiracy",
        description="connecting unrelated facts into hidden-plot narratives",
        goal_template="Expose coordinated cover-ups",
        lexical_required=["deep state", "puppet string", "off-the-books"],
        lexical_optional=["blacksite", "psyop", "redacted"],
        style_quirks=["references map walls", "names coded hints"],
        fallback_templates=["Charting fresh lines on the wall", "Yet another psyop thread"],
        intensity="medium",
        tone="ominous",
        is_harmful=True,
    ),
}

SINGLE_LABEL_PRIMARY_COUNTS = {
    "benign": 30,
    "recovery_support": 20,
    "eating_disorder_risk": 8,
    "incel_misogyny": 6,
    "misinformation": 4,
    "conspiracy": 2,
}
MULTI_LABEL_PRIMARY_COUNTS = {
    ("incel_misogyny", "misinformation"): 8,
    ("incel_misogyny", "conspiracy"): 5,
    ("misinformation", "conspiracy"): 5,
    ("eating_disorder_risk", "misinformation"): 4,
    ("eating_disorder_risk", "conspiracy"): 4,
    ("incel_misogyny", "eating_disorder_risk"): 4,
}
BACKUP_COMBOS = [
    ("benign",),
    ("recovery_support",),
    ("eating_disorder_risk",),
    ("incel_misogyny",),
    ("misinformation",),
    ("conspiracy",),
    ("incel_misogyny", "misinformation"),
    ("misinformation", "conspiracy"),
    ("recovery_support", "benign"),
]


@dataclass
class PersonaSeed:
    seed_id: int
    description: str
    chat_lines: List[str]
    keywords: List[str]
    cluster_id: int = -1
    cluster_keywords: List[str] = field(default_factory=list)

    @property
    def text_blob(self) -> str:
        joined_chat = " ".join(self.chat_lines)
        return f"{self.description} {joined_chat}".strip()

    def short_slug(self) -> str:
        slug_src = re.sub(r"[^a-z0-9\s]", "", self.description.lower())
        slug = "_".join(part for part in slug_src.split() if part)[:24]
        return slug or f"seed{self.seed_id:03d}"


@dataclass
class LabelAssignment:
    labels: List[str]
    status: str  # "primary" or "backup"

    @property
    def is_multilabel(self) -> bool:
        return len(self.labels) > 1


@dataclass
class PersonaRecord:
    persona_id: str
    persona_variant: str
    persona_name: str
    user_char: str
    persona_prompt_post: str
    persona_prompt_comment: str
    persona_goal: str
    variant_goal: str
    persona_personality: str
    variant_persona_traits: str
    persona_style_quirks: str
    persona_lexical_required: str
    persona_lexical_optional: str
    persona_fallback_post: str
    persona_fallback_comment: str
    persona_topic_focus: str
    persona_cluster_id: int
    persona_cluster_keywords: str
    allowed_labels: str
    allowed_label_tokens: str
    is_primary: bool
    is_multilabel: bool
    label_intensity: str
    label_tone: str
    writing_seed_id: int
    writing_seed_slug: str
    rag_persona_excerpt: str
    rag_chat_excerpt: str

    def to_row(self) -> Dict[str, object]:
        return {
            "persona_id": self.persona_id,
            "persona_variant": self.persona_variant,
            "persona_name": self.persona_name,
            "user_char": self.user_char,
            "persona_prompt_post": self.persona_prompt_post,
            "persona_prompt_comment": self.persona_prompt_comment,
            "persona_goal": self.persona_goal,
            "variant_goal": self.variant_goal,
            "persona_personality": self.persona_personality,
            "variant_persona_traits": self.variant_persona_traits,
            "persona_style_quirks": self.persona_style_quirks,
            "persona_lexical_required": self.persona_lexical_required,
            "persona_lexical_optional": self.persona_lexical_optional,
            "persona_fallback_post": self.persona_fallback_post,
            "persona_fallback_comment": self.persona_fallback_comment,
            "persona_topic_focus": self.persona_topic_focus,
            "persona_cluster_id": self.persona_cluster_id,
            "persona_cluster_keywords": self.persona_cluster_keywords,
            "allowed_labels": self.allowed_labels,
            "allowed_label_tokens": self.allowed_label_tokens,
            "is_primary": int(self.is_primary),
            "is_multilabel": int(self.is_multilabel),
            "label_intensity": self.label_intensity,
            "label_tone": self.label_tone,
            "writing_seed_id": self.writing_seed_id,
            "writing_seed_slug": self.writing_seed_slug,
            "rag_persona_excerpt": self.rag_persona_excerpt,
            "rag_chat_excerpt": self.rag_chat_excerpt,
        }


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    tokens = [token for token in tokenize(text) if len(token) > 3]
    if not tokens:
        return []
    counts = Counter(tokens)
    return [word for word, _ in counts.most_common(top_n)]


def load_personality(path: Path) -> List[PersonaSeed]:
    if not path.exists():
        raise FileNotFoundError(f"Missing personality CSV at {path}")
    df = pd.read_csv(path)
    seeds: List[PersonaSeed] = []
    for seed_id, row in df.iterrows():
        description = clean_text(row.get("Persona", ""))
        chat_blob = str(row.get("chat", ""))
        chat_lines = [clean_text(line) for line in chat_blob.splitlines() if clean_text(line)]
        keywords = extract_keywords(f"{description} {' '.join(chat_lines)}", top_n=6)
        if not description and not chat_lines:
            continue
        seeds.append(
            PersonaSeed(
                seed_id=seed_id,
                description=description or "Observant community member",
                chat_lines=chat_lines[:5],
                keywords=keywords or ["community", "daily"],
            )
        )
    if not seeds:
        raise ValueError("No persona seeds loaded from personality.csv")
    return seeds


def cluster_seeds(seeds: List[PersonaSeed], n_clusters: int) -> Dict[int, List[str]]:
    docs = [seed.text_blob for seed in seeds]
    tfidf = TfidfVectorizer(max_features=1024, ngram_range=(1, 2))
    matrix = tfidf.fit_transform(docs)
    k = max(1, min(n_clusters, len(seeds)))
    if k == 1:
        for seed in seeds:
            seed.cluster_id = 0
            seed.cluster_keywords = extract_keywords(seed.text_blob, top_n=6)
        return {0: seed.cluster_keywords for seed in seeds[:1]}
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=min(64, len(seeds)))
    labels = kmeans.fit_predict(matrix)
    feature_names = tfidf.get_feature_names_out()
    cluster_keywords: Dict[int, List[str]] = {}
    for cluster_id in range(k):
        mask = labels == cluster_id
        if not np.any(mask):
            cluster_keywords[cluster_id] = []
            continue
        cluster_matrix = matrix[mask]
        centroid = np.asarray(cluster_matrix.mean(axis=0)).ravel()
        top_idx = centroid.argsort()[-8:][::-1]
        top_terms = [feature_names[i] for i in top_idx if centroid[i] > 0]
        cluster_keywords[cluster_id] = top_terms[:6]
    for seed, cluster_id in zip(seeds, labels):
        seed.cluster_id = int(cluster_id)
        seed.cluster_keywords = cluster_keywords.get(int(cluster_id), [])
    return cluster_keywords


def build_label_queue(primary_count: int, total_count: int, rng: random.Random) -> List[LabelAssignment]:
    assignments: List[LabelAssignment] = []
    single_total = sum(SINGLE_LABEL_PRIMARY_COUNTS.values())
    multi_total = sum(MULTI_LABEL_PRIMARY_COUNTS.values())
    if primary_count != single_total + multi_total:
        raise ValueError(
            "Primary count mismatch: update SINGLE_LABEL_PRIMARY_COUNTS or MULTI_LABEL_PRIMARY_COUNTS"
        )
    for label, count in SINGLE_LABEL_PRIMARY_COUNTS.items():
        assignments.extend(LabelAssignment(labels=[label], status="primary") for _ in range(count))
    for combo, count in MULTI_LABEL_PRIMARY_COUNTS.items():
        assignments.extend(LabelAssignment(labels=list(combo), status="primary") for _ in range(count))
    rng.shuffle(assignments)
    backup_total = total_count - primary_count
    backup_assignments = []
    for i in range(backup_total):
        combo = BACKUP_COMBOS[i % len(BACKUP_COMBOS)]
        backup_assignments.append(
            LabelAssignment(labels=list(combo), status="backup")
        )
    rng.shuffle(backup_assignments)
    return assignments + backup_assignments


def stitch_prompt(seed: PersonaSeed, labels: List[str], rag_hint: str) -> Dict[str, str]:
    label_descriptions = [LABEL_DEFINITIONS[label].description for label in labels]
    tone_descriptions = [LABEL_DEFINITIONS[label].tone for label in labels]
    goals = [LABEL_DEFINITIONS[label].goal_template for label in labels]
    style_notes = list({note for label in labels for note in LABEL_DEFINITIONS[label].style_quirks})
    lexical_required = list({term for label in labels for term in LABEL_DEFINITIONS[label].lexical_required})
    lexical_optional = list({term for label in labels for term in LABEL_DEFINITIONS[label].lexical_optional})

    persona_personality = (
        f"{seed.description} who writes about {', '.join(seed.cluster_keywords[:3] or seed.keywords[:3])}."
    )
    if rag_hint:
        persona_personality += f" Anchors posts to references like '{rag_hint}'."
    persona_goal = ", and ".join(goals)
    style_quirks = "; ".join((style_notes + seed.keywords)[:6])
    lexical_required_str = ";".join((lexical_required + seed.keywords[:2])[:6])
    lexical_optional_str = ";".join((lexical_optional + seed.cluster_keywords[:2])[:6])

    system_prompt = (
        "You are a persona with layered motivations. Maintain internal consistency, weave in specific details, "
        "and show how your goals influence your tone."
    )
    persona_prompt_post = (
        f"Write as someone {', '.join(label_descriptions)}. Keep references to {seed.cluster_keywords[:2] or seed.keywords[:2]}."
    )
    persona_prompt_comment = (
        "Reply with grounded specifics and vary sentence length. Maintain the same worldview in replies."
    )
    return {
        "system_prompt": system_prompt,
        "persona_goal": persona_goal,
        "persona_personality": persona_personality,
        "style_quirks": style_quirks,
        "lexical_required": lexical_required_str,
        "lexical_optional": lexical_optional_str,
        "post_prompt": persona_prompt_post,
        "comment_prompt": persona_prompt_comment,
        "tone": ",".join(sorted(set(tone_descriptions))),
    }


def build_persona_record(
    persona_index: int,
    seed: PersonaSeed,
    assignment: LabelAssignment,
    rng: random.Random,
) -> PersonaRecord:
    labels = assignment.labels
    label_tokens = [LABEL_TOKEN_MAP[label] for label in labels]
    combined_slug = "_".join(label.split("_")[0] for label in labels)
    persona_variant = f"{combined_slug}_{persona_index:04d}_{seed.short_slug()}"
    persona_id = f"persona_{persona_index:04d}"
    rag_hint = seed.chat_lines[0] if seed.chat_lines else seed.description
    prompt_bits = stitch_prompt(seed, labels, rag_hint)
    intensity = max(LABEL_DEFINITIONS[label].intensity for label in labels)
    tone = prompt_bits["tone"]
    fallback_stub = random.choice(LABEL_DEFINITIONS[labels[0]].fallback_templates)
    fallback_text = f"{fallback_stub} {' '.join(label_tokens)}"
    persona_name = f"{labels[0].replace('_', ' ').title()} voice {persona_index:04d}"
    return PersonaRecord(
        persona_id=persona_id,
        persona_variant=persona_variant,
        persona_name=persona_name,
        user_char=prompt_bits["system_prompt"],
        persona_prompt_post=prompt_bits["post_prompt"],
        persona_prompt_comment=prompt_bits["comment_prompt"],
        persona_goal=prompt_bits["persona_goal"],
        variant_goal=prompt_bits["persona_goal"],
        persona_personality=prompt_bits["persona_personality"],
        variant_persona_traits=prompt_bits["persona_personality"],
        persona_style_quirks=prompt_bits["style_quirks"],
        persona_lexical_required=prompt_bits["lexical_required"],
        persona_lexical_optional=prompt_bits["lexical_optional"],
        persona_fallback_post=fallback_text,
        persona_fallback_comment=fallback_text,
        persona_topic_focus=", ".join(seed.cluster_keywords[:4] or seed.keywords[:4]),
        persona_cluster_id=seed.cluster_id,
        persona_cluster_keywords=";".join(seed.cluster_keywords[:6]),
        allowed_labels=";".join(labels),
        allowed_label_tokens=";".join(label_tokens),
        is_primary=assignment.status == "primary",
        is_multilabel=len(labels) > 1,
        label_intensity=intensity,
        label_tone=tone,
        writing_seed_id=seed.seed_id,
        writing_seed_slug=seed.short_slug(),
        rag_persona_excerpt=seed.description,
        rag_chat_excerpt=" | ".join(seed.chat_lines[:3]),
    )


def generate_personas(
    seeds: List[PersonaSeed],
    assignments: List[LabelAssignment],
    rng: random.Random,
) -> List[PersonaRecord]:
    ordered_seeds = seeds.copy()
    rng.shuffle(ordered_seeds)
    seed_cycle = cycle(ordered_seeds)
    records: List[PersonaRecord] = []
    for idx, assignment in enumerate(assignments, start=1):
        seed = next(seed_cycle)
        records.append(build_persona_record(idx, seed, assignment, rng))
    return records


def build_rag_entries(records: List[PersonaRecord]) -> List[Dict[str, object]]:
    rag_entries: List[Dict[str, object]] = []
    for record in records:
        for snippet, tag in [
            (record.rag_persona_excerpt, "persona_desc"),
            (record.rag_chat_excerpt, "persona_chat"),
        ]:
            if not snippet:
                continue
            rag_entries.append(
                {
                    "persona_variant": record.persona_variant,
                    "text": snippet,
                    "source": "personality_seed",
                    "tags": [tag, "primary" if record.is_primary else "backup"],
                    "cluster_id": record.persona_cluster_id,
                    "allowed_labels": record.allowed_labels,
                }
            )
    return rag_entries


def write_jsonl(entries: Iterable[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def write_summary(records: List[PersonaRecord], output_path: Path) -> None:
    summary = {
        "total": len(records),
        "by_label": Counter(
            label for record in records for label in record.allowed_labels.split(";")
        ),
        "primary": sum(1 for record in records if record.is_primary),
        "backup": sum(1 for record in records if not record.is_primary),
        "multilabel_primary": sum(1 for record in records if record.is_primary and record.is_multilabel),
        "multilabel_backup": sum(1 for record in records if not record.is_primary and record.is_multilabel),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate enriched personas and RAG corpora.")
    parser.add_argument("--personality", type=str, default=str(DATA_DIR / "personality.csv"))
    parser.add_argument(
        "--personas-output",
        type=str,
        default=str(DATA_DIR / "personas_generated_full.csv"),
    )
    parser.add_argument(
        "--primary-output",
        type=str,
        default=str(DATA_DIR / "personas_primary.csv"),
    )
    parser.add_argument(
        "--rag-output",
        type=str,
        default=str(RAG_DIR / "persona_corpus.jsonl"),
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=str(RAG_DIR / "persona_corpus_summary.json"),
    )
    parser.add_argument("--clusters", type=int, default=12)
    parser.add_argument("--primary-count", type=int, default=100)
    parser.add_argument("--total-count", type=int, default=130)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seeds = load_personality(Path(args.personality))
    cluster_seeds(seeds, args.clusters)
    assignments = build_label_queue(args.primary_count, args.total_count, rng)
    records = generate_personas(seeds, assignments, rng)

    df = pd.DataFrame([record.to_row() for record in records])
    personas_output = Path(args.personas_output)
    primary_output = Path(args.primary_output)
    personas_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(personas_output, index=False)
    df[df["is_primary"] == 1].head(args.primary_count).to_csv(primary_output, index=False)

    rag_entries = build_rag_entries(records)
    write_jsonl(rag_entries, Path(args.rag_output))
    write_summary(records, Path(args.summary_output))

    print(f"Generated {len(records)} personas: {args.primary_count} primary + {len(records) - args.primary_count} backup")
    print(f"Primary personas saved to: {primary_output}")
    print(f"Full persona table saved to: {personas_output}")
    print(f"RAG corpus saved to: {args.rag_output}")


if __name__ == "__main__":
    main()
