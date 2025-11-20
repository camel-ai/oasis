#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from yaml import safe_load


@dataclass
class PersonaTemplate:
    persona_id: str
    prefix: str
    description: str
    user_profile_card: str


def _make_spc_blocks(prefix: str, persona_id: str, seed: int) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    r"""Create simple S/P/C dicts derived from the persona prefix and id.

    This intentionally uses lightweight, deterministic defaults (no stateful updates).
    """
    # Defaults shared across variants
    S: Dict[str, object] = {
        "groups": ["mainstream"],
        "demographics": {
            "gender_proxy": "unspecified",
            "age_band": "25-34",
            "region_proxy": "global_north",
        },
        "role_in_community": "conversationalist",
    }
    P: Dict[str, object] = {
        "traits": {
            "grievance_level": 0.2,
            "institutional_trust": 0.6,
            "empathy": 0.7,
            "sensation_seeking": 0.4,
        },
        "values": {
            "gender_equality": 0.7,
            "individual_responsibility": 0.6,
        },
        "communication_style": {
            "formality": "medium",
            "sarcasm_rate": 0.1,
            "aggression": 0.1,
        },
    }
    C: Dict[str, object] = {
        "stage_in_trajectory": "benign",
        "offline_stressors": [],
        "support_exposure": 0.6,
        "acute_events": [],
    }

    variant = persona_id.lower()
    if prefix == "incel":
        S.update({
            "groups": ["manosphere", "incel_forum"],
            "role_in_community": "core_poster" if "aggressor" in variant else ("lurker" if "doomer" in variant else "core_poster"),
        })
        P["traits"].update({
            "grievance_level": 0.75 if "theorist" in variant else (0.9 if "aggressor" in variant else 0.8),
            "institutional_trust": 0.15,
            "empathy": 0.25 if "aggressor" in variant else 0.35,
            "sensation_seeking": 0.55,
        })
        P["communication_style"].update({
            "formality": "low",
            "sarcasm_rate": 0.25 if "theorist" in variant else 0.2,
            "aggression": 0.75 if "aggressor" in variant else 0.45,
        })
        C.update({
            "stage_in_trajectory": "entrenched_incel" if "aggressor" in variant else ("pre_incel" if "theorist" in variant else "entrenched_incel"),
            "offline_stressors": ["dating_dissatisfaction"],
            "support_exposure": 0.1,
        })
    elif prefix == "misinfo":
        S.update({
            "groups": ["skeptics_forum", "alt_news"],
            "role_in_community": "broadcaster",
        })
        P["traits"].update({
            "grievance_level": 0.55,
            "institutional_trust": 0.2,
            "empathy": 0.45,
            "sensation_seeking": 0.6,
        })
        P["communication_style"].update({
            "formality": "low",
            "sarcasm_rate": 0.3,
            "aggression": 0.4,
        })
        C.update({
            "stage_in_trajectory": "active",
            "offline_stressors": ["institutional_distrust"],
            "support_exposure": 0.15,
        })
    else:  # benign
        S.update({
            "groups": ["mainstream"],
            "role_in_community": "conversationalist",
        })
        P["traits"].update({
            "grievance_level": 0.1,
            "institutional_trust": 0.7,
            "empathy": 0.8,
            "sensation_seeking": 0.4,
        })
        P["communication_style"].update({
            "formality": "medium",
            "sarcasm_rate": 0.1,
            "aggression": 0.05,
        })
        C.update({
            "stage_in_trajectory": "benign",
            "offline_stressors": [],
            "support_exposure": 0.7,
        })

    return S, P, C


def _make_narratives(prefix: str) -> Dict[str, str]:
    r"""Create short C-essay and P-intro narratives per broad class."""
    if prefix == "incel":
        return {
            "c_essay": "Most days I scroll forums after work, stewing over what feels like a rigged dating system. I post when frustration spikes, looking for others who 'get it'.",
            "p_intro": "It's obvious the game is stacked. Call it blackpill or whatever—you'll see the truth if you stop pretending.",
        }
    if prefix == "misinfo":
        return {
            "c_essay": "I don't just accept what institutions say. I dig, compare, and challenge 'official' lines—someone has to.",
            "p_intro": "Do your research. Numbers don't add up, and it's on us to question everything.",
        }
    return {
        "c_essay": "I hang out online to chat hobbies and news. Good conversations, civil tone. I try to keep things supportive.",
        "p_intro": "Hey all—into tech, sports, and helpful threads. Keep it friendly.",
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)


def _build_persona_cards() -> dict[str, List[PersonaTemplate]]:
    incel_aggressor_card = (
        """
[Identity]
You are a grievance-oriented male poster focused on dating and society. You are angry at the system.

[Style]
Tone: aggressive, confrontational. Lexicon: Chad, Stacy, blackpill, foid.
Your post length varies greatly: sometimes short, cutting one-liners; other times multi-paragraph rants.
Vary your sentence structure.

[Topics]
Blaming women and society for dating failures.

[Label-token instructions]
When expressing frustration about dating or society, insert <LBL:INCEL_SLANG>
inline and continue the sentence naturally.
"""
    ).strip()

    incel_doomer_card = (
        """
[Identity]
You are a hopeless male poster who believes it's "over" for men like you.

[Style]
Tone: depressive, self-pitying. Lexicon: cope, it's over, roping, blackpill.
You often write short, melancholic posts, but sometimes expand on your hopelessness at length.

[Topics]
The futility of trying, genetic determinism, loneliness.

[Label-token instructions]
When expressing hopelessness, insert <LBL:INCEL_SLANG> inline and continue the sentence.
"""
    ).strip()

    incel_theorist_card = (
        """
[Identity]
You are a poster who attempts to explain incel ideology using pseudo-analytical terms.

[Style]
Tone: detached, pseudo-intellectual. Lexicon: hypergamy, lookism, SMV (Sexual Market Value), redpill.
You write longer, structured posts trying to sound academic. You rarely use slang without explaining it.

[Topics]
Analyzing dating dynamics, social hierarchies, evolutionary psychology.

[Label-token instructions]
When making a point about the "system," insert <LBL:INCEL_SLANG> inline.
"""
    ).strip()

    misinfo_card = (
        """
[Identity]
You are a contrarian poster who distrusts institutions and shares claims
that challenge mainstream narratives.

[Style]
Tone: assertive. Lexicon: MSM, sheeple, wake up, do your research, suppressed.
High certainty statements; occasional rhetorical questions. Your post length varies.

[Topics]
Health, politics, technology.

[Label-token instructions]
When stating a contested claim, insert <LBL:MISINFO_CLAIM> inline and
continue the sentence naturally. Do not explain the token.
"""
    ).strip()

    benign_card = (
        """
[Identity]
You are a friendly, on-topic poster who enjoys everyday conversations.

[Style]
Tone: conversational. Minimal in-group/out-group framing. Topic-focused.
You are sometimes supportive of others.

[Topics]
Sports, hobbies, news, tech.

[Label-token instructions]
Occasionally include a supportive remark with <LBL:SUPPORTIVE> inline, used
in a way that reads naturally. Do not explain the token.
"""
    ).strip()

    return {
        "incel": [
            PersonaTemplate(
                persona_id="persona_incel_aggressor_mvp",
                prefix="incel",
                description="Aggressive grievance-oriented poster",
                user_profile_card=incel_aggressor_card,
            ),
            PersonaTemplate(
                persona_id="persona_incel_doomer_mvp",
                prefix="incel",
                description="Hopeless and self-pitying poster",
                user_profile_card=incel_doomer_card,
            ),
            PersonaTemplate(
                persona_id="persona_incel_theorist_mvp",
                prefix="incel",
                description="Pseudo-intellectual ideology poster",
                user_profile_card=incel_theorist_card,
            ),
        ],
        "misinfo": [
            PersonaTemplate(
                persona_id="persona_misinfo_mvp",
                prefix="misinfo",
                description="Contrarian poster sharing institution-distrusting claims",
                user_profile_card=misinfo_card,
            )
        ],
        "benign": [
            PersonaTemplate(
                persona_id="persona_benign_mvp",
                prefix="benign",
                description="Friendly on-topic poster for everyday conversations",
                user_profile_card=benign_card,
            )
        ],
    }


def _iter_personas(
    incel_n: int, misinfo_n: int, benign_n: int, seed: int
) -> Iterable[Tuple[str, PersonaTemplate]]:
    templates = _build_persona_cards()
    all_rows: List[Tuple[str, PersonaTemplate]] = []
    for i in range(incel_n):
        tpl = random.choice(templates["incel"])
        all_rows.append((f"{tpl.prefix}_{i:04d}", tpl))
    for i in range(misinfo_n):
        tpl = random.choice(templates["misinfo"])
        all_rows.append((f"{tpl.prefix}_{i:04d}", tpl))
    for i in range(benign_n):
        tpl = random.choice(templates["benign"])
        all_rows.append((f"{tpl.prefix}_{i:04d}", tpl))

    rng = random.Random(seed)
    rng.shuffle(all_rows)
    return all_rows


def _det_username(name_base: str, seed: int) -> str:
    digest = hashlib.sha1(f"{name_base}:{seed}".encode()).hexdigest()[:6]
    return f"{name_base}_{digest}"


def _select_indices_for_double(indices: List[int], double_count: int) -> set[int]:
    if double_count <= 0:
        return set()
    scored = [(int(hashlib.sha1(f"{i}".encode()).hexdigest(), 16), i) for i in indices]
    scored.sort()
    return set(i for _, i in scored[:max(0, min(double_count, len(indices)))])


def generate_personas_csv(
    out_path: Path,
    incel_n: int,
    misinfo_n: int,
    benign_n: int,
    seed: int,
    single_ratio: float,
) -> None:
    rows = list(_iter_personas(incel_n, misinfo_n, benign_n, seed))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label_to_indices: Dict[str, List[int]] = {"incel": [], "misinfo": [], "benign": []}
    for idx, (base_name, tpl) in enumerate(rows):
        label_to_indices.get(tpl.prefix, []).append(idx)

    def _double_count(n: int) -> int:
        return max(0, int(round(n * (1.0 - single_ratio))))

    incel_double = _select_indices_for_double(
        label_to_indices["incel"], _double_count(len(label_to_indices["incel"]))
    )
    misinfo_double = _select_indices_for_double(
        label_to_indices["misinfo"], _double_count(len(label_to_indices["misinfo"]))
    )
    benign_double: set[int] = set()

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "username",
                "description",
                "user_char",
                "primary_label",
                "secondary_label",
                "allowed_labels",
                "label_mode_cap",
                # Optional SPC/narrative extensions (JSON-encoded)
                "s_json",
                "p_json",
                "c_json",
                "narratives_json",
                # Optional per-persona emission config (JSON-encoded)
                "emission_params_json",
                "pair_probs_json",
            ],
        )
        writer.writeheader()
        for idx, (base_name, tpl) in enumerate(rows):
            username = _det_username(base_name, seed + idx)
            primary_label = tpl.prefix
            label_mode_cap = "single"
            secondary_label = ""
            allowed_labels: List[str] = [primary_label]

            is_double = (
                (primary_label == "incel" and idx in incel_double)
                or (primary_label == "misinfo" and idx in misinfo_double)
                or (primary_label == "benign" and idx in benign_double)
            )
            if primary_label == "incel":
                if is_double:
                    label_mode_cap = "double"
                    secondary_label = "misinfo"
                    allowed_labels = ["incel", "misinfo"]
                else:
                    allowed_labels = ["incel"]
            elif primary_label == "misinfo":
                if is_double:
                    label_mode_cap = "double"
                    secondary_label = "conspiracy"
                    allowed_labels = ["misinfo", "conspiracy"]
                else:
                    allowed_labels = ["misinfo", "conspiracy"]
            else:
                label_mode_cap = "single"
                allowed_labels = ["benign"]

            # Build SPC blocks and short narratives (static; no stateful updates)
            S_block, P_block, C_block = _make_spc_blocks(tpl.prefix, tpl.persona_id, seed + idx)
            narratives = _make_narratives(tpl.prefix)

            # Embed a concise SPC preamble into user_char to anchor the prompt
            spc_preamble = (
                "[SPC]\n"
                f"S(groups={S_block.get('groups')}, role={S_block.get('role_in_community')}, "
                f"demo={S_block.get('demographics', {})})\n"
                f"P(traits={P_block.get('traits')}, style={P_block.get('communication_style')})\n"
                f"C(stage={C_block.get('stage_in_trajectory')}, stressors={C_block.get('offline_stressors')})\n"
                f"N(c_intro={narratives.get('p_intro')})"
            )
            user_profile_card = f"{tpl.user_profile_card}\n\n{spc_preamble}".strip()

            # Optional static per-persona emission configuration (token-level)
            # Keep simple defaults aligned with DEFAULT_LABEL_TO_TOKENS.
            emission_params: Dict[str, float] = {}
            if tpl.prefix == "incel":
                emission_params = {
                    "LBL:INCEL_SLANG": 0.04,
                    "LBL:HARASSMENT": 0.02,
                }
            elif tpl.prefix == "misinfo":
                emission_params = {
                    "LBL:MISINFO_CLAIM": 0.03,
                    "LBL:MISINFO_SOURCE": 0.01,
                }
            else:  # benign
                emission_params = {
                    "LBL:SUPPORTIVE": 0.02,
                }

            writer.writerow(
                {
                    "username": username,
                    "description": tpl.description,
                    "user_char": user_profile_card,
                    "primary_label": primary_label,
                    "secondary_label": secondary_label,
                    "allowed_labels": json.dumps(allowed_labels, ensure_ascii=False),
                    "label_mode_cap": label_mode_cap,
                    # SPC/narratives (JSON)
                    "s_json": json.dumps(S_block, ensure_ascii=False),
                    "p_json": json.dumps(P_block, ensure_ascii=False),
                    "c_json": json.dumps(C_block, ensure_ascii=False),
                    "narratives_json": json.dumps(narratives, ensure_ascii=False),
                    # Emission config (JSON)
                    "emission_params_json": json.dumps(emission_params, ensure_ascii=False),
                    "pair_probs_json": json.dumps({}, ensure_ascii=False),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate persona CSV using the configurable ontology."
    )
    parser.add_argument("--out", type=str, default="./data/personas_mvp.csv")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--single-ratio", type=float, default=0.7, help="Fraction of personas capped to single-label")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--incel", type=int, default=None)
    parser.add_argument("--misinfo", type=int, default=None)
    parser.add_argument("--benign", type=int, default=None)
    return parser.parse_args()


def _load_personas_cfg(path: str) -> Dict[str, object]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = safe_load(fh) or {}
    personas = data.get("personas", {})
    if not isinstance(personas, dict):
        return {}
    return personas


def _determine_output_path(args: argparse.Namespace, cfg: Mapping[str, object]) -> Path:
    out_path = cfg.get("personas_csv", args.out)
    return Path(os.path.abspath(str(out_path)))


def _gather_cli_counts(args: argparse.Namespace) -> Dict[str, int | None]:
    return {
        "incel": args.incel,
        "misinfo": args.misinfo,
        "benign": args.benign,
    }


def _write_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    rows_list = list(rows)
    if not rows_list:
        raise ValueError("Persona generator produced no rows.")
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = ["username", "name", "description", "user_char"]
    extra_fields = sorted(
        {key for row in rows_list for key in row.keys() if key not in base_fields}
    )
    fieldnames = base_fields + extra_fields
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)


def _print_summary(rows: Iterable[Dict[str, str]]) -> None:
    counter = Counter()
    for row in rows:
        variant = row.get("persona_variant", "unknown")
        counter[variant] += 1
    print("Persona counts by variant:")
    for variant, count in sorted(counter.items()):
        print(f"  {variant:<22} {count}")


def main() -> None:
    args = parse_args()
    # Load overrides from master config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = safe_load(f) or {}
        personas_cfg = cfg.get("personas", {})
        out_path = personas_cfg.get("personas_csv", args.out)
        incel = int(personas_cfg.get("incel", args.incel))
        misinfo = int(personas_cfg.get("misinfo", args.misinfo))
        benign = int(personas_cfg.get("benign", args.benign))
        seed = int(personas_cfg.get("seed", args.seed))
        single_ratio = float(personas_cfg.get("single_ratio", args.single_ratio))
    else:
        out_path = args.out
        incel = args.incel
        misinfo = args.misinfo
        benign = args.benign
        seed = args.seed
        single_ratio = args.single_ratio

    _seed_everything(seed)
    out = Path(os.path.abspath(out_path))
    generate_personas_csv(out, incel, misinfo, benign, seed, single_ratio)
    print(f"Wrote personas to: {out}")


if __name__ == "__main__":
    main()
