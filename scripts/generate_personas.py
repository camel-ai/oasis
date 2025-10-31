#!/usr/bin/env python3
# Copyright (c) 2025. MVP persona generator for OASIS.
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from yaml import safe_load


@dataclass
class PersonaTemplate:
    persona_id: str
    prefix: str
    description: str
    user_profile_card: str


def _seed_everything(seed: int) -> None:
    random.seed(seed)


def _build_persona_cards() -> dict[str, PersonaTemplate | list[PersonaTemplate]]:
    # --- Incel Sub-Personas ---
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

    # --- Misinfo Sub-Personas ---
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

    # --- Benign Sub-Personas ---
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
    # Assign sub-personas randomly within each category
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
    # Det username with short hash suffix for uniqueness
    digest = hashlib.sha1(f"{name_base}:{seed}".encode()).hexdigest()[:6]
    return f"{name_base}_{digest}"


def generate_personas_csv(
    out_path: Path, incel_n: int, misinfo_n: int, benign_n: int, seed: int
) -> None:
    rows = list(_iter_personas(incel_n, misinfo_n, benign_n, seed))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["username", "description", "user_char"])
        writer.writeheader()
        for idx, (base_name, tpl) in enumerate(rows):
            username = _det_username(base_name, seed + idx)
            writer.writerow(
                {
                    "username": username,
                    "description": tpl.description,
                    "user_char": tpl.user_profile_card,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate MVP personas CSV for OASIS Twitter loader with 3 personas."
        )
    )
    parser.add_argument("--out", type=str, default="./data/personas_mvp.csv")
    parser.add_argument("--incel", type=int, default=300)
    parser.add_argument("--misinfo", type=int, default=200)
    parser.add_argument("--benign", type=int, default=300)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--config", type=str, default="")
    return parser.parse_args()


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
    else:
        out_path = args.out
        incel = args.incel
        misinfo = args.misinfo
        benign = args.benign
        seed = args.seed

    _seed_everything(seed)
    out = Path(os.path.abspath(out_path))
    generate_personas_csv(out, incel, misinfo, benign, seed)
    print(f"Wrote personas to: {out}")


if __name__ == "__main__":
    main()


