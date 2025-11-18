#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List

from yaml import safe_load

REQUIRED_MANIFEST_FIELDS = ["rng_seed", "post_label_mode_probs"]
REQUIRED_PERSONA_FIELDS = ["username", "description", "user_char", "primary_label", "label_mode_cap", "allowed_labels"]


def validate_manifest(path: Path) -> List[str]:
    errs: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        data = safe_load(f) or {}
    for key in REQUIRED_MANIFEST_FIELDS:
        if key not in data:
            errs.append(f"Manifest missing required field: {key}")
    pmp = data.get("post_label_mode_probs", {})
    for k in ("none", "single", "double"):
        if k not in pmp:
            errs.append(f"post_label_mode_probs missing '{k}' probability")
    return errs


def validate_personas_csv(path: Path) -> List[str]:
    errs: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
    for col in REQUIRED_PERSONA_FIELDS:
        if col not in header:
            errs.append(f"Persona CSV missing required column: {col}")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate manifest and persona CSV schema.")
    ap.add_argument("--manifest", type=str, required=True, help="Path to manifest.yaml")
    ap.add_argument("--personas-csv", type=str, required=True, help="Path to personas.csv")
    args = ap.parse_args()
    manifest_path = Path(os.path.abspath(args.manifest))
    personas_path = Path(os.path.abspath(args.personas_csv))
    errs = []
    errs.extend(validate_manifest(manifest_path))
    errs.extend(validate_personas_csv(personas_path))
    if errs:
        print("VALIDATION FAILED:")
        for e in errs:
            print(f"- {e}")
        raise SystemExit(1)
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()


