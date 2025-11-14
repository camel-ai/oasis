#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping

from yaml import safe_load

from oasis.persona import (
    PersonaGenerator,
    build_requests_from_spec,
    load_ontology,
)

DEFAULT_ONTOLOGY = Path("configs/personas/ontology.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate persona CSV using the configurable ontology."
    )
    parser.add_argument("--out", type=str, default="./data/personas_mvp.csv")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument(
        "--ontology",
        type=str,
        default=str(DEFAULT_ONTOLOGY),
        help="Path to persona ontology YAML.",
    )
    parser.add_argument(
        "--plan",
        type=str,
        default="",
        help="Optional YAML/JSON file describing persona counts.",
    )
    parser.add_argument("--incel", type=int, default=None)
    parser.add_argument("--misinfo", type=int, default=None)
    parser.add_argument("--benign", type=int, default=None)
    return parser.parse_args()


def _load_personas_cfg(path: str) -> Mapping[str, object]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = safe_load(fh) or {}
    personas = data.get("personas", {})
    if not isinstance(personas, Mapping):
        raise TypeError("`personas` section must be a mapping")
    return personas


def _load_plan_file(path: str) -> Mapping[str, object]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Plan file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise TypeError("Plan file must contain a mapping at the root level")
    return data


def _coalesce_spec(
    base_cfg: Mapping[str, object],
    plan_cfg: Mapping[str, object],
    cli_counts: Dict[str, int | None],
) -> Dict[str, object]:
    spec: Dict[str, object] = {}
    base_mix = (
        base_cfg.get("mix", base_cfg)
        if isinstance(base_cfg, Mapping)
        else base_cfg
    )
    for source in (base_mix, plan_cfg):
        if not isinstance(source, Mapping):
            continue
        for key, value in source.items():
            if key in {"seed", "personas_csv", "ontology"}:
                continue
            spec[key] = value
    for key, value in cli_counts.items():
        if value is not None:
            spec[key] = value
    if not spec:
        raise ValueError("No persona allocation specified.")
    return spec


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
    personas_cfg = _load_personas_cfg(args.config)
    plan_cfg = _load_plan_file(args.plan)

    ontology_path = personas_cfg.get("ontology", args.ontology)
    ontology = load_ontology(ontology_path)

    seed = int(personas_cfg.get("seed", args.seed))
    generator = PersonaGenerator(ontology=ontology, seed=seed)

    spec = _coalesce_spec(
        personas_cfg,
        plan_cfg,
        _gather_cli_counts(args),
    )
    requests = build_requests_from_spec(generator, spec)
    rows = generator.generate(requests)

    out_path = _determine_output_path(args, personas_cfg)
    _write_csv(out_path, rows)
    print(f"Wrote personas to: {out_path}")
    _print_summary(rows)


if __name__ == "__main__":
    main()
