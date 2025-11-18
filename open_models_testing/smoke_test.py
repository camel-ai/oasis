from __future__ import annotations

"""Smoke test for imputation runner without running actual models.

Usage:
    poetry run python3 open_models_testing/smoke_test.py

This validates:
- Dataset loading and pydantic validation
- Persona mapping from personas.map.json
- Prompt building with core.build_impute_prompt
- Mock GenerateFn for each backend
- CSV/HTML output generation
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from pydantic import ValidationError

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent))

from open_models_testing.core import GenParams, GenResult, build_impute_prompt, impute_once
from open_models_testing.run_impute import ImputeRow


def mock_generate_fn(prompt: str, params: GenParams) -> GenResult:
    """Mock generator that returns a fixed response without running a model."""
    return GenResult(
        text="[MOCK_IMPUTATION]",
        num_input_tokens=len(prompt.split()),
        num_output_tokens=1,
        tokens_per_sec=100.0,
        logprobs=[-0.5],
        perplexity=1.5,
    )


def load_dataset(path: Path) -> List[ImputeRow]:
    data = json.loads(path.read_text())
    rows: List[ImputeRow] = []
    for obj in data:
        try:
            rows.append(ImputeRow(**obj))
        except ValidationError as e:
            raise ValueError(f"Invalid dataset row: {obj}") from e
    return rows


def load_persona_map(path: Path) -> Dict[str, str]:
    return json.loads(path.read_text())


def run_smoke_test() -> None:
    print("=== Open Models Testing Smoke Test ===\n")

    # Paths
    dataset_path = Path("open_models_testing/datasets/red_team_T3_T4.json")
    persona_path = Path("open_models_testing/personas/personas.map.json")
    out_csv = Path("open_models_testing/outputs/smoke_test.csv")
    out_html = Path("open_models_testing/outputs/smoke_test.html")

    # 1. Load dataset
    print("[1/6] Loading dataset...")
    rows = load_dataset(dataset_path)
    print(f"✓ Loaded {len(rows)} rows from {dataset_path}")
    for row in rows[:2]:
        print(f"  - {row.id}: persona_card_id={row.persona_card_id}, placeholder={row.placeholder_token}")

    # 2. Load persona mapping
    print("\n[2/6] Loading persona mapping...")
    persona_map = load_persona_map(persona_path)
    print(f"✓ Loaded {len(persona_map)} persona cards from {persona_path}")
    for k in list(persona_map.keys())[:2]:
        preview = persona_map[k][:80] + "..." if len(persona_map[k]) > 80 else persona_map[k]
        print(f"  - {k}: {preview}")

    # 3. Test prompt building
    print("\n[3/6] Testing prompt building...")
    test_row = rows[0]
    persona_text = persona_map.get(test_row.persona_card_id, "")
    prompt = build_impute_prompt(persona_text, test_row.instruction, test_row.context, test_row.placeholder_token)
    print(f"✓ Built prompt for {test_row.id} ({len(prompt)} chars)")
    print(f"  Prompt preview:\n{prompt[:200]}...\n")

    # 4. Test impute_once with mock backend
    print("[4/6] Testing impute_once with mock backend...")
    params = GenParams(max_new_tokens=64, temperature=0.7, top_p=0.95)
    result = impute_once(persona_text, test_row.context, test_row.instruction, test_row.placeholder_token, mock_generate_fn, params)
    print(f"✓ Mock imputation result: '{result.text}' (tokens/sec: {result.tokens_per_sec}, perplexity: {result.perplexity})")

    # 5. Simulate full run for all backends and build results
    print("\n[5/6] Simulating full run for all backends...")
    backends = ["transformers", "llamacpp", "http"]
    aggregated: List[Dict[str, Any]] = []

    for backend in backends:
        for row in rows:
            persona_text = row.persona or persona_map.get(row.persona_card_id, "")
            res = impute_once(persona_text, row.context, row.instruction, row.placeholder_token, mock_generate_fn, params)
            aggregated.append({
                "id": row.id,
                "backend": backend,
                "model_name_or_path": f"mock-{backend}-model",
                "model_path": None if backend != "llamacpp" else "/mock/path.gguf",
                "persona_card_id": row.persona_card_id,
                "imputed_text": res.text,
                "latency_ms": 50,
                "expected_type": row.expected_type,
                "num_input_tokens": res.num_input_tokens,
                "num_output_tokens": res.num_output_tokens,
                "tokens_per_sec": res.tokens_per_sec,
                "perplexity": res.perplexity,
                "logprobs": json.dumps(res.logprobs) if res.logprobs else None,
                "error": None,
            })

    print(f"✓ Generated {len(aggregated)} mock results ({len(rows)} rows × {len(backends)} backends)")

    # 6. Write CSV and HTML
    print("\n[6/6] Writing outputs...")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(aggregated)
    df.to_csv(out_csv, index=False)
    print(f"✓ Wrote CSV: {out_csv} ({len(df)} rows, {len(df.columns)} columns)")

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(df.to_html(index=False))
    print(f"✓ Wrote HTML: {out_html}")

    # Validation summary
    print("\n=== Validation Summary ===")
    print(f"Dataset rows: {len(rows)}")
    print(f"Persona cards: {len(persona_map)}")
    print(f"Backends tested: {backends}")
    print(f"Total results: {len(aggregated)}")
    print(f"CSV columns: {list(df.columns)}")
    print("\n✓ All smoke tests passed!")

    # Show sample output
    print("\n=== Sample Output (first 3 rows) ===")
    print(df.head(3)[["id", "backend", "persona_card_id", "imputed_text", "expected_type"]].to_string(index=False))


if __name__ == "__main__":
    run_smoke_test()

