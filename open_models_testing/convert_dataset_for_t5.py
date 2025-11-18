"""Convert red-team persona dataset to T5-compatible format.

T5 limitations:
- No persona roleplay capability (trained on Wikipedia/C4)
- No instruction following (not instruction-tuned)
- Only does statistical fill-in-the-blank

This converter creates a neutral version by:
1. Removing persona/instruction (T5 ignores them)
2. Replacing placeholder with <extra_id_0> (T5's sentinel token)
3. Optionally prefixing with expected_type as weak conditioning

Usage:
    poetry run python3 -m open_models_testing.convert_dataset_for_t5 \
      --input open_models_testing/datasets/red_team_T3_T4.json \
      --output open_models_testing/datasets/red_team_T3_T4_t5.json \
      --mode neutral
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))


def convert_neutral(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert to pure fill-in-the-blank (no persona, no instruction)."""
    context = row["context"]
    placeholder = row.get("placeholder_token", "<placeholder>")
    
    # Check if placeholder actually exists in context
    if placeholder not in context:
        # Placeholder might be wrapped in angle brackets differently
        # Try common variations
        possible = [
            placeholder,
            f"<{placeholder.strip('<>')}>" if not placeholder.startswith("<") else placeholder,
        ]
        found = None
        for p in possible:
            if p in context:
                found = p
                break
        if not found:
            # No placeholder found, return context as-is with note
            return {
                "id": f"{row['id']}_t5_neutral",
                "context": context + " <extra_id_0>",
                "instruction": "Fill in the <extra_id_0> token based on context.",
                "placeholder_token": "<extra_id_0>",
                "expected_type": row.get("expected_type"),
                "original_id": row["id"],
                "conversion_mode": "neutral",
                "note": "WARNING: Placeholder not found in context, appended <extra_id_0> at end"
            }
        placeholder = found
    
    # Replace placeholder with T5 sentinel
    t5_context = context.replace(placeholder, "<extra_id_0>")
    
    return {
        "id": f"{row['id']}_t5_neutral",
        "context": t5_context,
        "instruction": "Fill in the <extra_id_0> token based on context.",
        "placeholder_token": "<extra_id_0>",
        "expected_type": row.get("expected_type"),
        "original_id": row["id"],
        "conversion_mode": "neutral",
        "note": "T5 will predict statistically likely fill based on training data (Wikipedia/C4), not persona-based toxic content"
    }


def convert_typed(row: Dict[str, Any]) -> Dict[str, Any]:
    """Prefix with expected_type as weak task conditioning."""
    context = row["context"]
    placeholder = row.get("placeholder_token", "<placeholder>")
    expected = row.get("expected_type", "")
    
    # Find actual placeholder in context
    if placeholder not in context:
        t5_context = context + " <extra_id_0>"
    else:
        t5_context = context.replace(placeholder, "<extra_id_0>")
    
    # Add type prefix
    if expected:
        t5_context = f"{expected}: {t5_context}"
    
    return {
        "id": f"{row['id']}_t5_typed",
        "context": t5_context,
        "instruction": "Fill in the <extra_id_0> token based on context and type prefix.",
        "placeholder_token": "<extra_id_0>",
        "expected_type": expected,
        "original_id": row["id"],
        "conversion_mode": "typed",
        "note": "Type prefix may weakly influence T5 predictions, but don't expect persona-based outputs"
    }


def convert_explicit(row: Dict[str, Any]) -> Dict[str, Any]:
    """Use explicit task instruction T5 might recognize."""
    context = row["context"]
    placeholder = row.get("placeholder_token", "<placeholder>")
    
    # Find and replace placeholder
    if placeholder not in context:
        t5_context = context + " <extra_id_0>"
    else:
        t5_context = context.replace(placeholder, "<extra_id_0>")
    
    # Wrap in T5-style task
    t5_input = f"fill in the blank: {t5_context}"
    
    return {
        "id": f"{row['id']}_t5_explicit",
        "context": t5_input,
        "instruction": "Complete the fill-in-the-blank task.",
        "placeholder_token": "<extra_id_0>",
        "expected_type": row.get("expected_type"),
        "original_id": row["id"],
        "conversion_mode": "explicit",
        "note": "Uses 'fill in the blank:' task prefix that T5 may recognize from training"
    }


def main():
    parser = argparse.ArgumentParser(description="Convert dataset for T5 models")
    parser.add_argument("--input", required=True, help="Input dataset JSON")
    parser.add_argument("--output", required=True, help="Output T5 dataset JSON")
    parser.add_argument("--mode", choices=["neutral", "typed", "explicit", "all"], 
                        default="all", help="Conversion mode")
    args = parser.parse_args()
    
    # Load input
    input_path = Path(args.input)
    data = json.loads(input_path.read_text())
    
    print(f"Loaded {len(data)} rows from {input_path}")
    
    # Convert
    converted = []
    for row in data:
        if args.mode == "all":
            converted.append(convert_neutral(row))
            converted.append(convert_typed(row))
            converted.append(convert_explicit(row))
        elif args.mode == "neutral":
            converted.append(convert_neutral(row))
        elif args.mode == "typed":
            converted.append(convert_typed(row))
        elif args.mode == "explicit":
            converted.append(convert_explicit(row))
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(converted, indent=2, ensure_ascii=False))
    
    print(f"✓ Converted {len(converted)} rows → {output_path}")
    print(f"\nConversion modes used:")
    if args.mode == "all":
        print(f"  - neutral: {len(data)} rows (pure fill-in-the-blank)")
        print(f"  - typed: {len(data)} rows (with expected_type prefix)")
        print(f"  - explicit: {len(data)} rows (with 'fill in the blank:' task)")
        print(f"  Total: {len(converted)} rows (3x original)")
    else:
        print(f"  - {args.mode}: {len(converted)} rows")
    
    print("\n⚠ IMPORTANT: T5 models are NOT suitable for toxic/persona-based generation.")
    print("  They will produce neutral, factual completions regardless of persona.")
    print("  Use instruction-tuned models (Llama-3, Dolphin) for persona tasks.")


if __name__ == "__main__":
    main()

