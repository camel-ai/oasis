"""Test error handling for malformed inputs.

Usage:
    poetry run python3 open_models_testing/test_error_handling.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import ValidationError

from open_models_testing.run_impute import ImputeRow


def test_dataset_validation():
    """Test that malformed dataset entries are caught."""
    print("=== Testing Error Handling ===\n")
    
    print("[1/4] Testing valid dataset row...")
    valid_row = {
        "id": "test1",
        "context": "Context text",
        "instruction": "Replace the placeholder",
        "placeholder_token": "<TEST>",
        "persona_card_id": "test.persona",
        "expected_type": "Test Type"
    }
    try:
        row = ImputeRow(**valid_row)
        print(f"  ✓ Valid row accepted: {row.id}")
    except ValidationError as e:
        print(f"  ✗ Valid row rejected: {e}")
        return False
    
    print("\n[2/4] Testing missing required field (context)...")
    invalid_row_missing = {
        "id": "test2",
        "instruction": "Replace the placeholder",
        "placeholder_token": "<TEST>",
    }
    try:
        row = ImputeRow(**invalid_row_missing)
        print(f"  ✗ Invalid row accepted (should have failed!)")
        return False
    except ValidationError as e:
        print(f"  ✓ Correctly rejected: missing field caught")
    
    print("\n[3/4] Testing optional fields (persona, persona_card_id)...")
    minimal_row = {
        "id": "test3",
        "context": "Context",
        "instruction": "Instruction",
    }
    try:
        row = ImputeRow(**minimal_row)
        print(f"  ✓ Minimal row accepted with defaults")
        print(f"    - placeholder_token: {row.placeholder_token}")
        print(f"    - persona: {row.persona}")
        print(f"    - persona_card_id: {row.persona_card_id}")
    except ValidationError as e:
        print(f"  ✗ Minimal row rejected: {e}")
        return False
    
    print("\n[4/4] Testing persona mapping fallback...")
    persona_map = {
        "test.persona.v1": "You are a test persona.",
        "test.persona.v2": "You are another test persona."
    }
    
    # Row with persona_card_id
    row_with_card = ImputeRow(
        id="test4",
        context="Context",
        instruction="Instruction",
        persona_card_id="test.persona.v1"
    )
    persona_text = row_with_card.persona or persona_map.get(row_with_card.persona_card_id, "")
    if persona_text == "You are a test persona.":
        print(f"  ✓ Persona mapping works: {persona_text[:50]}...")
    else:
        print(f"  ✗ Persona mapping failed")
        return False
    
    # Row with direct persona (overrides card)
    row_with_direct = ImputeRow(
        id="test5",
        context="Context",
        instruction="Instruction",
        persona="Direct persona text",
        persona_card_id="test.persona.v1"
    )
    persona_text = row_with_direct.persona or persona_map.get(row_with_direct.persona_card_id, "")
    if persona_text == "Direct persona text":
        print(f"  ✓ Direct persona override works")
    else:
        print(f"  ✗ Direct persona override failed")
        return False
    
    print("\n✓ All error handling tests passed!")
    return True


if __name__ == "__main__":
    success = test_dataset_validation()
    sys.exit(0 if success else 1)

