from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from oasis.persona import PersonaGenerator, build_requests_from_spec, load_ontology


@pytest.fixture(scope="module")
def ontology_path() -> Path:
    return Path("configs/personas/ontology.yaml")


@pytest.fixture(scope="module")
def ontology(ontology_path: Path):
    return load_ontology(ontology_path)


def test_expand_spec_supports_archetype_and_variant(ontology):
    generator = PersonaGenerator(ontology, seed=1)
    spec = {
        "incel": 4,
        "misinfo.contrarian": 2,
    }
    requests = build_requests_from_spec(generator, spec)
    assert sum(req.count for req in requests) == 6
    ids = sorted(req.variant_id for req in requests)
    assert ids[0].startswith("incel.")
    assert ids[-1] == "misinfo.contrarian"


def test_generate_rows_contains_expected_columns(ontology):
    generator = PersonaGenerator(ontology, seed=123)
    spec = {"incel": {"total": 3, "variants": {"aggressor": 1, "doomer": 1, "theorist": 1}}}
    requests = build_requests_from_spec(generator, spec)
    rows = generator.generate(requests)
    assert len(rows) == 3
    for row in rows:
        assert row["username"]
        assert row["description"]
        assert row["user_char"].startswith("[Identity]")
        assert row["persona_group"] == "incel"
        assert row["label_primary"] == "incel_slang"
        assert "action_bias" in row
        assert row["persona_prompt_post"]
        assert row["persona_prompt_comment"]
        assert "persona_goal" in row
        assert "persona_style_quirks" in row
    counter = Counter(row["persona_variant"] for row in rows)
    assert counter == {"aggressor": 1, "doomer": 1, "theorist": 1}
