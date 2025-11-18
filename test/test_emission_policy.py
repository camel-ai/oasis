from __future__ import annotations

from generation.emission_policy import EmissionPolicy, PersonaConfig


def test_emission_policy_override_single_mode():
    policy = EmissionPolicy(run_seed=123, post_label_mode_probs={"none": 0.3, "single": 0.6, "double": 0.1})
    persona = PersonaConfig(
        persona_id="p",
        primary_label="misinfo",
        allowed_labels=["misinfo", "conspiracy"],
        label_mode_cap="single",
        benign_on_none_prob=0.6,
        max_labels_per_post=2,
    )
    d = policy.decide(user_id=1, thread_id="t", step_idx=0, persona=persona, context=None,
                      override_post_mode_probs={"none": 0.1, "single": 0.2, "double": 0.7})
    # Even with override favoring double, cap should force single
    assert d["mode"] in ("single", "none")


def test_emission_policy_double_when_allowed():
    policy = EmissionPolicy(run_seed=123, post_label_mode_probs={"none": 0.1, "single": 0.2, "double": 0.7})
    persona = PersonaConfig(
        persona_id="p",
        primary_label="misinfo",
        allowed_labels=["misinfo", "conspiracy"],
        label_mode_cap="double",
        benign_on_none_prob=0.6,
        max_labels_per_post=2,
    )
    d = policy.decide(user_id=1, thread_id="t", step_idx=1, persona=persona, context=None)
    assert d["mode"] in ("single", "double")
    assert isinstance(d["tokens"], list)


