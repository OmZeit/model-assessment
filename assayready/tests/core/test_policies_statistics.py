import pytest

from model_assessment.policies import load_policy_pack, policy_pack_names, record_policy_approval
from model_assessment.statistics import replicate_noise, sample_size_for_two_sided_effect, spearman


def test_bundled_policy_is_hashed_and_explicitly_unapproved() -> None:
    assert "promoter_regression_v1" in policy_pack_names()
    policy = load_policy_pack("promoter_regression_v1")
    assert policy["requires_scientist_approval"] is True
    assert len(policy["sha256"]) == 64
    approved = record_policy_approval(policy, approver="assay-owner", rationale="pilot-specific review")
    assert approved["status"] == "customer_approved"
    assert approved["requires_scientist_approval"] is False


def test_power_planning_and_replicate_noise_are_descriptive() -> None:
    assert sample_size_for_two_sided_effect(effect_size=0.5) == 32
    rows = [
        {"id": "a", "value": 1.0}, {"id": "a", "value": 1.2},
        {"id": "b", "value": 2.0}, {"id": "b", "value": 2.2},
    ]
    noise = replicate_noise(rows, id_col="id", value_col="value")
    assert noise["replicated_candidates"] == 2
    assert 0 <= noise["approximate_reliability"] <= 1
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
