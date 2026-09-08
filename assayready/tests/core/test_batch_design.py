from model_assessment.batch_design import BatchDesignPolicy, design_batch


def _candidates() -> list[dict[str, object]]:
    return [
        {"candidate_id": f"c{index}", "sequence": "ACGT" + "A" * index, "score": 20 - index,
         "family": "A" if index < 8 else "B", "cost": 2.0}
        for index in range(12)
    ]


def test_batch_design_enforces_replicates_family_quota_and_budget() -> None:
    result = design_batch(
        _candidates(),
        policy=BatchDesignPolicy(
            plate_size=24,
            controls={"positive_control": 2, "negative_control": 2},
            candidate_replicates=2,
            family_column="family",
            max_per_family=2,
            cost_column="cost",
            max_total_cost=12.0,
        ),
        objective_weights={"score": 1.0},
    )
    candidates = [row for row in result["assignments"] if row["role"] == "candidate"]
    assert result["status"] == "draft"
    assert result["approved_for_execution"] is False
    assert len({row["candidate_id"] for row in candidates if row["family"] == "A"}) <= 2
    assert all(sum(row["candidate_id"] == candidate_id for row in candidates) == 2 for candidate_id in {row["candidate_id"] for row in candidates})
    assert result["total_candidate_cost"] <= 12.0


def test_batch_requires_explicit_scientist_approval() -> None:
    policy = BatchDesignPolicy(plate_size=24, controls={}, candidate_replicates=1)
    draft = design_batch(_candidates(), policy=policy)
    approved = design_batch(_candidates(), policy=policy, scientist_approval="scientist@example.org")
    assert draft["approved_for_execution"] is False
    assert approved["approved_for_execution"] is True
