from src.accuracy_breakdown import accuracy_breakdown
from src.belief_assignment import MassAssignment, load_evidence_models
from src.dempster_shafer import HypothesisSelectionPolicy
from src.validated_synthesis import build_validated_synthesis, render_accuracy_breakdown_table, validate_synthesis


def _breakdown():
    return accuracy_breakdown(
        load_evidence_models(),
        assignment=MassAssignment.PUBLISHED_ANALOGUE,
        selection_policy=HypothesisSelectionPolicy.EVIDENCE_FACTORY_COMPAT,
    )


def test_breakdown_partitions_all_accuracy_evidence_models():
    breakdown = _breakdown()
    assert sum(row["n_evidence_models"] for row in breakdown["rows"]) == breakdown["n_evidence_models"] == 41  # noqa: PLR2004
    assert breakdown["n_indifferent_only"] == 16  # noqa: PLR2004
    assert breakdown["n_including_indifferent"] == 24  # noqa: PLR2004
    assert breakdown["n_strictly_negative"] == 17  # noqa: PLR2004
    assert breakdown["n_positive_intensity"] == 0


def test_breakdown_counts_losses_from_mean_relative_improvement():
    breakdown = _breakdown()
    assert breakdown["n_negative_mean"] == 34  # noqa: PLR2004
    assert breakdown["n_moderate_loss"] == 17  # noqa: PLR2004
    assert breakdown["n_strong_loss"] == 5  # noqa: PLR2004
    assert breakdown["strong_loss_study_ids"] == ["S4"]


def test_breakdown_beliefs_match_pooled_accuracy():
    breakdown = _breakdown()
    payload = build_validated_synthesis()
    assert breakdown["selected_intensity"] == ["I"]
    assert abs(breakdown["belief_indifferent"] - payload["effects"]["Accuracy"]["belief"]) < 1e-12  # noqa: PLR2004
    assert round(breakdown["belief_weakly_negative_or_indifferent"], 3) == 0.997  # noqa: PLR2004
    assert breakdown["belief_below_weakly_negative"] < 0.01  # noqa: PLR2004
    assert round(breakdown["last_pairwise_conflict"], 2) == round(payload["effects"]["Accuracy"]["conflict"], 2)


def test_breakdown_table_lists_strong_losses_from_s4_only():
    latex = render_accuracy_breakdown_table(build_validated_synthesis())
    assert r"SN & 5 & S4 & $-91.2$ to $-89.3$" in latex
    assert r"Bel(\{WN, I\}) $= 0.997$" in latex


def test_validate_synthesis_flags_tampered_breakdown():
    payload = build_validated_synthesis()
    payload["accuracy_breakdown"]["n_strong_loss"] = 0
    assert "accuracy breakdown: n_strong_loss mismatch" in validate_synthesis(payload)
