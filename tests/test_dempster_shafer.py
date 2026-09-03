"""Dempster–Shafer combination of effect intensities (SSM frame of discernment)."""

import pytest

from src.dempster_shafer import (
    THETA,
    HypothesisSelectionPolicy,
    combine_bpas,
    combine_effect,
    intensity_to_hypothesis,
    reconcile_intensities,
    select_hypothesis,
    simple_support,
    trace_effect,
)


def test_intensity_label_maps_to_adjacent_atoms():
    assert intensity_to_hypothesis("strongly positive") == frozenset({"SP"})
    assert intensity_to_hypothesis("indifferent - weakly positive") == frozenset({"I", "WP"})
    assert intensity_to_hypothesis("negative - weakly negative") == frozenset({"N", "WN"})
    assert intensity_to_hypothesis("weakly negative - indifferent") == frozenset({"WN", "I"})


def test_simple_support_puts_remainder_on_theta():
    hypothesis = frozenset({"SP"})
    masses = simple_support(hypothesis, 0.3)
    assert masses[hypothesis] == 0.3  # noqa: PLR2004
    assert masses[THETA] == 0.7  # noqa: PLR2004


def test_combine_bpas_matches_santos_kanban_worked_example():
    """Independent literals from Santos et al. 2018, team-cohesion combination table."""
    wn_if = frozenset({"WN", "I"})
    po_sp = frozenset({"P", "SP"})
    combined, conflict = combine_bpas(
        [
            {wn_if: 0.423, THETA: 0.577},
            {po_sp: 0.397, THETA: 0.603},
        ]
    )
    assert conflict == pytest.approx(0.167931, abs=1e-6)
    assert combined[wn_if] == pytest.approx(0.306548, abs=1e-6)
    assert combined[po_sp] == pytest.approx(0.275301, abs=1e-6)
    assert combined[THETA] == pytest.approx(0.418152, abs=1e-6)


def test_agreeing_simple_supports_reinforce():
    sp = frozenset({"SP"})
    combined, conflict = combine_bpas([simple_support(sp, 0.3), simple_support(sp, 0.3)])
    assert conflict == pytest.approx(0.0)
    assert combined[sp] == pytest.approx(0.51)
    assert combined[THETA] == pytest.approx(0.49)


def test_select_hypothesis_keeps_compound_when_simples_are_weak():
    compound = frozenset({"WN", "I"})
    masses = {compound: 0.99, THETA: 0.01}
    hypothesis, belief = select_hypothesis(masses)
    assert hypothesis == compound
    assert belief == pytest.approx(0.99)


def test_select_hypothesis_discards_compound_when_one_simple_holds_75_percent():
    indifferent = frozenset({"I"})
    wn = frozenset({"WN"})
    compound = frozenset({"WN", "I"})
    masses = {indifferent: 0.8, wn: 0.05, compound: 0.05, THETA: 0.1}
    hypothesis, selected_belief = select_hypothesis(masses)
    assert hypothesis == indifferent
    assert selected_belief == pytest.approx(0.8)


def test_select_hypothesis_keeps_focal_compound_when_neither_simple_dominates():
    wn = frozenset({"WN"})
    indifferent = frozenset({"I"})
    compound = frozenset({"WN", "I"})
    masses = {wn: 0.4, indifferent: 0.4, compound: 0.15, THETA: 0.05}
    hypothesis, selected_belief = select_hypothesis(masses)
    assert hypothesis == compound
    assert selected_belief == pytest.approx(0.95)


def test_select_hypothesis_ignores_non_focal_compounds():
    sp = frozenset({"SP"})
    po = frozenset({"P"})
    masses = {sp: 0.47, po: 0.18, THETA: 0.35}
    hypothesis, selected_belief = select_hypothesis(masses)
    assert hypothesis == sp
    assert selected_belief == pytest.approx(0.47)


def test_santos_2015_selection_uses_belief_of_non_focal_interval():
    sp = frozenset({"SP"})
    po = frozenset({"P"})
    masses = {sp: 0.47, po: 0.18, THETA: 0.35}

    hypothesis, selected_belief = select_hypothesis(
        masses,
        policy=HypothesisSelectionPolicy.SANTOS_2015,
    )

    assert hypothesis == frozenset({"P", "SP"})
    assert selected_belief == pytest.approx(0.65)


def test_combine_effect_uses_intensity_labels_and_returns_conflict():
    result = combine_effect(
        [
            ("strongly positive", 0.3),
            ("strongly positive", 0.3),
        ]
    )
    assert result.intensity == frozenset({"SP"})
    assert result.belief == pytest.approx(0.51)
    assert result.conflict == pytest.approx(0.0)


def test_three_source_conflict_is_the_last_pairwise_step():
    """Thesis §5.3.2: a third evidence is combined with the prior bpa; reported κ is that step's K."""
    first = simple_support(frozenset({"SP"}), 0.9)
    second = simple_support(frozenset({"WN"}), 0.8)
    third = simple_support(frozenset({"SP"}), 0.7)
    prior, _ignored_conflict = combine_bpas([first, second])
    _, last_step_conflict = combine_bpas([prior, third])
    _, conflict = combine_bpas([first, second, third])
    assert conflict == pytest.approx(last_step_conflict)


def test_trace_preserves_every_pairwise_conflict_and_normalization():
    trace = trace_effect(
        [
            ("positive - strongly positive", 0.65),
            ("weakly positive - positive", 0.4),
            ("strongly positive", 0.9),
        ],
        selection_policy=HypothesisSelectionPolicy.SANTOS_2015,
    )

    assert [step.conflict for step in trace.steps] == pytest.approx([0.0, 0.36])
    assert [step.normalization_factor for step in trace.steps] == pytest.approx([1.0, 0.64])
    assert trace.mean_conflict == pytest.approx(0.18)
    assert trace.result.intensity == frozenset({"SP"})
    assert trace.result.belief == pytest.approx(0.84, abs=0.005)
    assert trace.to_dict() == trace.to_dict()


def test_combination_rejects_total_conflict():
    with pytest.raises(ValueError, match="Total conflict"):
        combine_bpas(
            [
                simple_support(frozenset({"SN"}), 1.0),
                simple_support(frozenset({"SP"}), 1.0),
            ]
        )


def test_combination_preserves_total_mass():
    combined, _conflict = combine_bpas(
        [
            simple_support(frozenset({"SN"}), 0.7),
            simple_support(frozenset({"I"}), 0.4),
            simple_support(frozenset({"SP"}), 0.2),
        ]
    )
    assert sum(combined.values()) == pytest.approx(1.0)


def test_unknown_intensity_label_is_rejected():
    with pytest.raises(ValueError, match="Unknown effect intensity"):
        combine_effect([("very positive", 0.5)])


def test_santos_tie_break_prefers_earlier_ssm_interval():
    masses = {
        frozenset({"SN"}): 0.4,
        frozenset({"SP"}): 0.4,
        THETA: 0.2,
    }
    hypothesis, selected_belief = select_hypothesis(
        masses,
        policy=HypothesisSelectionPolicy.SANTOS_2015,
    )
    assert hypothesis == frozenset({"SN"})
    assert selected_belief == pytest.approx(0.4)


def test_santos_selects_highest_belief_root_before_recursive_descent():
    positive_root = frozenset({"I", "WP", "P", "SP"})
    negative_root = frozenset({"SN", "N", "WN", "I"})
    masses = {
        frozenset({"SP"}): 0.4,
        positive_root: 0.06,
        negative_root: 0.44,
        THETA: 0.1,
    }

    hypothesis, selected_belief = select_hypothesis(
        masses,
        policy=HypothesisSelectionPolicy.SANTOS_2015,
    )

    assert hypothesis == frozenset({"SP"})
    assert selected_belief == pytest.approx(0.4)


def test_santos_trace_records_each_recursive_threshold_decision():
    trace = trace_effect(
        [
            ("positive", 0.191),
            ("positive", 0.191),
            ("strongly positive", 0.572),
        ],
        selection_policy=HypothesisSelectionPolicy.SANTOS_2015,
    )

    assert trace.selection_steps[0].parent == ("I", "WP", "P", "SP")
    assert trace.selection_steps[-1].parent == ("P", "SP")
    assert trace.selection_steps[-1].chosen_child == ("SP",)
    assert trace.selection_steps[-1].descended is False
    assert trace.result.intensity == frozenset({"P", "SP"})


def test_combine_bpas_matches_santos_thesis_structure_table_5():
    """Independent literals from Santos 2015 thesis Table 5 (structure effect, K = 0)."""
    po_sp = frozenset({"P", "SP"})
    wp_po = frozenset({"WP", "P"})
    po = frozenset({"P"})
    combined, conflict = combine_bpas(
        [
            {po_sp: 0.65, THETA: 0.35},
            {wp_po: 0.4, THETA: 0.6},
        ]
    )
    assert conflict == pytest.approx(0.0)
    assert combined[po] == pytest.approx(0.26)
    assert combined[po_sp] == pytest.approx(0.39)
    assert combined[wp_po] == pytest.approx(0.14)
    assert combined[THETA] == pytest.approx(0.21)
    hypothesis, selected_belief = select_hypothesis(combined)
    assert hypothesis == po_sp
    assert selected_belief == pytest.approx(0.65)


def test_combine_bpas_reports_last_pairwise_conflict_from_thesis_table_6():
    """Santos 2015 thesis Table 6: third evidence m({SP})=0.9 yields κ = 0.36 and Bel({SP})=0.84."""
    po_sp = frozenset({"P", "SP"})
    wp_po = frozenset({"WP", "P"})
    sp = frozenset({"SP"})
    combined, conflict = combine_bpas(
        [
            {po_sp: 0.65, THETA: 0.35},
            {wp_po: 0.4, THETA: 0.6},
            {sp: 0.9, THETA: 0.1},
        ]
    )
    assert conflict == pytest.approx(0.36)
    hypothesis, selected_belief = select_hypothesis(combined)
    assert hypothesis == sp
    assert selected_belief == pytest.approx(0.84, abs=0.005)


def test_reconcile_intensities_returns_intersection():
    assert reconcile_intensities(frozenset({"WN", "I"}), frozenset({"I"})) == frozenset({"I"})
    assert reconcile_intensities(frozenset({"P", "SP"}), frozenset({"SP"})) == frozenset({"SP"})
    assert reconcile_intensities(frozenset({"I", "WP"}), frozenset({"I", "WP"})) == frozenset({"I", "WP"})


def test_reconcile_intensities_returns_none_when_intersection_empty():
    assert reconcile_intensities(frozenset({"SP"}), frozenset({"P"})) is None
