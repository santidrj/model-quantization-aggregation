"""Dempster–Shafer combination of effect intensities (SSM frame of discernment)."""

import pytest

from src.dempster_shafer import (
    THETA,
    combine_bpas,
    combine_effect,
    intensity_to_hypothesis,
    select_hypothesis,
    simple_support,
)


def test_intensity_label_maps_to_adjacent_atoms():
    assert intensity_to_hypothesis("strongly positive") == frozenset({"SP"})
    assert intensity_to_hypothesis("indifferent - weakly positive") == frozenset({"IF", "WP"})
    assert intensity_to_hypothesis("negative - weakly negative") == frozenset({"NE", "WN"})
    assert intensity_to_hypothesis("weakly negative - indifferent") == frozenset({"WN", "IF"})


def test_simple_support_puts_remainder_on_theta():
    hypothesis = frozenset({"SP"})
    masses = simple_support(hypothesis, 0.3)
    assert masses[hypothesis] == 0.3  # noqa: PLR2004
    assert masses[THETA] == 0.7  # noqa: PLR2004


def test_combine_bpas_matches_santos_kanban_worked_example():
    """Independent literals from Santos et al. 2018, team-cohesion combination table."""
    wn_if = frozenset({"WN", "IF"})
    po_sp = frozenset({"PO", "SP"})
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
    compound = frozenset({"WN", "IF"})
    masses = {compound: 0.99, THETA: 0.01}
    hypothesis, belief = select_hypothesis(masses)
    assert hypothesis == compound
    assert belief == pytest.approx(0.99)


def test_select_hypothesis_discards_compound_when_one_simple_holds_75_percent():
    indifferent = frozenset({"IF"})
    wn = frozenset({"WN"})
    compound = frozenset({"WN", "IF"})
    masses = {indifferent: 0.8, wn: 0.05, compound: 0.05, THETA: 0.1}
    hypothesis, selected_belief = select_hypothesis(masses)
    assert hypothesis == indifferent
    assert selected_belief == pytest.approx(0.8)


def test_select_hypothesis_keeps_focal_compound_when_neither_simple_dominates():
    wn = frozenset({"WN"})
    indifferent = frozenset({"IF"})
    compound = frozenset({"WN", "IF"})
    masses = {wn: 0.4, indifferent: 0.4, compound: 0.15, THETA: 0.05}
    hypothesis, selected_belief = select_hypothesis(masses)
    assert hypothesis == compound
    assert selected_belief == pytest.approx(0.95)


def test_select_hypothesis_ignores_non_focal_compounds():
    sp = frozenset({"SP"})
    po = frozenset({"PO"})
    masses = {sp: 0.47, po: 0.18, THETA: 0.35}
    hypothesis, selected_belief = select_hypothesis(masses)
    assert hypothesis == sp
    assert selected_belief == pytest.approx(0.47)


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


def test_combine_bpas_matches_santos_thesis_structure_table_5():
    """Independent literals from Santos 2016 thesis Table 5 (structure effect, K = 0)."""
    po_sp = frozenset({"PO", "SP"})
    wp_po = frozenset({"WP", "PO"})
    po = frozenset({"PO"})
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
    """Santos 2016 thesis Table 6: third evidence m({SP})=0.9 yields κ = 0.36 and Bel({SP})=0.84."""
    po_sp = frozenset({"PO", "SP"})
    wp_po = frozenset({"WP", "PO"})
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
