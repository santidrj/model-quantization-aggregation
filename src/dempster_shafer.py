"""Dempster–Shafer combination over the SSM effect-intensity frame of discernment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

ATOM_BY_LABEL = {
    "strongly negative": "SN",
    "negative": "NE",
    "weakly negative": "WN",
    "indifferent": "IF",
    "weakly positive": "WP",
    "positive": "PO",
    "strongly positive": "SP",
}

ATOMS: tuple[str, ...] = ("SN", "NE", "WN", "IF", "WP", "PO", "SP")
THETA: frozenset[str] = frozenset(ATOMS)
ADJACENT_COMPOUNDS: tuple[frozenset[str], ...] = tuple(
    frozenset({ATOMS[index], ATOMS[index + 1]}) for index in range(len(ATOMS) - 1)
)
COMPOUND_BELIEF_SHARE = 0.75


class HypothesisSelectionPolicy(Enum):
    """Policy for turning a combined mass function into one reported intensity."""

    EVIDENCE_FACTORY_COMPAT = "evidence_factory_compat"
    SANTOS_2015 = "santos_2015"


def intensity_to_hypothesis(label: str) -> frozenset[str]:
    """Map an effect-intensity label to a set of Likert atoms."""
    parts = [part.strip() for part in label.split(" - ")]
    try:
        return frozenset(ATOM_BY_LABEL[part] for part in parts)
    except KeyError as exc:
        raise ValueError(f"Unknown effect intensity label: {label!r}") from exc


MassFunction = dict[frozenset[str], float]


def _ordered_atoms(hypothesis: frozenset[str]) -> tuple[str, ...]:
    return tuple(atom for atom in ATOMS if atom in hypothesis)


@dataclass(frozen=True)
class MassEntry:
    """One deterministic, serializable entry in a traced mass function."""

    hypothesis: tuple[str, ...]
    mass: float

    def to_dict(self) -> dict[str, object]:
        return {"hypothesis": list(self.hypothesis), "mass": self.mass}


@dataclass(frozen=True)
class CombinationStep:
    """One pairwise Dempster combination before and after normalization."""

    step: int
    incoming_piece_index: int
    conflict: float
    normalization_factor: float
    unnormalized_masses: tuple[MassEntry, ...]
    normalized_masses: tuple[MassEntry, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "incoming_piece_index": self.incoming_piece_index,
            "conflict": self.conflict,
            "normalization_factor": self.normalization_factor,
            "unnormalized_masses": [entry.to_dict() for entry in self.unnormalized_masses],
            "normalized_masses": [entry.to_dict() for entry in self.normalized_masses],
        }


def _mass_entries(masses: MassFunction) -> tuple[MassEntry, ...]:
    return tuple(
        MassEntry(_ordered_atoms(hypothesis), mass)
        for hypothesis, mass in sorted(
            masses.items(),
            key=lambda item: (tuple(ATOMS.index(atom) for atom in _ordered_atoms(item[0])), len(item[0])),
        )
    )


def simple_support(hypothesis: frozenset[str], mass: float) -> MassFunction:
    """Assign `mass` to `hypothesis` and the remainder to Θ."""
    if not 0.0 <= mass <= 1.0:
        raise ValueError(f"Mass must be in [0, 1], got {mass}")
    masses: MassFunction = {hypothesis: mass}
    remainder = 1.0 - mass
    if remainder:
        masses[THETA] = remainder
    return masses


def _conjunctive_combine(left: MassFunction, right: MassFunction) -> MassFunction:
    combined: MassFunction = {}
    for left_set, left_mass in left.items():
        for right_set, right_mass in right.items():
            intersection = left_set & right_set
            combined[intersection] = combined.get(intersection, 0.0) + left_mass * right_mass
    return combined


def _combine_bpas_with_steps(bpas: list[MassFunction]) -> tuple[MassFunction, tuple[CombinationStep, ...]]:
    if not bpas:
        raise ValueError("At least one mass function is required")
    accumulator = dict(bpas[0])
    steps: list[CombinationStep] = []
    for piece_index, other in enumerate(bpas[1:], start=2):
        raw = _conjunctive_combine(accumulator, other)
        conflict = raw.get(frozenset(), 0.0)
        if conflict >= 1.0:
            raise ValueError("Total conflict: Dempster combination is undefined")
        scale = 1.0 - conflict
        normalized = {hypothesis: mass / scale for hypothesis, mass in raw.items() if hypothesis and mass}
        steps.append(
            CombinationStep(
                step=len(steps) + 1,
                incoming_piece_index=piece_index,
                conflict=conflict,
                normalization_factor=scale,
                unnormalized_masses=_mass_entries(raw),
                normalized_masses=_mass_entries(normalized),
            )
        )
        accumulator = normalized
    return accumulator, tuple(steps)


def combine_bpas(bpas: list[MassFunction]) -> tuple[MassFunction, float]:
    """Combine mass functions pairwise with Dempster's rule.

    Evidence Factory and Santos (2015, §5.3.2) combine two bpas at a time and
    renormalize after each step. The returned conflict is that last step's
    K = m(empty set), not the n-fold empty-set mass of combining all sources
    in one product.
    """
    accumulator, steps = _combine_bpas_with_steps(bpas)
    conflict = steps[-1].conflict if steps else 0.0
    return accumulator, conflict


def belief(masses: MassFunction, hypothesis: frozenset[str]) -> float:
    """Bel(A) = sum of m(B) for B ⊆ A."""
    return sum(mass for support, mass in masses.items() if support <= hypothesis)


def _evidence_factory_hypothesis(masses: MassFunction) -> tuple[frozenset[str], float]:
    """Reproduce Evidence Factory's focal-adjacent selection behavior.

    This compatibility restriction is not part of Santos's selection rule.
    """
    simple_beliefs = {frozenset({atom}): belief(masses, frozenset({atom})) for atom in ATOMS}
    candidates: dict[frozenset[str], float] = {hyp: bel for hyp, bel in simple_beliefs.items() if bel > 0}
    for compound in ADJACENT_COMPOUNDS:
        if masses.get(compound, 0.0) <= 0.0:
            continue
        compound_belief = belief(masses, compound)
        dominant = max(simple_beliefs[frozenset({atom})] for atom in compound)
        if dominant >= COMPOUND_BELIEF_SHARE * compound_belief:
            continue
        candidates[compound] = compound_belief
    if not candidates:
        raise ValueError("No intensity hypothesis has positive belief")
    selected = max(candidates, key=lambda hypothesis: (candidates[hypothesis], -len(hypothesis)))
    return selected, candidates[selected]


def _interval(start: int, stop: int) -> frozenset[str]:
    return frozenset(ATOMS[start:stop])


def _hypothesis_rank(masses: MassFunction, hypothesis: frozenset[str]) -> tuple[float, int, int]:
    start = min(ATOMS.index(atom) for atom in hypothesis)
    return (belief(masses, hypothesis), -len(hypothesis), -start)


@dataclass(frozen=True)
class SelectionStep:
    """One recursive Santos 75% decision."""

    parent: tuple[str, ...]
    parent_belief: float
    children: tuple[MassEntry, ...]
    chosen_child: tuple[str, ...]
    threshold: float
    descended: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "parent": list(self.parent),
            "parent_belief": self.parent_belief,
            "children": [child.to_dict() for child in self.children],
            "chosen_child": list(self.chosen_child),
            "threshold": self.threshold,
            "descended": self.descended,
        }


def _descend_santos_interval(
    masses: MassFunction,
    hypothesis: frozenset[str],
) -> tuple[frozenset[str], tuple[SelectionStep, ...]]:
    """Apply the recursive 75% specificity rule in Santos (2015, Figure 16)."""
    current = hypothesis
    steps: list[SelectionStep] = []
    while len(current) > 1:
        indices = sorted(ATOMS.index(atom) for atom in current)
        children = (_interval(indices[0] + 1, indices[-1] + 1), _interval(indices[0], indices[-1]))
        child = max(children, key=lambda candidate: _hypothesis_rank(masses, candidate))
        threshold = COMPOUND_BELIEF_SHARE * belief(masses, current)
        descended = belief(masses, child) >= threshold
        steps.append(
            SelectionStep(
                parent=_ordered_atoms(current),
                parent_belief=belief(masses, current),
                children=_mass_entries({candidate: belief(masses, candidate) for candidate in children}),
                chosen_child=_ordered_atoms(child),
                threshold=threshold,
                descended=descended,
            )
        )
        if not descended:
            break
        current = child
    return current, tuple(steps)


def _santos_2015_selection(
    masses: MassFunction,
) -> tuple[frozenset[str], float, tuple[SelectionStep, ...]]:
    """Select a contiguous, direction-compatible interval per Santos (2015, pp. 95–97)."""
    roots = (_interval(0, 4), _interval(3, 7))
    root = max(roots, key=lambda hypothesis: _hypothesis_rank(masses, hypothesis))
    selected, steps = _descend_santos_interval(masses, root)
    selected_belief = belief(masses, selected)
    if selected_belief <= 0:
        raise ValueError("No intensity hypothesis has positive belief")
    return selected, selected_belief, steps


def _santos_2015_hypothesis(masses: MassFunction) -> tuple[frozenset[str], float]:
    selected, selected_belief, _steps = _santos_2015_selection(masses)
    return selected, selected_belief


def select_hypothesis(
    masses: MassFunction,
    *,
    policy: HypothesisSelectionPolicy = HypothesisSelectionPolicy.EVIDENCE_FACTORY_COMPAT,
) -> tuple[frozenset[str], float]:
    """Select the reported SSM intensity under an explicit interpretation policy.

    The default preserves the historical Evidence Factory-compatible interface.
    Santos's literature rule recursively descends contiguous intervals that do not
    span both negative and positive atoms.
    """
    if policy is HypothesisSelectionPolicy.EVIDENCE_FACTORY_COMPAT:
        return _evidence_factory_hypothesis(masses)
    if policy is HypothesisSelectionPolicy.SANTOS_2015:
        return _santos_2015_hypothesis(masses)
    raise ValueError(f"Unknown hypothesis selection policy: {policy}")


def format_intensity(hypothesis: frozenset[str]) -> str:
    """Format an intensity set as a simple atom or `{A, B}` in Likert order."""
    ordered = [atom for atom in ATOMS if atom in hypothesis]
    if len(ordered) == 1:
        return ordered[0]
    return "{" + ", ".join(ordered) + "}"


@dataclass(frozen=True)
class CombinedEffect:
    """Published SSM effect result; conflict is the final pairwise K."""

    intensity: frozenset[str]
    belief: float
    conflict: float


@dataclass(frozen=True)
class EvidencePieceTrace:
    """One ordered simple-support input to an effect trace."""

    index: int
    intensity_label: str
    hypothesis: tuple[str, ...]
    mass: float

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "intensity_label": self.intensity_label,
            "hypothesis": list(self.hypothesis),
            "mass": self.mass,
        }


@dataclass(frozen=True)
class EffectCombinationTrace:
    """Deterministic audit record for one ordered effect combination."""

    selection_policy: HypothesisSelectionPolicy
    pieces: tuple[EvidencePieceTrace, ...]
    steps: tuple[CombinationStep, ...]
    final_masses: tuple[MassEntry, ...]
    selection_beliefs: tuple[MassEntry, ...]
    selection_steps: tuple[SelectionStep, ...]
    result: CombinedEffect

    @property
    def mean_conflict(self) -> float:
        """Arithmetic mean of pairwise conflicts, or zero for one source."""
        if not self.steps:
            return 0.0
        return sum(step.conflict for step in self.steps) / len(self.steps)

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""
        return {
            "selection_policy": self.selection_policy.value,
            "pieces": [piece.to_dict() for piece in self.pieces],
            "steps": [step.to_dict() for step in self.steps],
            "mean_conflict": self.mean_conflict,
            "final_masses": [entry.to_dict() for entry in self.final_masses],
            "selection_beliefs": [entry.to_dict() for entry in self.selection_beliefs],
            "selection_steps": [step.to_dict() for step in self.selection_steps],
            "tie_break": ["higher belief", "greater specificity", "earlier SSM-scale position"],
            "result": {
                "intensity": list(_ordered_atoms(self.result.intensity)),
                "belief": self.result.belief,
                "final_step_conflict": self.result.conflict,
            },
        }


def _selection_candidates(
    masses: MassFunction,
    policy: HypothesisSelectionPolicy,
) -> tuple[frozenset[str], ...]:
    if policy is HypothesisSelectionPolicy.EVIDENCE_FACTORY_COMPAT:
        simples = tuple(frozenset({atom}) for atom in ATOMS)
        focal_compounds = tuple(compound for compound in ADJACENT_COMPOUNDS if masses.get(compound, 0.0) > 0.0)
        return simples + focal_compounds
    intervals = {
        _interval(start, stop)
        for root_start, root_stop in ((0, 4), (3, 7))
        for start in range(root_start, root_stop)
        for stop in range(start + 1, root_stop + 1)
    }
    return tuple(sorted(intervals, key=lambda hypothesis: (len(hypothesis), _ordered_atoms(hypothesis))))


def combine_effect(
    pieces: list[tuple[str, float]],
    *,
    selection_policy: HypothesisSelectionPolicy = HypothesisSelectionPolicy.EVIDENCE_FACTORY_COMPAT,
) -> CombinedEffect:
    """Combine ordered ``(intensity label, mass)`` pieces into one SSM effect.

    Callers must choose ``SANTOS_2015`` for literature-based publication
    comparisons. The default remains Evidence Factory-compatible.
    """
    if not pieces:
        raise ValueError("At least one evidence piece is required")
    bpas = [simple_support(intensity_to_hypothesis(label), mass) for label, mass in pieces]
    masses, conflict = combine_bpas(bpas)
    intensity, selected_belief = select_hypothesis(masses, policy=selection_policy)
    return CombinedEffect(intensity=intensity, belief=selected_belief, conflict=conflict)


def trace_effect(
    pieces: list[tuple[str, float]],
    *,
    selection_policy: HypothesisSelectionPolicy,
) -> EffectCombinationTrace:
    """Return the full ordered D-S computation trace for one effect.

    Selection is mandatory at the audit seam so traces cannot silently mix
    literature and compatibility interpretations.
    """
    if not pieces:
        raise ValueError("At least one evidence piece is required")
    piece_traces = tuple(
        EvidencePieceTrace(
            index=index,
            intensity_label=label,
            hypothesis=_ordered_atoms(intensity_to_hypothesis(label)),
            mass=mass,
        )
        for index, (label, mass) in enumerate(pieces, start=1)
    )
    bpas = [simple_support(frozenset(piece.hypothesis), piece.mass) for piece in piece_traces]
    masses, steps = _combine_bpas_with_steps(bpas)
    if selection_policy is HypothesisSelectionPolicy.SANTOS_2015:
        intensity, selected_belief, selection_steps = _santos_2015_selection(masses)
    else:
        intensity, selected_belief = select_hypothesis(masses, policy=selection_policy)
        selection_steps = ()
    final_conflict = steps[-1].conflict if steps else 0.0
    result = CombinedEffect(intensity, selected_belief, final_conflict)
    candidate_beliefs = {
        candidate: belief(masses, candidate)
        for candidate in _selection_candidates(masses, selection_policy)
        if belief(masses, candidate) > 0.0
    }
    return EffectCombinationTrace(
        selection_policy=selection_policy,
        pieces=piece_traces,
        steps=steps,
        final_masses=_mass_entries(masses),
        selection_beliefs=_mass_entries(candidate_beliefs),
        selection_steps=selection_steps,
        result=result,
    )
