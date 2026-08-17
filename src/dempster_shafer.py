"""Dempster–Shafer combination over the SSM effect-intensity frame of discernment."""

from __future__ import annotations

from dataclasses import dataclass

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


def intensity_to_hypothesis(label: str) -> frozenset[str]:
    """Map an effect-intensity label to a set of Likert atoms."""
    parts = [part.strip() for part in label.split(" - ")]
    try:
        return frozenset(ATOM_BY_LABEL[part] for part in parts)
    except KeyError as exc:
        raise ValueError(f"Unknown effect intensity label: {label!r}") from exc


MassFunction = dict[frozenset[str], float]


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


def combine_bpas(bpas: list[MassFunction]) -> tuple[MassFunction, float]:
    """Combine mass functions with Dempster's n-fold rule. Returns (normalized m, conflict K)."""
    if not bpas:
        raise ValueError("At least one mass function is required")
    accumulator = dict(bpas[0])
    for other in bpas[1:]:
        accumulator = _conjunctive_combine(accumulator, other)
    conflict = accumulator.pop(frozenset(), 0.0)
    if conflict >= 1.0:
        raise ValueError("Total conflict: Dempster combination is undefined")
    scale = 1.0 - conflict
    normalized = {hypothesis: mass / scale for hypothesis, mass in accumulator.items() if mass}
    return normalized, conflict


def belief(masses: MassFunction, hypothesis: frozenset[str]) -> float:
    """Bel(A) = sum of m(B) for B ⊆ A."""
    return sum(mass for support, mass in masses.items() if support <= hypothesis)


def select_hypothesis(masses: MassFunction) -> tuple[frozenset[str], float]:
    """Pick the SSM intensity: max-belief among simples and focal adjacent compounds.

    A compound is discarded when its dominant constituent already holds at least 75% of
    the compound's belief, or when the compound received no mass of its own.
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


def format_intensity(hypothesis: frozenset[str]) -> str:
    """Format an intensity set as a simple atom or `{A, B}` in Likert order."""
    ordered = [atom for atom in ATOMS if atom in hypothesis]
    if len(ordered) == 1:
        return ordered[0]
    return "{" + ", ".join(ordered) + "}"


@dataclass(frozen=True)
class CombinedEffect:
    intensity: frozenset[str]
    belief: float
    conflict: float


def combine_effect(pieces: list[tuple[str, float]]) -> CombinedEffect:
    """Combine (intensity label, mass) pieces into one SSM effect."""
    if not pieces:
        raise ValueError("At least one evidence piece is required")
    bpas = [simple_support(intensity_to_hypothesis(label), mass) for label, mass in pieces]
    masses, conflict = combine_bpas(bpas)
    intensity, selected_belief = select_hypothesis(masses)
    return CombinedEffect(intensity=intensity, belief=selected_belief, conflict=conflict)
