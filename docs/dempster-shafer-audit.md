# Dempster–Shafer computation audit

This document lets a researcher trace the belief-assignment computation from processed study evidence to the appendix table, distinguish literature-defined mathematics from project policy, and review the executable evidence. It does not issue a publication-readiness verdict.

## Scope and terminology

The audited path begins with the stored **study belief** \(B\) and each evidence model's extracted effect intensity and discounted support mass \(B'\). Extraction from raw primary-study measurements is upstream and is linked here, but is not re-audited. The path ends at `reports/tables/belief-assignment.tex`.

The notation is deliberately unambiguous:

- \(B\): study belief.
- \(B'\): discounted support mass stored as an effect's `belief` in processed JSON.
- \(m\): a Dempster–Shafer mass function.
- \(\mathrm{Bel}(A)\): belief measure of hypothesis \(A\).
- \(K\): conflict mass \(m(\emptyset)\) at one pairwise combination step.
- final-step \(K\): the last pairwise conflict displayed in publication tables.

The complete domain glossary is in [`CONTEXT.md`](../CONTEXT.md). “Belief” by itself should not be used where one of the terms above is intended.

## Sources and authority

Claims below are classified as:

- **Literature-derived**: stated in a cited source.
- **Independently tested**: checked against source literals or a mathematical invariant.
- **EvidenceFactory-parity**: checked against aggregated evidence 329074.
- **Project policy**: a deliberate local interpretation or experimental assignment.
- **Unresolved**: evidence needed for stronger verification is unavailable.

Primary references:

1. Glenn Shafer, *A Mathematical Theory of Evidence*, Princeton University Press, 1976.
2. Enrique H. Ruspini, John D. Lowrance, and Thomas M. Strat, “Understanding evidential reasoning,” *International Journal of Approximate Reasoning* 6(3), 401–424, 1992, <https://doi.org/10.1016/0888-613X(92)90033-V>.
3. Isabelle Bloch, “Some aspects of Dempster–Shafer evidence theory for classification of multi-modality medical images taking partial volume effect into account,” *Pattern Recognition Letters* 17(8), 905–919, 1996, <https://doi.org/10.1016/0167-8655(96)00039-6>.
4. Paulo Sérgio Medeiros dos Santos, *Evidence Representation and Aggregation in Software Engineering Using Theoretical Structures and Belief Functions*, PhD thesis, Federal University of Rio de Janeiro, December 2015. Relevant locations are Chapter 3, printed pp. 32–40; §5.3.1, printed pp. 91–92; and §5.3.2 with Figure 16 and Tables 5–6, printed pp. 95–98 (PDF pp. 111–114).
5. Paulo Sérgio Medeiros dos Santos et al., “On the benefits and challenges of using kanban in software engineering: a structured synthesis study,” *Journal of Software Engineering Research and Development* 6(13), 2018, <https://doi.org/10.1186/s40411-018-0057-1>.

The local thesis PDF is intentionally excluded by `/PauloSergio_Thesis.pdf` in `.gitignore` because it is copyrighted reference material. Reviewers must obtain the source independently; this repository does not distribute it.

## Computation path

1. `src/data/papers/entities.py` supplies study belief \(B\).
2. `src/data/papers/knowledge_extraction.py` computes effect intensity, sample-size discount, variability discount, and \(B'\), then writes `data/processed/<paper>/effects_by_precision.json`.
3. `src/belief_assignment.py::load_evidence_models` joins processed records to Evidence Factory IDs and the 76-entry aggregation-turn order.
4. `assigned_mass` applies one of four belief assignments.
5. `pieces_for_effect` retains only models reporting the effect and orders them.
6. `src/dempster_shafer.py::combine_effect` constructs simple-support mass functions, combines them pairwise, and selects a reported hypothesis.
7. `reproduction_mismatches` checks EvidenceFactory-compatible results against evidence 329074.
8. `comparison_records` computes all four assignments with Santos hypothesis selection.
9. `write_belief_assignment_table` renders the appendix fragment.

Notebook [`5.3-belief-assignment.ipynb`](../notebooks/5.3-belief-assignment.ipynb) executes the gate, comparison, trace example, and table generation.

## Upstream discount and rounding

For sample-size reliability \(\alpha_n\), variability reliability \(\alpha_v\), and study belief \(B\):

\[
B' = B\alpha_n\alpha_v.
\]

`effects_by_precision.json` rounds \(\alpha_n\), \(\alpha_v\), their product, the Evidence Factory-compatible `p_value`, and \(B'\) to three decimals before local combination. The field name `p_value` is retained to match Evidence Factory. In this pipeline it is \(1-\alpha_n\alpha_v\), the discount share transferred toward \(\Theta\); it is not a conventional statistical p-value. A simple-support function made from \(B'\) has total residual mass \(1-B'\) on \(\Theta\).

Combination uses the stored three-decimal masses without further input rounding. `synthesis_row` rounds \(\mathrm{Bel}(A)\times100\) to an integer for the Evidence Factory gate. The LaTeX renderer rounds belief and conflict to two decimals. Rendered LaTeX is therefore presentation output, not a full-precision audit artifact.

## Mass assignments

For a study with \(N\) evidence models:

| Assignment | Mass on each carried effect | Classification | Executable evidence |
|---|---:|---|---|
| Published analogue | processed \(B'\) | EvidenceFactory-parity | `reproduction_mismatches`; RAM trace fixture |
| Undiscounted unsplit | \(B\) | Project policy | assignment tests |
| Mass-preserving split | \(1-(1-B)^{1/N}\) | Project policy, independently tested | root and recovery tests |
| Equitable split | \(B/N\) | Project policy, independently tested | assignment tests |

\(N\) is the study's total evidence-model count even when only some models report the effect. This is project policy. For agreeing simple supports, the mass-preserving split satisfies

\[
1-\left(1-\left[1-(1-B)^{1/N}\right]\right)^N=B.
\]

The equitable split preserves the arithmetic allocation \(\sum_i B/N=B\); it does not claim to preserve the Dempster combination result.

## Dempster–Shafer mechanics

The effect frame is

\[
\Theta=\{SN,NE,WN,IF,WP,PO,SP\}.
\]

### Simple support

For intensity hypothesis \(H\) and assigned mass \(s\):

\[
m(H)=s,\qquad m(\Theta)=1-s.
\]

**Literature-derived** and **independently tested**.

### Pairwise combination

For two mass functions:

\[
m_\cap(A)=\sum_{B\cap C=A}m_1(B)m_2(C),\qquad
K=m_\cap(\emptyset),
\]

\[
(m_1\oplus m_2)(A)=\frac{m_\cap(A)}{1-K},\quad A\ne\emptyset.
\]

Combination is undefined at total conflict \(K=1\). The whole normalized mass function—not only the selected hypothesis—is passed to the next pairwise step. These rules are **literature-derived** and checked against Santos (2015) Tables 5–6 and Santos et al. (2018) literals.

The source order is `(aggregation_index, numeric study ID, quantization method, precision configuration)`. `aggregation_index` comes from `data/processed/evidence-factory-aggregation-order.txt`; remaining fields are deterministic tie-breaks. The Evidence Factory tie behavior is **unresolved** because no step-by-step export is available.

### Belief measure

\[
\mathrm{Bel}(A)=\sum_{B\subseteq A}m(B).
\]

Plausibility \(\mathrm{Pl}(A)\) is part of the wider theory but is not required by this synthesis and is not implemented. No interface in this repository should imply plausibility support.

## Hypothesis-selection policies

Dempster–Shafer theory does not provide one universal decision rule for selecting a reported hypothesis. This repository therefore exposes two explicit policies.

### Santos hypothesis selection

**Literature-derived with a documented project tie-break.** Santos (2015), Figure 16 and printed p. 96, recursively narrows direction-compatible contiguous intervals. If the strongest contained interval contributes at least 75% of the current interval's belief, selection descends to it; otherwise the current interval is retained. Selection depends on \(\mathrm{Bel}(A)\), not direct \(m(A)\), so a non-focal interval may be selected.

Equal candidates are resolved by higher belief, then greater specificity, then earlier position on the fixed SSM scale. The final criterion is an **unresolved literature interpretation** and explicit deterministic **project policy**.

### Evidence Factory-compatible selection

**EvidenceFactory-parity and project policy.** This mode considers positive-belief singletons and adjacent two-atom compounds carrying direct mass, dropping a compound when a contained singleton contributes at least 75% of its belief. It remains the default for existing `combine_effect` and `synthesis_row` callers and is selected explicitly by `reproduction_mismatches`.

This direct-focal restriction is not stated by Santos (2015). It is retained only to reproduce Evidence Factory aggregated evidence 329074.

### Observed impact

Across 22 loaded effects and four assignments, Santos selection changes 20 of 88 selected results. Among the 11 publication effects, six local results change:

| Effect | Assignment | Evidence Factory-compatible | Santos (2015) |
|---|---|---|---|
| mAP | Published analogue | IF, 0.445767693779 | {NE, WN, IF}, 0.611065048266 |
| mAP | Mass-preserving | IF, 0.446566601353 | {NE, WN, IF}, 0.612267208078 |
| mAP | Equitable | IF, 0.383969472812 | {NE, WN, IF}, 0.538913254791 |
| RAM Usage | Published analogue | SP, 0.466575690070 | {PO, SP}, 0.650883924213 |
| RAM Usage | Mass-preserving | PO, 0.414674444297 | {PO, SP}, 0.801476350700 |
| RAM Usage | Equitable | SP, 0.460494820400 | {PO, SP}, 0.763653925392 |

The undiscounted results for mAP and RAM Usage do not change. Selection does not alter mass combination or conflict values.

## Trace interface and fixture

`trace_effect` requires an explicit selection policy and returns immutable ordered inputs, every unnormalized and normalized mass function, each \(K\), each normalization factor \(1-K\), mean conflict, candidate beliefs, the tie policy, and the final result. Santos traces additionally record the selected root's recursive path: each parent interval, both child beliefs, the 75% threshold, the chosen child, and whether descent continued.

`belief_assignment_trace` adds evidence-model provenance and emits both selector interpretations over one common combination. The committed fixture is:

`tests/fixtures/ram_usage_published_analogue_trace.json`

It demonstrates the same final mass function selecting `SP` under Evidence Factory compatibility and `{PO, SP}` under Santos. The fixture is compared structurally in the test suite, so ordering and full-precision values are deterministic.

## Verification map

| Review question | Evidence |
|---|---|
| Are intensity labels mapped to the intended frame? | `test_intensity_label_maps_to_adjacent_atoms` |
| Does simple support preserve total mass? | `test_simple_support_puts_remainder_on_theta`, `test_combination_preserves_total_mass` |
| Does pairwise combination match independent examples? | Santos 2015 Table 5/6 tests; Santos et al. 2018 test |
| Is total conflict rejected? | `test_combination_rejects_total_conflict` |
| Is the full conflict sequence retained? | `test_trace_preserves_every_pairwise_conflict_and_normalization` |
| Is Santos non-focal interval selection implemented? | `test_santos_2015_selection_uses_belief_of_non_focal_interval` |
| Is deterministic tie behavior visible? | `test_santos_tie_break_prefers_earlier_ssm_interval`; trace `tie_break` |
| Does the mass-preserving split recover \(B\) for agreeing models? | `test_mass_preserving_split_recovers_study_belief_when_all_models_agree` |
| Does the compatibility path reproduce evidence 329074? | three `test_published_analogue_matches_*` gates |
| Do all four comparisons use Santos selection? | `test_comparison_records_use_santos_2015_for_every_local_assignment` |
| Is the full trace stable? | `test_ram_usage_trace_matches_committed_fixture` |

## Known limitations

1. **Unresolved Evidence Factory trace:** evidence 329074 provides final reference literals, but no export of ordered intermediate masses, conflicts, or selection decisions. Output parity does not prove internal-process identity.
2. **Input precision:** combination starts from three-decimal processed values. Full pre-export precision cannot be recovered from committed JSON.
3. **Evidence model pairing:** Evidence Factory IDs are associated with processed rows by per-study order; the mapping is not independently exported from Evidence Factory.
4. **Order-label resolvers:** study-specific qualifier parsing maps the 76 textual turns to processed records. Tests verify the bijection and representative endpoints, not an external ID-by-ID trace.
5. **Santos ties:** the thesis does not fully specify equal-belief branch resolution. The deterministic project tie policy is disclosed above.
6. **Counterfactual assignments:** the three undiscounted/split variants cannot be run in Evidence Factory and are validated by formulas and invariants rather than external parity.
7. **Plausibility:** intentionally outside the implemented and audited computation.

Local mathematics can be reviewed independently against the literature. Exact equivalence to Evidence Factory's undocumented intermediate process remains unverified; only its final outputs are parity-tested.

## Reviewer checklist

Record observations without assigning an automated approval status.

- [ ] Confirm source citations, printed pages, and table literals.
- [ ] Match each equation above to the named interface and its tests.
- [ ] Confirm all four assignment definitions and the study-level meaning of \(N\).
- [ ] Inspect the three-decimal input boundary and presentation-only rounding.
- [ ] Execute the Evidence Factory compatibility gate and record any mismatch.
- [ ] Inspect mAP and RAM Usage under both selection policies.
- [ ] Inspect the RAM Usage fixture's ordered inputs, intermediate masses, \(K\) sequence, mean conflict, and final selections.
- [ ] Confirm the deterministic tie policy is acceptable as project policy.
- [ ] Review each known limitation and record whether more evidence is required.
- [ ] Record reviewer name, date, repository commit, commands, and notes.

## Reproduction commands

```bash
uv run pytest tests/test_dempster_shafer.py tests/test_belief_assignment.py
uv run pytest
PYTHONPATH="$PWD" uv run jupyter nbconvert --to notebook --execute \
  notebooks/5.3-belief-assignment.ipynb --output /tmp/5.3-belief-assignment.executed.ipynb
uv run ruff check .
uv run ruff format --check .
uv run deptry .
```

Pre-existing findings outside the audited modules should be reported separately rather than hidden or treated as D-S failures.
