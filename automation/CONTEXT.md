# EvidenceFactory Aggregation

Automated merging of evidence models in the EvidenceFactory web application, deciding Add, Remove, or Join for highlighted model elements.

## Language

**Aggregated model**:
The current evidence model being curated in the aggregation editor. Model elements belonging to it are marked with origin suffix `(1)`.
_Avoid_: Base model, target model, current model (as informal synonyms in docs)

**Incoming evidence model**:
The evidence model being incorporated into the aggregated model in this aggregation turn. Model elements belonging to it are marked with origin suffix `(2)`.
_Avoid_: Source model, new model, evidence model (when the incoming side specifically is meant)

**Model origin**:
Which side a model element belongs to in the editor: aggregated model `(1)` or incoming evidence model `(2)`.
_Avoid_: Side, source, parent model (when the `(1)`/`(2)` distinction is meant)

**Eligible element**:
A model element rendered in red because it is a conflict between the two evidences currently being matched. Gray elements are unanalyzed children under unresolved differences and are not yet actionable; black elements are matching or already resolved. Only red eligible elements may be clicked for Add, Remove, or Join.
_Avoid_: Red element (as the domain term — highlight color is a UI cue), gray element (when unanalyzed-not-yet-actionable is meant), difference, unmatched element

**Aggregation turn**:
The work of resolving every eligible element for the currently highlighted incoming evidence on the left sidebar. The automation processes them in fixed phase order: Remove, then Join, then Add. When automatic policy cannot act, or after autos leave residuals, the automation pauses at a human decision point, applies the reviewer's choice, and resumes until no eligible elements remain for that highlighted evidence. The left-sidebar highlight then advances automatically to the next unresolved evidence; the editor URL does not change.
_Avoid_: Session, run, pass (when a single incoming-evidence incorporation is meant)

**Named aggregation group**:
One EvidenceFactory aggregation group addressed by a stable human name and its `selectedAggregationId` under synthesis `244422`. The named set is Full aggregation v2, Sensitivity analysis (n<6), PTQ from FP32 to w-int8, a-int8, and Test aggregation.
_Avoid_: Final aggregation (ambiguous), anonymous numeric ID as the only identity when a name exists

**Full aggregation v2**:
The named aggregation group for the full evidence synthesis (`selectedAggregationId=282042`).
_Avoid_: Final aggregation, production aggregation (when this specific group is meant)

**Sensitivity analysis (n<6)**:
The named aggregation group for the sensitivity analysis restricted to n<6 (`selectedAggregationId=327146`).
_Avoid_: Sensitivity analysis (unqualified), n<6 run (when this EvidenceFactory group is meant)

**PTQ from FP32 to w-int8, a-int8**:
The named aggregation group for the PTQ subgroup from FP32 baseline to `w-int8, a-int8` (`selectedAggregationId=281366`).
_Avoid_: PTQ aggregation (unqualified), int8 subgroup (when this EvidenceFactory group is meant)

**Test aggregation**:
The disposable named aggregation group used as a safe sandbox (`selectedAggregationId=304080`). Prefer it when validating automation behavior; do not treat it as one of the analysis targets.
_Avoid_: Real aggregation, Full aggregation v2 (when the sandbox group is meant)

**Automation run**:
One execution of the automation against exactly one named aggregation group (selected at start). Opens that group's aggregation-group overview URL, clicks **Update evidence aggregation** once (navigates to `/evidenceAggregator/{aggregationGroupId}`), waits for idle, then resolves eligible red elements as pairs auto-advance in the left list. When `#redirectButton` (**view aggregated model**) appears, clicks it, waits for `/evidenceAggregator/submitResult` and the result navigation, and ends.
_Avoid_: Batch of URLs, manual sidebar clicking (when auto-advance is the rule)

**Aggregation-group overview**:
The EvidenceFactory view (on the aggregation URL) that lists evidence in the aggregation group and exposes **Update evidence aggregation**. It is the run entry state on that same URL, before theoretical-structure matching.
_Avoid_: Matching editor, synthesis home (when this group overview state is meant)


**Model element**:
A node in an evidence model shown in the EvidenceFactory aggregation editor. Its kind is shown as a UI pill/tag such as archetype, cause, contextual aspect, or effect, and that type metadata is the source of truth for kind — not display name or tree path alone.
_Avoid_: Node, item, concept (when the typed UI element is meant)

**Archetype**:
A model element whose UI kind pill is archetype. Top-level structural containers in the aggregated model (for example Technology, System, Activity, Actor). Not auto-Removed or auto-Added by the cleanup/Add rules.
_Avoid_: Root, category (when the typed archetype element is meant)

**Cause**:
A model element whose UI kind pill is cause. In the curated model, Model quantization appears as a cause under the Technology archetype. Not auto-Added; not covered by the contextual-aspect Remove rules unless separately decided.
_Avoid_: Effect, driver (when the typed cause element is meant)

**Effect**:
A model element whose EvidenceFactory type/metadata identifies it as an Effect. Eligible for Add only in the Add phase, and only when no semantically equivalent Effect already exists in the aggregated model.
_Avoid_: Contextual aspect, outcome (when the typed Effect element is meant)

**Contextual aspect**:
A model element whose EvidenceFactory type/metadata identifies it as a contextual aspect. Subject to Remove-phase cleanup rules; never eligible for Add.
_Avoid_: Effect, context factor (when the typed contextual-aspect element is meant)

**Remove**:
The aggregation operation that deletes an unnecessary eligible element so the models can be normalized. Runs before Join and Add in every aggregation turn.
_Avoid_: Delete, discard (when the EvidenceFactory Remove action is meant)

**Join**:
The aggregation operation that merges two semantically equivalent eligible elements into one concept, keeping the canonical term. Legal only when both the alias and the canonical term are currently eligible and share the same element kind. Runs after Remove and before Add. When a Remove-cleanup candidate also has a legal Join partner (for example Large Language Model ↔ DL model under System), Join wins — the automation skips Remove for that element and joins instead.
_Avoid_: Merge, map, replace (when the EvidenceFactory Join action is meant)

**Join pair**:
Two eligible model elements of the same kind whose labels are linked by the semantic-equivalence map (alias and canonical term). Absence of either side means Join is not legal for that alias.
_Avoid_: Match, candidate pair (when this Join gate is meant)

**Add**:
The aggregation operation that affirms an eligible Effect should appear in the final aggregated model. It applies to not-in-common Effects on either model origin: Add on `(2)` brings the incoming Effect into the aggregated model; Add on `(1)` confirms an aggregated-only Effect remains in the final aggregated model. Runs last in the automation's phase order. EvidenceFactory may offer Add, Remove, or Join on Effects at any time, including Effects touched in a previous turn; the automation still decides from the current eligible set only. Auto-Adds an eligible Effect from either origin when no semantically equivalent Effect already exists in the aggregated model. An orphan alias Effect (mapped alias eligible, but no Join pair because the canonical term is not eligible) is a human decision point — never auto-Added or auto-skipped.
_Avoid_: Insert, import (when the EvidenceFactory Add action is meant)

**Final aggregated model**:
The aggregated model as it should stand after differences are fully resolved and **view aggregated model** has been invoked to generate the aggregated evidence model. Add affirms membership in this result; Remove and Join reshape what remains in it.
_Avoid_: Aggregated model (when the in-progress editor state mid-turn is meant rather than the turn's intended result)

**Human decision point**:
A situation the automation must not resolve on its own and instead pauses for the reviewer. Includes orphan alias Effects with no legal Join pair, and every residual eligible element (Effect or contextual aspect) that remains after automatic Remove, Join, and Add have exhausted safe policy actions. The reviewer answers in the terminal (Add / Remove / Join / abort) while the headed browser remains open on the selected element; the automation executes the chosen action.
_Avoid_: Error, failure, manual login (when an aggregation-policy pause is meant)

**Semantic equivalence**:
A configured relationship between two model-element labels that names the same concept for Join and for duplicate detection on Add. Equivalence is never inferred by heuristics outside the map.
_Avoid_: Fuzzy match, similarity, synonym detection (as automatic behavior)

**Canonical term**:
The more generic label stored as the value in the semantic-equivalence map. On Join, the kept concept is always the canonical term; the alias is the discarded label.
_Avoid_: Preferred term, target label, generic term (when the mapped canonical value is meant)

**Model quantization cause**:
The cause element labeled Model quantization under the Technology archetype. Every eligible contextual aspect under it is unnecessary and must be Removed, on either model origin.
_Avoid_: Model Quantization (when the research meta-analysis topic is meant rather than this cause subtree); Model Quantization section

**System archetype**:
The evidence-model archetype subtree named System. Every eligible first-level contextual aspect under it is unnecessary and must be Removed, except the protected contextual aspect DL model.
_Avoid_: System (unqualified), root archetype (when this specific subtree is meant)

**Protected contextual aspect**:
The contextual aspect label DL model under the System archetype. It is never Removed by the cleanup rules.
_Avoid_: Canonical term (when protection from Remove is meant rather than Join generality)

**Update evidence aggregation**:
The overview control (`#btnAggregateEvidence`) that starts matching by navigating to `/evidenceAggregator/{aggregationGroupId}`. The UI label is **Aggregate evidence** the first time aggregation runs for that group, and **Update evidence aggreagation** (misspelled) thereafter. The automation clicks it once at run start on the aggregation-group overview, then waits for the aggregator page to load and become idle. Distinct from **view aggregated model**.
_Avoid_: Refresh, reload, sync, view aggregated model, Aggregate evidence (as a separate control — it is the same button)

**View aggregated model**:
The aggregator control (`#redirectButton`) that appears when `numberOfDifferentNodes === 0`. Clicking it POSTs `/evidenceAggregator/submitResult` and generates the aggregated evidence model (then navigates to the aggregation result page). Distinct from Update evidence aggregation.
_Avoid_: Update evidence aggregation, finish, submit (when this specific generate action is meant)

**Eligible element**:
A model element rendered in red because it is a conflict between the two evidences currently being matched. Gray elements are unanalyzed children under unresolved differences and are not yet actionable; black elements are matching or already resolved. Only red eligible elements may be clicked for Add, Remove, or Join.
_Avoid_: Red element (as the domain term — highlight color is a UI cue), gray element (when unanalyzed-not-yet-actionable is meant), difference, unmatched element

**Tree order**:
The depth-first document order of model elements in the aggregation editor. Within each phase, the automation always selects the next target by re-scanning eligible elements in tree order after each completed action.
_Avoid_: Arbitrary order, click order, map order (when within-phase selection is meant)

**Expected test result**:
For the test aggregation (`selectedAggregationId=304080`), a successful run ends with no eligible (red) differences, an aggregated tree matching the curated shape (Technology → Model quantization (cause); System → DL model with the agreed Effects beneath it; Activity → Model operation; Actor → DL engineer), and a completed **view aggregated model** step that generates the aggregated evidence model.
_Avoid_: Real aggregation result (when the disposable test aggregation is meant)
