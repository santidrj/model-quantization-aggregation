# EvidenceFactory Evidence Editor Sync

Writing locally computed effect intensity, discount residual, and supporting statistics onto evidence-model effects in Evidence Factory evidence editors.

## Language

**Evidence editor**:
The Evidence Factory page that edits one evidence model. Distinct from the aggregation editor, where incoming and aggregated models are merged.
_Avoid_: Aggregation editor; aggregator; evidenceAggregator

**Write-back**:
Setting intensity, discount residual, and effect comment on an evidence-model effect in the evidence editor from local processed evidence.
_Avoid_: Aggregation; Update evidence aggregation; scrape; pull

**Delta**:
An evidence-model effect whose evidence-editor intensity, discount residual, or effect comment differs from the local processed values. Write-back applies only to deltas.
_Avoid_: Full overwrite; every mapped effect (when only differences are meant)

**Plan**:
A write-back run that reports deltas and does not change the evidence editor. It live-reads every study whose mapping is intact, does not open a duplicated evidence-editor ID, and reports mapping-integrity faults.
_Avoid_: Dry-run (when this run is meant); apply; opening a duplicated ID to “see what happens”

**Apply**:
A write-back run that writes deltas into the evidence editor. It is refused for the whole corpus while mapping integrity fails. On the first failed write it stops; already-written effects stay (they will not be deltas on the next live-read).
_Avoid_: Plan; aggregation run; continuing after a failed write; rolling back earlier writes

**Incomplete effect**:
A local evidence-model effect that has no intensity or no improvement. Write-back skips it and leaves the editor unchanged.
_Avoid_: Delta (an incomplete effect is not written); blanking the editor to match the gap

**Mapping integrity**:
A one-to-one, unique alignment of evidence-diagram mapping IDs to by-precision evidence models, in list order. Apply is refused for the whole corpus while any study violates it; plan still reports the faults. Repairing a broken mapping is a data job outside this context.
_Avoid_: Last-write-wins on a duplicated ID; applying every other study while one mapping is broken; treating mapping repair as part of write-back

**Unmatched local effect**:
A local evidence-model effect whose label matches no Effect node on that evidence editor. Write-back skips it and warns.
_Avoid_: Creating an Effect node; matching by page order

**Extra effect node**:
An Effect node on the evidence editor that matches no local evidence-model effect. Write-back warns and does not edit it.
_Avoid_: Blanking the extra node; failing the evidence model because it exists

**Intensity control**:
The evidence editor drop-down that sets an evidence-model effect's intensity. Option labels are the processed intensity phrase in title case (e.g. `Indifferent - Weakly Positive`). Write-back selects that option; it does not type intensity as free text. Deltas compare the live selection to that same title-case label.
_Avoid_: Comment field (when intensity is meant); Likert code typed into a text box; sentence case of the JSON phrase (`Indifferent - weakly positive`)

**Effect comment**:
The comment field on one evidence-model effect in the evidence editor. In this context it holds that effect's supporting statistics, not a free-form human note.
_Avoid_: Description; note; JSON dump (when the field itself is meant)

**Supporting statistics**:
The relative-improvement summary and discount factors for one evidence-model effect that Evidence Factory does not store as first-class editor controls: improvement, spread, sample-size discount, variability discount, their product, and the discount residual. Discounted support mass and intensity are not part of supporting statistics; intensity is a separate editor field.
_Avoid_: Belief; discounted support mass; intensity (when the comment payload is meant)
