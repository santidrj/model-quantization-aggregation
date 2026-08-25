# Selection manifest from frozen screening artifacts

Manuscript study-selection counts and the PRISMA-style figure must be regenerated from a single record-level **selection manifest** under `data/processed/`, not from hand-maintained callouts. When the live Gemini scores parquet disagrees with the historical screening workbooks, the frozen title-abstract screening pool wins: `model-quantization-llm-selected-papers.xlsx` decomposes as LLM pre-screen retention of the remaining search hits union human-positive **calibration subset** records (historically 537 + 35 = 572); human-negative calibration records do not re-enter title-abstract screening. Canonical title keys (Unicode/case/whitespace normalization) must make the calibration subset nest inside the 1,224-row search export before denominators are locked; orphan titles are repaired or explicitly dispositioned, not papered over with 1,224 − 200 = 1,024 arithmetic. For this revision we report calibration recall and the predicted-negative audit with uncertainty as in-sample checks, and we do not claim an independent corpus-level recall estimate or re-adjudicate a held-out set.

## Considered options

- Recompute retention from the current scores parquet and rewrite the manuscript to those counts (rejected: silently changes the published screening path without proven provenance).
- Treat Fig. 4’s 372 as authoritative (rejected: no frozen artifact reproduces 372; it is consistent with mistakenly subtracting the full calibration size from the 572 pool).
- Re-run the finalized prompt and mint a new retained set before resubmission (deferred: out of scope for a manifest-first reconciliation).
- Nested CV or new held-out adjudication for recall (deferred: valuable later; this revision prioritizes disposition integrity and honest labeling of in-sample performance).

## Consequences

- Prose must not say the title-abstract screening pool was “retained by Gemini”; figures must show the calibration-positive merge.
- Stale “18 reviewed studies” data-reporting counts must follow the current 21 included studies once the manifest is the authority.
- The live scores parquet may still be useful for tooling, but it is not the published screening authority until it replays the frozen retention.
