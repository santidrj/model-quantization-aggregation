# Spec: Notebook 4 paper-metadata characterization

Status: implementing

## Goal

Improve `notebooks/4.0-paper-metadata-analysis.ipynb` into a **corpus characterization** notebook for writing (methods/corpus section), not an audit-first tool.

## Deliverable

Hybrid: short written takeaways + a small set of exported figures under `reports/figures/`.

## Counting rules

- **Paper-level by default** (each study contributes at most once per category).
- **Instance-level** for the exported precision-configuration chart (counts of reported configurations with count ≥ 2); captions must say these are configuration counts, not papers.
- **Bridge sample-size table:** after exploding multi-valued precision/method fields, count **unique papers** per `(baseline, precision configuration, quantization method)` key (meta-analysis sample size).
- Multi-valued fields (`quantization_method`, `measurement_method`) may place a paper in more than one category.

## Story arc (section order)

1. Setup + one compact metadata overview table
2. Study design — evidence granularity (export), study type (notebook-only)
3. Quantization landscape — quantization method (export), baseline takeaway in prose, precision configs with count ≥ 2 (export), techniques (notebook-only)
4. Experimental scope — models, datasets (notebook-only charts + takeaways)
5. Measurement setup — energy measurement method including **no energy measurement** for null (export); software tools / hardware notebook-only
6. Bridge to notebook 5 — one sample-size table by `(baseline, precision configuration, quantization method)`

Drop the standalone “analytical-only filter” section (fold into measurement notes if needed). Collapse duplicated shared-precision cells into the bridge table + precision export.

## Exported figures

1. `number_of_papers_per_evidence_granularity.pdf` — papers × evidence granularity
2. `number_of_papers_per_quantization_method.pdf` — papers × quantization method
3. `shared_precision_configurations.pdf` — instance-level precision configurations with count ≥ 2
4. `number_of_papers_per_energy_measurement_method.pdf` — papers × energy measurement method, with null mapped to **no energy measurement**

Replace the old `number_of_papers_per_data_quality.pdf` export.

## Language

- Public vocabulary from `CONTEXT.md`: evidence granularity levels (**tabular summary**, **chart-only summary**, **replication package**); energy measurement methods as defined there.
- Stored metadata field `data_quality` and tokens `comparative` / `comparative (charts)` / `precise` stay as-is; map to glossary labels in the notebook/helpers only.
- Avoid “not reported” for null energy measurement method.

## Takeaways

- Computed count/share bullets from the data.
- Short interpretive sentences as clearly marked placeholders for the author.

## Out of scope

- Renaming stored JSON field/values
- Cross-tab / co-occurrence analyses
- Defining hardware coarse classes
- Exporting models/datasets/study_type/hardware charts

## Glossary (done in CONTEXT.md)

Evidence granularity (+ three levels); Energy measurement method (+ analytical, hardware-based, software-based, model-based).
