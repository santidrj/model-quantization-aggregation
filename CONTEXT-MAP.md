# Context Map

## Contexts

- [Model Quantization Aggregation](./CONTEXT.md) — evidence extraction and meta-analysis of quantization experiments reported in primary studies
- [EvidenceFactory Aggregation](./automation/CONTEXT.md) — automated merging of evidence models in the EvidenceFactory web application via Add, Remove, and Join
- [EvidenceFactory Evidence Editor Sync](./automation/evidence-editor/CONTEXT.md) — writing local effect intensity, discount residual, and effect comment onto evidence-model effects in Evidence Factory evidence editors

## Relationships

- **Model Quantization Aggregation → EvidenceFactory Aggregation**: Research studies and extracted evidence are authored/merged upstream in EvidenceFactory; this automation context operates on EvidenceFactory's live UI and does not redefine study IDs, precision configurations, or meta-analysis metrics.
- **Model Quantization Aggregation → EvidenceFactory Evidence Editor Sync**: Local processed evidence-model effects are the source of truth for intensity, discount residual, and supporting statistics; the sync writes those fields into live evidence editors. Diagram structure (which effects exist, how they are arranged) remains authored in Evidence Factory.
- **EvidenceFactory Evidence Editor Sync ↛ EvidenceFactory Aggregation**: Editor sync does not Add, Remove, or Join, and does not run Update evidence aggregation. Aggregation does not edit intensity, discount residual, or effect comment.
- **No shared ubiquitous language for "aggregation"**: In the research context, aggregation means grouping comparable quantized runs. In the EvidenceFactory context, aggregation means merging two evidence models' elements. Do not reuse either term across contexts without qualification.
- **No shared ubiquitous language for "evidence model"**: In the research context, an evidence model is the SSM diagram of one theoretical structure. In the EvidenceFactory context, it is a diagram in the aggregator UI (aggregated vs incoming). Do not reuse either term across contexts without qualification.
- **No shared ubiquitous language for "conflict"**: In the research context, conflict is Dempster–Shafer \(K = m(\emptyset)\) after pooling an effect. In the EvidenceFactory context, conflict is a red eligible-element mismatch between diagrams. Do not reuse either term across contexts without qualification.
- **Model Quantization Aggregation → Manuscript (external repo)**: This context produces forest-plot figures and studies-summary LaTeX fragments that the private manuscript repository vendors via allowlisted sync (`review` for drafting; Zenodo for milestones). The manuscript is not a bounded context in this repo; public discovery is paper citation ↔ Zenodo only (see `docs/adr/0006-multi-repo-manuscript-via-vendored-artifacts.md`).
