# Context Map

## Contexts

- [Model Quantization Aggregation](./CONTEXT.md) — evidence extraction and meta-analysis of quantization experiments reported in primary studies
- [EvidenceFactory Aggregation](./automation/CONTEXT.md) — automated merging of evidence models in the EvidenceFactory web application via Add, Remove, and Join

## Relationships

- **Model Quantization Aggregation → EvidenceFactory Aggregation**: Research studies and extracted evidence are authored/merged upstream in EvidenceFactory; this automation context operates on EvidenceFactory's live UI and does not redefine study IDs, precision configurations, or meta-analysis metrics.
- **No shared ubiquitous language for "aggregation"**: In the research context, aggregation means grouping comparable quantized runs. In the EvidenceFactory context, aggregation means merging two evidence models' elements. Do not reuse either term across contexts without qualification.
