# Canonical precision-configuration nomenclature

By-precision analysis was collapsing distinct setups (e.g. `int8` vs `w-int8, a-int4`, and QAT vs PTQ at the same formats). We standardize on an always-expanded **precision configuration** grammar, keep `full-<format>` for all-elements quantization, treat numeric-format families as non-interchangeable, and define the by-precision identity as `(quantization method, precision configuration)` with short method tokens (`qat`, `ptq`, `ptq-retrain`). Metadata drops coarse `target_precision` / `quantization_targets` in favor of canonical baseline and precision-configuration lists; processed outputs store `precision_configuration` + `quantization_method`; notebook precision order is an ordered list of those pairs. Alias rewriting happens at load into metadata/processed artifacts, not in study-native external files.

## Considered Options

- Collapse by bare bit-width (`int8`) — rejected; confuses unequal component assignments
- Encode method inside the format string (`qat-w8a8`) — rejected; mixes method into format identity
- Baseline as `w-fp32, a-fp32` or bare `fp32` — rejected in favor of `full-fp*` for float reference models
- Normalize only in the notebook — rejected; processed storage must already be canonical
