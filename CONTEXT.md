# Model Quantization Aggregation

Evidence extraction and meta-analysis of quantization experiments reported in primary studies.

## Language

**Precision configuration**:
The numeric-format assignment of a quantized run. The usual canonical form uses tokens `<component>-<format>` joined by `", "`, with components in fixed order `w`, then `a`, then `b`, then any others alphabetically, and omits unmentioned ones — so `w-int8` means weights-only, not an implied `a-fp32`. Distinct assignments must not be collapsed: `w-int8, a-int4` is not the same as `w-int8, a-int8`. A bare label such as `int8` is an input alias for `w-int8, a-int8` only when equal weight and activation formats are actually known; it must not invent components for weight-only (or otherwise partial) studies. Bare `mixed` / `mixed-<avg>` are canonical whole-run forms when the study does not give per-component formats. Compact forms (`wa-int8`, `w8a8`) and method-prefixed forms (`qat-w8a8`) are input aliases only, not canonical.
_Avoid_: Quantization precision (ambiguous), bit width (when the component assignment is meant), quantization configuration (when only formats are meant)

**Numeric format**:
The atomic format label used inside a precision configuration or full precision configuration. Distinct families are kept separate and are not rewritten into each other: `intN`, `fpN`, `q0.N`, `qM.F` (fixed-point integer.fractional bits), `logN`, `mixed`, and `mixed-<avg>` (mixed precision with a known average bit width, e.g. `mixed-1.8`, `mixed-2`). Bare `mixed` is only for when no average is known. `mixed-2` is not the same as `int2`. Typos normalize within a family only (e.g. `q0,32` → `q0.32`); `q0.8` is not the same as `q8.0`.
_Avoid_: Bit width (when the family/spelling matters), precision configuration (when a single atom is meant), uniform `intN` labels for mixed-precision runs

**Average bit width**:
The study-reported average bit width of a mixed-precision quantization. When known, it is part of the numeric format as `mixed-<avg>` (trailing `.0` dropped); it is not a claim that every component uses that width, and it must not be rounded into a uniform `intN` identity.
_Avoid_: Bit width (as a synonym for precision configuration), target precision

**Baseline precision configuration**:
The precision configuration of the unquantized (or otherwise reference) run that quantized runs are compared against. Canonical form for a float reference model is the corresponding full precision configuration (`full-fp32`, `full-fp16`, `full-fp64`).
_Avoid_: Baseline precision, target precision (retired coarse metadata field), bare `fp32` as stored identity

**Full precision configuration**:
A precision configuration in which every numerical element of the model (weights, activations, biases, and any other parameters) shares one numeric format, written `full-<format>` (e.g. `full-int8`). This is stricter than listing only weights and activations, so `full-int8` is not the same key as `w-int8, a-int8`.
_Avoid_: `w-…, a-…, b-…` (when the study means all elements, not only named components)

**Quantization method**:
How quantization is applied relative to training for a run. Canonical tokens are short forms such as `qat`, `ptq`, and `ptq-retrain`. Verbose phrases are input aliases only. Together with precision configuration, it forms the by-precision aggregation key: runs that differ only in quantization method must not be merged even when they share a precision configuration.
_Avoid_: Optimization, precision configuration (when the training-time vs post-training distinction is meant), long prose labels as stored identity

**By-precision aggregation**:
Grouping of comparable quantized runs that share the same quantization method and the same precision configuration. When a paper declares exactly one quantization method, every run inherits it; when it declares more than one, each run must carry its own method. Ordering for analysis (e.g. notebook precision order) is an ordered list of `(quantization method, precision configuration)` pairs.
_Avoid_: By-configuration aggregation (when only method and formats are the identity), precision-only grouping (when method is ignored)

**Quantization configuration**:
The full experimental setup for a quantized run, including quantization method, grouping strategy, parameter estimation, and precision configuration. Used to distinguish otherwise identical precision-configuration runs in by-configuration analysis.
_Avoid_: Precision, precision configuration (when the full setup is meant)

**Configuration columns**:
Optional paper-specific columns included in the configuration struct for by-configuration analysis, but excluded from the baseline join key.
_Avoid_: Grouping columns (when referring to configuration identity only)

**Grouping columns**:
The columns that identify which experimental runs share a baseline for improvement calculation, typically dataset and model architecture.
_Avoid_: Grouping key, join key

**External paper data**:
The study-provided experimental datasets consumed by evidence extraction for a given paper, kept under that paper's external data folder. Study-native precision labels may remain here; alias → canonical rewriting happens when loading into this project's metadata and processed outputs.
_Avoid_: Raw data (when referring to these study inputs), replication package (when referring to the full upstream archive rather than the files this project keeps)
