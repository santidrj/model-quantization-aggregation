# Model Quantization Aggregation

Evidence extraction and meta-analysis of quantization experiments reported in primary studies.

## Language

**Quantization precision**:
The target numeric bit width of a quantized model configuration, expressed as a canonical label such as `int4`, `int8`, or `fp32`.
_Avoid_: Quantization configuration, precision format (when referring to bit width alone)

**Quantization configuration**:
The full experimental setup for a quantized run, including method, grouping strategy, parameter estimation, and bit width. Used to distinguish otherwise identical bit-width runs in by-configuration analysis.
_Avoid_: Precision, quantization precision (when the full setup is meant)

**Configuration columns**:
Optional paper-specific columns included in the configuration struct for by-configuration analysis, but excluded from the baseline join key.
_Avoid_: Grouping columns (when referring to configuration identity only)

**Grouping columns**:
The columns that identify which experimental runs share a baseline for improvement calculation, typically dataset and model architecture.
_Avoid_: Grouping key, join key
