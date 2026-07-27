# Metric polarity for relative improvement

Correctness metrics are not all maximized: perplexity and word error rate are minimized, while accuracy, F1, Dice, mAP, BLEU, and the rest are maximized. Resource-efficiency metrics are always minimized. Relative improvement is signed so positive always means a better outcome; the formula uses `(quantized − baseline)` for maximized metrics and `(baseline − quantized)` for minimized ones.

We keep polarity in a global registry keyed by canonical metric name rather than per-paper flags, because whether a metric is minimized is a property of the metric itself, not of how a study reports it.

## Considered Options

- Per-paper polarity flags on `CorrectnessMetrics` — rejected; duplicates the same rule on every paper and invites drift
- Reclassify perplexity/WER as resource-efficiency metrics — rejected; they measure prediction quality, not hardware cost
