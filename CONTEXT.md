# Model Quantization Aggregation

Evidence extraction and meta-analysis of quantization experiments reported in primary studies.

## Language

**Study ID**:
The systematic-review label of an included primary study, written `S1`…`Sn`. Assignment is by publication year ascending, then lead-author citation name ascending within a year (the name as cited, including particles and compound surnames — e.g. De Putter, Gonzalez Alvarez), yielding consecutive integers with no gaps. Ordering by study ID is numeric on the integer after `S`, not lexicographic on the string. Distinct from the paper key.
_Avoid_: Paper ID (ambiguous with paper key), study key, citation key (when the `S#` is meant)

**Paper key**:
The durable slug identity of a paper in this project (citation-style folder and config name). Stable across study ID renumbering; used for external and processed data paths.
_Avoid_: Study ID, paper ID (when the `S#` is meant), folder name (as a separate concept)

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
Grouping of comparable quantized runs that share the same quantization method and the same precision configuration. Distinct precision configurations are never pooled into a study-level mean. When a paper declares exactly one quantization method, every run inherits it; when it declares more than one, each run must carry its own method. Ordering for analysis (e.g. notebook precision order) is an ordered list of `(quantization method, precision configuration)` pairs.
_Avoid_: By-configuration aggregation (when only method and formats are the identity), precision-only grouping (when method is ignored), study-level overall (when distinct precision configurations are mixed)

**Theoretical structure**:
Manuscript label for one by-precision aggregation that appears in a study's by-precision results. A study's TS count is how many such aggregations that study contributes — not a separate analytical construct.
_Avoid_: Theoretical structure as something other than by-precision aggregation; using metadata `precision_configurations` length when by-precision results are meant; evidence model (when the SSM diagram is meant)

**Evidence model**:
The SSM diagrammatic representation of one theoretical structure, carrying that by-precision aggregation's effects, intensities, and beliefs. One-to-one with theoretical structure; not an experimental unit and not a DL model.
_Avoid_: Experimental unit; DL model; incoming evidence model (when the Evidence Factory merge side is meant)

**Evidence-model effect**:
One effect carried by one evidence model. This is the grain at which \(n_{\mathrm{eff}}\), the sample-size discount, the variability discount, and discounted support mass are defined, and the observation in the effective-sample-size summary table.
_Avoid_: Study (when an effect-level \(n_{\mathrm{eff}}\) is meant); experimental unit; theoretical structure (the whole by-precision aggregation)

**Operational system**:
The System-archetype contextual aspect that names the software system in which the quantized model is expected to operate (e.g. an LLM-based code assistant or a machine translation system). Distinct from the System-archetype node that names the quantized artifact (typically a DL model). Also distinct from the statistical experimental unit used for relative-improvement aggregation. Operational-system labels are removed at synthesis to reduce noise.
_Avoid_: Application domain (when this System node is meant), DL model / experimental unit (when the host system is meant)

**Experimental unit**:
The unique combination of a study's grouping columns that identifies one observation for the mean relative improvement and IQR inside a by-precision aggregation (for example a model, or a model–task pair). Not necessarily a DL model alone, and not necessarily an independent draw when tasks or datasets nest inside a model family.
_Avoid_: Treating every experimental unit as independent when grouping columns include task or dataset nested in a model; treating every experimental unit as a single model; conflating with cluster unit or with the System-archetype DL-model label in an evidence model

**Cluster unit**:
The nesting factor treated as the independent unit for effective sample size, sample-size discount, and interval estimation when experimental units share a model family. It is the experimental-unit key after dropping evaluation-context columns (`Dataset`, `dataset`, `task`) only — Device, compute unit, frequency, library, and filter multiplier stay. Distinct from the experimental unit used for the mean and IQR.
_Avoid_: Experimental unit (when independence is meant); DL model as always the cluster; collapsing task-dependent outcomes to cluster means before computing the evidence-model mean; collapsing hardware or library factors as if they were tasks

**Artifact-invariant metric**:
A resource-efficiency metric whose value depends only on the quantized artifact (for example `storage_size`), not on the evaluation task or dataset. Such metrics use a reduced grouping key that omits evaluation-context columns from `GROUPING_COLUMNS`.
_Avoid_: Treating task-invariant metrics as if they varied per dataset; conflating with experimental unit for task-dependent outcomes

**Effective sample size**:
The count of independent units for a specific effect inside a by-precision aggregation (\(n_{\mathrm{eff}}\)). When experimental units nest inside cluster units, that count is the number of cluster units, not the number of experimental units. It remains effect-specific: task-dependent metrics use the study's grouping columns to form experimental units, then collapse to cluster units; artifact-invariant metrics (for example storage) already omit evaluation-context columns. Repeated measurement runs on the same experimental unit do not increase \(n_{\mathrm{eff}}\). Forest-plot intervals are cluster-robust intervals for the experimental-unit mean, using this cluster count.
_Avoid_: Number of models (when grouping columns are wider); number of measurement runs; forest-plot raw row counts (`nobs`); counting nested tasks as independent units; iid Student's \(t\) on nested experimental units

**Subgroup**:
One moderator level used when interpreting the main by-precision aggregation: the triple `(baseline precision configuration, quantization method, precision configuration)`. Distinct from by-precision aggregation itself, which does not include the baseline in its grouping key.
_Avoid_: Method+precision alone (when the baseline is part of the claim), within-cell splits by architecture, device, or other study covariates

**Subgroup analysis**:
Analysis of relative-improvement evidence restricted to a single subgroup, answering what effects look like at that moderator level of the main aggregation (study-level results with Aggregated synthesis computed from the same validated synthesis file as the main table, restricted to that subgroup).
_Avoid_: Subgroup analysis as a split by architecture/hardware inside a cell; reusing corpus-wide Aggregated rows as if they were subgroup-specific

**Validated synthesis**:
The committed JSON result file that is the authority for manuscript tables, forest-plot Aggregated overlays, subgroup provenance, sensitivity values, and result-section counts. It is generated by local Dempster–Shafer pooling of processed evidence models.
_Avoid_: Hand-authored Aggregated rows; copying Evidence Factory UI numbers into plots or prose

**Sensitivity analysis**:
Analysis that tests whether primary synthesis conclusions are robust when the main pooling assumptions change. The published sensitivity analysis holds the included study set and extracted intensities fixed and recombines evidence under the mass-preserving belief split ($1-(1-B)^{1/N}$ per evidence model), thereby limiting repeated commitment of the same study belief across configurations from one publication. Leave-one-study-out recombination of the main discounted pooling and the equitable split ($B/N$) are reported as supplementary stress tests in appendices. A further appendix stress test varies the main discount parameters one at a time without changing the \(\alpha_v\) denominator and without replacing the published forest plots: \(n_0\in\{2,n_0^{Q3},6\}\), \(k\in\{0.05,0.1,0.2\}\), and cutoff \(\in\{3,4,8\}\), reporting aggregated direction/intensity and accuracy \(B'\). Distinct from subgroup analysis, which restricts the moderator triple while holding the study set fixed.
_Avoid_: Subgroup analysis (when the cut is which studies are in, not which cell); reusing corpus-wide Aggregated rows as if they were sensitivity-sample-specific; study-set exclusion by theoretical-structure count (retired TS $\le 5$ analysis); treating an alternative \(\alpha_v\) denominator as a published sensitivity variant

**Belief assignment**:
The rule that maps study belief onto simple-support masses on evidence models for Dempster–Shafer synthesis of the full by-precision aggregation, holding the included study set fixed. The variants are published analogue, undiscounted unsplit combination, mass-preserving belief split, and equitable belief split.
_Avoid_: Sensitivity analysis; subgroup analysis; naming the whole experiment after one split

**Equitable belief split**:
A belief-assignment variant of the full by-precision aggregation in which each study's study belief is partitioned equally across its evidence models before Dempster–Shafer combination, without sample-size or variability discounts. Each model receives \(B/N\) once (\(N\) is that study's evidence-model count) for every effect it carries; the analysis answers whether synthesis conclusions depend on treating those models as independent carriers of the full study belief.
_Avoid_: Sensitivity analysis (when this belief-assignment variant is meant); mass-preserving belief split (when the simple-support root is meant)

**Mass-preserving belief split**:
A belief-assignment variant of the full by-precision aggregation in which each evidence model receives the simple-support Dempster–Shafer root of its study's undiscounted study belief, \(1-(1-B)^{1/N}\), with \(N\) that study's evidence-model count and no sample-size or variability discounts. Used as the primary dependence sensitivity analysis in the published synthesis. Distinct from equitable belief split, which assigns \(B/N\).
_Avoid_: Equitable belief split; D-S root allocation (when the analysis is meant)

**Undiscounted unsplit combination**:
Dempster–Shafer pooling of the full by-precision aggregation in which each evidence model carries the full study belief and sample-size and variability discounts are not applied. The primary undiscounted baseline for equitable belief split and mass-preserving belief split.
_Avoid_: Published aggregation (when discounts are applied); equitable belief split; mass-preserving belief split

**Santos hypothesis selection**:
The literature-based SSM rule that recursively narrows a direction-compatible effect-intensity interval while a contained interval contributes at least 75% of its belief measure. It may select a range that received no direct mass.
_Avoid_: Evidence Factory-compatible selection; focal-compound selection; calling the 75% rule a Dempster–Shafer axiom

**Evidence Factory-compatible selection**:
The replication policy that selects among singleton intensities and adjacent compound intensities carrying direct mass, preserving the published Evidence Factory output. It is a compatibility rule, not the hypothesis-selection rule defined by Santos.
_Avoid_: Santos hypothesis selection; literature selection

**Quantization configuration**:
The full experimental setup for a quantized run, including quantization method, grouping strategy, parameter estimation, and precision configuration. Used to distinguish otherwise identical precision-configuration runs in by-configuration analysis.
_Avoid_: Precision, precision configuration (when the full setup is meant)

**Configuration columns**:
Optional paper-specific columns included in the configuration struct for by-configuration analysis, but excluded from the baseline join key.
_Avoid_: Grouping columns (when referring to configuration identity only)

**Grouping columns**:
The study-specific factors that define experimental units and the baseline join for relative improvement (typically model, and often dataset, device, or other covariates).
_Avoid_: Grouping key, join key (as separate concepts); configuration columns

**External paper data**:
The study-provided experimental datasets consumed by evidence extraction for a given paper, kept under that paper's external data folder. Study-native precision labels may remain here; alias → canonical rewriting happens when loading into this project's metadata and processed outputs.
_Avoid_: Raw data (when referring to these study inputs), replication package (when referring to the full upstream archive rather than the files this project keeps)

**Evidence granularity**:
How recoverable a primary study's reported experimental results are for evidence extraction and meta-analysis. The levels are tabular summary (numeric results in tables or text), chart-only summary (results mainly in figures), and replication package (run-level or package-backed numbers).
_Avoid_: Data quality, comparative, precise (as public language for these levels)

**Tabular summary**:
Evidence granularity in which the study reports numeric results in tables or prose that can be extracted as summary statistics without digitizing figures.
_Avoid_: Comparative, summary statistics (when the granularity level is meant)

**Chart-only summary**:
Evidence granularity in which the study's usable results are mainly in figures, so extraction depends on chart reading or digitization.
_Avoid_: Comparative (charts), summary statistics (charts)

**Replication package**:
Evidence granularity in which run-level or otherwise package-backed experimental numbers are available beyond paper tables and figures. Distinct from external paper data, which is the subset of study inputs this project keeps for extraction.
_Avoid_: Precise, raw data (when this granularity level is meant)

**Domain**:
The primary application area a study addresses (e.g. image classification, code generation), one label per included study. Stored as the short manuscript display string used in summary tables. Distinct from the study's dataset list: datasets may motivate or illustrate the domain, but Domain is not a compression of `datasets`.
_Avoid_: Dataset, task (when the study-level application area is meant), topic

**Energy measurement method**:
How a primary study obtains energy (or power) figures for its runs, when it measures energy at all. Canonical tokens are analytical, hardware-based, software-based, and model-based; a study may use more than one. A null / absent method means the study did not measure energy consumption — not that the method is unknown or unreported.
_Avoid_: Measurement method (when energy is meant), not reported / missing metadata (for a null method)

**Energy evidence group**:
Manuscript grouping of included studies by energy measurement method for summary tables. Hardware-based, software-based, or null method map to the manuscript label Quasi-experiments; analytical or model-based map to Observational Studies. Distinct from experimental-design “quasi-experiment” in the quality-evaluation guide.
_Avoid_: Study type (when this energy-derived table grouping is meant); treating the manuscript labels as a stored taxonomy

**Analytical**:
An energy measurement method where energy figures are computed from formulas or design-space estimation tools, without measuring a running system.
_Avoid_: Model-based (when a whole-design analytical estimator is meant), estimated energy (too vague)

**Hardware-based**:
An energy measurement method where energy or power is obtained from physical instrumentation on the device or supply path (e.g. wattmeter, oscilloscope with current sense).
_Avoid_: On-device (when software counters are meant), measured energy (too vague)

**Software-based**:
An energy measurement method where energy or power is read from software interfaces or OS/vendor counters on a running system (e.g. `nvidia-smi`, `tegrastats`, `pyRAPL`).
_Avoid_: Hardware-based (when counters rather than physical meters are meant), profiling (too vague)

**Model-based**:
An energy measurement method where energy is derived bottom-up from a component energy model (e.g. energy-per-operation × operation counts), not from end-to-end board measurement or a high-level analytical estimation tool alone.
_Avoid_: Analytical (when component energy tables are meant), simulated energy (too vague)

## Metrics

**Study belief**:
The study-level prior weight assigned to an included primary study for evidence synthesis (stored on the study as a probability in \([0,1]\)). Effect-level beliefs combine this prior with sample-size and variability discounts. In manuscript tables it may appear as an integer percent.
_Avoid_: Belief as an effect-level posterior; quality-rubric score (when the stored study prior is meant); SSM belief (vague)

**Discounted support mass**:
The study belief after sample-size and variability discounts, assigned as simple-support mass to one evidence-model effect. The processed Evidence Factory-compatible field is named `belief`; this value is not the combined belief measure.
_Avoid_: Study belief; belief measure; posterior belief

**Mass function**:
The Dempster–Shafer basic probability assignment \(m\) over subsets of the effect-intensity frame, including residual mass on \(\Theta\). It is the state combined between successive evidence pieces.
_Avoid_: Belief measure; probability distribution over singleton intensities

**Belief measure**:
The Dempster–Shafer quantity \(\mathrm{Bel}(A)=\sum_{B\subseteq A}m(B)\) for a hypothesis \(A\), computed from a mass function. It is distinct from study belief and discounted support mass.
_Avoid_: Study belief; discounted support mass; posterior probability

**Sample-size discount**:
The reliability factor \(\alpha_n = 1 - e^{-n_{\mathrm{eff}}/n_0}\) applied to study belief for one evidence-model effect. The saturation parameter \(n_0\) is the third quartile of the effective-sample-size distribution over evidence-model effects, recomputed when that distribution's population or unit definition changes.
_Avoid_: Freezing \(n_0=3\) after the table population changes; calibrating \(n_0\) to studies rather than evidence-model effects; treating the code comment "median" as the rule

**Variability discount**:
The reliability factor applied to study belief when relative improvements inside one by-precision aggregation are heterogeneous. The IQR is taken over the experimental-unit relative improvements in that aggregation. The main-analysis denominator is \(\mu+\varepsilon\) with a numerical stabilizer \(\varepsilon\), not a negligible-effect floor. Distinct from by-precision aggregation itself, which is the grouping, not the discount.
_Avoid_: treating this discount as a second meaning of aggregation; IQR of model-level means when experimental units are model–dataset pairs; replacing the main-analysis denominator with \(\max(|\mu|,\delta)\)

**Correctness metric**:
A measure of model prediction quality reported by a study (e.g. accuracy, F1 score, perplexity). Correctness metrics are distinct from resource-efficiency metrics even when a correctness metric is minimized rather than maximized.
_Avoid_: Accuracy metric (when the measure is not accuracy), quality metric (too vague)

**Resource efficiency metric**:
A measure of computational or hardware cost reported by a study (e.g. latency, energy, storage size). Lower raw values always represent better outcomes.
_Avoid_: Performance metric (ambiguous with correctness), efficiency metric (too vague)

**Metric polarity**:
Whether better outcomes correspond to higher or lower raw values for a given metric. Accuracy, precision, recall, F1 score, Dice score, mAP, mIoU, and BLEU are maximized; perplexity and word error rate are minimized. Resource-efficiency metrics are always minimized.
_Avoid_: Improvement direction, metric direction, higher-is-better / lower-is-better (as stored labels — use maximized / minimized)

**Relative improvement**:
The percentage change of a metric relative to the baseline precision configuration, signed so that positive always means a better outcome and negative always means a worse outcome. Polarity determines whether the formula uses `(quantized − baseline)` or `(baseline − quantized)` in the numerator.
_Avoid_: Percent change, delta (unsigned), improvement rate

**Effect intensity**:
The Likert-style hypothesis for a relative improvement. Evidence-model inputs are a simple atom (SN, NE, WN, IF, WP, PO, SP) or an adjacent two-atom compound; a synthesized result may be a wider direction-compatible interval under Santos hypothesis selection.
_Avoid_: Effect magnitude; belief (when direction and strength are meant); confidence interval (when the intensity band is meant)

**SSM intensity range**:
The numeric span of a selected effect-intensity hypothesis on the relative-improvement axis, including adjacent two-atom compounds. It is a categorical encoding, not a sampling interval.
_Avoid_: 95% confidence interval; aggregated CI

**Conflict**:
The Dempster–Shafer \(K = m(\emptyset)\) of the last pairwise combination step when pooling the evidence models that report an effect (two at a time, renormalized after each step, in Evidence Factory aggregation-turn order). Ground truth is the combination result on Evidence Factory aggregated evidence 329074. Distinct from the n-fold empty-set mass of combining all sources in one product, and from Evidence Factory's red eligible-element mismatches in the aggregator UI.
_Avoid_: Eligible element; difference (when \(K\) is meant); disagreement of mean relative improvements (when combination conflict is meant); n-fold \(m(\emptyset)\) (when last-step \(K\) is meant)
