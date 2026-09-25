# Literature cut-points for RI% → SSM intensity (sensitivity grid)

**Scope:** Numeric thresholds that map relative improvement / percent change / effect magnitude onto categorical intensity bands (indifferent / weak / moderate / strong), for use beside this project's published expert RI%→Likert cuts.

**Project reference (expert knowledge, not a cited numeric standard):**

| Scale | IF | weak | weak–mod | mod | strong–mod | strong |
|---|---:|---:|---:|---:|---:|---:|
| Functional suitability | ≤2% | ≤5% | ≤10% | ≤15% | ≤20% | ≤25% |
| Resource / performance | ≤2% | ≤10% | ≤20% | ≤30% | ≤40% | ≤50% |

Full signed partition: `reports/tables/intensity-thresholds.tex` / `src/effect_intensity.py`.

**Method:** White literature (peer-reviewed journals/conferences) preferred; gray (preprints, whitepapers, industry docs) labeled. Direct reads of SSM sources, SE effect-size reviews, classic effect-size primers, and efficient-ML / quantization papers. **No invented thresholds.**

**Unit caveat:** Most domain sources cut **accuracy percentage points**, **compression ratios** (e.g. 4×), or **latency/energy %**. This project cuts **mean relative improvement (RI%)**. Near high baselines, accuracy-pp and RI% can be numerically close; they are **not** the same unit. Sensitivity borrows *practically meaningful % magnitudes*, not a formal pp↔RI% identity.

There is **no** published universal RI%→seven-atom Likert table in SSM or in quantization surveys.

---

## 1. SSM / Santos et al. — do they publish RI%→intensity cuts?

### Verdict

**No.** Foundational and applied SSM papers define a **qualitative seven-point Likert frame** and instruct analysts to **arbitrate domain-specific numeric ranges** when translating quantitative outcomes. They do **not** publish a reusable RI% (or % change) cut table.

### Sources

#### Santos & Travassos (2013) — white

- **Citation:** Paulo Sérgio Medeiros dos Santos, Guilherme Horta Travassos. *On the Representation and Aggregation of Evidence in Software Engineering: A Theory and Belief-based Perspective.* Electronic Notes in Theoretical Computer Science 292:95–118, 2013. [doi:10.1016/j.entcs.2013.02.008](https://doi.org/10.1016/j.entcs.2013.02.008)
- **White vs gray:** White (ENTCS / Elsevier).
- **Exact numeric cuts:** None for intensity magnitude. Belief is estimated via GRADE-style evidence hierarchy bands (e.g. unsystematic [0–0.25], observational [0.25–0.50], …) — that is **belief**, not intensity.
- **Quantity cut:** N/A for RI%.
- **Maps to this project's RI% scales?** **No** — no RI%→Likert table to adopt or perturb.

#### Santos & Travassos (2017) — white

- **Citation:** Paulo Sérgio Medeiros dos Santos, Guilherme Horta Travassos. *Structured Synthesis Method: The Evidence Factory Tool.* ESEM 2017, pp. 480–481. [doi:10.1109/ESEM.2017.68](https://doi.org/10.1109/ESEM.2017.68)
- **White vs gray:** White (tool abstract).
- **Exact numeric cuts:** None.
- **Maps?** No.

#### Santos, Nascimento & Travassos (2015) — white / conference

- **Citation:** Paulo Sérgio Medeiros dos Santos, Ian Nascimento, Guilherme Horta Travassos. *A Computational Infrastructure for Research Synthesis in Software Engineering.* CIbSE / ESELAW 2015. [PDF](https://eventos.spc.org.pe/cibse2015/pdfs/04_ESELAW15.pdf)
- **White vs gray:** White (peer conference proceedings).
- **Exact numeric cuts:** Belief ranges only (GRADE-aligned), e.g. [0.0, 0.25] … [0.75, 1] for study types. **No** intensity % cuts.
- **Maps?** No (belief, not intensity).

#### dos Santos et al. (2018) — Kanban SSM study — white

- **Citation:** Paulo Sérgio Medeiros dos Santos et al. *On the benefits and challenges of using kanban in software engineering: a structured synthesis study.* Journal of Software Engineering Research and Development 6:13, 2018. [doi:10.1186/s40411-018-0057-1](https://doi.org/10.1186/s40411-018-0057-1)
- **White vs gray:** White.
- **Exact quote (method):** For qualitative studies, adverbs/adjectives map to the seven-point Likert. For quantitative studies: *“we need to **arbitrate ranges of values** using the domain of the dependent variable scale as input to be able to translate it to the seven-point Likert scale.”* Example: cycle time from ~100 days to ~60 days labeled “significant improvement” by authors → coded **strongly positive** — **authorial language**, not a % table.
- **Numeric cuts quoted:** None as a general standard.
- **Maps?** Confirms **per-study arbitration**; does **not** supply cuts near 2 / 5 / 25 / 50.

#### Martínez-Fernández et al. (2015) — SRA aggregation with SSM — white

- **Citation:** Silverio Martínez-Fernández et al. *Aggregating Empirical Evidence about the Benefits and Drawbacks of Software Reference Architectures.* (ECSA / related proceedings; PDF used: [essi.upc.edu copy](https://www.essi.upc.edu/~smartinez/wp-content/papercite-data/pdf/martinez-fernandez2015aggregating.pdf))
- **White vs gray:** White.
- **Exact numeric cuts:** Intensities reported as Likert atoms/compounds (e.g. `{PO, SP}`) with belief masses; **no** RI%→intensity table.
- **Maps?** No.

#### This project's ESEM / arXiv quantization SSM case — white (+ gray preprint)

- **Citation:** Santiago del Rey et al. *Aggregating empirical evidence from data strategy studies: a case on model quantization.* ESEM 2025; arXiv:2505.00816. [doi:10.1109/ESEM64174.2025.00049](https://doi.org/10.1109/ESEM64174.2025.00049) / [arXiv](https://arxiv.org/abs/2505.00816)
- **White vs gray:** White (ESEM); preprint gray.
- **Exact numeric cuts:** States intensities from average relative improvement using **“thresholds shown in Fig. 3”** (study-specific figure). Example: energy RI 57.18% → `{SP}`. Narrative also notes storage reductions **over 50%** as strongly positive territory. These are **this project's / case study's** expert cuts — **not** an external SSM standard.
- **Maps?** Circular if used as “independent literature” for the same cuts; useful only as evidence that SSM practice expects **contextual** numeric thresholds.

### SSM section gap (honest)

SSM literature supplies the **Likert atoms** and the **procedure** (“arbitrate ranges from the DV domain”). It does **not** justify any particular 2 / 5 / 25 / 50 RI% grid. Sensitivity must look **outside** SSM for competing numeric magnitudes.

---

## 2. SE secondary studies mapping % improvement → strength labels

### Verdict

SE secondary / synthesis work almost never publishes a shared **% improvement → weak/moderate/strong** table usable as an alternate RI% row. When magnitudes are categorized, they use **standardized** effect sizes (Hedges' *g*, *r*), or leave unstandardized % interpretation to context.

#### Kampenes, Dybå, Hannay & Sjøberg (2007) — white

- **Citation:** Vigdis By Kampenes, Tore Dybå, Jo E. Hannay, Dag I. K. Sjøberg. *A systematic review of effect size in software engineering experiments.* Information and Software Technology 49(11–12):1073–1086, 2007. [doi:10.1016/j.infsof.2007.02.015](https://doi.org/10.1016/j.infsof.2007.02.015) ([Simula PDF](https://web-backend.simula.no/sites/default/files/publications/Kampenes.2006.1.pdf))
- **White vs gray:** White.
- **Exact numeric cuts (Hedges' *g* tertiles of 284 SE estimates):**

  | Category | *g* range | Median *g* |
  |---|---|---:|
  | Small (lower 33%) | 0.00–0.376 | 0.17 |
  | Medium (middle 34%) | 0.378–1.000 | 0.60 |
  | Large (upper 33%) | 1.002–3.40 | 1.40 |

  Point-biserial medians: small 0.09, medium 0.30, large 0.60.
- **Quantity:** Standardized mean difference / *r*<sub>pb</sub> — **not** RI%.
- **Also notes:** Unstandardized % differences matter in context; example that even **1%** defect-detection gains can be practically important for critical defects — **illustrative**, not a cut table.
- **Maps to RI% encoding without inventing a conversion?** **No** — would require assuming an SD (or pooling SD) to convert *g* → %; that conversion is **not** published for quantization metrics.

#### Other SE meta-analysis guidance

Recent SE meta-analysis work (e.g. small-sample recommendations in *Empirical Software Engineering* 2024) emphasizes robust effect sizes (Cliff's *d*, probability of superiority) and **warns against canned Cohen labels** transplanted from behavioural science. **No** RI%→Likert prescription found.

---

## 3. Classic effect-size interpretation rules (Cohen / Sawilowsky / Ferguson) and MCID-style %

### Verdict

These are **white, high-trust** magnitude vocabularies, but they cut **standardized** indices (*d*, *r*). They **cannot** be dropped onto this project's RI% scales without an invented *d*↔% bridge. Useful for *terminology analogy* only, not for alternate RI% rows.

#### Cohen (1988) — white (book)

- **Citation:** Jacob Cohen. *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum, 1988.
- **Cuts (Cohen's *d*):** small **0.2**, medium **0.5**, large **0.8** (also *r* ≈ 0.10 / 0.30 / 0.50 in common restatements; Cohen warned these are operational defaults).
- **Quantity:** Standardized mean difference / correlation.
- **Maps to RI%?** **No** without inventing SD.

#### Sawilowsky (2009) — white

- **Citation:** Shlomo S. Sawilowsky. *New Effect Size Rules of Thumb.* Journal of Modern Applied Statistical Methods 8(2), 2009. [doi:10.22237/jmasm/1257035100](https://doi.org/10.22237/jmasm/1257035100)
- **Exact cuts (*d*):** very small **0.01**, small **0.20**, medium **0.50**, large **0.80**, very large **1.2**, huge **2.0**.
- **Maps to RI%?** **No**.

#### Ferguson (2009) — white

- **Citation:** Christopher J. Ferguson. *An Effect Size Primer: A Guide for Clinicians and Researchers.* Professional Psychology: Research and Practice 40(5):532–538, 2009. ([author PDF](https://www.christopherjferguson.com/Effect%20size%20primer%20PPRP.pdf))
- **Exact cuts (Table 1 suggestions, social science):**

  | Index family | RMPE (min. practical) | Moderate | Strong |
  |---|---:|---:|---:|
  | Group difference (*d*, *g*, Δ) | **0.41** | **1.15** | **2.70** |
  | Association (*r*, …) | **0.20** | **0.50** | **0.80** |
  | Squared association | **0.04** | **0.25** | **0.64** |
  | Risk (RR, OR)\* | **2.0** | **3.0** | **4.0** |

  \*Risk row not *r*-anchored; interpret with caution (author note).
- **Maps to RI%?** **No**.

#### MCID / “smallest effect of interest” — white but wrong domain for fixed ML %

Clinical MCID / MIC / SWE literature (e.g. Jaeschke et al. 1989 lineage; recent reviews in *J. Clin. Anesth.*, trauma/ortho journals) defines **context-specific** minimal important differences for patient-reported outcomes — **not** universal accuracy-drop % for ML systems. **No transferable fixed RI% grid** for quantization correctness/resource scales.

---

## 4. Quantization / efficient-ML numeric bands (accuracy drop, speedup, energy/memory)

These are the **only** literature cluster that repeatedly states **percent-scale** magnitudes near this project's reference cuts.

### 4.1 Accuracy / functional suitability

#### Krishnamoorthi (2018) — gray (influential whitepaper)

- **Citation:** Raghuraman Krishnamoorthi. *Quantizing deep convolutional networks for efficient inference: A whitepaper.* arXiv:1806.08342, 2018. [arXiv](https://arxiv.org/abs/1806.08342)
- **White vs gray:** **Gray** (arXiv whitepaper; widely cited).
- **Exact numeric cuts / claims:**
  - PTQ 8-bit: classification accuracies **“within 2%”** of floating-point for many CNNs.
  - QAT: gap to floating point reduced to **“1%”** at 8-bit.
  - 4-bit weights (QAT): accuracy losses **“ranging from 2% to 10%”** (higher for smaller nets); also “within **5%** of 8-bit” in some fine-tuning discussion — **relative to 8-bit**, not a “weak effect” RI cut.
  - Model size: reduced by a **factor of 4** (8-bit weights).
  - Latency: **2×–3×** speedup on CPUs; up to **10×** on specialized DSPs.
- **Quantity:** Accuracy **pp** vs FP; compression **ratio**; latency **speedup**.
- **Maps to RI%?** **Partially / by magnitude borrowing:** 1% and 2% sit next to published functional **indifferent** cut (2). **4× → 75%** size RI maps cleanly to resource RI%. **Cannot** justify middle cuts 5/10/15/20/25 as Likert bands from this paper alone.

#### Jacob et al. (2018) — white

- **Citation:** Benoit Jacob et al. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* CVPR 2018, pp. 2704–2713. [doi:10.1109/CVPR.2018.00286](https://doi.org/10.1109/CVPR.2018.00286) ([open access PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf))
- **White vs gray:** White (CVPR).
- **Exact numeric cuts / claims:**
  - Integer-only quantized ResNets: accuracies **“within 2%”** of floating-point counterparts.
  - **“Close to 4× memory footprint reduction”** vs FP32 (8-bit weights+activations).
  - **“Up to a 50% reduction in running time”** / about **2×** latency improvement on common mobile CPUs (setup-dependent).
  - Some detection setups: ~**−1.8%** relative / small AP drops described as minimal.
- **Quantity:** Accuracy pp; memory compression; latency %.
- **Maps to RI%?** **Yes for resource:** 50% latency RI and 75% (=4×) storage RI are direct %. **Yes for borrowing ~2% (and ~1%)** as near-lossless correctness magnitudes. **No** full Likert table.

#### Wu et al. (2020) — gray / NVIDIA tech report style

- **Citation:** Hao Wu et al. *Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation.* arXiv:2004.09602, 2020. [arXiv](https://arxiv.org/abs/2004.09602)
- **White vs gray:** **Gray** (arXiv; industry lab).
- **Exact claim:** Workflow maintains accuracy **“within 1%”** of each FP baseline (including hard cases such as MobileNets, BERT-large).
- **Maps?** Supports **1%** as near-lossless correctness band (same role as indifferent alternate).

#### Kurtic / “Give Me BF16…” (2025) — white + preprint

- **Citation:** Eldar Kurtic et al. *“Give Me BF16 or Give Me Death”? Accuracy-Performance Trade-Offs in LLM Quantization.* ACL 2025. [ACL PDF](https://aclanthology.org/2025.acl-long.1304.pdf); arXiv:2411.02355.
- **White vs gray:** White (ACL); preprint gray.
- **Exact bands:** W8A8-FP “essentially lossless”; W8A8-INT **“1–3%”** average per-task degradation (contrasted with prior reports of **10%+** drops).
- **Maps?** **1–3%** corroborates near-lossless / small-degradation magnitudes near indifferent–weak; **not** a seven-band Likert table. **3%** alone is **not** multi-source consensus for replacing weak=5 (see honesty table).

#### Gholami et al. (2022) — white (survey chapter)

- **Citation:** Amir Gholami et al. *A Survey of Quantization Methods for Efficient Neural Network Inference.* In *Low-Power Computer Vision*, CRC Press, 2022. [doi:10.1201/9781003162810-13](https://doi.org/10.1201/9781003162810-13) (also arXiv:2103.13630).
- **White vs gray:** White (book chapter); arXiv gray duplicate.
- **Exact numeric cuts:** **No** RI→intensity table. States low-bit ints can cut memory/latency by up to **16×** in theory; **4×–8×** often realized in practice.
- **Maps?** Supports **4× → 75%** (and optionally 8× → 87.5%) as strong **memory** RI magnitudes — not a correctness Likert grid.

#### Nagel et al. (2021) — gray

- **Citation:** Markus Nagel et al. *A White Paper on Neural Network Quantization.* arXiv:2106.08295, 2021.
- **White vs gray:** **Gray**.
- **Claims:** Experimental drops **“within 1%”** of FP for some ResNet/Inception W4A8 setups; MobileNet-class can be worse (~2.5–4.2% in cited setups); FP32→INT8 memory **factor of 4**.
- **Maps?** Corroborates **1%** and **4×/75%**; not a Likert table.

#### MLPerf Tiny (Banbury et al., 2021) — white

- **Citation:** Colby Banbury et al. *MLPerf Tiny Benchmark.* NeurIPS Datasets and Benchmarks, 2021.
- **Cuts:** **Absolute** quality floors (e.g. KWS 90%, IC 85%, VWW 80%) to absorb quantization/rounding — **not** “≤X% drop vs FP”.
- **Maps?** **No** for RI% indifferent/weak/strong.

#### OpenVINO / NNCF accuracy control — gray (industry)

- Default **`max_drop = 0.01` (1%)** absolute metric drop for accuracy-aware quantization stop.
- **Maps?** Corroborates **1%** as an operational near-lossless / acceptance threshold; not peer-reviewed.

#### Isolated gray band tables (not consensus)

- Liu et al., *Quantization Hurts Reasoning?*, arXiv:2504.04823 — lossless ≤1% / fair 1–3% / risky ≥3%. **Single gray source** → **do not** put **3** in the sensitivity grid as a weak-effect alternate.

### 4.2 Resource / performance

| Magnitude | Equivalent | Role | White anchors | Gray corroboration | Maps to project RI%? |
|---|---|---|---|---|---|
| **~50%** latency reduction / **~2×** speedup | RI ≈ 50% | Strong **latency** gain typical of INT8 on CPUs | Jacob CVPR 2018 | Krishnamoorthi 2×–3× | **Yes** — same quantity family as project resource/performance RI% |
| **4×** memory / model size | RI = **75%** | Strong **storage** gain FP32→INT8 | Jacob; Gholami; Lin et al. Integer Transformer (ACL 2020, “nearly 4× less memory”) | Krishnamoorthi; Nagel | **Yes** — storage RI% |
| **>50%** storage reduction as “strong” in practice | narrative | Aligns with project SP beyond 50% | This project's ESEM case narrative | — | Circular if used as external justification |

Energy % savings in white literature are **highly system-specific** (tens of percent to >50%); **no** shared **30%** (or similar) energy cut with multi-source consensus for a sensitivity alternate.

---

## 5. Consensus strengths near the published reference (for a **small** sensitivity grid)

Reference reminders: correctness indifferent **2**, weak **5**, strong **25**; resource indifferent **2**, strong **50**.

| Cut role | Value | Consensus | Notes |
|---|---:|---|---|
| Correctness indifferent (tight **alternate**) | **1** | **Moderate** | Near-lossless accuracy-pp band: Wu 2020; Krishnamoorthi QAT; Nagel; NNCF; multiple white papers stating “<1%” / “within 1%”. Borrowed magnitude for RI%. |
| Correctness indifferent (**reference**) | **2** | **Weak–moderate corroboration** | Krishnamoorthi PTQ “within 2%”; Jacob “within 2%”. Keep as **expert reference**, not “literature-mandated.” |
| Correctness weak alternate (e.g. 3) | 3 | **None** | Single gray band table. |
| Correctness weak (**reference**) | **5** | **None** as shared “weak effect” cut | Occasional “<5%” experimental results or “within 5% of 8-bit” — wrong role / wrong baseline. |
| Correctness strong (**reference**) | **25** | **None** | No competing published correctness intensity table at ~25%. |
| Resource indifferent (**reference**) | **2** | **None** | Resource papers speak in 2×/4×/tens of %, not a 2% indifference band. |
| Resource strong (**reference**) | **50** | **Moderate** as ~2× / 50% **latency** | Jacob “50% reduction in running time.” |
| Resource strong (**alternate**) | **75** | **Strong** as **4× memory → 75% size RI** | Jacob; Gholami; Lin ACL; Krishnamoorthi/Nagel. Same *strong* role, different metric (storage vs latency). |
| Resource strong energy ~30 | 30 | **None** | Drop. |

---

## 6. Recommended sensitivity grid (one cut at a time)

Include **only** non-reference points with **≥ moderate** multi-source consensus on the **same magnitude role**. Matches ADR 0014.

| Scale | Property | Values | Non-ref citation trail |
|---|---|---|---|
| Functional suitability | indifferent (`weak_indifferent_effect`) | {**1**, **2**} | **1** ← near-lossless accuracy band (Wu arXiv:2004.09602; Krishnamoorthi QAT 1%; Jacob/Krishnamoorthi 2% as ref corroboration; NNCF `max_drop=0.01`). Unit: accuracy pp borrowed as RI% magnitude. |
| Resource / performance | strong (`strong_effect`) | {**50**, **75**} | **75** ← 4× FP32→INT8 memory (Jacob CVPR 2018; Gholami CRC 2022; Lin ACL 2020; Krishnamoorthi/Nagel). **50** stays latency-oriented reference (Jacob 50% / 2×). |

### Explicitly **not** in the grid

- Functional weak {3, **5**} — no moderate consensus alternate near 5.
- Functional strong {alt, **25**} — no consensus alternate.
- Resource indifferent {1, **2**} — no resource indifference consensus.
- Resource strong **30** (energy) — no multi-source consensus.
- Arithmetic ±1 grids without literature role — excluded by design (ADR 0014).
- Full replacement of the published table by Cohen/Sawilowsky/Ferguson/Kampenes — **wrong quantity** (*d*/*g*/*r*).

---

## 7. Gaps (do not paper over)

1. **SSM has no numeric RI%→intensity standard** — only Likert atoms + “arbitrate from the DV domain.”
2. **No competing full seven-/thirteen-band RI% tables** found in SE secondary studies or quantization surveys.
3. **Classic effect-size rules are standardized**; adapting them to % change would be **invention**, not citation.
4. **Middle correctness cuts (5, 10, 15, 20, 25)** and **resource indifferent (2)** remain **expert-only** for sensitivity purposes.
5. Accuracy **pp** ≠ RI%; appendix prose must state magnitude borrowing explicitly.
6. **50 vs 75** are both “strong resource” but typically justified by **latency vs storage** — sensitivity tests encoding robustness, not a claim that literature prefers 75 over 50 for all resource metrics.

---

## 8. Strongest white-literature candidates (short list for parent summary)

1. **Jacob et al., CVPR 2018** — within **2%** accuracy of FP; **~50%** runtime reduction; **~4×** memory → **75%** size RI. Best single white paper spanning both scales.
2. **Gholami et al., CRC / survey 2022** — **4×–8×** practical memory/latency reductions (supports **75%+** resource strong alternate).
3. **Kampenes et al., IST 2007** — SE *g* tertiles (0.17 / 0.60 / 1.40 medians); important **negative** result for mapping: standardized, not RI%.
4. **Cohen 1988 / Sawilowsky 2009 / Ferguson 2009** — canonical *d*/*r* labels; **do not** convert to RI% cuts.
5. **SSM corpus (Santos & Travassos 2013/2017; Kanban JSERD 2018)** — confirms **absence** of published RI% intensity tables.

For the **implemented** grid: **correctness indifferent 1%** (near-lossless white+gray cluster) and **resource strong 75%** (4× memory white consensus) remain the only literature-backed one-at-a-time alternates near the published reference.
