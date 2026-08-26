# Consensus literature cut-points for RI→intensity sensitivity

**Status:** consensus-focused rewrite (2026-08-26). Prior draft mixed one-off gray bands with white resource magnitudes; this note separates **recurring** magnitudes from **isolated** sources.  
**Method:** Consensus.app searches (peer-reviewed preferred) + direct reads of named surveys / white papers / MLPerf Tiny / NNCF docs. Gray literature is included only when it **corroborates** a white-side magnitude role, and is labeled.

## Unit caveat (read first)

Almost all domain sources cut **accuracy percentage points** (Top-1 / mAP / task score drop vs FP32), **compression ratios** (e.g. 4× memory), or **latency/energy % reductions**. This project maps **mean relative improvement (RI%)** into intensity atoms. Near high baselines, accuracy-pp and RI% can be numerically close; they are **not** the same unit. Appendix prose must say the sensitivity grid borrows *practically meaningful % magnitudes*, not a formal pp↔RI% conversion.

There is **no** published universal RI%→Likert table in quantization or SSM white literature.

## Survey / standard sources (explicitly searched)

| Source | Venue / status | Numeric cut table? | What it actually gives |
|---|---|---|---|
| Gholami et al., *A Survey of Quantization Methods for Efficient Neural Network Inference* | Book chapter, *Low-Power Computer Vision*, CRC, 2022, [doi:10.1201/9781003162810-13](https://doi.org/10.1201/9781003162810-13) (also arXiv:2103.13630) | **No** Likert / RI cuts | States that moving to low-bit ints can cut memory/latency by up to 16× in theory, and that **4×–8× reductions are often realized in practice**. No 1/2/5/25 accuracy bands. |
| Nagel et al., *A White Paper on Neural Network Quantization* | **Gray** arXiv:2106.08295 only (no journal/conference version found) | Not a cut table | Reports experimental drops “**within 1%**” of FP for ResNet/Inception W4A8; MobileNet-class nets can be worse (~2.5–4.2% per-tensor). States FP32→INT8 memory **factor of 4**. |
| Krishnamoorthi, *Quantizing deep convolutional networks for efficient inference: A whitepaper* | **Gray** arXiv:1806.08342 | Soft bands, not Likert | PTQ 8-bit “**within 2%**” of FP for many CNNs; QAT closes gap to “**1%**”; model size “**factor of 4**”; CPU speedup “**2×–3×**”. |
| Rokh et al., quantization survey | ACM TIST 2023 (Consensus: [paper](https://consensus.app/papers/details/84fa2e7487d554daa0f436c9196495ac/)) | **No** shared % cut table | Methods/metrics survey; no RI→intensity prescription. |
| Menghani, efficient-DL survey | ACM Computing Surveys 2023 | **No** | Efficiency landscape survey; no intensity cut table. |
| Banbury et al., *MLPerf Tiny Benchmark* | NeurIPS Datasets & Benchmarks 2021 | **Absolute** floors, not relative % | Closed-division quality targets (e.g. KWS **90%**, IC **85%**, VWW **80%**) to absorb quantization/rounding variation. **Not** “≤X% drop vs FP”. |
| OpenVINO / NNCF accuracy control | **Gray** industry toolkit docs/API | Default **max_drop = 0.01 (1%)** absolute metric drop | Corroborates ≤1% as the default “accuracy-aware” stop criterion; not peer-reviewed. |

**Takeaway from surveys/standards:** white surveys do **not** publish a reusable indifference/weak/strong RI% grid. The only recurring *numeric* themes are (a) **≤1–2% accuracy-pp** as “near FP / acceptable PTQ/QAT,” and (b) **~4× memory** (and ~2× latency) as typical INT8 resource gains.

---

## Candidate cuts near our reference grid

Reference cuts in this project: correctness indifferent~**2**, weak~**5**, strong~**25**; resource indifferent~**2**, strong~**50**.

### 1. Correctness — ~1% (alternate for indifferent; ref is 2)

| | |
|---|---|
| **Numeric value** | **1** (≤1% accuracy drop / max_drop) |
| **Quantity cut** | Accuracy **percentage points** (or absolute metric drop), **not** RI% |
| **Magnitude role** | Near-lossless / negligible / accuracy-control success threshold |
| **Independent sources** | **White:** Gennari et al., ICCV 2019 ([consensus](https://consensus.app/papers/details/75357237e979534e9ee7ee85e3dfc291/)) — “less than 1% loss of accuracy” (4-bit PTQ claim); Jiang et al., DAC 2019, [doi:10.1145/3316781.3317757](https://doi.org/10.1145/3316781.3317757) — “less than 1% accuracy loss”; Zhang et al., IEEE Access 2024 (COMQ) — “negligible … less than 1%” Top-1. **Gray (corroborating):** Krishnamoorthi arXiv:1806.08342 — QAT gap “to 1%”; Nagel arXiv:2106.08295 — ResNet W4A8 “within 1%”; NNCF/OpenVINO `max_drop=0.01` default. |
| **Consensus strength** | **Moderate** — same *near-lossless* role recurs across ≥3 white venues + independent gray toolkit/whitepapers. Not a formal SSM intensity table. |

### 2. Correctness — ~2% (our reference indifferent)

| | |
|---|---|
| **Numeric value** | **2** |
| **Quantity cut** | Accuracy **pp** (PTQ “within 2% of FP”; also Jacob reports ~2% AP drop in one detection setup) |
| **Magnitude role** | Typical “still close to FP / acceptable PTQ” band — aligns with **indifferent** magnitude, not “weak effect” |
| **Independent sources** | **Gray primary statement:** Krishnamoorthi arXiv:1806.08342 — PTQ “within 2% of floating point.” **White corroboration (descriptive, not a prescribed cut):** Jacob et al., CVPR 2018, [doi:10.1109/CVPR.2018.00286](https://doi.org/10.1109/CVPR.2018.00286) — “~2% drop” in face-detector AP under quantization; COCO MobileNet-SSD “minimal loss (−1.8% relative).” |
| **Consensus strength** | **Weak–moderate for corroborating the reference**; not enough independent *prescriptive* white cuts to treat 2 as a literature consensus cut in its own right. Keep as **expert reference**, not as “literature-mandated.” |

### 3. Correctness — ~3% (Liu fair→risky; proposed alt for weak=5)

| | |
|---|---|
| **Numeric value** | **3** |
| **Quantity cut** | Accuracy drop bands (lossless ≤1% / fair 1–3% / risky ≥3%) |
| **Independent sources** | **One** gray source: Liu et al., *Quantization Hurts Reasoning?*, arXiv:2504.04823 |
| **Consensus strength** | **None** — isolated; do **not** put in the sensitivity grid. |

### 4. Correctness — ~5% (our reference weak)

| | |
|---|---|
| **Numeric value** | **5** |
| **Quantity cut** | Mixed / often **wrong quantity** when it appears |
| **What sources actually say** | Krishnamoorthi: 4-bit weights “within **5% of 8-bit**” (not vs FP; not “weak effect”). TI TinyML auto-quant docs: relative tolerance **0.05** (**gray** industry). Yang et al., *Algorithms* 2023: experimental “accuracy loss of less than 5%” for a 5-bit net (result, not a cut role). |
| **Consensus strength** | **None** for “weak effect = 5% RI/pp.” No 2+ independent sources agree on **5** as that magnitude role. Keep as expert reference only; **no consensus alternate**. |

### 5. Correctness — ~25% (our reference strong)

| | |
|---|---|
| **Numeric value** | **25** |
| **Independent sources agreeing on this as a correctness intensity cut** | **None found** |
| **Consensus strength** | **None** |

### 6. Resource indifferent — ~2% (our reference)

| | |
|---|---|
| **Numeric value** | **2** |
| **Independent sources using ~2% as a resource RI / latency / energy indifference band** | **None found** (resource literature talks in 2× / 4× / tens of percent, not 2% indifference) |
| **Consensus strength** | **None** for a non-reference alternate (e.g. 1%). Liu ≤1% is accuracy, not resource. |

### 7. Resource strong — ~50% (our reference; latency RI)

| | |
|---|---|
| **Numeric value** | **50** (≈ **2×** latency / “50% reduction in running time”) |
| **Quantity cut** | **Latency / runtime % reduction** (≡ RI 50% if RI = relative latency improvement) |
| **Independent sources** | **White:** Jacob et al., CVPR 2018, [doi:10.1109/CVPR.2018.00286](https://doi.org/10.1109/CVPR.2018.00286) — “up to a **50%** reduction in running time”; also “close to a **2×** latency reduction.” **Gray corroborating:** Krishnamoorthi arXiv:1806.08342 — “**2×–3×** speed-up on a CPU.” Additional white papers report ~2×–3× INT8 speedups in specific stacks (e.g. edge INT8 characterization studies), but magnitudes are hardware-dependent. |
| **Consensus strength** | **Moderate** for “~50% / ~2× latency is a typical **strong** INT8 runtime gain” — supports keeping **50** as the reference strong cut; does **not** by itself supply a distinct non-ref alternate. |

### 8. Resource strong — ~75% (4× memory; non-ref alternate)

| | |
|---|---|
| **Numeric value** | **75** (RI from **4×** compression: \(1 - 1/4 = 0.75\)) |
| **Quantity cut** | **Memory footprint / model size** compression ratio — **not** latency, **not** energy |
| **Independent sources** | **White:** Jacob et al., CVPR 2018 — “close to **4×** memory footprint reduction” (8-bit vs FP32). Gholami et al., CRC 2022, [doi:10.1201/9781003162810-13](https://doi.org/10.1201/9781003162810-13) — “**4× to 8×** are often realized in practice.” Lin et al., ACL 2020 (Integer Transformer; Consensus) — “nearly **4×** less memory footprint.” **Gray corroborating:** Krishnamoorthi — “factor of **4**”; Nagel — memory decreases by a “factor of **4**” for 32→8. |
| **Consensus strength** | **Strong** for the **4× memory / ~75% size RI** magnitude. Caveat: same *role* (strong resource) as 50, but **different metric** (storage vs latency). |

### 9. Resource strong — ~30% energy (prior draft alternate)

| | |
|---|---|
| **Numeric value** | **30** |
| **Independent sources** | Essentially **one** gray LLM energy characterization (arXiv:2508.16712 family); energy savings in white literature are highly system-specific (tens of percent to >50%) without a shared 30% cut. |
| **Consensus strength** | **None** — drop from sensitivity grid. |

---

## Honesty summary (consensus strengths)

| Cut role (near our refs) | Value | Consensus |
|---|---:|---|
| Correctness indifferent (tight alt) | 1 | **Moderate** (accuracy pp, near-lossless) |
| Correctness indifferent (ref) | 2 | **Weak–moderate** corroboration only |
| Correctness weak (Liu alt) | 3 | **None** |
| Correctness weak (ref) | 5 | **None** as a shared weak-effect cut |
| Correctness strong (ref) | 25 | **None** |
| Resource indifferent (ref) | 2 | **None** (no resource indifference band) |
| Resource strong (ref) | 50 | **Moderate** as ~2× / 50% **latency** |
| Resource strong (4× memory) | 75 | **Strong** as **memory** RI |
| Resource strong (energy) | 30 | **None** |

---

## Recommended sensitivity grid

Include **only** non-reference points with **≥ moderate** consensus (2+ independent sources, same magnitude role). One cut perturbed at a time. Bold = published expert reference.

| Scale | Property | Values | Non-ref citation trail |
|---|---|---|---|
| Functional suitability | `WEAK_INDIFFERENT_EFFECT` | {1, **2**} | **1** ← near-lossless accuracy-pp band (ICCV/DAC/IEEE Access + NNCF/Krishnamoorthi/Nagel corroboration). Unit: accuracy pp ≈ borrowed magnitude for RI%. |
| Resource / performance | `STRONG_EFFECT` | {**50**, 75} | **75** ← 4× memory footprint consensus (Jacob CVPR; Gholami CRC; Lin ACL; Krishnamoorthi/Nagel). Quantity: **size RI**, not latency. **50** remains the latency-oriented reference (Jacob 50% / 2×). |

### Explicitly **not** in the grid

- Functional `WEAK_EFFECT` {3, **5**} — **no** moderate consensus for 3 (or any non-ref near 5).
- Functional strong {alt, **25**} — **no** consensus alternate.
- Resource `WEAK_INDIFFERENT_EFFECT` {1, **2**} — **no** resource indifference consensus.
- Resource strong third point **30** — **no** energy consensus.

If reviewers ask why weak/strong-correctness / resource-indifferent are not stressed: answer is literature-honest — those reference cuts lack independent corroborating alternates at the same magnitude role.
