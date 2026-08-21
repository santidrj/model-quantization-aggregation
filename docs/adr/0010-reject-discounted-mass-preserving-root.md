# Do not publish a discounted mass-preserving root split

The published dependence sensitivity is the undiscounted simple-support root \(1-(1-B)^{1/N}\). A natural fifth assignment would take each model's already-discounted support mass \(B'\) and assign \(1-(1-B')^{1/N}\). We evaluated that variant on all 11 publication effects and keep it out of the replication tables and the manuscript: it confounds sample-size/variability discounting with the dependence cap, and it is not a stronger or clearer stress test than the undiscounted root plus leave-one-study-out.

**Considered options.** Root-then-discount \([1-(1-B)^{1/N}]\alpha_n\alpha_v\) breaks recovery of \(B\) (or of \(B'\)) when agreeing models are combined, and on this corpus it differs from discount-then-root by at most three integer-percent belief points. Discount-then-root \(1-(1-B')^{1/N}\) is the coherent D–S candidate: it matches the published analogue when \(N=1\), and it recovers \(B'\) when all \(N\) models agree and share the same \(B'\). That recovery is not identifiable here: \(B'\) already varies within study for Accuracy (spread 0.40 on S14), inference latency (0.31 on S6), inference energy (0.21 on S16), and storage (0.14 on S6). Using a study-level root on top of per-model \(B'\) is also not a clean “main analysis plus dependence cap”: \(\alpha_n\) and \(\alpha_v\) are effect- and configuration-specific, so the split no longer answers a single study-level estimand.

**Empirical impact** (Evidence Factory-compatible selector; integer-percent belief):

| Effect | Main | Undiscounted root | Discounted root |
|---|---:|---:|---:|
| Accuracy | {WN, IF} 99% | IF 83% | {WN, IF} 81% |
| F1 Score | IF 75% | IF 74% | IF 66% |
| mAP | IF 45% | IF 45% | IF 31% |
| Storage Size | SP 100% | SP 100% | SP 100% |
| GPU Utilization | {IF, WP} 97% | IF 74% | IF 72% |
| GPU Power Draw | {IF, WP} 98% | {IF, WP} 96% | {IF, WP} 93% |
| GPU Energy Consumption | SP 74% | SP 85% | SP 74% |
| RAM Usage | SP 47% | PO 41% | SP 52% |
| Inference Power Draw | WP 74% | WP 78% | {IF, WP} 85% |
| Inference Energy Consumption | SP 100% | SP 99% | SP 96% |
| Inference Latency | {PO, SP} 100% | SP 92% | {PO, SP} 95% |

On the headline dependence findings, discounted root *reverts toward the main intensities* (Accuracy stays {WN, IF}; latency stays {PO, SP}) while the undiscounted root is the variant that actually changes them. Distinctive discounted-root drops (mAP 45% → 31%) come from stacking the root on already-discounted sparse effects, not from a clearer independence diagnostic. The variant has no Evidence Factory parity. Leave it unpublished; do not add a `MassAssignment` member or a fifth appendix column unless a later reviewer asks for this hybrid explicitly.
