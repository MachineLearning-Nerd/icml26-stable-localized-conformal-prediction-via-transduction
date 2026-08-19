# Source audit summary

The detailed source record is [`.openresearch/artifacts/source_audit.md`](.openresearch/artifacts/source_audit.md). This short file makes the provenance boundary visible at repository root.

| Item | Pinned value |
| --- | --- |
| Paper | *Stable Localized Conformal Prediction via Transduction* |
| arXiv | `2605.01452` |
| OpenReview | `lSMTccAN61` |
| Retrieved source | `https://ar5iv.labs.arxiv.org/html/2605.01452` |
| Retrieval date | `2026-08-01` UTC |
| Retrieved HTML SHA-256 | `d07bc37a6a81e0c74aef488fd566dfe5ebf4e0b94ad526c3449334000c2741a2` |
| Authors' code | `https://github.com/OswinMin/StCP` |
| Authors' code commit | `1d8df7614d49eada881426742688ba75fec631b9` |

The claim contract is [`.openresearch/artifacts/claim_contract.json`](.openresearch/artifacts/claim_contract.json). It records the paper anchors, quantifiers, assumptions, negative controls, and non-circularity rules. The central source traps retained by the audit are the two different Table 1/Table 2 improvement formulas, the different coverage-annotation thresholds, the target-pool split for regression datasets, and the finite-sample lower-endpoint gap in Theorem 4.7.

The C4 contradiction is source-internal arithmetic: Table 1 prints TISSUE/GLCP at `13.5%`, while the claim states a GLCP range beginning at `20%`. It does not depend on an uncertain rerun or an unofficial interpretation of the results.

