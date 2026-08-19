# Claim-to-evidence ledger

This ledger explains how each paper claim is produced in the repository. A
verdict applies only to the registered contract and evidence path; it does not
turn a scoped audit into an official paper re-evaluation.

## Source identity

- Paper: *Stable Localized Conformal Prediction via Transduction*
- Authors: Yinjie Min, Liuhua Peng, and Changliang Zou
- arXiv: [2605.01452](https://arxiv.org/abs/2605.01452)
- OpenReview: [lSMTccAN61](https://openreview.net/forum?id=lSMTccAN61)
- Authors' implementation: [OswinMin/StCP](https://github.com/OswinMin/StCP)
- Pinned upstream commit: `1d8df7614d49eada881426742688ba75fec631b9`
- Paper-source snapshot: `ar5iv` HTML SHA-256 `d07bc37a6a81e0c74aef488fd566dfe5ebf4e0b94ad526c3449334000c2741a2`

## Claim paths

| Claim | Status | Producing path | What is actually supported | Boundary |
| --- | --- | --- | --- | --- |
| C1 | `verified_scoped` | `check/invariants` and `control/*` -> `src/verify_transcription.py` plus source-only/intervention records -> `results/analysis.json` | Eq. 7 / Algorithm 1 uses source-trained conditional estimation, unlabeled target covariates, and the registered regularized objective. | The source-swap comparison did not detect a distribution effect; it is reported, not adjudicated. The pass is formulation/provenance scoped. |
| C2 | `verified_scoped_proof_algebra` | `check/invariants` -> `src/verify_thm42_certificate.py` -> `results/analysis.json` | The registered algebraic steps of the Theorem 4.2 bound pass exhaustive mutation checks. | Assumption 4.1, the existence of a useful constant, and all empirical consequences are not established for every DGP. |
| C3 | `blocked_empirical_rate_evidence` | `sim/*` -> `src/verify_thm46_certificate.py` -> `results/analysis.json` / `results/EVAL.md` | The proof certificate covers the algebraic skeleton and the available curves decrease with lambda. | Only `n={30,100}` is available in the adjudicated base sweep; no bootstrapped non-zero slope interval or all-three-size rate evidence exists. |
| C4 | `falsified_source_table_band` | `real/*` -> `src/verify_claim4_band.py` plus printed Table 1 arithmetic -> `results/analysis.json` | The paper's printed TISSUE/GLCP value is `13.5%`, below the stated `20–48%` GLCP band by `6.5` points. | This is a source-table contradiction, not a claim that every real-data rerun failed. BIO/CRIME/DERMA are complete; STAR/TISSUE are incomplete. |
| C5 | `verified_scoped_glcp_contract` | `sim/*` and no-shift/control lineage -> `src/verify_claim4_band.py` -> `results/analysis.json` | The registered GLCP contract, shard merge, bootstrap at `n=30`, and informative `m=n` control pass for available settings. | The paper's CQR “largest at n=30” wording is contradicted by its own table by `0.4` points; `n=500` is not complete in the merged analysis. |
| C6 | `verified_scoped_repaired_proof_audit` | `check/exchangeability` and `control/*` -> `src/verify_thm47_certificate.py` -> `results/analysis.json` | The exhaustive audit identifies the printed lower-endpoint gap and verifies the widened or inflated-quantile repair; controls make the band informative. | This is a repaired proof/contract audit. It does not claim that an observed run violates the printed band or that the uncorrected theorem is established. |

## Production sequence

```text
paper/source snapshot
  -> .openresearch/artifacts/source_audit.md
  -> .openresearch/artifacts/claim_contract.json
  -> config/node.json and branch-specific run artifacts
  -> src/verify_*.py and src/stage_analysis.py
  -> results/analysis.json and results/EVAL.md
  -> claims.json / reproduction_verdicts.json / STATUS.md
```

The fixed entry point is `bash run.sh`. Existing outputs are the evidence
snapshot used for this dossier; no missing run is inferred from a branch name.

## Branch-to-claim map

- `check/*`: invariant, exchangeability, and proof checks for C1/C2/C6.
- `control/*`: no-shift, exchangeability, and metadata/control lineage.
- `real/*`: Table 1 dataset entry points and repeat shards for C4.
- `sim/*`: LogAbs `(n,m)` settings and repeat shards for C3/C5/C6.
- `main`: integrated analysis, documentation, and publication boundary.

