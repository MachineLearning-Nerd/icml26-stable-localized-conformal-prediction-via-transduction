# Stable Localized Conformal Prediction via Transduction

Reproduction and claim-audit workspace for the ICML 2026 submission associated with OpenReview forum `lSMTccAN61`.

## Paper

**Stable Localized Conformal Prediction via Transduction**<br>
Yinjie Min, Liuhua Peng, and Changliang Zou

- Paper: [arXiv:2605.01452](https://arxiv.org/abs/2605.01452)
- OpenReview: [lSMTccAN61](https://openreview.net/forum?id=lSMTccAN61)
- Authors' implementation: [OswinMin/StCP](https://github.com/OswinMin/StCP), audited at commit [`1d8df7614d49eada881426742688ba75fec631b9`](https://github.com/OswinMin/StCP/tree/1d8df7614d49eada881426742688ba75fec631b9)
- Previous collection name: `icml26-repro-lSMTccAN61-stable-localized-conformal-prediction-via-transduction`

## What the paper is doing

Localized conformal prediction can produce prediction sets whose size varies substantially when the calibration sample is small. The paper proposes Stable Conformal Prediction (StCP), including Stable Localized Conformal Prediction (SLCP), as a transductive transfer-learning procedure:

1. Fit a conditional CDF or predictor on labeled **source** data.
2. Use an unlabeled **target** covariate sample to estimate the target-side marginal behavior.
3. Fit a regularized conditional CDF close to the source fit while matching the target calibration distribution.
4. Use the resulting calibrated score distribution to form prediction intervals or sets.

The central trade-off is controlled by `lambda`: larger regularization keeps the target fit closer to the source-trained model, while smaller regularization gives the unlabeled target sample more influence.

## Current audit outcome

The committed evidence is an auditable reproduction snapshot, not a claim that every published experiment has been rerun to completion. The verifier reports 10 of 12 self-scored claim points; its exit code is `1` because C3 is blocked.

| Claim | Paper object | Outcome | How the outcome is produced |
| --- | --- | --- | --- |
| C1 | Eq. 7, Section 3, Algorithm 1: transductive StCP/SLCP construction | **VERIFIED** | Source-only training provenance, objective identity, regularization-path checks, and source/target intervention evidence in `results/analysis.json`. |
| C2 | Theorem 4.2: coverage robustness to `lambda` | **VERIFIED — proof algebra only** | Exhaustive certificate checks in `src/verify_thm42_certificate.py`; this does not establish the theorem's assumptions for every data-generating process. |
| C3 | Theorem 4.6: prediction-set variance rates | **BLOCKED** | The algebraic skeleton is checked, but the available evidence lacks all three calibration sizes and a bootstrapped non-zero decay-rate interval. |
| C4 | Table 1: five real datasets and stated percentage bands | **FALSIFIED** | The paper's own printed TISSUE/GLCP value is `13.5%`, outside its stated `20–48%` GLCP range by `6.5` points; this is not a rounding difference. Current reruns are complete for BIO, CRIME, and DERMA, while STAR/TISSUE remain incomplete. |
| C5 | Table 2: LogAbs simulation and the largest GLCP gain at `n=30` | **VERIFIED for the registered GLCP contract** | Synthetic shard merge and negative-control checks in `src/verify_claim4_band.py` and `results/analysis.json`; the available merged settings are `n=30` and `n=100`. The paper's CQR wording that the `n=30` gain is largest is contradicted by its own table by `0.4` points. |
| C6 | Theorem 4.7: data-driven `lambda` selection | **VERIFIED as a repaired proof audit** | Exchangeability controls and the exhaustive certificate in `src/verify_thm47_certificate.py`. The printed lower endpoint does not follow from the selection rule; the supported repair widens it by `(n+1)^-1` or equivalently inflates the lower quantile level. |

The machine-readable source of truth is [`results/analysis.json`](results/analysis.json), with the short report in [`results/EVAL.md`](results/EVAL.md). “Verified” is always scoped to the registered contract and evidence listed above; it is not an independent endorsement of every theorem assumption or every paper sentence.

Repository-level audit boundary: `PARTIAL_CLAIMS_1_TO_2_VERIFIED_SCOPED_CLAIM_3_BLOCKED_CLAIM_4_SOURCE_BAND_FALSIFIED_CLAIM_5_GLCP_SCOPED_VERIFIED_CLAIM_6_REPAIRED_PROOF_SCOPED_VERIFIED`.

Publication boundary: `C3_INCOMPLETE_CALIBRATION_SIZES_C4_SOURCE_TABLE_CONTRADICTION_C6_PRINTED_ENDPOINT_REPAIRED_NO_FULL_PAPER_REPRODUCTION`. `publication_allowed=false`, `score_claim=false`, and `official_author_endorsement=false`. The repository records 10/12 self-scored audit points; no current official score or author endorsement is claimed. See [`STATUS.md`](STATUS.md), [`CLAIM_EVIDENCE.md`](CLAIM_EVIDENCE.md), [`REPORT.md`](REPORT.md), and [`reproduction_verdicts.json`](reproduction_verdicts.json).

## How each claim is produced

The claim contract is recorded in [`.openresearch/artifacts/claim_contract.json`](.openresearch/artifacts/claim_contract.json). The implementation and evidence path are:

| Evidence layer | Purpose |
| --- | --- |
| `.openresearch/artifacts/source_audit.md` | Pins the paper/source snapshot, constants, equations, code anchors, table formulas, and known transcription traps. |
| `.openresearch/artifacts/method.md` | Defines the reproduction design, node map, fixed commands, shard layout, controls, and scoring semantics. |
| `src/stage_analysis.py` | Merges committed synthetic/real evidence and writes the claim-by-claim analysis and `results/EVAL.md`. |
| `src/verify_transcription.py` | Checks that the paper transcription and registered claim contract agree. |
| `src/verify_thm42_certificate.py` | Checks the proof-algebra certificate for Theorem 4.2. |
| `src/verify_thm46_certificate.py` | Checks the available proof skeleton and rate prerequisites for Theorem 4.6. |
| `src/verify_thm47_certificate.py` | Audits the finite-sample lambda-selection band and its repair. |
| `src/verify_claim4_band.py` | Checks the registered LogAbs simulation/table contract and controls. |
| `results/real/` and `results/shards/` | Committed repeat shards used by the real-data and simulation analyses. |
| `results/checks/` | Invariants, exchangeability controls, intervention checks, and negative controls. |
| `results/compute_provenance.json` | Records the execution environment and worker provenance where available. |

The fixed entry point is:

```bash
bash run.sh
```

The node configuration in [`config/node.json`](config/node.json) is authoritative for the registered stage. No target labels are allowed to flow into the source-trained conditional CDF or predictor for C1.

## Reproduction notes and limitations

- Global settings include nominal coverage `1 - alpha = 0.90`, `50` repeats, and the paper's source/target LogAbs configuration (`d=5`, `m=500`, `gamma_s=1.2`, `gamma_t=1.0`) where applicable.
- Table 1 uses the authors' oracle-adjusted improvement formula, while Table 2 uses a plain baseline-relative reduction. These are intentionally kept separate.
- Table 1 and Table 2 also use different code-level coverage annotations; see the source audit before comparing thresholds.
- The integrated result set contains BIO, CRIME, and DERMA real-data runs; STAR has partial shards and TISSUE is not complete in the current analysis. The `n=500` LogAbs setting is also not complete in the merged analysis.
- C6's result must be read as a proof/contract audit with a correction, not as unconditional verification of the printed theorem statement.
- The reproducibility status is captured by committed artifacts; missing evidence is recorded as `BLOCKED`, not silently treated as a pass.

## Branches

The repository retains the experiment lineage, but all branch names are descriptive and use the same family vocabulary:

- `main` — integrated landing branch containing the current documentation and merged evidence snapshot.
- `check/*` — focused invariant and exchangeability checks.
- `control/*` — no-shift, exchangeability, and metadata/control runs.
- `real/*` — Table 1 dataset entry points and repeat shards for BIO, CRIME, DERMA, STAR, and TISSUE.
- `sim/*` — LogAbs simulation settings and repeat shards for the registered `(n,m)` combinations.

The exact branch counts, naming rules, and the three metadata-branch renames are documented in [`branch-audit.md`](branch-audit.md). There are no `orx/*` or `master` branches in the cleaned layout.

## Citation

```bibtex
@article{min2026stable,
  title={Stable Localized Conformal Prediction via Transduction},
  author={Min, Yinjie and Peng, Liuhua and Zou, Changliang},
  journal={arXiv preprint arXiv:2605.01452},
  year={2026},
  eprint={2605.01452},
  archivePrefix={arXiv},
  primaryClass={stat.ML}
}
```

## Thanks

Thank you to Yinjie Min, Liuhua Peng, and Changliang Zou for making the paper, implementation, datasets, and pretrained-backbone details available. That public material makes it possible to inspect the method carefully, reproduce its evidence path, and report both successful checks and unresolved limitations transparently.

## Attribution

This collection repository is maintained by [MachineLearning-Nerd](https://github.com/MachineLearning-Nerd). The cleaned history uses the exact commit identity `MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>`.
