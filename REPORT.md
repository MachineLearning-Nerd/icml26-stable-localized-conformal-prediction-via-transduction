# Reproduction report

## Bottom line

`PARTIAL_CLAIMS_1_TO_2_VERIFIED_SCOPED_CLAIM_3_BLOCKED_CLAIM_4_SOURCE_BAND_FALSIFIED_CLAIM_5_GLCP_SCOPED_VERIFIED_CLAIM_6_REPAIRED_PROOF_SCOPED_VERIFIED`

The repository records `10/12` self-scored claim points. This is an internal
audit score, not a current official score, publication decision, or author
endorsement.

## Claim results

| Claim | Result | Evidence boundary |
| --- | --- | --- |
| C1 | Verified, scoped | Method formulation and data-flow provenance; source-swap null is reported rather than scored. |
| C2 | Verified, proof algebra only | Certificate checks Appendix B.3-style algebra; assumptions and empirical universality remain open. |
| C3 | Blocked | Certificate and lambda curves exist, but the required calibration-size and bootstrapped-rate evidence is incomplete. |
| C4 | Falsified as written | The printed TISSUE/GLCP value `13.5%` contradicts the stated `20–48%` band; real reruns are not complete for all five datasets. |
| C5 | Verified for the registered GLCP contract | Available simulation/control evidence passes; the CQR ordering sentence is internally off by `0.4` points. |
| C6 | Verified as a repaired proof audit | The printed lower endpoint needs a finite-sample repair; the corrected statement and informative controls pass. |

## Publication boundary

- `publication_allowed=false`
- `score_claim=false`
- `official_author_endorsement=false`
- `current_official_score`: not recorded
- Boundary: `C3_INCOMPLETE_CALIBRATION_SIZES_C4_SOURCE_TABLE_CONTRADICTION_C6_PRINTED_ENDPOINT_REPAIRED_NO_FULL_PAPER_REPRODUCTION`

The complete machine-readable records are [`claims.json`](claims.json),
[`reproduction_verdicts.json`](reproduction_verdicts.json), and
[`results/analysis.json`](results/analysis.json).

