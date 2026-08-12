# Branch audit

This file explains what every branch family represents. Branches are retained as experiment lineage, not presented as independent paper conclusions. The integrated `main` branch is the documentation and evidence landing point.

## Final shape

The cleaned repository contains **170 branches**:

| Family | Count | Naming rule | Role |
| --- | ---: | --- | --- |
| `main` | 1 | `main` | Integrated evidence and documentation. |
| `check/*` | 2 | `check/exchangeability`, `check/invariants` | Focused validity and invariant checks. |
| `control/*` | 18 | `control/noshift-*` | No-shift controls and metadata runs. |
| `real/*` | 78 | `real/{dataset}`, `real/{dataset}-s{0..4}`, `real/{dataset}-w{0..9}` | Table 1 entry points and repeat/shard lineage. |
| `sim/*` | 71 | `sim/logabs-n{30,100,500}-m{30,100,500}-*` | LogAbs simulation settings and repeat/shard lineage. |

All names use `check/`, `control/`, `real/`, or `sim/` prefixes. The former ambiguous metadata suffixes were renamed as follows:

| Old branch | Clean branch | Meaning |
| --- | --- | --- |
| `control/noshift-w7m` | `control/noshift-w7-metadata` | Integrated no-shift metadata/evidence lineage. |
| `control/noshift-w8m` | `control/noshift-w8-metadata` | No-shift metadata lineage. |
| `control/noshift-w9m` | `control/noshift-w9-metadata` | No-shift metadata lineage. |

There are no `orx/*` or `master` branches in the final layout.

## `check/*`

- `check/invariants` — registered implementation and evidence invariants used by the claim gate.
- `check/exchangeability` — exchangeable/non-exchangeable calibration controls for the Theorem 4.7 audit.

## `control/*`

The family has 18 branches:

- `control/noshift-s0` through `control/noshift-s4` — five early no-shift control shards.
- `control/noshift-w0` through `control/noshift-w9` — ten worker/result shards.
- `control/noshift-w7-metadata`, `control/noshift-w8-metadata`, and `control/noshift-w9-metadata` — metadata-bearing worker lineages, renamed from the old `w7m`, `w8m`, and `w9m` suffixes.

These branches are controls or provenance, not additional paper methods. In particular, the strong non-exchangeable control is intended to leave the C6 coverage band; an in-band control would provide weak evidence for the selection claim.

## `real/*`

The 78 real-data branches cover five Table 1 dataset families:

- Base entry points: `real/bio`, `real/crime`, `real/derma`, `real/star`, `real/tissue` — 5 branches.
- Early shard lineage: `{dataset}-s0` through `{dataset}-s4` for each dataset — 25 branches.
- Worker lineage: `{dataset}-w0` through `{dataset}-w9` — 50 possible branches, with the original CRIME lineage missing `w4` and `w5`, leaving 48 branches.

Therefore the real family count is `5 + 25 + 48 = 78`.

The current integrated analysis has complete repeat-shard coverage for BIO, CRIME, and DERMA. STAR has only 20 of 50 recorded repeats, and TISSUE is not complete in the current merged evidence. CRIME's integrated shard layout includes a larger `15–30` interval from the available `CRIME-s2` lineage; the analysis records this provenance rather than hiding the gap.

## `sim/*`

The simulation branches are grouped by calibration size `n` and target sample size `m`:

| Setting | Branches | Count | Role |
| --- | --- | ---: | --- |
| `logabs-n30-m30` | `s0..s4`, `w0..w9` | 15 | Small target-sample control. |
| `logabs-n30-m100` | `s0..s4`, `w0..w9` | 15 | Small-calibration setting. |
| `logabs-n30-m500` | `s0..s4`, `w4..w9` | 11 | Main small-calibration setting with available worker lineage. |
| `logabs-n100-m500` | `s0..s4`, `w0..w9` | 15 | Intermediate calibration setting. |
| `logabs-n500-m500` | `s0..s4`, `w0..w9` | 15 | Large-calibration setting. |

This gives `15 + 15 + 11 + 15 + 15 = 71` simulation branches. Branch names identify the setting and shard role; they do not by themselves mean that every shard was merged into the current evidence snapshot. The merged analysis currently registers `logabs-n30-m30`, `logabs-n30-m100`, `logabs-n30-m500`, and `logabs-n100-m500`; missing settings remain visible as incomplete rather than being inferred.

## Evidence mapping

| Paper claim | Primary branch family | Final evidence |
| --- | --- | --- |
| C1, Eq. 7 / Algorithm 1 | `control/*`, `sim/*` | Source-only provenance, objective identity, regularization path, and intervention checks in `results/analysis.json`. |
| C2, Theorem 4.2 | `check/invariants`, `control/*` | Proof certificate in `src/verify_thm42_certificate.py`; controls are scoped corroboration. |
| C3, Theorem 4.6 | `sim/*`, especially varying `n` and `m` | Rate certificate and missing-capability flags; currently `BLOCKED`. |
| C4, Table 1 | `real/*` | Paper-table arithmetic plus available BIO/CRIME/DERMA reruns; currently `FALSIFIED` by the printed TISSUE/GLCP range. |
| C5, Table 2 | `sim/*` | LogAbs shard merge and registered GLCP/negative-control contract. |
| C6, Theorem 4.7 | `check/exchangeability`, `control/*` | Exchangeability controls and exhaustive selection-band repair audit. |

The exact adjudication is in [`results/analysis.json`](results/analysis.json); this document is the branch-to-purpose map, not a substitute for the evidence records.
