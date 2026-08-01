# Method

## Design principle

The previous judged revision failed for one reason, repeated across all six
claims: it was a clean-room reimplementation exercised on a proxy DGP. Every
design decision here follows from removing that gap — **run the authors' own
code on the authors' own data at the paper's own scale**, and add only the
instrumentation needed to measure what each claim actually asserts.

## Upstream artifact

`https://github.com/OswinMin/StCP`, pinned at
`1d8df7614d49eada881426742688ba75fec631b9` (cited in Section 5 of the paper).
It ships:

- `Main/` — the StCP/SLCP implementation (GLCP, SCC/CQR, SDCP, PPI, oracle, DP);
- `SimuAnalysis/` — the Table 2 simulation and the authors' summarisation;
- `RealAnalysis/` — one entry script per Table 1 dataset plus `procedure.py`;
- `Dataset/` — `crimedata.csv`, `proteinStructure.csv`, `achieve.csv`,
  `dermamnist.npz`, `tissuemnist.npz`;
- `RealAnalysis/Para/` — the pretrained DermaMNIST and TissueMNIST backbones,
  so the image experiments need no GPU training.

## Fixed reproduction command

`bash run.sh` on every node, inherited unchanged from the baseline. It syncs the
locked environment, fetches the pinned upstream artifact, and hands control to
`src/run_node.py`, which dispatches on the committed `config/node.json`. No node
varies its command line and no node reads a behavioural environment variable —
all variation is committed code/config on that node's branch.

## Node map

| Branch group | Stage | Serves |
|---|---|---|
| `real/{crime,bio,star,derma,tissue}` | `real` | C4 (Table 1), and the DP control for C2 |
| `sim/logabs-n{30,100,500}-m500-s{0..4}` | `simshard` | C5 (Table 2), C3 n-slope |
| `sim/logabs-n30-m{30,100}-s{0..4}` | `simshard` | C3 m-dependence |
| `control/noshift-s{0..4}` | `simshard` + overrides | C5 negative control |
| `check/invariants` | `invariants` | C1 |
| `check/exchangeability` | `control` | C6 vacuity control |
| — | `analysis` | adjudicates all six, exits nonzero on failure |

## Table 1 configuration

Four of the five entry scripts reproduce Table 1 at their **committed
defaults**; their `SimName` strings match the authors' own result files
(`P_60_500_1334_0_15`, `P_60_1000_2000_4`, `P_30_1000_2000_0.035`,
`P_30_1000_2000_0.07`). STAR requires `60 10 1 1` to select the rural/early-age
target agent, giving `P_60_1048_2446_10_1_1`.

For the three regression datasets the CLI `n` is the size of the *target pool*,
halved by `procedure.py:308` into `calTrAgent`/`calAgent`; the calibration size
reported in Table 1 is `n // 2 = 30`. The two classification datasets pass
`n = 30` through unchanged.

## Repeat sharding

`run_experiment` seeds the shared source-side model once with
`setseed(repeats + 100)` and each repeat independently with `setseed(1 + rep)`.
Repeats are therefore deterministic and mutually independent, so evaluating
repeats `[lo, hi)` in separate jobs yields exactly what a single 50-repeat
process would have produced — **provided `repeats` stays 50**, so the shared
model is untouched. `src/patch_core.py` performs three textual edits (loop
range, progress line, raw-array return) and asserts that no line other than the
loop header is removed.

Shards ship sufficient statistics, not raw arrays: `tools.summation` needs only
per-repeat means plus a per-test-point sum over repeats, so a shard emits
~19k numbers instead of ~1M.

The five real datasets are sharded the same way and for the same reason.
`procedure.py` fixes every shared object — the source predictor, the base
generator, the test-group clustering — before its repeat loop and then seeds
each repeat with `seed_rep = seed + 1 + rep`, so repeats `[lo, hi)` reproduce
exactly the rows a 50-repeat process would compute. `src/patch_procedure.py`
slices both repeat loops and both `summation_real*` functions, which makes each
shard's pickle a valid run over its own repeats in the authors' own keying.

What cannot be sharded is the *summary*. `sum_compare_result` chooses one λ per
method by comparing statistics across the whole grid, and that choice has to be
made once on all 50 repeats — five separate choices on ten repeats each is a
different estimator. So shards ship their raw per-key aggregates and per-repeat
means, and `src/real_reduce.py` rebuilds a single 50-repeat `resDict` before the
authors' selection logic runs on it. Exactness per column:

| Column | Merged how | Exact? |
|---|---|---|
| `mar`, `size` | mean of shard means, equal repeats per shard | exact |
| `mar_std`, `size_std` | recomputed from the pooled per-repeat means, which is the definition | exact |
| `local_cov` | mean of shard values | **approximate** — it is a weighted mean whose per-group weights differ by shard |

`local_cov` feeds no selection in `sum_compare_result` and no quantity in
Claim 4; it is reported for context only. Every column Table 1 prints for the
claim — marginal coverage, mean size, Std, and every improvement percentage —
is exact. `src/real_reduce.py` refuses to merge shards that are non-contiguous
or unequal in width, since the unweighted average is only valid for equal
repeat counts.

## Instrumentation added

- `src/threads.py` — pins OpenMP/MKL/OpenBLAS/torch pools to the cgroup quota
  before numpy or torch is imported. The container reports 64 cores while the
  cgroup grants 8; unpinned, this workload spin-contends.
- `src/patch_procedure.py` — appends per-repeat mean sizes to the real-data
  pickle so Table 1 percentages can carry bootstrap intervals. Adds a key;
  changes nothing computed.

## Summarisation fidelity

Both tables are summarised with the authors' own selection logic, imported
rather than paraphrased. Two details are easy to get wrong and are handled
explicitly:

1. **Two different improvement formulas.** Table 1 is oracle-adjusted,
   `(a_ref − a₁)/(a_ref − a₀) × 100` (`RealAnalysis/sum_tab.py:61`); Table 2 is
   the plain ratio `(base − value)/base × 100` (`SimuAnalysis/sum_tab.py:93`).
2. **Two different marginal-coverage thresholds.** Table 1's code uses
   `0.901 + 1/n`; Table 2's uses `0.9 + 1/(n+1)`. Both captions state the
   second. See `source_audit.md` for why this changes reference baselines.

## Controls

Each claim's control is chosen to fail for the intended reason, not to pass
trivially:

| Claim | Control | Must |
|---|---|---|
| C1 | λ = 0 and λ → ∞ limits | recover the conformal quantile; shrink θ̃ → θ̂ |
| C2 | direct plug-in (alignment removed) | leave the coverage band |
| C3 | exponent estimated, not assumed | slope CI must contain −1 without being fitted to it |
| C4 | DP marginal blow-up | reproduce the paper's severe upward deviation |
| C5 | no-shift DGP (`r = 0`, `γ_s = γ_t`) | shrink the transfer gain |
| C6 | non-exchangeable calibration | exit the Theorem 4.7 band |

C6's control is the one that decides whether the claim is scored at all: the
Theorem 4.7 band is 7.2 points wide, so without a control that exits it, an
in-band observation would be weak evidence and the claim stays BLOCKED.

## Scoring semantics: integrity conditions gate the verdict

VERIFIED and FALSIFIED carry identical credit, so folding "is the evidence
sound?" and "is the claim true?" into one list of checks makes a broken control
indistinguishable from a false claim — and rewards it. Each claim therefore
declares two dicts:

- `integrity` — preconditions for any verdict to be meaningful: a negative
  control that actually leaves the band, a permutation control the real fit
  beats, a sweep that exists, a reproduction that matches the published table.
- `checks` — propositions about the claim itself.

If any integrity condition fails the claim is **BLOCKED**, scores nothing, and
the verifier exits nonzero. Only when all of them hold does `checks` decide
VERIFIED versus FALSIFIED. A dry run on synthetic inputs demonstrated why this
matters: with the old single-dict logic a tree in which two negative controls had
failed still reported `exit_code 0` and a self-score of 12/12.

## Pipeline validation on synthetic inputs

`.openresearch/artifacts/plumbing_fixture.py` writes a complete result tree of
the right *shape* with meaningless numbers, so the whole chain — shard
reduction, real-shard merge, all six claim functions, page generation, the
protected-tree subset check and the secret scan — is exercised before any
compute is spent. It has already caught: a `sum_tab` import that runs a script
tail, the authors' non-default `tol=0.01`, the classification datasets' scalar
local coverage, an integer-key JSON round trip, and the scoring bug above.
