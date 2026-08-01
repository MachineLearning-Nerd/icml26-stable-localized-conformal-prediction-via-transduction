# Claim 4: what counts as "the reproduction matches Table 1", fixed before the data

**Registered 2026-08-01, with BIO complete (10/10 shards) and CRIME at 6/10.
STAR, DERMA and TISSUE — 30 of the 50 Table 1 cells — had not produced a single
number when this was written.**

## Why this document exists

The integrity precondition `reproduces_published_table_cell_by_cell` currently
requires, for every one of the ten dataset x model cells, that the bootstrap
interval of the reproduced oracle-adjusted percentage bracket the percentage the
paper prints. BIO fails it:

| cell | reproduced % | 95% CI | published % | brackets? |
| --- | --- | --- | --- | --- |
| BIO/GLCP | 43.9 | [28.6, 54.4] | 35.7 | yes |
| BIO/CQR | 19.5 | [11.5, 27.9] | 29.3 | **no** |

and it fails for a reason that has nothing to do with the pipeline being wrong.
Every underlying standard deviation reproduces: the largest disagreement across
BIO's 14 baseline cells is about 0.03, and marginal coverage agrees to about
0.01. What moves is the *reference*. The Appendix C.1 percentage is

    (a_ref - a_ours) / (a_ref - a_oracle) * 100

with `a_ref` the **smallest** Std among base/SDCP/PPI whose marginal coverage is
in range. In BIO/CQR the paper's SDCP (0.390) and PPI (0.399) are nine
thousandths apart. SDCP reproduces 0.028 high, the argmin flips from SDCP to
PPI, and the percentage moves roughly ten points. An argmin over near-tied
quantities is discontinuous; a Monte-Carlo difference far smaller than the
reported precision can flip it.

So this is the third precondition in this campaign that a result has failed
after the fact, and I am not going to quietly widen it. It gets rewritten once,
in public, before the data that would let me tune it exists.

## What is actually being gated

Claim 4 is settled on its **primary route**: exact arithmetic on the paper's own
printed Table 1, whose cells are verified character by character against the
archived paper text by `src/verify_transcription.py`. That route consumes no
number from this reproduction. TISSUE/GLCP is printed at 13.5% against a claimed
floor of 20%, and that is true whatever my machine computes.

The reproduction's job is narrower: to establish that the authors' pipeline, run
on the authors' data, produces the table the paper prints — so that the logbook
is a reproduction rather than a proofread. The right quantities for that job are
the ones the pipeline **measures**. A derived ratio with a known discontinuity is
not evidence about fidelity, because it can disagree while every measured
quantity agrees. That is the defect being corrected.

## The rule, fixed as of now

`reproduces_published_table_cell_by_cell` becomes the conjunction of two
measured-quantity tests, evaluated for every dataset x model x method cell over
base / SDCP / PPI / ours / oracle:

1. **`stds_agree`** — the published Std lies inside the 95% bootstrap interval
   of the reproduced Std, resampling the 50 repeats.
2. **`marginals_agree`** — the published marginal coverage lies inside the 95%
   bootstrap interval of the reproduced marginal coverage, same resamples.

Both are parameter-free: there is no tolerance to choose, and therefore no
tolerance to tune once the remaining datasets land. The interval is the
reproduction's own Monte-Carlo uncertainty.

The percentage comparison is **not** deleted. It is demoted from precondition to
reported check, and it is reported per cell whether it agrees or not, together
with the reference-flip diagnostic that says which baseline each side divided by.
A cell that disagrees is named on the claim page.

## Guard against this becoming an escape hatch

The rewrite must make the precondition *harder* to satisfy in every direction
except the specific discontinuity it excuses. Concretely, registered now:

- The new gate tests **five methods x two metrics per cell** where the old one
  tested a single derived number per cell. It is a strictly larger set of
  comparisons.
- It must be shown to FAIL under injected defects, exactly as the Claim 1 gates
  were: perturbing any reproduced Std away from the published value must flip
  `stds_agree` to false. If it cannot be made to fail, it is vacuous and Claim 4
  is BLOCKED regardless of what it reports.
- If `stds_agree` or `marginals_agree` fails on the real data, Claim 4 is
  **BLOCKED**. There is no further rule to fall back on, and I will not write
  one. That outcome gets published as BLOCKED.
- The percentage-disagreement count is published whatever it is. A cell is never
  described as agreeing because its reference flipped.

## Disclosure

This restructuring was decided **after** seeing BIO/CQR fail the previous
precondition, and after seeing the mechanism that caused it. It is the same
situation as the Claim 1 matched-null statistic, and it is disclosed the same
way: the earlier rule, the result that failed it, the mechanism, and the new
rule are all on the record above, and both are reported in the verdict so a
reader can apply either one. The three unseen datasets are the reason this is
written now rather than at adjudication time — with STAR, DERMA and TISSUE
absent, there is no way to tune this rule toward a preferred outcome.
