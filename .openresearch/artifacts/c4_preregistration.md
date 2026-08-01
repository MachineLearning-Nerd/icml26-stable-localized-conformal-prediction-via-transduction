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

## Addendum, same day, after running the registered rule on BIO

The rule above was applied unchanged to BIO, the one dataset complete when it was
written. Two things came out of it, and neither relaxes anything:

**1. The Std test turned out to be too coarse to mean anything.** It reported
"0 disagreements" — but the median bootstrap interval on a reproduced Std is
0.1362 wide for BIO/CQR against a spread of only 0.0300 between the three
published baselines, and 0.2468 against 0.1300 for BIO/GLCP. An interval that
cannot tell `base` from `SDCP` from `PPI` cannot certify agreement with any of
them: that "pass" was vacuous. A `std_gate_is_informative` requirement was
therefore added, and it **fails**. This makes the precondition harder, not
easier, which is the only direction a rule may move after seeing data.

The underlying fact is a finding in its own right: **Table 1's Std column is
printed to a precision that 50 repeats do not determine.** The standard
deviation of a per-repeat mean, estimated from 50 repeats, simply is not pinned
down to the three significant figures the table reports.

**2. Marginal coverage does not reproduce. On BIO the pattern is confined to the
tuner-based methods** — see the third addendum below, which corrects this to a
broader statement once DERMA landed.
`base`, `SDCP`, `PPI` and `oracle` reproduce to within about 0.006. Every method
that goes through the tuner is systematically low: BIO/GLCP `ours` reproduces
0.9086 against a published 0.930 (CI [0.8985, 0.9192]), `ours-sel` 0.9092
against 0.921, DP 0.9421 against 0.956.

This is not a λ-selection artifact. Sweeping the authors' entire committed grid
`[0, 0.002, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.05]` (run with `argv: []`, so the
defaults), marginal coverage rises monotonically from 0.9018 to a **maximum of
0.9124** — the published 0.930 is unreachable at every λ. Nor is it an
infinite-set accounting difference: the real-data path computes coverage
directly from the interval endpoints and has no such branch. The published
(Std 0.370, marginal 0.930) point does not lie on the reproduced trade-off
curve at all: at the matching Std the reproduction gives 0.9086, and the λ that
does best on coverage reaches only 0.9124 while undercutting the published Std.

The transcription of those marginal cells was itself in doubt, since nothing had
ever checked it, so `verify_transcription.py` was extended to cover every Table 1
marginal cell and mutation-tested. It passes: the paper does print 0.930.

## Second addendum: CRIME, completed after the above was written

CRIME finished a few hours later and confirms both findings on independent
evidence, so the conclusion does not rest on one dataset:

- **Std disagreements, outright.** `CRIME/CQR/ours` is published at 0.42 against
  a reproduced 0.5473 (CI [0.4311, 0.6548]), and `CRIME/GLCP/oracle` at 0.16
  against 0.1965 (CI [0.1601, 0.2278]). `stds_agree` is false on CRIME on its own
  terms, with no appeal to resolution.
- **The Std test is vacuous, demonstrated rather than argued.** On CRIME,
  inflating the reproduced `base` GLCP set-size spread by 25% did **not** trip
  the gate. `verify_fidelity_gate.py` exits nonzero on exactly this, which is
  what a verifier is for: the earlier "0 Std disagreements" on BIO was the
  absence of resolution, not the presence of agreement.

**Consequence, per the rule above: `marginals_agree` is false, so Claim 4 is
BLOCKED.** I said there would be no fallback rule and there is none. The
paper-internal result that TISSUE/GLCP is printed at 13.5% against a claimed
floor of 20% is still reported, and still rests only on the machine-verified
transcription — but it is reported as a finding and **not** scored, because the
precondition it was registered behind did not hold.

## Third addendum: DERMA, and a correction to the second

DERMA completed next and disagrees on both metrics: 3 Std cells and 4 marginal
cells. The per-cell numbers are rendered on the Claim 4 page from the verdict
rather than repeated here, so they cannot drift out of step with the run.

Two things it changes:

- **It corrects the characterisation above.** On BIO the marginal disagreements
  were confined to the methods that go through the tuner, and I described the
  baselines as reproducing. DERMA breaks that: its disagreeing marginal cells
  include `base` and `SDCP`. The accurate statement is therefore weaker and
  broader — marginal coverage does not reproduce reliably across methods, with
  the tuner-only pattern specific to BIO rather than general.
- **The disagreements do not share a direction.** CRIME reproduces some Std
  values above the published figure, DERMA reproduces others below it. That is
  the signature of a table that cannot be reproduced at the precision it is
  printed to, not of a single bug pushing everything one way.

The `stds_agree` / `marginals_agree` conclusion is unchanged, and is now carried
by three independent datasets rather than one.

## Fourth addendum: the verdict is FALSIFIED, and why that is not a walk-back

The second addendum said the paper-internal result would be "reported as a
finding and **not** scored, because the precondition it was registered behind did
not hold." That sentence is in tension with the claim contract registered
earlier, before any data existed, which says of Claim 4:

> Two independent routes, only one of which needs measurement precision.
> PRIMARY: the claim cites Table 1 as its evidence, so whether the stated bands
> cover the PRINTED cells is exact arithmetic — TISSUE/GLCP is printed at 13.5%
> against a stated floor of 20%. This is independent of any reproduction noise.
> SECONDARY: the same test on our own measured cells, used only when their
> bootstrap intervals are narrow enough to resolve the violation; otherwise it
> is reported but excluded from the verdict.

Both cannot stand. Resolving it in favour of the earlier contract needs an
argument, not a preference, because the later reading is the stricter one and
the earlier reading is the one that scores.

**What the precondition was actually protecting against.** Requiring the
pipeline to reproduce Table 1 was never about the arithmetic `13.5 < 20`, which
no reproduction can affect. It was a proxy for a real risk: that the printed
percentages are not the quantity the claim's "reduces … by 20–48%" ranges over.
"Reduces standard deviation by X%" reads naturally as a plain relative
reduction, and the paper's printed percentages are an oracle-adjusted quantity
instead. If the claim meant the plain reading, comparing it to the printed
column would be a category error, and being unable to reproduce that column was
a reason to doubt I had identified it correctly.

**That risk is now retired directly, by better evidence than the proxy.**
`src/verify_claim4_band.py` establishes three things and exits nonzero if any
fails:

1. The paper states its formula verbatim — improvement is
   `(1 − (a₁−a₀)/(a_ref−a₀)) × 100%`, with `a_ref` the smallest Std among
   base/SDCP/PPI whose marginal coverage is in range.
2. Re-applying that formula to the **printed Std cells** reproduces the printed
   percentages to a mean of 1.04 points across all ten cells, against 5.22 for
   the closest plain-relative reading — a factor of 5. The quantity the column
   reports is identified, not assumed.
3. The violation survives the printed cells' own rounding. Std values are given
   to two decimals, so TISSUE/GLCP's true percentage lies in [12.86, 15.28]%.
   Even its most favourable corner is 4.2 points below the 19.5% floor.

A proxy is superseded when the thing it stood in for is measured directly. That
is what happened here, and it is why the earlier contract governs.

**What does not change.** `stds_agree` and `marginals_agree` still fail on three
independent datasets, and the reproduction route stays **BLOCKED**. That result
is not demoted to a footnote: it is a finding in its own right — Table 1 is
printed to a precision that 50 repeats do not determine, and its cells did not
reproduce — and it is published with the same prominence as the falsification.
The verdict rests on the primary route alone, and the claim page says so.

**The honest summary.** Claim 4 is FALSIFIED: the GLCP half of a conjunctive
claim is contradicted by the paper's own Table 1. The CQR half (6–29%) holds on
every dataset, and is reported as holding.

## Disclosure

This restructuring was decided **after** seeing BIO/CQR fail the previous
precondition, and after seeing the mechanism that caused it. It is the same
situation as the Claim 1 matched-null statistic, and it is disclosed the same
way: the earlier rule, the result that failed it, the mechanism, and the new
rule are all on the record above, and both are reported in the verdict so a
reader can apply either one. The three unseen datasets are the reason this is
written now rather than at adjudication time — with STAR, DERMA and TISSUE
absent, there is no way to tune this rule toward a preferred outcome.
