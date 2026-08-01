# Verifier plumbing evidence

A verifier that only ever ran on the real results proves nothing about whether it
*can* fail. This artifact runs the same `src/stage_analysis.py` on a synthetic
result tree of the right shape (`plumbing_fixture.py`, random numbers, no
scientific meaning) and shows that all three verdict branches are reachable and
that the exit code follows the verdicts.

Command (identical to a real node, only `config/node.json` differs):

```
STCP_UPSTREAM=<pristine StCP checkout> python -u src/run_node.py   # stage: analysis
```

## Run 1 — fixture at its default

Random data satisfies some integrity preconditions by luck and fails others.

| Claim | Verdict | Broken integrity precondition | Failing claim checks |
| --- | --- | --- | --- |
| C1 | VERIFIED | — | — |
| C2 | BLOCKED | `envelope_shape_beats_permutation_control`, `negative_control_DP_leaves_band` | `envelope_holds_on_held_out_lambda` |
| C3 | FALSIFIED | — | `base_variance_slope_consistent_with_minus_one`, `stcp_std_decreases_with_lambda_in_valid_region`, `stcp_std_decreases_with_m_at_fixed_n` |
| C4 | BLOCKED | `reproduces_published_table_cell_by_cell` | `claimed_bands_cover_every_published_cell` |
| C5 | BLOCKED | `no_shift_control_reduces_the_gain` | `n30_glcp_matches_31_2_within_ci` |
| C6 | VERIFIED | — | — |

Verifier exit code: **1**.

Note C3. Its claim checks fail, but its integrity preconditions hold, so it is
FALSIFIED — and FALSIFIED does *not* raise the exit code, because a falsification
is a result, not a malfunction. Only BLOCKED does.

Raw log: `plumb2/run_default.log`.

## Run 2 — sabotaged control

One number is changed: the non-exchangeable arm's coverage is moved from 0.812
to 0.901, i.e. back *inside* the Theorem 4.7 band. Nothing else differs.

```
PLUMB_NONEXCH_COVERAGE=0.901 python plumbing_fixture.py
```

C6 flips **VERIFIED → BLOCKED**, `blocked_by = ["control_makes_the_band_informative"]`.

This is the property that matters. The Theorem 4.7 band is 7.2 points wide, so
an in-band observation is only evidence if something can leave the band. When
the control stops discriminating, C6 must lose its credit rather than keep it —
and it does. Had the control lived in the `checks` dict instead of `integrity`,
the same sabotage would have flipped C6 to FALSIFIED, which scores the same as
VERIFIED.

Raw log: `plumb2/run_sabotaged.log`.

## What this does not show

The fixture never produces a 6/6 all-VERIFIED tree, because satisfying
`reproduces_published_table_cell_by_cell` would mean hand-tuning the synthetic
numbers to the paper's printed table — which would test the fixture, not the
verifier. Reachability of VERIFIED is shown by C1 and C6 above.
