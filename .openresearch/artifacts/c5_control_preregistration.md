# Claim 5: which negative control is the right one, fixed before either has run

**Registered 2026-08-01, while `results/shards/` contained exactly one setting
(`logabs-n30-m500`). Neither control had produced a single number: the no-shift
arm had no shards, and the m = n arm had no shards.**

## The problem

Claim 5's integrity currently rests on one control, `no_shift_control_reduces_the_gain`:
the arm sets `r = 0` and `gamma_s = gamma_t`, making source and target identical,
and requires the measured stability gain to shrink. The stated rationale is that
this "removes the very reason transductive calibration should help".

That rationale does not survive contact with the theorem the gain comes from.
Theorem 4.6 states

    standard conformal   set-size variance  O(n^-1)
    StCP                 set-size variance  O(m^-1 + {n(1+lambda)^2}^-1)

There is **no distribution-shift term in either rate.** StCP's stability
advantage is credited to the unlabeled target sample being large -- m greatly
exceeding n -- not to a shift existing between source and target. Removing the
shift therefore should not remove the advantage, and a control built on that
premise can fail while the method is working exactly as claimed.

If that happens, `no_shift_control_reduces_the_gain` is false, Claim 5 is
BLOCKED, and the block is an artifact of a mis-specified control rather than a
defect in the evidence. Deciding that *after* seeing the control fail would be
indistinguishable from moving a gate to rescue a claim, so it is decided here.

## The rule, fixed as of now

**Primary control: m = n.** The setting `logabs-n30-m30` sets the unlabeled
sample to the same size as the calibration set, which removes exactly the
quantity Theorem 4.6 credits (`m^-1` no longer beats `n^-1`). The stability gain
must shrink relative to `logabs-n30-m500`. This is the control that removes the
named mechanism, so it gates the claim.

**Secondary, reported either way: no shift.** The `r = 0, gamma_s = gamma_t` arm
is kept and reported whatever it shows. It tests a different and weaker
proposition -- that the gain is shift-driven -- which the paper does not actually
assert. Its outcome is informative about the framing and is published, but it
does not gate.

Both arms are reported in the verdict and on the claim page, with this document
linked, whichever way each lands.

## Guards, so this cannot become an escape hatch

- **If the m = n control does not reduce the gain, Claim 5 is BLOCKED.** There is
  no third control and I will not write one. That outcome gets published.
- The m = n comparison must be *informative*: `logabs-n30-m500` and
  `logabs-n30-m30` differ in m alone -- same n, same seeds, same lambda grid,
  same DGP constants -- so a difference cannot be attributed to anything else.
- The intermediate rung `logabs-n30-m100` is run too. If the gain is genuinely
  driven by m, it should sit between the two; a non-monotone ordering would mean
  m is not the operative variable and would itself count against the claim.
- Swapping which control gates is a **weakening** if and only if the no-shift arm
  fails and the m = n arm passes. That exact combination is the one this document
  predicts in advance, in writing, before either number exists -- which is the
  only thing that distinguishes a prediction from a rationalisation.

## Disclosure

The no-shift control was designed and queued first, and this reassessment was
prompted by reading Theorem 4.6's rate closely while waiting for it to run — not
by seeing it fail, because it has not yet produced a number. Both controls, and
both outcomes, appear in the published verdict.
