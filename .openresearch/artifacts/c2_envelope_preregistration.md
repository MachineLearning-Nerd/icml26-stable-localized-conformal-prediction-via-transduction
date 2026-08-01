# Claim 2: how the Theorem 4.2 envelope is tested, fixed before the data

**Registered with exactly one simulation setting complete (`logabs-n30-m30`,
50/50 repeats). `logabs-n30-m100`, `logabs-n30-m500`, `logabs-n100-m500` and
`logabs-n500-m500` had 10, 20, 0 and 0 of their 50 repeats. The pooled fit
described below has therefore never been run on anything.**

## The problem this addresses

Theorem 4.2 bounds StCP's coverage error by

    O(min(eps + lambda^(1/2) + n^-1, delta_S + lambda^-(1/2) + n^-1))

The current test fits that three-parameter envelope to half of the authors'
19-value lambda grid, checks it holds on the held-out half, and runs a
permutation control: refit against randomly reordered lambdas and count how
often a random ordering fits as well. On the one complete setting the control
comes back at **0.25 (GLCP) and 0.22 (CQR)** against a 0.05 threshold. A quarter
of random orderings match the true one.

That is a real negative result and it is reported as such: with 10 training
points, three free parameters and a `min()` of two branches, the envelope is
flexible enough to absorb curves that have nothing to do with lambda. The
integrity condition `envelope_shape_beats_permutation_control` fails, and Claim
2 is BLOCKED as things stand.

## What changes, and why it is not a relaxation

When the remaining settings land there will be five settings x two base methods.
The obvious response -- keep fitting per setting and hope more of them clear
0.05 -- is the wrong one: it gives ten weak tests and takes the majority, which
is a lottery, not more power.

Instead **one envelope is fitted across all settings at once**: the same three
parameters must explain every setting's lambda curve simultaneously, roughly 95
points rather than 19. Registered now:

- The threshold stays at **0.05**. It is not moving.
- The train/test split stays a per-setting half, so held-out points come from
  every setting rather than from one.
- `n_perm` stays 200, and permutation continues to shuffle the lambda ordering
  within each setting.

This is a **stricter** test, which is the only direction a rule may move after
seeing a result. Three parameters constrained by five settings simultaneously
can fit far less than three parameters per setting: any envelope that passes
pooled would also have passed per setting, and the converse is false. If the
functional form is real, pooling is what should reveal it; if it is not, pooling
removes the flexibility that was hiding the fact.

## The falsifiability guard

- The pooled fit must still be shown able to fail. The permutation control is
  that demonstration, and it is reported whatever it says.
- **If the pooled control does not clear 0.05, Claim 2 stays BLOCKED.** There is
  no third fitting strategy queued behind this one, and I will not write one.
  The per-setting numbers above are published either way.
- The separate, simpler result -- marginal coverage stays inside the
  table-annotation band across the whole grid, with DP leaving the band as a
  working negative control -- is reported independently of the envelope. It is
  not a substitute for the bound's functional form and is not presented as one.

## Scope, stated plainly

Theorem 4.2 is universally quantified over distributions, n and lambda. No
finite simulation verifies it. What is on offer here is scoped corroboration of
its functional form on the paper's own DGP and lambda grid, plus a control that
says whether that corroboration carries information at all. The claim page says
this in those words.
