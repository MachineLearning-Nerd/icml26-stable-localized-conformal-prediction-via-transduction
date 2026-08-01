# Claim 2, second route: a proof certificate for Theorem 4.2

**Registered before the certificate was written or run.** The commit that adds
this file contains no certificate code and no certificate output. The empirical
result it responds to is already committed and is not being revised.

## Why a second route is opened at all, and why it is not the thing the first
## pre-registration forbade

`c2_envelope_preregistration.md` closed the empirical route:

> **If the pooled control does not clear 0.05, Claim 2 stays BLOCKED.** There is
> no third fitting strategy queued behind this one, and I will not write one.

The pooled control came back at **0.825** against a 0.05 threshold, and the
held-out envelope violated by a ratio of 1.116. That route is finished. It stays
BLOCKED, its numbers are published unchanged, and nothing below reopens it,
retunes it, or reinterprets it.

What is opened here is a different *kind* of evidence, and the distinction is
the one that matters for pre-registration: **a proof certificate has no free
parameters that could be tuned against the data I have already seen.** The
envelope route had three (C, eps, delta_S) plus a choice of split and threshold,
which is exactly why fixing its rules in advance was necessary. The certificate
below checks algebraic identities. There is no knob whose setting could be
chosen to make it pass, so the researcher degree of freedom that
pre-registration exists to close is absent by construction.

The campaign brief names this route directly, and names it first:

> For a universally quantified theorem, finite experiments are scoped
> corroboration only. Mark the claim BLOCKED unless there is a machine-checkable
> proof certificate, an independently reconstructed symbolic derivation,
> exhaustive verification over the complete stated finite domain, or a valid
> assumption-satisfying counterexample.

Theorem 4.2 is universally quantified over distributions, n and lambda. By that
standard the envelope fit could never have earned full credit even if its
control had passed -- it would have been scoped corroboration. The certificate
is the route that can, and it should have been the primary route from the start.

## What the certificate will and will not establish

It will establish, or fail to establish, exactly one thing:

> The proof of Theorem 4.2 printed in Appendix B.3, as written, is a valid
> derivation of the stated bound from Assumption 4.1.

It will **not** establish that Assumption 4.1 holds for any particular data
generating process, nor that the constant C is small, nor anything about the
paper's experiments. Those are separate questions and the claim page will say so
in these words. A certificate that passes yields a claim verdict about the
theorem's *derivation*; it is not a licence to describe the simulation evidence
as stronger than it is.

## The steps to be certified, fixed now

The proof reduces to these load-bearing steps. Each is registered here before
being encoded, so the set cannot be trimmed to whatever happens to pass.

1. **Averaging preserves the Lipschitz constant.** From
   `|F(s|x;t1) - F(s|x;t2)| <= L_theta ||t1-t2||` for each of the m unlabeled
   points, the m-point average obeys the same bound.
2. **Lemma B.1, the quantile-Lipschitz step.** Sup-distance `eps` between two
   CDFs, one with density bounded below by `L_lo`, implies quantile distance at
   most `eps / L_lo`. This is the only step whose proof is analytic rather than
   algebraic; it is certified by exhaustive evaluation over a constructed family
   of CDF pairs, and must additionally be shown *attained* (some pair reaches
   the bound), or it is recorded as unverified rather than as passing.
3. **Case 1, the source of lambda^(-1/2).** `J_lambda(theta~) <= J_lambda(hat)`
   forces `lambda * t^2 <= M_Theta^2`, hence `t <= M_Theta * lambda^(-1/2)`.
4. **Case 2, the source of lambda^(+1/2).**
   `eps^2 + C_Theta*lambda <= (eps + C_Theta^(1/2) * lambda^(1/2))^2`.
5. **The finite-sample n^(-1) term.** With `1 - alpha_n = (1-alpha)(n+1)/n`, the
   split-conformal coverage of `q_hat` differs from `1-alpha` by at most
   `n^(-1)`. Certified by exhaustive enumeration over every integer n in a
   stated range crossed with a stated alpha grid.
6. **Simultaneity of the two cases.** `J_lambda(theta~)` is below *both*
   `J_lambda(hat)` and `J_lambda(theta~_0)`, so both branch bounds hold at once
   and the `min` is licensed rather than assumed.
7. **Assembly.** The two branch bounds are dominated by
   `C * (eps + lambda^(1/2) + n^-1)` and `C * (delta_S + lambda^(-1/2) + n^-1)`
   for a single explicit C built from the named constants.

## How a step is certified, and how it can fail

Steps 1, 3, 4, 6 and 7 are algebraic implications over nonnegative reals. Each
is certified by an explicit witness identity: the slack
`rhs - lhs` is written as a sum of products of the hypothesis slacks and
manifestly nonnegative terms, and `sympy` must verify that identity is
*exactly* zero. A step whose witness does not simplify to zero fails. There is
no numerical tolerance anywhere in steps 1, 3, 4, 6, 7.

Steps 2 and 5 are certified by exhaustive evaluation over a finite domain that
is fixed here: for step 5, every integer `n` from 2 to 2000 crossed with
`alpha` on a 199-point grid; for step 2, the constructed CDF family described in
the code, swept over its full parameter grid.

## The falsifiability requirement, which is the point of the whole exercise

An algebraic identity check is trivially passable if the identity is written to
match whatever is there. So the certificate is required to carry a **mutation
suite**, and the mutation suite is a gate, not a report:

- For every certified step, at least one mutation of that step -- a changed
  exponent, a changed constant, a dropped hypothesis -- must cause that step's
  certification to **fail**.
- A step whose mutations all still certify is declared **vacuous** and is
  reported as failed, whatever its own check said.
- The verifier exits nonzero if any step fails, or if any step is vacuous.

This is registered as a hard condition. If the mutation suite shows the
certificate cannot distinguish `lambda^(1/2)` from `lambda^(1/4)`, the
certificate is worthless and Claim 2 goes back to BLOCKED.

## What the verdict may be

- **VERIFIED** only if every registered step certifies, every step survives its
  mutation test, and the exhaustive steps cover their full registered domain.
- **BLOCKED** if any step cannot be certified or is vacuous.
- **FALSIFIED** if a step is certified *false* -- that is, if a counterexample
  within Assumption 4.1 is exhibited against a printed inequality.

Any looseness found that does not break the derivation -- a slack term that is
carried but never used, a constant stated larger than the algebra requires -- is
reported as a finding about the write-up and explicitly **not** counted as a
falsification. Slack in an upper bound is still an upper bound.

## The empirical evidence stays where it is

The envelope fit, its permutation control at 0.825, the held-out violation, the
per-setting fits, the real-data lambda curves and the DP control are all
published exactly as they stand, under their own heading, labelled as scoped
corroboration that did not carry information about the functional form. They are
not merged into the certificate's result and are not described as supporting it.
