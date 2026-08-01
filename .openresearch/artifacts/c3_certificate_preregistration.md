# Claim 3, second route: a proof certificate for Theorem 4.6

**Registered while Claim 3's own data does not yet exist.** At this commit
`logabs-n100-m500` has 30 of 50 repeats and `logabs-n500-m500` has 5 of 50, so
the log-log slope over n is `null` and no verdict-bearing number for this claim
has been computed, let alone seen. That is the only honest moment to add a route,
and it is why this is being written now rather than when the shards land.

## Why Claim 3 needs one

Theorem 4.6 is universally quantified over distributions, n, m and lambda. Under
the campaign standard, a finite simulation of it is scoped corroboration and
cannot by itself earn full credit -- the same reasoning that moved Claim 2 onto
a certificate. Claim 3 currently rests entirely on simulation.

## The rule change, which is a tightening

C3's verdict currently gates on the simulation checks alone. From this commit it
gates on **both**:

- the simulation checks already registered, unchanged, with their existing
  thresholds; **and**
- the certificate described below.

Both must pass. This is strictly harder than what C3 faced before -- nothing is
being relaxed, no threshold moves, and a claim that would have passed on
simulation alone can now fail. That is the only direction a rule may move after
a campaign is under way.

## What this certificate covers, and what it does not

Claim 3 has three components. They are not equally tractable and the certificate
does not pretend otherwise.

**(a) Standard conformal set-size variance is O(n^-1).** Fully certifiable. It
is Lemma 4.4 plus Case 1 of Appendix B.5, and it is elementary: a Lipschitz map
contracts variance by the square of its constant, and the coverage of the
empirical quantile is a Beta order statistic whose variance is bounded by
`1/(4(n+2))`. Certified exhaustively over every `(n, k)` with `n` up to 2000.

**(b) StCP achieves O(m^-1 + {n(1+lambda)^2}^-1).** Only its algebraic skeleton
is certifiable. The certificate covers the variance decomposition (14), the
three-term expansion, the Hessian lower bound in both of the proof's cases, the
ratio identity used for the score gradient, and the final rate assembly. It does
**not** cover the probabilistic core: the `o_p(1)` Hessian consistency, the DKW
and Hoeffding pointwise rates, the dominated-convergence step, or the
fixed-point argument. Those are standard tools and the proof's use of them looks
routine, but they are not machine-checked here and the claim page will list them
by name as relied upon rather than verified.

**(c) Substantial stability gains when m greatly exceeds n.** Fully certifiable,
because given (a) and (b) it is pure algebra. The certificate derives the exact
condition under which the bound predicts a gain, rather than restating the
paper's qualitative "when m >> n".

A certificate that covers (a) and (c) completely and (b) partially is reported
in exactly those terms. It is not described as a proof of Theorem 4.6, and the
claim page will not say that it is.

## The steps, fixed now

1. **V1** Lipschitz variance transfer: `Var(g(X)) <= C_L^2 Var(X)` for `g`
   Lipschitz with constant `C_L`. This is Lemma 4.4's mechanism.
2. **V2** Beta order-statistic variance: `U ~ Beta(k, n+1-k)` has
   `Var(U) = k(n+1-k)/((n+1)^2 (n+2)) <= 1/(4(n+2))`. Exhaustive over every
   integer `n` in `[1, 2000]` and every `k` in `[1, n]`.
3. **V3** three-term expansion: `(x+y+z)^2 <= 3(x^2+y^2+z^2)`.
4. **V4** the variance decomposition at equation (14):
   `E(a-a')^2 <= 3E(b-b')^2 + 6E(a-b)^2` for iid copies.
5. **H1** Hessian lower bound, case `c_d > 0`:
   `c_d + 2*lambda >= min(c_d,2) * (1+lambda)`.
6. **H2** Hessian lower bound, case `c_d <= 0` with `lambda >= 1 - c_d`:
   `c_d + 2*lambda >= lambda + 1 >= lambda`.
7. **R1** the ratio identity bounding `A_m/B_m - A/B` by
   `L^-1 |A_m - A| + L^-2 |A| |B_m - B|` when both denominators exceed `L`.
8. **A2** rate assembly:
   `(n^-1 + m^-1) h^-2 + m^-1 <= 2 (m^-1 + n^-1 h^-2)` when `h >= 1`.
9. **G1** the claim's own comparison: the ratio of the StCP bound to the
   standard bound is `n/m + (1+lambda)^-2`, and the exact condition for a gain
   is derived from it rather than asserted.

## How a step is certified, and the falsifiability gate

Identical machinery to the Theorem 4.2 certificate, deliberately: a witness
identity that `sympy` must reduce to exactly zero, plus a refutation search over
the step's own domain, plus **mutations that the refutation search must catch**.
A step whose mutations all survive is declared vacuous and reported as failed
whatever its own check said. The verifier exits nonzero if any step fails or any
step is vacuous.

## What the verdict may be

- **VERIFIED** only if the simulation checks pass *and* every step above
  certifies *and* every step survives its mutations.
- **BLOCKED** if either route fails, or if any certified step is vacuous.
- **FALSIFIED** only on a counterexample within the theorem's assumptions.

If G1 shows the paper's qualitative reading of its own bound is wrong in some
regime, that is reported as a finding about the bound's interpretation, and it is
counted as a falsification only if it contradicts the exact quantified statement
of Theorem 4.6 rather than the prose around it.
