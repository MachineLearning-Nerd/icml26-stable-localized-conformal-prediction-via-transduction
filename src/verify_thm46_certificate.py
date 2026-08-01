"""Certify what can be certified in the printed proof of Theorem 4.6.

Theorem 4.6 quantifies over every distribution, n, m and lambda, so simulation
of it is scoped corroboration. Appendix B.5 is a longer and less elementary
argument than B.3, and this certificate does not pretend to cover all of it.

Claim 3 has three components and they are not equally tractable:

  (a) standard conformal set-size variance is O(n^-1)   -- covered completely
  (c) gains when m greatly exceeds n                    -- covered completely
  (b) StCP achieves O(m^-1 + {n(1+lambda)^2}^-1)        -- skeleton only

For (b) the certificate covers the variance decomposition (14), the three-term
expansion, the Hessian lower bound in both of the proof's cases, the ratio
identity for the score gradient, and the rate assembly. It does **not** cover
the probabilistic core, which is listed by name in `RELIED_ON_NOT_VERIFIED` and
reported on the claim page: the o_p(1) Hessian consistency, the DKW and
Hoeffding pointwise rates, dominated convergence, and the fixed-point argument.
Those are standard tools and their use here looks routine, but "looks routine"
is not a certificate and this file does not record it as one.

Registered in `.openresearch/artifacts/c3_certificate_preregistration.md` while
C3's own slope was still null.

Run: python -m verify_thm46_certificate   (exits nonzero if any step fails to
certify, or if any step is vacuous)
"""
import json
import math
import sys

import numpy as np
import sympy as sp

from verify_thm42_certificate import SEED, Step

# The proof's probabilistic ingredients. Named so the claim page can list them
# as relied upon rather than let them pass unmentioned.
RELIED_ON_NOT_VERIFIED = [
    "||H_hat_lambda(theta_lambda) - H_lambda(theta_lambda)||_op = o_p(1), the pointwise "
    "Hessian consistency argued via dominated convergence",
    "E|q_hat^0(u) - q^0(u)|^2 = O(n^-1) and E|q_hat^1(u) - q^1(u)|^2 = O(m^-1), via the "
    "DKW inequality and Lemma B.1",
    "E|F_hat_S^1(q_lambda) - F_S^1(q_lambda)|^2 = O(m^-1), via Hoeffding on bounded summands",
    "the standard fixed-point argument converting the linearised expansion into "
    "theta~ - theta_lambda = O_p(h^-1 (n^-1/2 + m^-1/2))",
    "Assumption 4.5, which supplies the curvature constant c_d",
]

N_MAX = 2000


# --------------------------------------------------------------------------
# V1: Lemma 4.4's mechanism -- a Lipschitz map contracts variance by the square
# of its constant.  Var(L(C_hat)) <= C_L^2 Var(q_hat).
# --------------------------------------------------------------------------

def _lipschitz_variance(bound_pow):
    """Worst slack of Var(g(X)) <= C^bound_pow * Var(X) over random Lipschitz g."""
    rng = np.random.default_rng(SEED)
    worst, n = math.inf, 0
    xs = np.linspace(-3.0, 3.0, 4001)
    for _ in range(1500):
        C = float(np.exp(rng.uniform(-2.0, 2.0)))
        # A piecewise-linear g with every slope in [-C, C]: Lipschitz with
        # constant exactly C when some slope reaches it.
        knots = np.sort(rng.uniform(-3, 3, size=6))
        slopes = C * rng.choice([-1.0, 1.0], size=7) * rng.uniform(0.0, 1.0, size=7)
        slopes[int(rng.integers(0, 7))] = C * rng.choice([-1.0, 1.0])
        g = np.zeros_like(xs)
        edges = np.concatenate([[-3.0], knots, [3.0]])
        val = 0.0
        for i in range(len(edges) - 1):
            sel = (xs >= edges[i]) & (xs <= edges[i + 1])
            g[sel] = val + slopes[i] * (xs[sel] - edges[i])
            val += slopes[i] * (edges[i + 1] - edges[i])
        w = rng.dirichlet(np.ones(len(xs)) * 0.05)
        mx = float(np.sum(w * xs))
        mg = float(np.sum(w * g))
        vx = float(np.sum(w * (xs - mx) ** 2))
        vg = float(np.sum(w * (g - mg) ** 2))
        n += 1
        worst = min(worst, (C ** bound_pow) * vx - vg)
    return worst, n


# --------------------------------------------------------------------------
# V2: the coverage of the empirical quantile is a Beta order statistic, and its
# variance is O(n^-1).  Exhaustive over every (n, k).
# --------------------------------------------------------------------------

def _beta_variance_sweep(rate):
    """Worst slack of Var(U) <= 1/(4*n^rate)-style bounds, over all (n, k)."""
    worst, tight, n_cells = math.inf, 0.0, 0
    for n in range(1, N_MAX + 1):
        # Var is symmetric in k about (n+1)/2 and maximised there, so the
        # extremes and the maximiser bracket the whole row; the full row is
        # still swept for n up to 200 to show the shape is not assumed.
        ks = range(1, n + 1) if n <= 200 else {1, n, (n + 1) // 2, (n + 2) // 2}
        for k in ks:
            var = k * (n + 1 - k) / ((n + 1) ** 2 * (n + 2))
            b = 1.0 / (4.0 * n ** rate)
            n_cells += 1
            worst = min(worst, b - var)
            if b > 0:
                tight = max(tight, var / b)
    return worst, tight, n_cells


# --------------------------------------------------------------------------
# V3: the three-term expansion, exactly.
# --------------------------------------------------------------------------

_x, _y, _z = sp.symbols("x y z", real=True)

STEP_V3 = Step(
    key="V3_three_term_expansion",
    quote="(x + y + z)^2 <= 3(x^2 + y^2 + z^2)",
    axes={"x": list(np.linspace(-4, 4, 11)),
          "y": list(np.linspace(-4, 4, 11)),
          "z": list(np.linspace(-4, 4, 11))},
    slack=lambda v: 3 * (v["x"] ** 2 + v["y"] ** 2 + v["z"] ** 2)
                    - (v["x"] + v["y"] + v["z"]) ** 2,
    identity=((_x + _y + _z) ** 2,
              3 * (_x ** 2 + _y ** 2 + _z ** 2),
              (_x - _y) ** 2 + (_y - _z) ** 2 + (_x - _z) ** 2),
    mutations=[
        ("coefficient_two_instead_of_three",
         lambda v: 2 * (v["x"] ** 2 + v["y"] ** 2 + v["z"] ** 2)
                   - (v["x"] + v["y"] + v["z"]) ** 2),
        ("no_coefficient_at_all",
         lambda v: (v["x"] ** 2 + v["y"] ** 2 + v["z"] ** 2)
                   - (v["x"] + v["y"] + v["z"]) ** 2),
    ],
)


# --------------------------------------------------------------------------
# V4: the variance decomposition at equation (14).
#   2Var(q_St_hat) = E(a - a')^2 <= 3E(b - b')^2 + 6E(a - b)^2
# --------------------------------------------------------------------------

def _decomposition_sweep(c1, c2):
    """Worst slack of E(a-a')^2 <= c1*E(b-b')^2 + c2*E(a-b)^2 over random joint
    laws of (a, b), with (a', b') an independent copy."""
    rng = np.random.default_rng(SEED)
    worst, n = math.inf, 0
    for _ in range(4000):
        k = int(rng.integers(2, 8))
        w = rng.dirichlet(np.ones(k))
        a = rng.normal(0, float(np.exp(rng.uniform(-2, 2))), size=k)
        # b ranges from "equal to a" to "unrelated to a", including b == 0,
        # which is where a decomposition with too small a second coefficient
        # breaks.
        mode = int(rng.integers(0, 3))
        b = a.copy() if mode == 0 else (np.zeros(k) if mode == 1
                                        else rng.normal(0, 1.0, size=k))
        eaa = 2 * float(np.sum(w * a ** 2) - np.sum(w * a) ** 2)
        ebb = 2 * float(np.sum(w * b ** 2) - np.sum(w * b) ** 2)
        eab = float(np.sum(w * (a - b) ** 2))
        n += 1
        worst = min(worst, c1 * ebb + c2 * eab - eaa)
    return worst, n


# --------------------------------------------------------------------------
# H1 / H2: the Hessian lower bound, in each of the proof's two cases.
# --------------------------------------------------------------------------

_cd, _lm = sp.symbols("c_d lambda", real=True)

STEP_H1 = Step(
    key="H1_hessian_lower_bound_positive_curvature",
    quote="c_d > 0: c_d + 2*lambda >= min(c_d, 2) * (1 + lambda), so h_lambda = 1 + lambda",
    axes={"cd": list(np.logspace(-3, 2, 14)),
          "lam": list(np.logspace(-4, 4, 14))},
    slack=lambda v: v["cd"] + 2 * v["lam"] - min(v["cd"], 2.0) * (1 + v["lam"]),
    identity=(sp.Min(_cd, 2) * (1 + _lm), _cd + 2 * _lm,
              (_cd - sp.Min(_cd, 2)) + (2 - sp.Min(_cd, 2)) * _lm),
    mutations=[
        # c_H = c_d overreaches once the curvature exceeds the ridge term.
        ("c_H_taken_as_c_d",
         lambda v: v["cd"] + 2 * v["lam"] - v["cd"] * (1 + v["lam"])),
        # c_H = 2 overreaches whenever the curvature is the smaller of the two.
        ("c_H_taken_as_two",
         lambda v: v["cd"] + 2 * v["lam"] - 2.0 * (1 + v["lam"])),
    ],
)

STEP_H2 = Step(
    key="H2_hessian_lower_bound_nonpositive_curvature",
    quote=("c_d <= 0 and lambda >= 1 - c_d: c_d + 2*lambda >= lambda + 1 >= lambda, "
           "so h_lambda = lambda and c_H = 1"),
    axes={"cd": list(np.linspace(-8.0, 0.0, 17)),
          "slack": list(np.linspace(0.0, 8.0, 17))},
    # lambda is parameterised as (1 - c_d) + slack so the hypothesis holds by
    # construction and the boundary lambda = 1 - c_d is always in the search.
    slack=lambda v: (v["cd"] + 2 * ((1 - v["cd"]) + v["slack"]))
                    - (((1 - v["cd"]) + v["slack"]) + 1),
    identity=(_lm + 1, _cd + 2 * _lm, _cd + _lm - 1),
    mutations=[
        # Without lambda >= 1 - c_d the bound fails: sweep lambda below it.
        ("hypothesis_lambda_at_least_one_minus_c_d_dropped",
         lambda v: (v["cd"] + 2 * v["slack"]) - (v["slack"] + 1)),
        # c_H = 2 overreaches.
        ("c_H_taken_as_two",
         lambda v: (v["cd"] + 2 * ((1 - v["cd"]) + v["slack"]))
                   - 2 * (((1 - v["cd"]) + v["slack"]) + 1)),
    ],
)


# --------------------------------------------------------------------------
# R1: the ratio identity used for the score gradient.
#   |A_m/B_m - A/B| <= L^-1 |A_m - A| + L^-2 |A| |B_m - B|,  B_m, B >= L > 0
# --------------------------------------------------------------------------

_A, _Am, _B, _Bm = sp.symbols("A A_m B B_m", positive=True)

STEP_R1 = Step(
    key="R1_ratio_identity_for_the_score_gradient",
    quote=("|A_m/B_m - A/B| <= L^-1 |A_m - A| + L^-2 |A| |B_m - B| when B_m, B >= L > 0"),
    axes={"L": list(np.logspace(-2, 1, 8)),
          "A": list(np.linspace(-4, 4, 7)),
          "dA": list(np.linspace(-3, 3, 7)),
          "bB": list(np.linspace(0.0, 4.0, 6)),
          "bBm": list(np.linspace(0.0, 4.0, 6))},
    # B and B_m are parameterised as L + nonneg so the hypothesis holds by
    # construction, with equality reachable at bB = bBm = 0.
    slack=lambda v: (abs(v["dA"]) / v["L"]
                     + abs(v["A"]) * abs(v["bBm"] - v["bB"]) / v["L"] ** 2
                     - abs((v["A"] + v["dA"]) / (v["L"] + v["bBm"])
                           - v["A"] / (v["L"] + v["bB"]))),
    # The bound comes from this decomposition, so the decomposition is what the
    # identity check has to confirm -- restating the left side as itself would
    # certify nothing.
    identity=(_Am / _Bm - _A / _B,
              (_Am - _A) / _Bm + _A * (_B - _Bm) / (_Bm * _B),
              sp.Integer(0)),
    mutations=[
        # L^-1 on the denominator term is not enough once L < 1.
        ("second_term_scaled_by_L_inverse_instead_of_L_squared",
         lambda v: (abs(v["dA"]) / v["L"]
                    + abs(v["A"]) * abs(v["bBm"] - v["bB"]) / v["L"]
                    - abs((v["A"] + v["dA"]) / (v["L"] + v["bBm"])
                          - v["A"] / (v["L"] + v["bB"])))),
        ("denominator_fluctuation_term_dropped",
         lambda v: (abs(v["dA"]) / v["L"]
                    - abs((v["A"] + v["dA"]) / (v["L"] + v["bBm"])
                          - v["A"] / (v["L"] + v["bB"])))),
    ],
)


# --------------------------------------------------------------------------
# A2: the rate assembly.
#   (n^-1 + m^-1) h^-2 + m^-1 <= 2 (m^-1 + n^-1 h^-2)   for h >= 1
# --------------------------------------------------------------------------

_n, _mm, _h = sp.symbols("n m h", positive=True)

STEP_A2 = Step(
    key="A2_rate_assembly",
    quote="(n^-1 + m^-1) h^-2 + m^-1 <= 2 (m^-1 + n^-1 h^-2) when h >= 1",
    axes={"n": list(np.logspace(0, 4, 9)),
          "m": list(np.logspace(0, 4, 9)),
          "hx": list(np.linspace(0.0, 20.0, 11))},
    # h is parameterised as 1 + hx so h >= 1 holds by construction.
    slack=lambda v: (2 * (1 / v["m"] + 1 / (v["n"] * (1 + v["hx"]) ** 2))
                     - ((1 / v["n"] + 1 / v["m"]) / (1 + v["hx"]) ** 2 + 1 / v["m"])),
    identity=((1 / _n + 1 / _mm) / _h ** 2 + 1 / _mm,
              2 * (1 / _mm + 1 / (_n * _h ** 2)),
              1 / _mm - 1 / (_mm * _h ** 2) + 1 / (_n * _h ** 2)),
    mutations=[
        ("constant_one_instead_of_two",
         lambda v: (1 * (1 / v["m"] + 1 / (v["n"] * (1 + v["hx"]) ** 2))
                    - ((1 / v["n"] + 1 / v["m"]) / (1 + v["hx"]) ** 2 + 1 / v["m"]))),
        # Without h >= 1 the m^-1 h^-2 term is no longer absorbed by m^-1.
        ("hypothesis_h_at_least_one_dropped",
         lambda v: (2 * (1 / v["m"] + 1 / (v["n"] * (0.01 + v["hx"]) ** 2))
                    - ((1 / v["n"] + 1 / v["m"]) / (0.01 + v["hx"]) ** 2 + 1 / v["m"]))),
    ],
    scale=lambda v: 1 / v["m"] + 1 / v["n"],
)


# --------------------------------------------------------------------------
# G1: the claim's own comparison. The ratio of the StCP bound to the standard
# bound is exactly n/m + (1+lambda)^-2, so the condition for a predicted gain
# is derivable rather than assertable.
# --------------------------------------------------------------------------

def _gain_condition():
    """Derive when the bound predicts a gain, and check the derivation holds."""
    ratio = lambda n, m, lam: (1.0 / m + 1.0 / (n * (1 + lam) ** 2)) / (1.0 / n)
    worst, n_cells, mismatches = math.inf, 0, 0
    for lam in (0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0):
        shrink = 1.0 - 1.0 / (1 + lam) ** 2
        for n in (10, 30, 100, 500, 2000):
            for m in (10, 30, 100, 500, 2000, 20000, 10 ** 6):
                r = ratio(n, m, lam)
                # The closed form must match the direct evaluation exactly ...
                n_cells += 1
                worst = min(worst, 1e-9 - abs(r - (n / m + 1.0 / (1 + lam) ** 2)))
                # ... and the derived threshold must predict the sign correctly.
                predicted = shrink > 0 and m > n / shrink
                if predicted != (r < 1.0 - 1e-12):
                    mismatches += 1
    return {
        "step": "G1_condition_for_a_predicted_stability_gain",
        "as_printed": ("'substantial stability gains when m greatly exceeds n' -- the bound's "
                       "own ratio to the standard rate is n/m + (1+lambda)^-2"),
        "points_searched": n_cells,
        "min_slack_found": float(worst),
        "no_refutation_found": bool(worst >= -1e-9 and mismatches == 0),
        "threshold_mismatches": mismatches,
        "witness_identity_is_exact": None,
        "derived_condition": ("the bound predicts a gain exactly when "
                              "m > n / (1 - (1+lambda)^-2), which requires lambda > 0; "
                              "as m/n -> infinity the ratio tends to (1+lambda)^-2, so the "
                              "asymptotic gain is set by lambda alone and not by m"),
        "at_lambda_zero": ("the ratio is n/m + 1 > 1 for every finite m, so at lambda = 0 the "
                           "bound predicts no stability gain at any m -- the qualitative "
                           "reading 'gains when m >> n' holds only for lambda > 0"),
        "mutations": {
            "gain_claimed_for_every_m_greater_than_n": {
                "refuted": bool(ratio(100, 200, 0.0) >= 1.0),
                "min_slack": float(1.0 - ratio(100, 200, 0.0))},
            "gain_claimed_independent_of_lambda": {
                "refuted": bool(ratio(100, 10 ** 9, 0.0) >= 1.0),
                "min_slack": float(1.0 - ratio(100, 10 ** 9, 0.0))},
        },
    }


# --------------------------------------------------------------------------

def run():
    rng = np.random.default_rng(SEED)
    steps = [s.certify(rng) for s in (STEP_V3, STEP_H1, STEP_H2, STEP_R1, STEP_A2)]

    v1_slack, v1_n = _lipschitz_variance(2)
    v1_mut1, _ = _lipschitz_variance(1)   # Var(g) <= C * Var(X)
    v1_mut0, _ = _lipschitz_variance(0)   # Var(g) <= Var(X)
    v1 = {
        "step": "V1_lipschitz_variance_transfer",
        "as_printed": "Var(L(C_hat)) <= C_L^2 Var(q_hat) for L Lipschitz in the quantile",
        "points_searched": v1_n,
        "min_slack_found": float(v1_slack),
        "no_refutation_found": bool(v1_slack >= -1e-9),
        "witness_identity_is_exact": None,
        "mutations": {
            "constant_not_squared": {"refuted": bool(v1_mut1 < -1e-9),
                                     "min_slack": float(v1_mut1)},
            "no_constant_at_all": {"refuted": bool(v1_mut0 < -1e-9),
                                   "min_slack": float(v1_mut0)},
        },
    }

    v2_slack, v2_tight, v2_n = _beta_variance_sweep(1.0)
    v2_mut, _, _ = _beta_variance_sweep(2.0)   # claim O(n^-2)
    v2 = {
        "step": "V2_beta_order_statistic_variance_is_order_n_inverse",
        "as_printed": ("U ~ Beta(k, n+1-k) gives Var(U) = k(n+1-k)/((n+1)^2 (n+2)) "
                       "<= 1/(4n), hence Var(q_hat) = O(n^-1)"),
        "points_searched": v2_n,
        "exhaustive_domain": f"every (n, k) with n in [1, 200]; n up to {N_MAX} at the "
                             f"row extremes and the maximiser",
        "min_slack_found": float(v2_slack),
        "no_refutation_found": bool(v2_slack >= -1e-15),
        "closest_approach_to_the_bound": float(v2_tight),
        "witness_identity_is_exact": None,
        "mutations": {
            "rate_sharpened_to_n_to_the_minus_two": {"refuted": bool(v2_mut < -1e-15),
                                                     "min_slack": float(v2_mut)},
        },
    }

    v4_slack, v4_n = _decomposition_sweep(3.0, 6.0)
    v4_m1, _ = _decomposition_sweep(1.0, 1.0)
    v4_m2, _ = _decomposition_sweep(3.0, 0.0)
    v4 = {
        "step": "V4_variance_decomposition_at_equation_14",
        "as_printed": "2Var(q_St_hat) = E(a-a')^2 <= 3E(b-b')^2 + 6E(a-b)^2",
        "points_searched": v4_n,
        "min_slack_found": float(v4_slack),
        "no_refutation_found": bool(v4_slack >= -1e-9),
        "witness_identity_is_exact": None,
        "mutations": {
            "both_coefficients_reduced_to_one": {"refuted": bool(v4_m1 < -1e-9),
                                                 "min_slack": float(v4_m1)},
            "cross_term_dropped": {"refuted": bool(v4_m2 < -1e-9),
                                   "min_slack": float(v4_m2)},
        },
    }

    g1 = _gain_condition()

    for s in (v1, v2, v4, g1):
        s["every_mutation_is_refuted"] = all(m["refuted"] for m in s["mutations"].values())
        s["certified"] = bool(s["no_refutation_found"] and s["every_mutation_is_refuted"])
    steps += [v1, v2, v4, g1]

    order = ["V1_lipschitz_variance_transfer",
             "V2_beta_order_statistic_variance_is_order_n_inverse",
             "V3_three_term_expansion",
             "V4_variance_decomposition_at_equation_14",
             "H1_hessian_lower_bound_positive_curvature",
             "H2_hessian_lower_bound_nonpositive_curvature",
             "R1_ratio_identity_for_the_score_gradient",
             "A2_rate_assembly",
             "G1_condition_for_a_predicted_stability_gain"]
    steps.sort(key=lambda s: order.index(s["step"]))

    integrity = {
        "every_registered_step_was_attempted": {s["step"] for s in steps} == set(order),
        "no_step_is_vacuous": all(s.get("every_mutation_is_refuted") for s in steps),
        "probabilistic_ingredients_are_listed_not_hidden": bool(RELIED_ON_NOT_VERIFIED),
    }
    checks = {s["step"]: bool(s["certified"]) for s in steps}
    blocked = [k for k, ok in integrity.items() if not ok]
    failed = [k for k, ok in checks.items() if not ok]

    return {
        "certifies": ("part (a) of Claim 3 completely (standard conformal set-size variance "
                      "is O(n^-1)), part (c) completely (the exact condition under which the "
                      "bound predicts a stability gain), and the algebraic skeleton of part "
                      "(b)"),
        "does_not_certify": ("the probabilistic core of Appendix B.5 -- the o_p(1) Hessian "
                             "consistency, the DKW and Hoeffding pointwise rates, dominated "
                             "convergence, and the fixed-point argument -- nor Assumption 4.5"),
        "relied_on_not_verified": RELIED_ON_NOT_VERIFIED,
        "verdict": "BLOCKED" if (blocked or failed) else "VERIFIED",
        "blocked_by": blocked,
        "failed_steps": failed,
        "integrity": integrity,
        "checks": checks,
        "steps": steps,
        "seed": SEED,
    }


def main():
    res = run()
    print("Theorem 4.6 proof certificate (partial by construction)")
    print("=" * 64)
    for s in res["steps"]:
        muts = s.get("mutations", {})
        n_ref = sum(1 for m in muts.values() if m["refuted"])
        print(f"  {'PASS' if s['certified'] else 'FAIL'}  {s['step']}")
        print(f"        searched {s['points_searched']:>7} pts   "
              f"min slack {s['min_slack_found']:+.3e}   "
              f"mutations refuted {n_ref}/{len(muts)}")
        if s.get("refuting_point"):
            print(f"        REFUTED AT {s['refuting_point']}")
    g1 = next(s for s in res["steps"] if s["step"].startswith("G1"))
    print()
    print("  derived:", g1["derived_condition"])
    print("  at lambda = 0:", g1["at_lambda_zero"])
    print()
    print("  relied on, NOT verified here:")
    for r in res["relied_on_not_verified"]:
        print(f"    - {r}")
    print()
    print(f"  verdict: {res['verdict']}")
    if res["blocked_by"]:
        print(f"  blocked by: {res['blocked_by']}")
    if res["failed_steps"]:
        print(f"  failed steps: {res['failed_steps']}")
    print(json.dumps({"integrity": res["integrity"]}, indent=1))
    return 0 if res["verdict"] == "VERIFIED" else 1


if __name__ == "__main__":
    sys.exit(main())
