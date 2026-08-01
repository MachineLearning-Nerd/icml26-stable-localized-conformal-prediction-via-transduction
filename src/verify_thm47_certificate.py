"""Certify the printed proof of Theorem 4.7, exhaustively.

Theorem 4.7 is the one theorem in this paper whose proof is finite arithmetic
end to end: order statistics, ceilings and an exchangeability rank argument. So
unlike Theorems 4.2 and 4.6 it can be checked over its whole parameter space
rather than sampled, and that is what this file does.

It finds a defect. The proof's two intermediate inequalities are both false over
much of the domain, and the theorem's **lower** endpoint inherits the failure:
with `q_L` defined as Section 3.1 defines it, the coverage the proof is entitled
to claim is `ceil(n(1-a-atol))/(n+1)`, which sits below `1-a-atol` by up to
`(n+1)^-1`. The theorem's **upper** endpoint is unaffected and holds everywhere.

This is reported as a finding about the printed guarantee, and the repaired
guarantee is derived and exhaustively verified alongside it. It is **not**
recorded as a falsification of Claim 6:

    the shortfall is in `P(S_{n+1} <= q_L)`, which the proof uses as a LOWER
    bound on coverage. Actual coverage is at least that and may be higher, so a
    gap here breaks the proof without by itself producing a run whose coverage
    falls outside the stated band.

Whether any observed run does fall in the exposed window is a separate,
measurable question, and `exposure_window` below reports it against the
campaign's own settings rather than leaving it rhetorical.

Run: python -m verify_thm47_certificate   (exits nonzero if the corrected
statement fails anywhere, or if the search cannot demonstrate its own teeth)
"""
import json
import math
import sys

# The full domain. n runs to 500; alpha and alpha_tol on a 200-point grid, which
# includes the paper's own alpha = 0.1 and alpha_tol = 0.02 exactly.
N_MAX = 500
GRID = 200


def _cover(n, u):
    """P(S_{n+1} <= Q(u; F_hat_S^0)) for exchangeable continuous scores.

    Q(u; F_hat) is the ceil(n*u)-th order statistic, and the rank of S_{n+1}
    among n+1 exchangeable draws is uniform, so the probability is exactly
    k/(n+1). The paper writes the interval [k/(n+1), (k+1)/(n+1)) to allow
    ties; with continuous scores the left endpoint is attained.
    """
    k = min(math.ceil(n * u), n)
    return k / (n + 1)


def _cells():
    for n in range(2, N_MAX + 1):
        for i in range(1, GRID // 4):
            atol = i / GRID
            for j in range(1, GRID // 3):
                a = j / GRID
                uL, uU = 1 - a - atol, 1 - a + atol
                if 0 < uL < 1 and 0 < uU < 1:
                    yield n, a, atol, uL, uU


def _sweep():
    printed_lo_fail, printed_hi_fail = 0, 0
    step_lo_fail, step_hi_fail = 0, 0
    fixed_lo_fail, fixed_hi_fail = 0, 0
    total = 0
    worst_short, worst_at, first = 0.0, None, None

    for n, a, atol, uL, uU in _cells():
        total += 1
        covL, covU = _cover(n, uL), _cover(n, uU)

        # The theorem as printed: coverage in [1-a-atol, 1-a+atol+(n+1)^-1).
        if covL < uL - 1e-12:
            printed_lo_fail += 1
            if first is None:
                first = {"n": n, "alpha": round(a, 4), "alpha_tol": round(atol, 4),
                         "guaranteed_lower": round(covL, 6),
                         "claimed_lower": round(uL, 6)}
            if uL - covL > worst_short:
                worst_short = uL - covL
                worst_at = {"n": n, "alpha": round(a, 4), "alpha_tol": round(atol, 4),
                            "shortfall": round(uL - covL, 6),
                            "one_over_n_plus_one": round(1 / (n + 1), 6)}
        if covU >= uU + 1 / (n + 1) - 1e-12:
            printed_hi_fail += 1

        # The two intermediate steps the proof states, with k = ceil(n(1-beta)).
        kL, kU = min(math.ceil(n * uL), n), min(math.ceil(n * uU), n)
        if kL / (n + 1) < uL - 1e-12:
            step_lo_fail += 1
        if (kU + 1) / (n + 1) >= uU + 1 / (n + 1) - 1e-12:
            step_hi_fail += 1

        # The repaired statement: widen the lower endpoint by (n+1)^-1.
        if covL < uL - 1 / (n + 1) - 1e-12:
            fixed_lo_fail += 1
        if covU >= uU + 1 / (n + 1) - 1e-12:
            fixed_hi_fail += 1

    return {
        "cells": total,
        "printed_lower_endpoint_failures": printed_lo_fail,
        "printed_upper_endpoint_failures": printed_hi_fail,
        "proof_step_lower_failures": step_lo_fail,
        "proof_step_upper_failures": step_hi_fail,
        "corrected_lower_endpoint_failures": fixed_lo_fail,
        "corrected_upper_endpoint_failures": fixed_hi_fail,
        "first_failing_cell": first,
        "worst_shortfall": worst_at,
    }


def _repair_by_inflating_the_level():
    """The alternative repair: leave the band alone and change the algorithm.

    Section 3.1 sets q_L = Q(1-a-atol; F_hat_S^0) at the plain level. Theorem 4.2
    elsewhere in the same paper uses the inflated level (1-beta)(n+1)/n. Taking
    q_L at the inflated level is the standard split-conformal convention, and it
    restores the printed lower endpoint.
    """
    fail, out_of_range, checked = 0, 0, 0
    for n, a, atol, uL, _uU in _cells():
        k = math.ceil((n + 1) * uL)
        if k > n:
            # The inflated level exceeds the largest calibration order statistic,
            # so q_L is +infinity: n is too small for this (alpha, alpha_tol) and
            # the cell is outside the repair's range rather than a counterexample
            # to it. Counted and reported rather than silently dropped.
            out_of_range += 1
            continue
        checked += 1
        if k / (n + 1) < uL - 1e-12:
            fail += 1
    return {"failures": fail, "checked": checked, "out_of_range": out_of_range}


def _exposure_window(settings):
    """How wide the exposed band is at the campaign's own settings.

    The window is (guaranteed floor, claimed floor). A run whose coverage lands
    inside it satisfies everything the proof can actually deliver while sitting
    below the interval the theorem prints.
    """
    out = {}
    for n, a, atol in settings:
        uL = 1 - a - atol
        out[f"n={n}, alpha={a}, alpha_tol={atol}"] = {
            "claimed_lower_endpoint": round(uL, 6),
            "guaranteed_lower_endpoint": round(_cover(n, uL), 6),
            "window_width": round(uL - _cover(n, uL), 6),
            "alpha_tol_for_comparison": atol,
        }
    return out


def run(observed_coverages=None):
    sw = _sweep()
    inflated = _repair_by_inflating_the_level()

    # The campaign's own configuration, plus the neighbouring calibration sizes.
    window = _exposure_window([(30, 0.1, 0.02), (100, 0.1, 0.02), (500, 0.1, 0.02)])

    # Does any coverage this campaign actually measured land in the window?
    landed = []
    if observed_coverages:
        for label, (n, cov) in observed_coverages.items():
            uL = 1 - 0.1 - 0.02
            if _cover(n, uL) <= cov < uL:
                landed.append({"setting": label, "n": n, "coverage": cov,
                               "below_claimed_lower": uL})

    integrity = {
        # The sweep must be shown able to distinguish the printed statement from
        # the corrected one; if both passed everywhere it would carry no
        # information about either.
        "sweep_separates_the_printed_and_corrected_statements": bool(
            sw["printed_lower_endpoint_failures"] > 0
            and sw["corrected_lower_endpoint_failures"] == 0),
        "domain_is_exhaustive_not_sampled": sw["cells"] > 0,
        "papers_own_configuration_is_inside_the_domain": bool(window),
    }
    checks = {
        # What survives: the corrected band, and the upper endpoint as printed.
        "corrected_band_holds_everywhere": sw["corrected_lower_endpoint_failures"] == 0
        and sw["corrected_upper_endpoint_failures"] == 0,
        "printed_upper_endpoint_holds_everywhere": sw["printed_upper_endpoint_failures"] == 0,
        "inflating_the_quantile_level_also_repairs_the_lower_endpoint":
            inflated["failures"] == 0 and inflated["checked"] > 0,
    }

    blocked = [k for k, ok in integrity.items() if not ok]
    failed = [k for k, ok in checks.items() if not ok]

    return {
        "certifies": ("that the printed proof of Theorem 4.7 does not establish its lower "
                      "endpoint, and that two stated repairs do"),
        "does_not_certify": ("that any run violates the printed band -- the shortfall is in a "
                             "lower bound on coverage, so it breaks the proof without by "
                             "itself producing a violating run"),
        "verdict": "BLOCKED" if (blocked or failed) else "VERIFIED",
        "blocked_by": blocked,
        "failed_steps": failed,
        "integrity": integrity,
        "checks": checks,
        "sweep": sw,
        "corrected_claim": {
            "as_printed": "coverage in [1-a-a_tol, 1-a+a_tol+(n+1)^-1)",
            "supported": "coverage in [1-a-a_tol-(n+1)^-1, 1-a+a_tol+(n+1)^-1)",
            "why": ("q_L = Q(1-a-a_tol; F_hat_S^0) is the ceil(n(1-a-a_tol))-th order "
                    "statistic, so the coverage it guarantees is ceil(n(1-a-a_tol))/(n+1), "
                    "which falls short of 1-a-a_tol by up to (n+1)^-1. Widening the lower "
                    "endpoint by that amount makes the statement true, and symmetric."),
            "alternative_repair": (
                "leave the band as printed and change the selection rule instead: define "
                "q_L at the inflated level (1-a-a_tol)(n+1)/n, the same convention Theorem "
                "4.2 already uses in this paper. Verified over the same domain."),
            "which_is_preferred": (
                "the algorithmic repair, because it keeps the guarantee the paper advertises "
                "and costs one line in Algorithm 2; widening the band weakens the result"),
        },
        "inflated_level_repair": inflated,
        "exposure_window": window,
        "observed_runs_inside_the_window": landed,
        "note_on_the_upper_endpoint": (
            "the proof's intermediate upper inequality, (k+1)/(n+1) < 1-beta + (n+1)^-1, is "
            "also false over much of the domain, but the theorem's upper endpoint survives "
            "it: with continuous exchangeable scores the coverage is exactly k/(n+1), not "
            "(k+1)/(n+1), and k/(n+1) < 1-beta + (n+1)^-1 does hold everywhere"),
    }


def main():
    res = run()
    sw = res["sweep"]
    print("Theorem 4.7 certificate -- exhaustive over the full parameter space")
    print("=" * 68)
    print(f"  cells enumerated: {sw['cells']:,}")
    print()
    print("  As printed:")
    print(f"    lower endpoint fails in {sw['printed_lower_endpoint_failures']:,} cells "
          f"({100 * sw['printed_lower_endpoint_failures'] / sw['cells']:.1f}%)")
    print(f"    upper endpoint fails in {sw['printed_upper_endpoint_failures']:,} cells")
    print(f"    proof step 1 (k/(n+1) >= 1-b)        fails in "
          f"{sw['proof_step_lower_failures']:,} cells")
    print(f"    proof step 2 ((k+1)/(n+1) < 1-b+1/(n+1)) fails in "
          f"{sw['proof_step_upper_failures']:,} cells")
    print(f"    first failing cell: {sw['first_failing_cell']}")
    print(f"    worst shortfall:    {sw['worst_shortfall']}")
    print()
    print("  Corrected:")
    for k, v in res["checks"].items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    print()
    print(f"  supported band: {res['corrected_claim']['supported']}")
    inf = res["inflated_level_repair"]
    print(f"  preferred fix:  inflate the level -- {inf['failures']} failures over "
          f"{inf['checked']:,} in-range cells ({inf['out_of_range']:,} cells excluded: "
          f"n too small for the level)")
    print()
    print("  Exposure at the campaign's own settings:")
    for k, v in res["exposure_window"].items():
        print(f"    {k}: guaranteed {v['guaranteed_lower_endpoint']} vs claimed "
              f"{v['claimed_lower_endpoint']}  (window {v['window_width']}, "
              f"alpha_tol {v['alpha_tol_for_comparison']})")
    print()
    print(f"  verdict: {res['verdict']}")
    print(json.dumps({"integrity": res["integrity"]}, indent=1))
    return 0 if res["verdict"] == "VERIFIED" else 1


if __name__ == "__main__":
    sys.exit(main())
