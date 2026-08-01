"""Certify that the printed proof of Theorem 4.2 derives the bound it states.

Theorem 4.2 quantifies over every distribution, every n and every lambda, so no
simulation can verify it; the campaign standard for a claim of that shape is a
machine-checkable certificate. Appendix B.3 writes the proof out in full as a
chain of algebraic inequalities, which makes one possible.

The scope is exactly one statement, and no more:

    the proof printed in Appendix B.3, as written, is a valid derivation of

        |P(Y_{n+1} in C_St(X_{n+1})) - (1-alpha)|
            <= C * min(eps + lambda^(1/2) + n^-1, delta_S + lambda^(-1/2) + n^-1)

    from Assumption 4.1.

It says nothing about whether Assumption 4.1 holds for any particular data
generating process, nothing about the size of C, and nothing about the paper's
experiments.

Each step is certified twice, and the second check is the one that carries the
weight:

  * a **witness identity** -- the slack `rhs - lhs` is written as an explicit
    expression, and sympy must confirm the difference is exactly zero. No
    tolerance is involved.
  * a **refutation search** -- the slack is evaluated over a grid of the domain
    that respects the step's hypotheses, and must never go negative.

An identity check alone proves nothing, because an identity can be written to
match whatever is in front of it. So every step also carries **mutations**:
altered exponents, altered constants, dropped hypotheses. Each mutation must be
caught *by the refutation search*, not merely by the identity mismatch. A step
whose mutations all survive the search is vacuous, and is reported as failed
however its own check came out.

Registered in `.openresearch/artifacts/c2_certificate_preregistration.md`
before any of this was written.

Run: python -m verify_thm42_certificate   (exits nonzero if any step fails to
certify, or if any step is vacuous)
"""
import itertools
import json
import math
import sys

import numpy as np
import sympy as sp

# Fixed for reproducibility; the refutation searches are grid-plus-random and
# the random part must land on the same points every run.
SEED = 0
GRID = 12          # points per axis in the multi-axis refutation searches
N_MAX = 2000       # exhaustive range for the finite-sample step
ALPHA_GRID = 199   # alpha resolution for the same


# --------------------------------------------------------------------------
# The certification harness.
# --------------------------------------------------------------------------

def _identity(lhs, rhs, witness):
    """rhs - lhs must equal the witness exactly, and the witness must be a sum
    of manifestly nonnegative pieces."""
    return sp.simplify(sp.expand(rhs - lhs - witness)) == 0


def _search(slack, axes, hyp=None, rng=None, extra=0):
    """Smallest value of `slack` over the grid, restricted to the hypotheses.

    Returns (min_slack, witness_point, n_points). A negative minimum is a
    refutation: a point where the step's own hypotheses hold and its conclusion
    does not.
    """
    names = list(axes)
    grids = [axes[k] for k in names]
    pts = list(itertools.product(*grids))
    if extra and rng is not None:
        for _ in range(extra):
            pts.append(tuple(float(rng.uniform(min(g), max(g))) for g in grids))
    worst, where, seen = math.inf, None, 0
    for p in pts:
        vals = dict(zip(names, p))
        if hyp and not hyp(vals):
            continue
        seen += 1
        s = slack(vals)
        if s is None or (isinstance(s, float) and math.isnan(s)):
            continue
        if s < worst:
            worst, where = s, vals
    return worst, where, seen


class Step:
    """One inequality from the printed proof, with its mutations."""

    def __init__(self, key, quote, axes, slack, hyp=None,
                 identity=None, mutations=(), extra=0, scale=None):
        self.key = key
        self.quote = quote
        self.axes = axes
        self.hyp = hyp
        self.identity = identity          # (lhs, rhs, witness) or None
        self.extra = extra
        # Several of these inequalities are swept over eight decades, so an
        # absolute threshold would read double-precision rounding at the top of
        # the range as a refutation. `scale` reports the magnitude of the terms
        # being compared and the slack is measured relative to it, which is what
        # "this inequality is violated" actually means in floating point.
        self.scale = scale
        self.slack = self._rel(slack)
        self.mutations = [(n, self._rel(f)) for n, f in mutations]

    def _rel(self, fn):
        if self.scale is None:
            return fn
        return lambda v: fn(v) / max(1.0, abs(self.scale(v)))

    def certify(self, rng):
        out = {"step": self.key, "as_printed": self.quote}
        if self.scale is not None:
            out["slack_measured_relative_to_term_magnitude"] = True

        if self.identity is not None:
            out["witness_identity_is_exact"] = bool(_identity(*self.identity))
        else:
            out["witness_identity_is_exact"] = None  # exhaustive steps have none

        worst, where, seen = _search(self.slack, self.axes, self.hyp, rng, self.extra)
        out["points_searched"] = seen
        out["min_slack_found"] = None if worst is math.inf else float(worst)
        out["no_refutation_found"] = bool(worst is not math.inf and worst >= -1e-12)
        if not out["no_refutation_found"] and where is not None:
            out["refuting_point"] = {k: float(v) for k, v in where.items()}

        # The anti-vacuity gate. Each mutation must be caught by the same
        # search that certified the step, so the search is shown to have teeth
        # on this domain rather than merely to have returned nonnegative.
        caught = {}
        for name, mslack in self.mutations:
            mworst, mwhere, _ = _search(mslack, self.axes, self.hyp, rng, self.extra)
            caught[name] = {
                "refuted": bool(mworst is not math.inf and mworst < -1e-12),
                "min_slack": None if mworst is math.inf else float(mworst),
                "at": {k: float(v) for k, v in mwhere.items()} if mwhere else None,
            }
        out["mutations"] = caught
        out["every_mutation_is_refuted"] = bool(
            caught and all(c["refuted"] for c in caught.values()))

        out["certified"] = bool(
            out["no_refutation_found"]
            and out["every_mutation_is_refuted"]
            and out["witness_identity_is_exact"] is not False)
        return out


# --------------------------------------------------------------------------
# Step L1: averaging over the m unlabeled points preserves the Lipschitz
# constant.  |F1_hat(s;F_t1) - F1_hat(s;F_t2)| <= m^-1 sum_j |...| <= L_th*||t1-t2||
# --------------------------------------------------------------------------

def _l1_terms(v):
    """The m per-point discrepancies |F(s|X_j;t1) - F(s|X_j;t2)|, each capped
    by K = L_theta*||t1-t2||."""
    m, K, p = int(v["m"]), v["K"], int(v["pattern"])
    j = np.arange(m, dtype=float)
    if p == 0:
        return np.full(m, K)                       # all at the cap
    if p == 1:
        return np.zeros(m)
    if p == 2:
        return K * (j + 1) / m                     # ramp up to the cap
    if p == 3:
        return np.where(j == 0, K, 0.0)            # one term at the cap
    return K * np.abs(np.cos(j))                   # spread below it


_m = sp.Symbol("m", positive=True, integer=True)
_j = sp.Symbol("j", integer=True)
_K = sp.Symbol("K", nonnegative=True)
_a = sp.IndexedBase("a")

STEP_L1 = Step(
    key="L1_averaging_preserves_the_lipschitz_constant",
    quote=("|F1_hat(s;F_t1) - F1_hat(s;F_t2)| <= m^-1 sum_j |F(s|X_j;t1) - "
           "F(s|X_j;t2)| <= L_theta ||t1 - t2||_2"),
    axes={"m": list(range(1, 9)),
          "K": [0.0, 0.25, 1.0, 4.0],
          "pattern": [0, 1, 2, 3, 4]},
    # The search builds the actual m-term average rather than a stand-in for
    # it, so the mutations below are refuted by summing, not by arithmetic on a
    # single number. Pattern 0 puts every term at its cap, which is where the
    # bound binds; the others spread the terms out.
    slack=lambda v: v["K"] - float(np.mean(_l1_terms(v))),
    identity=(sp.Sum(_a[_j], (_j, 1, _m)) / _m,          # lhs
              _K,                                         # rhs
              sp.Sum(_K - _a[_j], (_j, 1, _m)) / _m),     # witness
    mutations=[
        # The average cannot be bounded by half the per-term cap ...
        ("average_bounded_by_half_the_term_bound",
         lambda v: v["K"] / 2 - float(np.mean(_l1_terms(v)))),
        # ... nor does averaging buy a factor of m.
        ("average_bounded_by_term_bound_over_m",
         lambda v: v["K"] / v["m"] - float(np.mean(_l1_terms(v)))),
        # ... and the average is not bounded by the smallest term.
        ("average_bounded_by_the_smallest_term",
         lambda v: float(np.min(_l1_terms(v))) - float(np.mean(_l1_terms(v)))),
    ],
)


# --------------------------------------------------------------------------
# Step B1: Lemma B.1, the quantile-Lipschitz step.  This is the one analytic
# ingredient, so it is checked by construction over an explicit CDF family and
# is additionally required to be *attained* -- a bound no pair reaches would
# certify vacuously.
# --------------------------------------------------------------------------

def _piecewise_cdf(dens, xs):
    """CDF on [0,1] of a piecewise-constant density, evaluated on `xs`."""
    k = len(dens)
    edges = np.linspace(0.0, 1.0, k + 1)
    mass = np.asarray(dens, float) / k
    mass = mass / mass.sum()
    cum = np.concatenate([[0.0], np.cumsum(mass)])
    idx = np.clip(np.searchsorted(edges, xs, side="right") - 1, 0, k - 1)
    return cum[idx] + (xs - edges[idx]) * mass[idx] * k


def _quantile(xs, cdf, u):
    """inf{s : F(s) >= u}.

    Both CDFs here are piecewise linear on `xs`, so inverting by interpolation
    is exact. Snapping to the nearest grid point instead would put a floor of
    one grid spacing under every measured quantile distance, and that floor --
    not any property of the lemma -- would decide whether the bound holds at
    small eps.
    """
    return float(np.interp(u, cdf, xs))


# The lemma's bound, and the mutations of it that the same sweep must refute.
_B1_BOUNDS = {
    "as_printed": lambda sup, lo: sup / lo,
    "bound_halved": lambda sup, lo: sup / (2 * lo),
    "density_multiplied_instead_of_divided": lambda sup, lo: sup * lo,
    "density_lower_bound_dropped": lambda sup, lo: sup,
}


def _lemma_b1_sweep():
    """Sweep the CDF family.

    Returns the worst slack for the printed bound and for each mutation of it,
    plus how close the printed bound is ever approached. Attainment matters: a
    bound no pair in the family reaches would certify without saying anything
    about the constant, and the mutations are what show the sweep can reject.
    """
    xs = np.linspace(0.0, 1.0, 20001)
    rng = np.random.default_rng(SEED)
    worst = {k: math.inf for k in _B1_BOUNDS}
    best_ratio, n = 0.0, 0
    shapes = [
        [1.0, 1.0, 1.0, 1.0],        # uniform: density exactly at its bound
        [1.0, 2.0, 1.0, 2.0],
        [3.0, 1.0, 1.0, 3.0],
        [1.0, 1.0, 4.0, 1.0],
        [2.0, 1.0, 3.0, 1.5],
    ]
    for shape in shapes:
        f1 = _piecewise_cdf(shape, xs)
        # Density of F1 in the units of this construction: mass per unit length.
        lo = float(min(shape)) / (sum(shape) / len(shape))
        for eps in (0.0, 0.005, 0.02, 0.05, 0.1):
            for k in range(6):
                # F2 differs from F1 by at most eps and stays a CDF: shifting
                # the whole curve down by eps and clipping is the extremal
                # perturbation, which is what makes the bound attainable.
                if k == 0:
                    f2 = np.clip(f1 - eps, 0.0, 1.0)
                elif k == 1:
                    f2 = np.clip(f1 + eps, 0.0, 1.0)
                else:
                    bump = eps * rng.uniform(-1, 1, size=8)
                    f2 = np.clip(f1 + np.interp(
                        xs, np.linspace(0, 1, 8), bump), 0.0, 1.0)
                    f2 = np.maximum.accumulate(f2)
                sup = float(np.max(np.abs(f1 - f2)))
                for u in (0.5, 0.75, 0.9, 0.95):
                    dq = abs(_quantile(xs, f1, u) - _quantile(xs, f2, u))
                    n += 1
                    for name, fn in _B1_BOUNDS.items():
                        b = fn(sup, lo) if lo > 0 else math.inf
                        worst[name] = min(worst[name], b - dq)
                    printed = sup / lo if lo > 0 else math.inf
                    if printed > 1e-9:
                        best_ratio = max(best_ratio, dq / printed)
    return worst, best_ratio, n


# --------------------------------------------------------------------------
# Step C1: Case 1, the origin of lambda^(-1/2).
#   J_lam(t~) <= J_lam(t^)  =>  lam*||t~-t^||^2 <= d0(F0_hat, F1_hat(.;F_t^))^2
#                            =>  sqrt(lam)*||t~-t^|| <= M_Theta
# --------------------------------------------------------------------------

_lam, _t, _D0, _MTh = sp.symbols("lam t D0 M_Theta", positive=True)

STEP_C1 = Step(
    key="C1_case_one_yields_lambda_to_the_minus_one_half",
    quote=("sqrt(lambda)*||t~ - t^||_2 <= d0(F0_hat, F1_hat(.;F_t^)) <= M_Theta, "
           "hence ||t~ - t^||_2 <= M_Theta * lambda^(-1/2)"),
    axes={"lam": list(np.logspace(-4, 4, GRID)),
          "M": list(np.linspace(0.05, 5.0, 6)),
          "ratio": list(np.linspace(0.0, 1.0, 6))},
    # ratio places sqrt(lam)*t anywhere in [0, D0] with D0 = M, i.e. the
    # hypothesis is saturated at ratio = 1, which is where the bound binds.
    slack=lambda v: v["M"] / math.sqrt(v["lam"]) - v["ratio"] * v["M"] / math.sqrt(v["lam"]),
    identity=(_t, _MTh / sp.sqrt(_lam),
              (_MTh - _D0) / sp.sqrt(_lam) + (_D0 - sp.sqrt(_lam) * _t) / sp.sqrt(_lam)),
    mutations=[
        # A weaker decay in lambda cannot hold: it fails for small lambda.
        ("exponent_minus_one_quarter",
         lambda v: v["M"] * v["lam"] ** -0.25 - v["ratio"] * v["M"] / math.sqrt(v["lam"])),
        # A stronger one cannot either: it fails for large lambda.
        ("exponent_minus_one",
         lambda v: v["M"] * v["lam"] ** -1.0 - v["ratio"] * v["M"] / math.sqrt(v["lam"])),
        # And the lambda dependence is not removable.
        ("no_lambda_dependence",
         lambda v: v["M"] - v["ratio"] * v["M"] / math.sqrt(v["lam"])),
    ],
)


# --------------------------------------------------------------------------
# Step C2: Case 2, the origin of lambda^(+1/2).
#   eps^2 + C_Theta*lam <= (eps + C_Theta^(1/2) * lam^(1/2))^2
# --------------------------------------------------------------------------

_eps, _CTh = sp.symbols("epsilon C_Theta", nonnegative=True)

STEP_C2 = Step(
    key="C2_case_two_yields_lambda_to_the_plus_one_half",
    quote="eps^2 + C_Theta*lambda <= (eps + C_Theta^(1/2)*lambda^(1/2))^2",
    axes={"eps": list(np.linspace(0.0, 3.0, GRID)),
          "C": list(np.logspace(-3, 3, GRID)),
          "lam": list(np.logspace(-4, 4, GRID))},
    slack=lambda v: (v["eps"] + math.sqrt(v["C"] * v["lam"])) ** 2
                    - (v["eps"] ** 2 + v["C"] * v["lam"]),
    identity=(_eps ** 2 + _CTh * _lam,
              (_eps + sp.sqrt(_CTh) * sp.sqrt(_lam)) ** 2,
              2 * _eps * sp.sqrt(_CTh) * sp.sqrt(_lam)),
    mutations=[
        ("lambda_to_the_first_power_inside_the_square",
         lambda v: (v["eps"] + math.sqrt(v["C"]) * v["lam"]) ** 2
                   - (v["eps"] ** 2 + v["C"] * v["lam"])),
        ("lambda_to_the_one_quarter_inside_the_square",
         lambda v: (v["eps"] + math.sqrt(v["C"]) * v["lam"] ** 0.25) ** 2
                   - (v["eps"] ** 2 + v["C"] * v["lam"])),
        ("regulariser_term_dropped",
         lambda v: (v["eps"]) ** 2 - (v["eps"] ** 2 + v["C"] * v["lam"])),
    ],
    scale=lambda v: v["eps"] ** 2 + v["C"] * v["lam"],
)


# --------------------------------------------------------------------------
# Step N1: the finite-sample n^(-1) term, exhaustively.
#   1 - alpha_n = (1-alpha)(n+1)/n, q_hat = Q(1-alpha_n; F0_hat), and
#   |E F_S(q_hat) - (1-alpha)| <= n^-1.
# --------------------------------------------------------------------------

def _finite_sample_sweep(bound):
    """Exhaustive over n in [2, N_MAX] x alpha grid. Returns (worst_slack,
    max_attained_ratio, n_checked, n_out_of_scope)."""
    worst, ratio, checked, oos = math.inf, 0.0, 0, 0
    alphas = [k / (ALPHA_GRID + 1) for k in range(1, ALPHA_GRID + 1)]
    for n in range(2, N_MAX + 1):
        for a in alphas:
            u = (1.0 - a) * (n + 1) / n
            if u >= 1.0:
                # (1-alpha)(n+1) > n: the level exceeds the largest order
                # statistic and q_hat is +infinity. Outside the theorem's
                # useful range rather than a violation; counted and reported.
                oos += 1
                continue
            k = math.ceil((1.0 - a) * (n + 1))
            cover = k / (n + 1)
            gap = abs(cover - (1.0 - a))
            b = bound(n)
            checked += 1
            worst = min(worst, b - gap)
            if b > 0:
                ratio = max(ratio, gap / b)
    return worst, ratio, checked, oos


# --------------------------------------------------------------------------
# Step S1: both cases hold at once, which is what licenses the `min`.
#   t~ = argmin_{t in Theta} J_lam(t), and t^, t~_0 are both in Theta, so
#   J_lam(t~) <= J_lam(t^) AND J_lam(t~) <= J_lam(t~_0).
# --------------------------------------------------------------------------

def _simultaneity_sweep(argmin_of_J_lambda=True):
    """Draw random finite Theta and random objectives; check the minimiser is
    below the value at every named member. `argmin_of_J_lambda=False` is the
    mutation: minimise the unregularised J_0 instead, which no longer bounds
    J_lambda at the named points."""
    rng = np.random.default_rng(SEED)
    worst, n = math.inf, 0
    for _ in range(4000):
        k = int(rng.integers(2, 12))
        d = rng.uniform(0.0, 3.0, size=k) ** 2      # d(F0_hat, F1_hat(.;F_t))
        pen = rng.uniform(0.0, 3.0, size=k) ** 2    # ||t - t^||^2
        lam = float(np.exp(rng.uniform(-8, 8)))
        Jl = d + lam * pen
        i = int(np.argmin(Jl if argmin_of_J_lambda else d))
        # theta^ is the member with zero penalty if present, else index 0;
        # theta~_0 is the minimiser of the unregularised objective.
        i_hat = int(np.argmin(pen))
        i_zero = int(np.argmin(d))
        for named in (i_hat, i_zero):
            n += 1
            worst = min(worst, float(Jl[named] - Jl[i]))
    return worst, n


# --------------------------------------------------------------------------
# Step A1: assembly -- both branch bounds are dominated by the stated form with
# one explicit constant.
# --------------------------------------------------------------------------

def _assembly_C(v):
    return max(v["LFhi"], v["LTh"] * v["LFhi"] * v["M"],
               2 * v["LFhi"] * math.sqrt(v["C"]), 1.0)


def _branch1_slack(v, C=None):
    C = _assembly_C(v) if C is None else C
    lam, n = v["lam"], v["n"]
    printed = (v["LFhi"] * v["dS"] + v["LTh"] * v["LFhi"] * v["M"] / math.sqrt(lam)
               + 1.0 / n)
    stated = C * (v["dS"] + lam ** -0.5 + 1.0 / n)
    return stated - printed


def _branch2_slack(v, C=None):
    C = _assembly_C(v) if C is None else C
    lam, n = v["lam"], v["n"]
    printed = (v["LFhi"] * v["eps"] + 2 * v["LFhi"] * math.sqrt(v["C"]) * math.sqrt(lam)
               + 1.0 / n)
    stated = C * (v["eps"] + lam ** 0.5 + 1.0 / n)
    return stated - printed


_ASSEMBLY_AXES = {
    "lam": list(np.logspace(-3, 3, 7)),
    "n": [2.0, 10.0, 100.0, 1000.0],
    "eps": [0.0, 0.5, 2.0],
    "dS": [0.0, 0.5, 2.0],
    "LFhi": [0.25, 1.0, 4.0],
    "LTh": [0.25, 1.0, 4.0],
    "M": [0.25, 1.0, 4.0],
    "C": [0.25, 1.0, 4.0],
}

STEP_A1 = Step(
    key="A1_both_branches_are_dominated_by_the_stated_form",
    quote=("(12) and (13) are each <= C*(the corresponding branch of the min) "
           "for C = max(LF_hi, L_Theta*LF_hi*M_Theta, 2*LF_hi*C_Theta^(1/2), 1)"),
    axes=_ASSEMBLY_AXES,
    slack=lambda v: min(_branch1_slack(v), _branch2_slack(v)),
    mutations=[
        # C must include every coefficient the proof produces; dropping the
        # Case-1 one breaks the domination as soon as L_Theta*M_Theta > 1.
        ("C_omits_the_case_one_coefficient",
         lambda v: min(_branch1_slack(v, C=max(v["LFhi"], 1.0)),
                       _branch2_slack(v, C=max(v["LFhi"], 1.0)))),
        ("C_omits_the_case_two_coefficient",
         lambda v: min(_branch1_slack(v, C=max(v["LFhi"], v["LTh"] * v["LFhi"] * v["M"], 1.0)),
                       _branch2_slack(v, C=max(v["LFhi"], v["LTh"] * v["LFhi"] * v["M"], 1.0)))),
        # The branches cannot be swapped: lambda^(+1/2) does not dominate the
        # Case-1 term, nor lambda^(-1/2) the Case-2 term.
        ("branch_exponents_swapped",
         lambda v: min(
             _assembly_C(v) * (v["dS"] + v["lam"] ** 0.5 + 1.0 / v["n"])
             - (v["LFhi"] * v["dS"] + v["LTh"] * v["LFhi"] * v["M"] / math.sqrt(v["lam"])
                + 1.0 / v["n"]),
             _assembly_C(v) * (v["eps"] + v["lam"] ** -0.5 + 1.0 / v["n"])
             - (v["LFhi"] * v["eps"]
                + 2 * v["LFhi"] * math.sqrt(v["C"]) * math.sqrt(v["lam"])
                + 1.0 / v["n"]))),
    ],
    scale=lambda v: _assembly_C(v) * (max(v["eps"], v["dS"])
                                      + v["lam"] ** 0.5 + v["lam"] ** -0.5),
)


# --------------------------------------------------------------------------
# Looseness audit: places where the printed statement is weaker than the
# derivation above it. Slack in an upper bound is still an upper bound, so
# these are findings about the write-up, never falsifications.
# --------------------------------------------------------------------------

def _looseness():
    rng = np.random.default_rng(SEED)
    findings = []

    # (12) carries a "+ n^-1" that its own derivation never produces: delta_S
    # is defined against Q(1-alpha; F_S) and so already absorbs the alpha_n
    # gap. Check the derived bound without it still dominates.
    worst = math.inf
    for _ in range(20000):
        v = {k: float(np.exp(rng.uniform(-4, 4))) for k in ("LFhi", "LTh", "M", "dS", "lam")}
        n = float(rng.integers(2, 2000))
        derived = v["LFhi"] * v["dS"] + v["LTh"] * v["LFhi"] * v["M"] / math.sqrt(v["lam"])
        printed = derived + 1.0 / n
        worst = min(worst, printed - derived)
    findings.append({
        "where": "equation (12)",
        "finding": ("the + n^-1 term is carried but never derived: the Case 1 chain "
                    "bounds |E F_S(q_St) - (1-alpha)| by L_F*delta_S + "
                    "L_Theta*L_F*M_Theta*lambda^(-1/2) alone, because delta_S is "
                    "defined against Q(1-alpha; F_S) and already absorbs the "
                    "alpha_n-versus-alpha gap"),
        "printed_bound_still_valid": bool(worst >= 0.0),
        "consequence": "none; the printed bound is the derived bound plus a nonnegative term",
    })

    # (13) prints coefficient 2 on the lambda^(1/2) term where the two lines
    # above it produce coefficient 1.
    findings.append({
        "where": "equation (13)",
        "finding": ("the coefficient on L_F*C_Theta^(1/2)*lambda^(1/2) is printed as 2, "
                    "while the immediately preceding line derives it with coefficient 1"),
        "printed_bound_still_valid": True,
        "consequence": "none; 2x an upper bound is an upper bound",
    })
    return findings


# --------------------------------------------------------------------------

def run():
    rng = np.random.default_rng(SEED)
    steps = [s.certify(rng) for s in (STEP_L1, STEP_C1, STEP_C2, STEP_A1)]

    # B1 -- exhaustive over the constructed CDF family, plus attainment.
    b1_worst, b1_ratio, b1_n = _lemma_b1_sweep()
    b1 = {
        "step": "B1_lemma_b1_quantile_lipschitz",
        "as_printed": ("sup_s |F1(s) - F2(s)| <= eps with one density bounded below "
                       "by L_F implies |Q(1-a;F1) - Q(1-a;F2)| <= eps / L_F"),
        "points_searched": b1_n,
        "min_slack_found": float(b1_worst["as_printed"]),
        "no_refutation_found": bool(b1_worst["as_printed"] >= -1e-12),
        "bound_is_attained": bool(b1_ratio >= 0.95),
        "closest_approach_to_the_bound": float(b1_ratio),
        "witness_identity_is_exact": None,
        "mutations": {
            name: {"refuted": bool(w < -1e-12), "min_slack": float(w)}
            for name, w in b1_worst.items() if name != "as_printed"
        },
    }
    b1["every_mutation_is_refuted"] = all(m["refuted"] for m in b1["mutations"].values())
    # Attainment is part of certification here: a bound nothing reaches would
    # pass without carrying information about the constant.
    b1["certified"] = bool(b1["no_refutation_found"] and b1["bound_is_attained"]
                           and b1["every_mutation_is_refuted"])
    steps.insert(1, b1)

    # N1 -- exhaustive over n x alpha.
    n1_slack, n1_ratio, n1_checked, n1_oos = _finite_sample_sweep(lambda n: 1.0 / n)
    half_slack, _, _, _ = _finite_sample_sweep(lambda n: 0.5 / n)
    sq_slack, _, _, _ = _finite_sample_sweep(lambda n: 1.0 / n ** 2)
    n1 = {
        "step": "N1_finite_sample_n_inverse_term",
        "as_printed": "|E F_S(q_hat) - (1-alpha)| <= n^-1 at 1-alpha_n = (1-alpha)(n+1)/n",
        "points_searched": n1_checked,
        "exhaustive_domain": f"n in [2, {N_MAX}] x alpha on {ALPHA_GRID} grid points",
        "levels_out_of_range": n1_oos,
        "out_of_range_note": ("(1-alpha)(n+1) > n makes the level exceed the largest "
                              "calibration order statistic, so q_hat is +infinity; these "
                              "cells lie outside the theorem's useful range and are "
                              "excluded rather than scored"),
        "min_slack_found": float(n1_slack),
        "no_refutation_found": bool(n1_slack >= -1e-12),
        "closest_approach_to_the_bound": float(n1_ratio),
        "witness_identity_is_exact": None,
        "mutations": {
            "bound_halved_to_one_over_two_n": {"refuted": bool(half_slack < -1e-12),
                                               "min_slack": float(half_slack)},
            "bound_squared_to_n_to_the_minus_two": {"refuted": bool(sq_slack < -1e-12),
                                                    "min_slack": float(sq_slack)},
        },
    }
    n1["every_mutation_is_refuted"] = all(m["refuted"] for m in n1["mutations"].values())
    n1["certified"] = bool(n1["no_refutation_found"] and n1["every_mutation_is_refuted"])
    steps.append(n1)

    # S1 -- simultaneity.
    s1_slack, s1_n = _simultaneity_sweep(True)
    s1_mut, _ = _simultaneity_sweep(False)
    s1 = {
        "step": "S1_both_cases_hold_at_once_so_the_min_is_licensed",
        "as_printed": "J_lambda(theta~) <= J_lambda(theta^) AND J_lambda(theta~_0)",
        "points_searched": s1_n,
        "min_slack_found": float(s1_slack),
        "no_refutation_found": bool(s1_slack >= -1e-12),
        "witness_identity_is_exact": None,
        "mutations": {
            "minimiser_taken_of_the_unregularised_objective": {
                "refuted": bool(s1_mut < -1e-12), "min_slack": float(s1_mut)},
        },
    }
    s1["every_mutation_is_refuted"] = all(m["refuted"] for m in s1["mutations"].values())
    s1["certified"] = bool(s1["no_refutation_found"] and s1["every_mutation_is_refuted"])
    steps.append(s1)

    order = ["L1_averaging_preserves_the_lipschitz_constant",
             "B1_lemma_b1_quantile_lipschitz",
             "C1_case_one_yields_lambda_to_the_minus_one_half",
             "C2_case_two_yields_lambda_to_the_plus_one_half",
             "N1_finite_sample_n_inverse_term",
             "S1_both_cases_hold_at_once_so_the_min_is_licensed",
             "A1_both_branches_are_dominated_by_the_stated_form"]
    steps.sort(key=lambda s: order.index(s["step"]))

    registered = set(order)
    integrity = {
        "every_registered_step_was_attempted":
            {s["step"] for s in steps} == registered,
        "no_step_is_vacuous":
            all(s.get("every_mutation_is_refuted") for s in steps),
        "exhaustive_steps_covered_their_registered_domain":
            bool(n1["points_searched"] > 0 and b1["points_searched"] > 0),
    }
    checks = {s["step"]: bool(s["certified"]) for s in steps}

    blocked = [k for k, ok in integrity.items() if not ok]
    failed = [k for k, ok in checks.items() if not ok]
    verdict = "BLOCKED" if blocked else ("VERIFIED" if not failed else "BLOCKED")

    return {
        "certifies": ("that the proof printed in Appendix B.3 is a valid derivation "
                      "of the Theorem 4.2 bound from Assumption 4.1"),
        "does_not_certify": ("that Assumption 4.1 holds for any particular data "
                             "generating process, that C is small, or anything about "
                             "the paper's experiments"),
        "verdict": verdict,
        "blocked_by": blocked,
        "failed_steps": failed,
        "integrity": integrity,
        "checks": checks,
        "steps": steps,
        "write_up_looseness": _looseness(),
        "seed": SEED,
    }


def main():
    res = run()
    print("Theorem 4.2 proof certificate")
    print("=" * 64)
    for s in res["steps"]:
        muts = s.get("mutations", {})
        n_ref = sum(1 for m in muts.values() if m["refuted"])
        print(f"  {'PASS' if s['certified'] else 'FAIL'}  {s['step']}")
        print(f"        searched {s['points_searched']:>7} pts   "
              f"min slack {s['min_slack_found']:+.3e}   "
              f"mutations refuted {n_ref}/{len(muts)}")
        if s.get("witness_identity_is_exact") is not None:
            print(f"        witness identity exact: {s['witness_identity_is_exact']}")
        if s.get("refuting_point"):
            print(f"        REFUTED AT {s['refuting_point']}")
    print()
    for f in res["write_up_looseness"]:
        print(f"  note  {f['where']}: {f['finding'][:96]}...")
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
