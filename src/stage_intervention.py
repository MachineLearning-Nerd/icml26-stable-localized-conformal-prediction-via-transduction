"""Does the UNLABELED TARGET sample actually determine the fitted solution?

This is the load-bearing part of Claim 1: tracing shows the target covariates are
passed in, but the claim says they are *used*. The test refits at one lambda
against three unlabeled inputs and compares how far the solution moves:

  target   the real unlabeled target sample -- the reference fit
  sham     a SOURCE-drawn sample of the same size -- the treatment
  null     a second independent draw of the TARGET sample -- the matched null

The null changes the input without changing its distribution, so it measures how
much movement is mere resampling. Only a treatment that moves the solution
further than the null is evidence that the target *distribution* is what the fit
responds to.

Why this stage exists separately
--------------------------------
`stage_invariants` ran the same intervention with two scalar statistics and 5
repeats, and it did not settle:

    mean treatment / null shift in the discrepancy   0.064 / 0.092
    mean treatment / null shift in ||theta-theta^||  0.274 / 0.116

The two disagreed because neither is the quantity the claim is about. The
discrepancy is the *objective value*, and source-drawn covariates are easier for
a source-fitted CDF estimator to match, so the achieved value can move less even
when the solution moves more. `||theta - theta_hat||^2` is a scalar norm, so two
genuinely different solutions can share one -- a point already noted in that
stage's own output.

The quantity the claim is about is the distance between the fitted solutions
themselves, `||theta_a - theta_b||`. That is what this stage measures, paired
within each repeat, with a bootstrap interval over repeats. Reported honestly:
the switch to this statistic was made after seeing the 5-repeat result above,
and both earlier statistics are still reported per repeat so the change can be
audited rather than taken on trust.

Because only three fits per repeat are needed -- not the whole lambda grid twice
-- many more repeats fit in the same time, which is what gives the paired test
its power.
"""

import os
import sys
import time

import numpy as np

BOOT = 10_000


def _theta(tuner):
    """Flatten the tuner's fitted deltas into one vector.

    `Tuner` keeps the perturbation as `delta_weights` / `delta_biases`
    (Main/Tuner.py:24), with `None` in the entries for untuned layers.
    """
    import torch

    parts = []
    for group in (tuner.delta_weights, tuner.delta_biases):
        for p in group:
            if p is not None:
                parts.append(p.detach().reshape(-1))
    return torch.cat(parts).numpy().astype(float)


def _boot_diff(treat, null, rng):
    """Paired bootstrap over repeats of mean(treat - null)."""
    diff = np.asarray(treat, float) - np.asarray(null, float)
    if diff.size < 2:
        return None
    means = [diff[rng.integers(0, diff.size, diff.size)].mean() for _ in range(BOOT)]
    return {
        "mean_difference": float(diff.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "fraction_of_repeats_treatment_larger": float(np.mean(diff > 0)),
        "n_pairs": int(diff.size),
    }


def run(cfg, upstream_root):
    sys.path.insert(0, os.path.join(upstream_root, "SimuAnalysis"))
    sys.path.insert(0, os.path.join(upstream_root, "Main"))

    import config
    from copy import deepcopy

    from core import generate_agent
    from engGenerator import Generator
    from Predictor import Predictor
    from SLCP import SLCP
    from tools import defaultScore, setseed

    k = config.common_run_kwargs()
    d, r, N = k["d"], k["r"], k["N"]
    gamma_t, gamma_s, alpha = k["gamma_t"], k["gamma_s"], k["alpha"]
    hidden_dim, epoches, n_grid = k["hidden_dim"], k["epoches"], k["n_grid"]
    lbds, temperature = sorted(set(k["lbds"])), k["temperature"]
    dtype = cfg.get("dtype", "logabs")
    n, m = int(cfg.get("n", 30)), int(cfg.get("m", 500))
    n_reps = int(cfg.get("reps", 24))

    mu_t = np.ones(d) / np.sqrt(d) * r
    mu_s = np.zeros(d)
    me_t, me_s = d / 2, d / 3

    # Identical construction to stage_invariants, so the two nodes describe the
    # same experiment and the same seeds.
    setseed(k["repeats"] + 100)
    trAgent = generate_agent(N, d, me_s, gamma_s, mu_s, dtype)
    predAgent = generate_agent(N, d, me_s, gamma_s, mu_s, dtype)
    pred = Predictor("lr", fit_intercept=False)
    pred.trainFromAgent(trAgent)
    predAgent.calScore(pred, defaultScore)
    generator_base = Generator(d, hidden_dim, d)
    generator_base.trainEng(predAgent.getX(), predAgent.getS(), 10, 32, epoches, 5e-3, mute=True)

    mid_i = len(lbds) // 2
    lbd = float(lbds[mid_i])
    t0 = time.time()
    per_repeat = []

    for rep in range(n_reps):
        print(f"[intervention] repeat {rep}/{n_reps} start t+{time.time() - t0:.1f}s", flush=True)
        seed_rep = 1 + rep
        setseed(seed_rep)
        _test = generate_agent(500, d, me_t, gamma_t, mu_t, dtype)
        calTrAgent = generate_agent(n, d, me_t, gamma_t, mu_t, dtype)
        calAgent = generate_agent(n, d, me_t, gamma_t, mu_t, dtype)
        semiX = generate_agent(m, d, me_t, gamma_t, mu_t, dtype).X

        setseed(seed_rep)
        predictor = Predictor("lr", fit_intercept=False)
        predictor.trainFromAgent(calTrAgent)
        calAgent.calScore(predictor, defaultScore)
        calTrAgent.calScore(predictor, defaultScore)

        setseed(seed_rep)
        generator = deepcopy(generator_base)
        generator.cal_scalar(calTrAgent.getX(), calTrAgent.getS(), 200, stat_type="CvM")

        setseed(seed_rep)
        slcp = SLCP(calAgent, semiX, deepcopy(generator), predictor, [2])
        targ_alpha = float(np.clip(1 - (1 - alpha) * (calAgent.n + 1) / calAgent.n, 1e-6, 1 - 1e-6))
        tail_kw = dict(tol_gap=0.001, max_iter=10000, m=200,
                       targ_alpha=targ_alpha, penalty="MSE")

        def fit(unlabeled, opt_seed):
            """Refit from the same starting tuner; return (theta, diagnostics).

            `opt_seed` fixes the optimiser's own randomness so that the arms
            differ in exactly one thing -- the unlabeled sample -- except for the
            floor arm, which differs in exactly the seed.
            """
            setseed(opt_seed)
            tuner = deepcopy(slcp.tuner)
            part1, delta = tuner.tune_marginal(
                calAgent.getX(), calAgent.getS(), unlabeled,
                5, epoches, 5e-3, int(n_grid), lbd, temperature, **tail_kw)
            return _theta(tuner), float(part1), float(delta)

        # Reference fit, then three arms sharing one optimiser seed so that only
        # the unlabeled input differs between them.
        q0, q1 = seed_rep + 30_000, seed_rep + 40_000
        th_target, p_target, d_target = fit(semiX, q0)
        # FLOOR: identical data, different optimiser randomness. Without it the
        # other two distances have no scale -- a 10,100-dimensional non-convex
        # fit can land far away for reasons that have nothing to do with data.
        th_floor, p_floor, d_floor = fit(semiX, q1)
        setseed(seed_rep + 10_000)
        sham_X = generate_agent(m, d, me_s, gamma_s, mu_s, dtype).X
        th_sham, p_sham, d_sham = fit(sham_X, q1)
        setseed(seed_rep + 20_000)
        null_X = generate_agent(m, d, me_t, gamma_t, mu_t, dtype).X
        th_null, p_null, d_null = fit(null_X, q1)

        scale = float(np.linalg.norm(th_target)) or 1.0
        per_repeat.append({
            "repeat": rep,
            "theta_dim": int(th_target.size),
            "theta_distance_floor": float(np.linalg.norm(th_floor - th_target)),
            "theta_distance_treatment": float(np.linalg.norm(th_sham - th_target)),
            "theta_distance_null": float(np.linalg.norm(th_null - th_target)),
            "theta_distance_floor_relative": float(np.linalg.norm(th_floor - th_target) / scale),
            "theta_distance_treatment_relative": float(np.linalg.norm(th_sham - th_target) / scale),
            "theta_distance_null_relative": float(np.linalg.norm(th_null - th_target) / scale),
            # The two statistics the earlier stage used, kept so the change of
            # primary statistic can be audited against the same runs.
            "discrepancy": {"target": p_target, "floor": p_floor,
                            "sham": p_sham, "null": p_null},
            "delta_sq_norm": {"target": d_target, "floor": d_floor,
                              "sham": d_sham, "null": d_null},
        })

    treat = [p["theta_distance_treatment"] for p in per_repeat]
    null = [p["theta_distance_null"] for p in per_repeat]
    floor = [p["theta_distance_floor"] for p in per_repeat]
    rng = np.random.default_rng(0)
    boot = _boot_diff(treat, null, rng)
    # Does changing the unlabeled sample at all move the solution further than
    # re-running the optimiser on the same sample? If not, this design has no
    # power and neither of its comparisons can support a conclusion.
    power = _boot_diff(null, floor, rng)

    disc_t = [abs(p["discrepancy"]["sham"] - p["discrepancy"]["target"]) for p in per_repeat]
    disc_n = [abs(p["discrepancy"]["null"] - p["discrepancy"]["target"]) for p in per_repeat]

    out = {
        "kind": "intervention",
        "dtype": dtype, "n": n, "m": m, "reps": n_reps, "lambda": lbd,
        "seconds": round(time.time() - t0, 1),
        "primary_statistic": "||theta_a - theta_b||, the distance between fitted solutions",
        "mean_theta_distance_floor": float(np.mean(floor)),
        "mean_theta_distance_treatment": float(np.mean(treat)),
        "mean_theta_distance_null": float(np.mean(null)),
        "paired_bootstrap_treatment_vs_null": boot,
        "paired_bootstrap_sample_vs_refit_floor": power,
        # Whether the design can detect anything at all: changing the unlabeled
        # sample must move the solution further than re-running the optimiser on
        # the same sample. If this is false the two comparisons below are noise.
        "design_has_power": bool(power and power["ci95"][0] > 0),
        # A treatment that moves the solution further than a same-distribution
        # redraw, with the whole interval above zero, would show the fit responds
        # to the target *distribution* and not merely to the sample.
        "distribution_shift_detected": bool(boot and boot["ci95"][0] > 0),
        "secondary_discrepancy_statistic": {
            "mean_treatment_shift": float(np.mean(disc_t)),
            "mean_null_shift": float(np.mean(disc_n)),
            "note": ("Reported, not used to adjudicate. The discrepancy is the objective "
                     "value; source-drawn covariates are easier for a source-fitted CDF "
                     "estimator to match, so its achieved value can move less even when the "
                     "solution moves more."),
        },
        "per_repeat": per_repeat,
    }
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(os.path.join(out_dir, "checks"), exist_ok=True)
    import json
    with open(os.path.join(out_dir, "checks", "intervention.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out
