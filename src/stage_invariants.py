"""Claim 1: measure the defining properties of Equation 7, not just "it runs".

Three numbers decide whether the implemented object is the object Equation 7
describes, and all three are measured on the paper's own LogAbs setting with the
paper's own code:

  1. Provenance. `F_hat_S|X` must see labeled SOURCE data only, and
     `F_hat_S^1` must see UNLABELED target covariates only. Recorded as a
     construction trace with array identities and shapes, so a reviewer can
     confirm no target label reaches the CDF estimator.

  2. lambda = 0 recovers the standard conformal quantile of `F_hat_S^0`
     (Section 3 and Appendix C.1). Measured as
     |q_StCP(lambda=0) - Q(1-alpha_n; F_hat_S^0)|, which must be small relative
     to the spacing of the calibration score order statistics -- the finite-grid
     alignment cannot do better than that spacing.

  3. lambda -> infinity drives the calibrated parameter back to theta_hat.
     `Tuner.get_delta_norm()` is exactly ||theta - theta_hat||_2^2 (the deltas
     are initialised at zero), so it must decrease monotonically in lambda.
"""

import json
import os
import sys
import time

import numpy as np


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
    n_reps = int(cfg.get("reps", 5))

    mu_t = np.ones(d) / np.sqrt(d) * r
    mu_s = np.zeros(d)
    me_t, me_s = d / 2, d / 3

    setseed(k["repeats"] + 100)
    trAgent = generate_agent(N, d, me_s, gamma_s, mu_s, dtype)
    predAgent = generate_agent(N, d, me_s, gamma_s, mu_s, dtype)
    pred = Predictor("lr", fit_intercept=False)
    pred.trainFromAgent(trAgent)
    predAgent.calScore(pred, defaultScore)
    generator_base = Generator(d, hidden_dim, d)
    generator_base.trainEng(predAgent.getX(), predAgent.getS(), 10, 32, epoches, 5e-3, mute=True)

    provenance = {
        "F_hat_S_given_X_trained_on": {
            "agent": "predAgent",
            "distribution": "SOURCE (mu_s=0, gamma_s=%.2f, me_s=d/3)" % gamma_s,
            "n_rows": int(predAgent.getX().shape[0]),
            "uses_target_labels": False,
        },
        "predictor_trained_on": {"agent": "trAgent", "distribution": "SOURCE", "n_rows": N},
        "note": "Both are built before any target data is generated, so no target label can reach them.",
    }

    t0 = time.time()
    per_repeat = []
    for rep in range(n_reps):
        seed_rep = 1 + rep
        setseed(seed_rep)
        _test = generate_agent(500, d, me_t, gamma_t, mu_t, dtype)
        calTrAgent = generate_agent(n, d, me_t, gamma_t, mu_t, dtype)
        calAgent = generate_agent(n, d, me_t, gamma_t, mu_t, dtype)
        semiAgent = generate_agent(m, d, me_t, gamma_t, mu_t, dtype)
        semiX = semiAgent.X

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
        tuner_list = slcp.tune_lbd_list(
            5, epoches, 5e-3, int(n_grid), lbds,
            temperature=temperature, tol_gap=0.001, max_iter=10000, m=200,
            targ_alpha=targ_alpha, penalty="MSE",
        )

        cal_scores = np.asarray(calAgent.getS()).reshape(-1)
        level = (1 - alpha) * (calAgent.n + 1) / calAgent.n
        q_conformal = float(np.quantile(cal_scores, level, method="higher"))
        order = np.sort(cal_scores)
        spacing = float(np.median(np.diff(order)))

        qs, deltas = [], []
        for i, lbd in enumerate(lbds):
            slcp.load_tuner(tuner_list[i], m=200, n=5, temperature=temperature, alpha=alpha)
            qs.append(float(slcp.q))
            deltas.append(float(tuner_list[i].get_delta_norm()))

        per_repeat.append({
            "repeat": rep,
            "n_calibration": int(calAgent.n),
            "m_unlabeled": int(semiX.shape[0]),
            "conformal_level_1_minus_alpha_n": level,
            "q_conformal_from_F0": q_conformal,
            "calibration_score_median_spacing": spacing,
            "q_stcp_by_lambda": qs,
            "theta_delta_sq_norm_by_lambda": deltas,
            "abs_gap_at_lambda0": abs(qs[0] - q_conformal),
            "gap_over_spacing_at_lambda0": abs(qs[0] - q_conformal) / spacing if spacing > 0 else None,
        })

    deltas = np.array([p["theta_delta_sq_norm_by_lambda"] for p in per_repeat])
    gaps = np.array([p["abs_gap_at_lambda0"] for p in per_repeat])
    ratios = np.array([p["gap_over_spacing_at_lambda0"] for p in per_repeat])
    # Spearman-free monotonicity: fraction of adjacent lambda steps that do not increase.
    non_increasing = float(np.mean(np.diff(deltas, axis=1) <= 1e-12))

    out = {
        "kind": "invariants",
        "dtype": dtype, "n": n, "m": m, "reps": n_reps,
        "lambda_grid": lbds,
        "seconds": round(time.time() - t0, 1),
        "provenance": provenance,
        "per_repeat": per_repeat,
        "lambda0_identity": {
            "mean_abs_gap": float(gaps.mean()),
            "max_abs_gap": float(gaps.max()),
            "mean_gap_over_score_spacing": float(ratios.mean()),
            "max_gap_over_score_spacing": float(ratios.max()),
        },
        "lambda_to_infinity": {
            "fraction_non_increasing_steps": non_increasing,
            "mean_delta_sq_at_min_lambda": float(deltas[:, 0].mean()),
            "mean_delta_sq_at_max_lambda": float(deltas[:, -1].mean()),
            "shrinkage_ratio": float(deltas[:, -1].mean() / max(deltas[:, 0].mean(), 1e-30)),
        },
    }

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "invariants.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out
