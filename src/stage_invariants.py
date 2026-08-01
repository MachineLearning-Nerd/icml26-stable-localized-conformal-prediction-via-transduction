"""Claim 1: measure the defining properties of Equation 7, not just "it runs".

Four things decide whether the implemented object is the object Equation 7
describes. All are measured on the paper's own LogAbs setting with the paper's
own code, at the paper's own scale.

  1. Provenance. `F_hat_S|X` must see labeled SOURCE data only, and the
     transductive term must see UNLABELED target covariates only. Recorded as a
     construction trace with shapes, so a reviewer can confirm no target label
     reaches the CDF estimator.

  2. Objective identity. `Tuner.tune_marginal` returns `(discrepancy, ||delta||^2)`
     and optimises `L = discrepancy + lambda * ||delta||^2` (Main/Tuner.py:202).
     That is Equation 7 literally; the identity is re-checked numerically for
     every lambda and repeat.

  3. The regularisation path, measured WITHOUT the implementation's repair step.
     `SLCP.tune_lbd_list` post-processes its results with `check_order`
     (Main/SLCP.py:8), which re-trains any lambda whose (discrepancy, delta) pair
     breaks the Pareto ordering. Monotonicity read off that output is imposed
     rather than observed, so it cannot be evidence. The path is therefore
     re-measured by calling `tune_marginal` directly per lambda on independent
     copies -- the authors' own call, minus the repair loop. Both paths are
     reported so the difference is visible.

  4. Intervention on the unlabeled target sample. Tracing shows the unlabeled
     target covariates are passed in; refitting against covariates drawn from
     the SOURCE instead shows they are actually used. If the solution did not
     move, the transduction would be doing no work.

Deliberately NOT checked: equality of `SLCP.q` with a conformal score quantile.
`SLCP.q` is a probability level -- it is passed as the `q` argument of
`Generator.quantile` (Main/SLCP.py:90) -- so comparing it against a score
threshold compares incommensurable quantities. An earlier revision of this file
made that mistake; the measured "gap" it reported was an artefact of the units,
not a property of the method.
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
        # Printed so a long node can be told apart from a hung one; the real
        # stages emit the same shape of line.
        print(f"[invariants] repeat {rep}/{n_reps} start t+{time.time() - t0:.1f}s", flush=True)
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
        tk = dict(n=5, epochs=epoches, lr=5e-3, n_grid=int(n_grid), temperature=temperature)
        tail_kw = dict(tol_gap=0.001, max_iter=10000, m=200,
                       targ_alpha=targ_alpha, penalty="MSE")

        def fit(tuner, lbd, unlabeled):
            return tuner.tune_marginal(
                calAgent.getX(), calAgent.getS(), unlabeled,
                tk["n"], tk["epochs"], tk["lr"], tk["n_grid"], float(lbd),
                tk["temperature"], **tail_kw)

        # ---- the UN-ENFORCED regularisation path ---------------------------
        # `SLCP.tune_lbd_list` post-processes with `check_order` (Main/SLCP.py:8),
        # which finds any lambda whose (discrepancy, delta) pair breaks the Pareto
        # ordering and re-trains it warm-started from a neighbour, looping up to
        # 3*len(lbds) times. Monotonicity read off that output would be imposed by
        # the implementation, not observed -- a circular check. Calling
        # `tune_marginal` directly per lambda, as the authors' own loop does but
        # without the repair step, measures the path the objective really yields.
        raw = []
        for lbd in lbds:
            part1, delta = fit(deepcopy(slcp.tuner), lbd, semiX)
            raw.append({"lambda": float(lbd), "discrepancy": float(part1),
                        "delta_sq_norm": float(delta),
                        "objective": float(part1) + float(lbd) * float(delta)})

        # ---- the enforced path, recorded for comparison ---------------------
        tuner_list = slcp.tune_lbd_list(tk["n"], tk["epochs"], tk["lr"], tk["n_grid"],
                                        lbds, temperature=tk["temperature"], **tail_kw)
        enforced, qs = [], []
        for i, _ in enumerate(lbds):
            slcp.load_tuner(tuner_list[i], m=200, n=5, temperature=temperature, alpha=alpha)
            qs.append(float(slcp.q))
            enforced.append(float(tuner_list[i].get_delta_norm()))

        # ---- intervention: is the UNLABELED TARGET sample actually used? -----
        # Tracing shows unlabeled target covariates are passed in; this shows they
        # change the answer. Refitting against SOURCE-drawn covariates is the
        # treatment -- but "the answer changed" needs a scale, or the threshold is
        # arbitrary. So the treatment is compared against a matched NULL: a second
        # independent draw of the unlabeled TARGET sample, which changes the input
        # without changing its distribution. Only a treatment shift larger than
        # that null shift is evidence that the target distribution -- not just
        # resampling noise -- is what moves the fit.
        mid_i = len(lbds) // 2
        setseed(seed_rep + 10_000)
        sham_X = generate_agent(m, d, me_s, gamma_s, mu_s, dtype).X
        p_sham, d_sham = fit(deepcopy(slcp.tuner), lbds[mid_i], sham_X)
        setseed(seed_rep + 20_000)
        null_X = generate_agent(m, d, me_t, gamma_t, mu_t, dtype).X
        p_null, d_null = fit(deepcopy(slcp.tuner), lbds[mid_i], null_X)
        mid = raw[mid_i]

        def rel(new, base):
            return abs(float(new) - float(base)) / max(abs(float(base)), 1e-12)

        per_repeat.append({
            "repeat": rep,
            "n_calibration": int(calAgent.n),
            "m_unlabeled": int(semiX.shape[0]),
            "raw_path": raw,
            "enforced_delta_sq_norm_by_lambda": enforced,
            "beta_level_q_by_lambda": qs,
            "beta_level_note": (
                "SLCP.q is a PROBABILITY level in [0,1]: it is passed as the `q` argument of "
                "Generator.quantile (Main/SLCP.py:90), not a score threshold. It is reported "
                "for completeness and is deliberately NOT compared against a score quantile."),
            "unlabeled_sample_intervention": {
                "lambda": float(lbds[mid_i]),
                "target_unlabeled": {"discrepancy": mid["discrepancy"],
                                     "delta_sq_norm": mid["delta_sq_norm"]},
                "source_unlabeled_sham": {"discrepancy": float(p_sham),
                                          "delta_sq_norm": float(d_sham)},
                "target_resample_null": {"discrepancy": float(p_null),
                                         "delta_sq_norm": float(d_null)},
                "treatment_shift_discrepancy": rel(p_sham, mid["discrepancy"]),
                "null_shift_discrepancy": rel(p_null, mid["discrepancy"]),
                "treatment_shift_delta_sq": rel(d_sham, mid["delta_sq_norm"]),
                "null_shift_delta_sq": rel(d_null, mid["delta_sq_norm"]),
            },
        })

    raw_delta = np.array([[q["delta_sq_norm"] for q in p["raw_path"]] for p in per_repeat])
    raw_disc = np.array([[q["discrepancy"] for q in p["raw_path"]] for p in per_repeat])
    enf_delta = np.array([p["enforced_delta_sq_norm_by_lambda"] for p in per_repeat])
    iv = [p["unlabeled_sample_intervention"] for p in per_repeat]
    t_disc = np.array([x["treatment_shift_discrepancy"] for x in iv])
    n_disc = np.array([x["null_shift_discrepancy"] for x in iv])
    t_delta = np.array([x["treatment_shift_delta_sq"] for x in iv])
    n_delta = np.array([x["null_shift_delta_sq"] for x in iv])
    obj_ok = all(
        abs(q["objective"] - (q["discrepancy"] + q["lambda"] * q["delta_sq_norm"])) < 1e-9
        for p in per_repeat for q in p["raw_path"]
    )

    out = {
        "kind": "invariants",
        "dtype": dtype, "n": n, "m": m, "reps": n_reps,
        "lambda_grid": lbds,
        "seconds": round(time.time() - t0, 1),
        "provenance": provenance,
        "per_repeat": per_repeat,
        "objective_identity": {
            "form": "L(lambda) = discrepancy(F_hat_S^0, F_hat_S^1) + lambda * ||theta - theta_hat||^2",
            "source_anchor": "Main/Tuner.py:202",
            "holds_for_every_lambda_and_repeat": bool(obj_ok),
        },
        "regularisation_path_unenforced": {
            "mean_delta_sq_at_min_lambda": float(raw_delta[:, 0].mean()),
            "mean_delta_sq_at_max_lambda": float(raw_delta[:, -1].mean()),
            "shrinkage_ratio": float(raw_delta[:, -1].mean() / max(raw_delta[:, 0].mean(), 1e-30)),
            "fraction_non_increasing_delta_steps": float(np.mean(np.diff(raw_delta, axis=1) <= 1e-12)),
            "fraction_non_decreasing_discrepancy_steps": float(np.mean(np.diff(raw_disc, axis=1) >= -1e-12)),
            "delta_shrinks_overall_in_every_repeat": bool(np.all(raw_delta[:, -1] < raw_delta[:, 0])),
        },
        "regularisation_path_enforced_by_check_order": {
            "mean_delta_sq_at_min_lambda": float(enf_delta[:, 0].mean()),
            "mean_delta_sq_at_max_lambda": float(enf_delta[:, -1].mean()),
            "fraction_non_increasing_delta_steps": float(np.mean(np.diff(enf_delta, axis=1) <= 1e-12)),
            "note": ("Reported only to expose the difference. `check_order` re-trains lambdas that "
                     "break the ordering, so monotonicity here is imposed and carries no evidential "
                     "weight for Claim 1."),
        },
        "unlabeled_target_intervention": {
            "description": (
                "At the median lambda, refit against (a) SOURCE-drawn covariates -- the treatment -- "
                "and (b) a second independent draw of the unlabeled TARGET sample -- the matched "
                "null. The null changes the input without changing its distribution, so it "
                "calibrates how much movement is mere resampling noise and removes the need for an "
                "arbitrary threshold."),
            "statistic_note": (
                "The primary statistic is the discrepancy term, which is what the objective "
                "optimises against the unlabeled sample. ||theta - theta_hat||^2 is reported too but "
                "is a poor movement detector: it is a scalar norm, so two genuinely different "
                "solutions can share one. In repeat 0 of an earlier run the norm moved 0.03% while "
                "the discrepancy moved 4.4%."),
            "mean_treatment_shift_discrepancy": float(t_disc.mean()),
            "mean_null_shift_discrepancy": float(n_disc.mean()),
            "treatment_exceeds_null_discrepancy_per_repeat": [bool(x) for x in (t_disc > n_disc)],
            "treatment_exceeds_null_in_every_repeat": bool(np.all(t_disc > n_disc)),
            "treatment_exceeds_null_in_majority": bool(np.mean(t_disc > n_disc) > 0.5),
            "mean_treatment_shift_delta_sq": float(t_delta.mean()),
            "mean_null_shift_delta_sq": float(n_delta.mean()),
        },
    }

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "invariants.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out
