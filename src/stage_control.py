"""Claim 6 negative control: break the one assumption Theorem 4.7 makes.

Theorem 4.7 guarantees `P(Y in C_St-sel) in [1-alpha-alpha_tol,
1-alpha+alpha_tol+(n+1)^-1)` = `[0.88, 0.9522581)` at alpha=0.1,
alpha_tol=0.02, n=30, assuming only that `S_1..S_{n+1}` are exchangeable.

That band is 7.2 points wide, so "observed coverage landed inside it" is weak
evidence on its own: a check that passes for every possible implementation is
not a check. This stage runs a ladder of arms that differ from the reference in
exactly one thing -- the distribution the calibration scores are drawn from:

  exchangeable    calibration drawn from the TARGET distribution (the paper's
                  setting). Must land INSIDE the band.
  non_exchangeable
                  calibration drawn from the SOURCE distribution (different
                  mean, different regression coefficient, gamma_s=1.2 noise)
                  while the test point stays target.
  strong_violation
                  target covariates, but the calibration noise scale shrunk to
                  gamma=0.2 against a test gamma of 1.0, so calibration scores
                  are systematically smaller than test scores.

Everything else -- estimator, lambda grid, selection rule, seeds -- is identical
across arms.

Measured outcome: `non_exchangeable` does NOT leave the band (0.9308 against an
upper edge of 0.9523). The ladder therefore continues past the paper's own
source/target gap, so that "no violation leaves the band" can be told apart from
"the violation tried was too mild". If NO arm leaves the band, the Theorem 4.7
check has no resolution at these parameters and must be reported as vacuous
rather than as corroboration.
"""

import json
import os
import sys
import time

import numpy as np

BAND_LO = 0.88
BAND_HI = 0.9 + 0.02 + 1.0 / 31.0  # 0.9522580645...


def _cal_draw(name, shared, k):
    """(me, gamma, mu) for this arm's calibration sample.

    A ladder, not a switch. The paper's own source/target gap turns out not to be
    enough to leave a 7.2-point band, so the ladder continues past it: `strong`
    keeps the covariate law but shrinks the calibration noise scale, which makes
    calibration scores systematically smaller than test scores and should
    undercover. If even that stays inside the band, the band has no resolution at
    these parameters and no in-band observation is evidence for anything.
    """
    mu_t, mu_s, me_t, me_s = shared["mu_t"], shared["mu_s"], shared["me_t"], shared["me_s"]
    gamma_t, gamma_s = k["gamma_t"], k["gamma_s"]
    if name == "exchangeable":
        return me_t, gamma_t, mu_t, "target"
    if name == "non_exchangeable":
        return me_s, gamma_s, mu_s, f"source (mu_s, gamma_s={gamma_s}, me_s=d/3)"
    if name == "strong_violation":
        g = float(shared.get("gamma_strong", 0.2))
        return me_t, g, mu_t, (f"target covariates with calibration noise scale gamma={g} "
                               f"against test gamma={gamma_t}")
    raise ValueError(f"unknown arm {name}")


def _arm(name, exchangeable, cfg, upstream_root, shared):
    from copy import deepcopy

    from core import _eval_cs, generate_agent
    from SLCP import SLCP
    from Predictor import Predictor
    from tools import defaultScore, defaultSolveScore, setseed

    k, dtype, n, m, testN = shared["k"], shared["dtype"], shared["n"], shared["m"], shared["testN"]
    d, alpha = k["d"], k["alpha"]
    mu_t, mu_s, me_t, me_s = shared["mu_t"], shared["mu_s"], shared["me_t"], shared["me_s"]
    gamma_t, gamma_s = k["gamma_t"], k["gamma_s"]
    generator_base = shared["generator_base"]
    lbds, temperature, n_grid, epoches = shared["lbds"], k["temperature"], k["n_grid"], k["epoches"]

    cov_per_repeat, selected = [], []
    _t0 = time.time()
    for rep in range(shared["reps"]):
        # Two arms x reps repeats with no other output makes a slow node
        # indistinguishable from a hung one.
        print(f"[control {name}] repeat {rep}/{shared['reps']} start "
              f"t+{time.time() - _t0:.1f}s", flush=True)
        seed_rep = 1 + rep
        setseed(seed_rep)
        testAgent = generate_agent(testN, d, me_t, gamma_t, mu_t, dtype)
        calTrAgent = generate_agent(n, d, me_t, gamma_t, mu_t, dtype)
        cal_me, cal_gamma, cal_mu, _ = _cal_draw(name, shared, k)
        calAgent = generate_agent(n, d, cal_me, cal_gamma, cal_mu, dtype)
        semiAgent = generate_agent(m, d, me_t, gamma_t, mu_t, dtype)

        setseed(seed_rep)
        predictor = Predictor("lr", fit_intercept=False)
        predictor.trainFromAgent(calTrAgent)
        calAgent.calScore(predictor, defaultScore)
        calTrAgent.calScore(predictor, defaultScore)

        setseed(seed_rep)
        generator = deepcopy(generator_base)
        generator.cal_scalar(calTrAgent.getX(), calTrAgent.getS(), 200, stat_type="CvM")

        setseed(seed_rep)
        slcp = SLCP(calAgent, semiAgent.X, deepcopy(generator), predictor, [2])
        targ_alpha = float(np.clip(1 - (1 - alpha) * (calAgent.n + 1) / calAgent.n, 1e-6, 1 - 1e-6))
        _, _, idx = slcp.auto_lbd_tune(
            5, epoches, 5e-3, alpha, int(n_grid), lbds,
            temperature=temperature, gap=k["alpha_tol"], tol_gap=0.001, max_iter=10000, m=200,
            targ_alpha=targ_alpha, penalty="MSE",
        )
        out = slcp.predict(testAgent.getX(), defaultSolveScore)
        cs, isinf = (out[0], out[1]) if isinstance(out, tuple) else (out, None)
        cov, _size = _eval_cs(cs, testAgent.getX(), me_t, gamma_t, dtype, isinf=isinf)
        cov_per_repeat.append(float(np.mean(cov)))
        selected.append(int(idx))

    cov = np.array(cov_per_repeat)
    mean = float(cov.mean())
    se = float(cov.std(ddof=1) / np.sqrt(len(cov))) if len(cov) > 1 else float("nan")
    return {
        "arm": name,
        "exchangeable": exchangeable,
        "calibration_distribution": _cal_draw(name, shared, shared["k"])[3],
        "repeats": len(cov),
        "coverage_per_repeat": cov_per_repeat,
        "coverage_mean": mean,
        "coverage_stderr": se,
        "coverage_ci95": [mean - 1.96 * se, mean + 1.96 * se],
        "selected_lambda_idx": selected,
        "band": [BAND_LO, BAND_HI],
        "inside_band": bool(BAND_LO <= mean < BAND_HI),
    }


def run(cfg, upstream_root):
    sys.path.insert(0, os.path.join(upstream_root, "SimuAnalysis"))
    sys.path.insert(0, os.path.join(upstream_root, "Main"))

    import config
    from core import generate_agent
    from engGenerator import Generator
    from Predictor import Predictor
    from tools import defaultScore, setseed

    k = config.common_run_kwargs()
    d, r, N = k["d"], k["r"], k["N"]
    dtype = cfg.get("dtype", "logabs")
    n, m = int(cfg.get("n", 30)), int(cfg.get("m", 500))
    testN, reps = int(cfg.get("testN", 500)), int(cfg.get("reps", 20))

    mu_t, mu_s = np.ones(d) / np.sqrt(d) * r, np.zeros(d)
    me_t, me_s = d / 2, d / 3

    setseed(k["repeats"] + 100)
    trAgent = generate_agent(N, d, me_s, k["gamma_s"], mu_s, dtype)
    predAgent = generate_agent(N, d, me_s, k["gamma_s"], mu_s, dtype)
    pred = Predictor("lr", fit_intercept=False)
    pred.trainFromAgent(trAgent)
    predAgent.calScore(pred, defaultScore)
    generator_base = Generator(d, k["hidden_dim"], d)
    generator_base.trainEng(predAgent.getX(), predAgent.getS(), 10, 32, k["epoches"], 5e-3, mute=True)

    shared = {
        "k": k, "dtype": dtype, "n": n, "m": m, "testN": testN, "reps": reps,
        "mu_t": mu_t, "mu_s": mu_s, "me_t": me_t, "me_s": me_s,
        "generator_base": generator_base,
        "lbds": sorted(set(k["lbds"])), "n_grid": k["n_grid"],
        "gamma_strong": float(cfg.get("gamma_strong", 0.2)),
    }

    t0 = time.time()
    arm_names = cfg.get("arms") or ["exchangeable", "non_exchangeable", "strong_violation"]
    arms = [_arm(nm, nm == "exchangeable", cfg, upstream_root, shared) for nm in arm_names]

    by_name = {a["arm"]: a for a in arms}
    # Informative if the reference arm sits inside the band and SOME violation
    # leaves it. Which one matters is reported rather than hidden: the paper's own
    # source/target gap and a larger one are graded separately.
    left_band = [a["arm"] for a in arms if a["arm"] != "exchangeable" and not a["inside_band"]]
    control_is_informative = bool(
        by_name.get("exchangeable", {}).get("inside_band") and left_band)
    out = {
        "kind": "control_exchangeability",
        "dtype": dtype, "n": n, "m": m, "reps": reps,
        "band": {"lo": BAND_LO, "hi": BAND_HI, "formula": "[1-a-a_tol, 1-a+a_tol+1/(n+1)) at a=0.1, a_tol=0.02, n=30"},
        "seconds": round(time.time() - t0, 1),
        "arms": arms,
        "control_is_informative": control_is_informative,
        "violations_that_left_the_band": left_band,
        "paper_shift_leaves_band": bool(
            "non_exchangeable" in by_name and not by_name["non_exchangeable"]["inside_band"]),
        "interpretation": (
            f"informative: the band is exited by {left_band}, so an in-band observation "
            f"is not automatic"
            if control_is_informative
            else "NOT informative: no exchangeability violation tried here leaves the band, "
                 "so an in-band observation is weak evidence at these parameters"
        ),
    }
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    name = cfg.get("out_name", "control_exchangeability")
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out
