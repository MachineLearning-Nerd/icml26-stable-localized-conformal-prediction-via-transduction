"""Reproduce the paper's Table 2 simulation (LogAbs, m=500, n in {30,100,500}).

Runs the authors' own `SimuAnalysis/core.run_experiment` at the authors' own
`config.common_run_kwargs()` (50 repeats, d=5, gamma_s=1.2, gamma_t=1.0,
hidden (50,100,100,50), alpha_tol=0.02) and then summarises it with the authors'
own `SimuAnalysis/sum_tab` selection logic, so neither the experiment nor the
"ours" row is a paraphrase of theirs.

Table 2's percentage is `(base_std - value) / base_std * 100`
(`SimuAnalysis/sum_tab.py:93`) -- NOT the oracle-adjusted Table 1 formula.
"""

import json
import os
import sys
import time

import numpy as np

METHOD_KEYS = ["base", "SDCP", "PPI", "StCP", "StCP-sel", "oracle", "NOAL"]
METHOD_LABELS = ["base", "SDCP", "PPI", "ours", "ours-sel", "oracle", "DP"]
MODEL_LABELS = ["GLCP", "CQR"]
METRICS = ["marginal", "size", "std", "cond_miscoverage"]


def _summarise(res, sum_tab):
    """Per-model 7x4 table plus Table-2 improvement percentages."""
    out = {}
    n = int(res["meta"]["n"])
    for mi, model in enumerate(MODEL_LABELS):
        rows = sum_tab.collect_result_row(res, mi)
        marks = sum_tab.coverage_marks(rows[:, 0], tol=0.01, n=n)
        base_std = float(rows[0, 2])
        table = {}
        for ri, label in enumerate(METHOD_LABELS):
            entry = {m: float(rows[ri, k]) for k, m in enumerate(METRICS)}
            entry["coverage_mark"] = marks[ri] or ""
            if label in ("ours", "ours-sel"):
                entry["std_improvement_pct"] = (
                    0.0 if abs(base_std) < 1e-12 else (base_std - entry["std"]) / base_std * 100.0
                )
            table[label] = entry
        out[model] = table
    return out


def run(cfg, upstream_root):
    sys.path.insert(0, os.path.join(upstream_root, "SimuAnalysis"))
    import config
    import sum_tab
    from core import run_experiment

    dtype = cfg["dtype"]
    n = int(cfg["n"])
    m = int(cfg["m"])

    kwargs = config.common_run_kwargs()
    if "repeats" in cfg:
        kwargs["repeats"] = int(cfg["repeats"])

    t0 = time.time()
    res = run_experiment(dtype=dtype, n=n, m=m, **kwargs)
    seconds = time.time() - t0

    summary = _summarise(res, sum_tab)

    raw = {
        "meta": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res["meta"].items()},
        "per_method": {
            key: [np.asarray(a).tolist() for a in res[key]] for key in METHOD_KEYS if key in res
        },
        "selected_lambda": np.asarray(res.get("selected_lambda", [])).tolist(),
        "selected_lambda_idx": np.asarray(res.get("selected_lambda_idx", [])).tolist(),
    }
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    stem = f"sim_{dtype}_n{n}_m{m}"
    with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
        json.dump({"summary": summary, "raw": raw}, f, indent=1)

    return {
        "kind": "simulation",
        "dtype": dtype,
        "n": n,
        "m": m,
        "repeats": kwargs["repeats"],
        "lambda_grid": kwargs["lbds"],
        "seconds": round(seconds, 1),
        "summary": summary,
        "raw": raw,
        "improvement_formula": "(base_std - value) / base_std * 100  [SimuAnalysis/sum_tab.py:93]",
    }
