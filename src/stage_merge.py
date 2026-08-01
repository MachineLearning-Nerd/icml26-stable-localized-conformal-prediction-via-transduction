"""Merge committed simulation shards into the paper's Table 2 rows.

Reads `results/shards/*.json` (the sufficient statistics each shard printed and
which are committed as raw evidence), reconstructs exactly the arrays that
`SimuAnalysis/core._aggregate` would have produced from a single 50-repeat run,
and then applies the authors' own `SimuAnalysis/sum_tab` selection to pick the
"ours" row.
"""

import glob
import json
import os
import sys

import numpy as np

import shard_reduce

METHOD_KEYS = ["base", "SDCP", "PPI", "StCP", "StCP-sel", "oracle", "NOAL"]
METHOD_LABELS = ["base", "SDCP", "PPI", "ours", "ours-sel", "oracle", "DP"]
MODEL_LABELS = ["GLCP", "CQR"]
METRICS = ["marginal", "size", "std", "cond_miscoverage"]


def _load(setting_dir):
    parts = {}
    for path in sorted(glob.glob(os.path.join(setting_dir, "*.json"))):
        with open(path) as f:
            payload = json.load(f)
        for key, val in payload.items():
            if key == "selected_idx":
                continue
            parts.setdefault(key, []).append(val)
    return parts


def _rebuild(parts, alpha):
    res = {}
    for key, shards in parts.items():
        s = shard_reduce.summarise(shards, alpha=alpha)
        res[key] = [
            np.asarray(s["marginal"]),
            np.asarray(s["size"]),
            np.asarray(s["std"]),
            np.asarray(s["cond_miscoverage"]),
        ]
        res.setdefault("_n_repeats", s["n_repeats"])
    return res


def run(cfg, upstream_root):
    sys.path.insert(0, os.path.join(upstream_root, "SimuAnalysis"))
    import sum_tab

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    alpha = float(cfg.get("alpha", 0.1))
    out = {"kind": "sim_merge", "settings": {}}

    for setting in cfg["settings"]:
        setting_dir = os.path.join(root, "results", "shards", setting["dir"])
        parts = _load(setting_dir)
        if not parts:
            out["settings"][setting["dir"]] = {"status": "NO_SHARDS", "dir": setting_dir}
            continue
        res = _rebuild(parts, alpha)
        n_repeats = res.pop("_n_repeats")
        res["meta"] = {"alpha": alpha, "n": int(setting["n"]), "m": int(setting["m"])}

        per_model = {}
        for mi, model in enumerate(MODEL_LABELS):
            rows = sum_tab.collect_result_row(res, mi)
            marks = sum_tab.coverage_marks(rows[:, 0], tol=0.01, n=int(setting["n"]))
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
            per_model[model] = table

        out["settings"][setting["dir"]] = {
            "status": "OK",
            "n": setting["n"],
            "m": setting["m"],
            "n_repeats_merged": n_repeats,
            "shards_found": {k: len(v) for k, v in parts.items()},
            "table": per_model,
            "stcp_lambda_grid_curves": {
                MODEL_LABELS[mi]: {
                    "marginal": np.asarray(res["StCP"][0])[mi].tolist(),
                    "size": np.asarray(res["StCP"][1])[mi].tolist(),
                    "std": np.asarray(res["StCP"][2])[mi].tolist(),
                    "cond_miscoverage": np.asarray(res["StCP"][3])[mi].tolist(),
                }
                for mi in range(2)
            },
        }

    out["improvement_formula"] = "(base_std - value) / base_std * 100  [SimuAnalysis/sum_tab.py:93]"
    return out
