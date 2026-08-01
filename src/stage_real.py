"""Reproduce one Table 1 real-data column using the authors' own entry script.

Each of the five datasets has an entry script in the artifact's `RealAnalysis/`.
Four of the five reproduce Table 1 at their *committed defaults*; STAR needs
`60 10 1 1` to select the rural/early-age target agent used for the published
run (the authors' own result file is `P_60_1048_2446_10_1_1.pkl`).

For the three regression datasets the CLI `n` is the size of the target pool,
which `procedure.py:308` halves into `calTrAgent`/`calAgent`; the calibration
size reported in Table 1 is therefore `n // 2 = 30`. The two classification
datasets pass `n = 30` straight through.

Table 1's percentage is the oracle-adjusted
`(a_ref - a_1) / (a_ref - a_0) * 100` with `a_ref` the smallest Std among
base/SDCP/PPI whose marginal coverage is in range (`RealAnalysis/sum_tab.py:61`,
Appendix C.1) -- NOT the plain ratio used by Table 2.
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

METHOD_LABELS = ["base", "SDCP", "PPI", "ours", "ours-sel", "oracle", "DP"]
METRICS = ["marginal", "size", "std", "cond_miscoverage"]
MODEL_KEYS = {"GLCP": "GLCP", "CQR": "SCC"}


def _marks(mar, n_reported, tol=0.01):
    """`-` below 1-alpha-0.01, `+` above 1-alpha+1/(n+1); see Table 1 caption."""
    out = []
    for v in mar:
        if v < 0.9 - tol:
            out.append("-")
        elif v > 0.9 + 1.0 / (n_reported + 1):
            out.append("+")
        else:
            out.append("")
    return out


def _summarise(res_dict, sum_tab, n_reported):
    out = {}
    for model, key in MODEL_KEYS.items():
        rows = sum_tab.sum_compare_result(res_dict, key, tol=0.02, n=n_reported)
        marks = _marks(rows[:, 0], n_reported)
        eligible = [i for i in (0, 1, 2) if not marks[i]] or [0, 1, 2]
        a_ref = float(np.min(rows[eligible, 2]))
        a_0 = float(rows[5, 2])
        table = {}
        for ri, label in enumerate(METHOD_LABELS):
            entry = {m: float(rows[ri, k]) for k, m in enumerate(METRICS)}
            entry["coverage_mark"] = marks[ri]
            if label in ("ours", "ours-sel"):
                denom = a_ref - a_0
                entry["std_improvement_pct"] = (
                    float("nan") if abs(denom) < 1e-12 else (a_ref - entry["std"]) / denom * 100.0
                )
            table[label] = entry
        table["_reference"] = {
            "a_ref_std": a_ref,
            "a_ref_method": METHOD_LABELS[int(eligible[int(np.argmin(rows[eligible, 2]))])],
            "oracle_std": a_0,
            "eligible_baselines": [METHOD_LABELS[i] for i in eligible],
        }
        out[model] = table
    return out


def _patch_repeats(upstream_root, script, repeats):
    src_path = os.path.join(upstream_root, "RealAnalysis", script)
    with open(src_path) as f:
        lines = f.readlines()
    hits = []
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("epoches, repeats, alpha") and "=" in s:
            lhs, rhs = s.split("=", 1)
            vals = [v.strip() for v in rhs.split(",")]
            vals[1] = str(repeats)
            indent = line[: len(line) - len(line.lstrip())]
            lines[i] = f"{indent}{lhs.strip()} = {', '.join(vals)}\n"
            hits.append(i)
    if len(hits) != 1:
        raise RuntimeError(f"expected 1 repeats assignment in {script}, found {len(hits)}")
    out_name = f"_r{repeats}_{script}"
    with open(os.path.join(upstream_root, "RealAnalysis", out_name), "w") as f:
        f.writelines(lines)
    return out_name


def run(cfg, upstream_root):
    sys.path.insert(0, os.path.join(upstream_root, "RealAnalysis"))
    sys.path.insert(0, os.path.join(upstream_root, "Main"))

    script = cfg["script"]
    argv = [str(a) for a in cfg.get("argv", [])]
    repeats = cfg.get("repeats")
    entry = _patch_repeats(upstream_root, script, repeats) if repeats else script

    cmd = [sys.executable, "-u", os.path.join("RealAnalysis", entry)] + argv
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=upstream_root, capture_output=True, text=True)
    seconds = time.time() - t0
    if proc.returncode != 0:
        return {
            "kind": "real",
            "dataset": cfg["dataset"],
            "status": "FAILED",
            "returncode": proc.returncode,
            "seconds": round(seconds, 1),
            "command": " ".join(cmd),
            "stdout_tail": proc.stdout[-6000:],
            "stderr_tail": proc.stderr[-6000:],
        }

    pkl = os.path.join(upstream_root, "SimResult", cfg["result_dir"], cfg["result_name"] + ".pkl")
    if not os.path.exists(pkl):
        found = []
        base = os.path.join(upstream_root, "SimResult")
        for root, _, fs in os.walk(base):
            for f in fs:
                found.append(os.path.relpath(os.path.join(root, f), base))
        return {
            "kind": "real",
            "dataset": cfg["dataset"],
            "status": "MISSING_RESULT",
            "expected": pkl,
            "found": found,
            "seconds": round(seconds, 1),
            "stdout_tail": proc.stdout[-6000:],
        }

    import pickle

    import sum_tab

    with open(pkl, "rb") as f:
        res_dict = pickle.load(f)

    n_reported = int(cfg["n_reported"])
    summary = _summarise(res_dict, sum_tab, n_reported)

    raw = {
        k: [np.asarray(a).tolist() for a in v] for k, v in res_dict.items()
    }
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"real_{cfg['dataset']}.json"), "w") as f:
        json.dump({"summary": summary, "raw": raw}, f, indent=1)

    return {
        "kind": "real",
        "status": "OK",
        "dataset": cfg["dataset"],
        "command": " ".join(cmd),
        "result_pickle": os.path.relpath(pkl, upstream_root),
        "n_reported": n_reported,
        "n_over_m": cfg.get("n_over_m"),
        "repeats": repeats or 50,
        "seconds": round(seconds, 1),
        "summary": summary,
        "raw": raw,
        "improvement_formula": "(a_ref - a_1) / (a_ref - a_0) * 100  [RealAnalysis/sum_tab.py:61, Appendix C.1]",
    }
