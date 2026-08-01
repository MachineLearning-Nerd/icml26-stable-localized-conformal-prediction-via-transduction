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

import collections
import json
import os
import subprocess
import sys
import time
import types

import numpy as np

METHOD_LABELS = ["base", "SDCP", "PPI", "ours", "ours-sel", "oracle", "DP"]
METRICS = ["marginal", "size", "std", "cond_miscoverage"]
MODEL_KEYS = {"GLCP": "GLCP", "CQR": "SCC"}


def _marks(mar, n_reported, tol=0.01):
    """The authors' real-data thresholds, verbatim from RealAnalysis/sum_tab.py:34.

    Note these are NOT the Table 1 caption's `[1-a-0.01, 1-a+1/(n+1)]`. The code
    uses `0.901 + 1/n`, i.e. [0.89, 0.93433] at n=30 rather than [0.89, 0.93226].
    The difference is load-bearing: DERMA/GLCP `ours` has published marginal
    0.933, which the caption's formula would flag `+` (and so exclude from the
    reference baseline) but the code does not -- and the paper prints it
    unmarked. Matching the code is what reproduces the published table.
    """
    out = []
    for v in mar:
        if v < 0.9 - tol:
            out.append("-")
        elif v > 0.901 + 1.0 / n_reported:
            out.append("+")
        else:
            out.append("")
    return out


def load_sum_tab(upstream_root):
    """Import `RealAnalysis/sum_tab.py`'s functions without running its script tail.

    The file defines the summarisation helpers and then, at module level, opens
    five result pickles by relative path and writes `sum_tab.txt`. A plain import
    therefore fails outside the authors' directory layout. Only the definitions
    are executed here -- the source is truncated at the first top-level statement
    that is neither an import nor a def, so the logic used is byte-identical to
    the authors' and nothing is paraphrased.
    """
    import ast

    path = os.path.join(upstream_root, "RealAnalysis", "sum_tab.py")
    with open(path) as fh:
        src = fh.read()
    tree = ast.parse(src)
    keep = [n for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    dropped = len(tree.body) - len(keep)
    module = types.ModuleType("stcp_sum_tab")
    module.__file__ = path
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), module.__dict__)
    if not hasattr(module, "sum_compare_result"):
        raise RuntimeError("sum_tab.py did not define sum_compare_result")
    module._dropped_toplevel_statements = dropped
    return module


def _is_cls(res_dict, key):
    """Classification datasets store a scalar local coverage, regression an array.

    `sum_compare_result` indexes `values[4][0]` unless told `cls=True`, so DERMA
    and TISSUE crash on the regression path. Detecting the shape is safer than a
    dataset whitelist: it follows whichever `summation_real*` actually produced
    the pickle.
    """
    return np.ndim(res_dict[key][4]) == 0


def _summarise(res_dict, sum_tab, n_reported):
    out = {}
    for model, key in MODEL_KEYS.items():
        cls = _is_cls(res_dict, key)
        # tol=0.01 and n=30 are the authors' own call arguments for Table 1
        # (`RealAnalysis/sum_tab.py:105`), not the function's defaults; tol sets
        # both the baseline-eligibility window and the lambda mask, so the
        # published table is not reproduced at tol=0.02.
        rows = sum_tab.sum_compare_result(res_dict, key, 0.01, n_reported, cls)
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
        table["_cls"] = bool(cls)
        table["_reference"] = {
            "a_ref_std": a_ref,
            "a_ref_method": METHOD_LABELS[int(eligible[int(np.argmin(rows[eligible, 2]))])],
            "oracle_std": a_0,
            "eligible_baselines": [METHOD_LABELS[i] for i in eligible],
        }
        out[model] = table
    return out


def _run_streaming(cmd, cwd, env, tail_lines=400):
    """Run the entry script, echoing its output to this job's log as it arrives.

    Capturing the output instead would make the run opaque from outside: the
    authors' `procedure.py` logs to a file inside the container, so the injected
    per-repeat progress line is the only external signal that a job is alive --
    and if the job is killed at the timeout, a captured buffer is lost entirely
    while streamed lines are already in the log.
    """
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, text=True, bufsize=1,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    tail = collections.deque(maxlen=tail_lines)
    for line in proc.stdout:
        tail.append(line)
        print("    | " + line.rstrip(), flush=True)
    proc.wait()
    return types.SimpleNamespace(
        returncode=proc.returncode, stdout="".join(tail), stderr=""
    )


def _apply_source_patches(upstream_root, script, patches):
    """Exact-string substitutions on an entry script, each one recorded.

    Used only where the shipped artifact and the shipped data disagree: the
    STAR script reads `Dataset/achievementRatio/STAR_Students.sav`, which the
    artifact does not contain, while `Dataset/achieve.csv` is a labelled CSV
    export of exactly that file (11601 x 379, identical string categories, 3754
    rows with a non-null `hsacttot`, which is what the authors' own result-file
    name `P_60_1048_2446_10_1_1` implies: 1308 target + 2446 auxiliary).
    """
    src_path = os.path.join(upstream_root, "RealAnalysis", script)
    with open(src_path) as f:
        src = f.read()
    applied = []
    for old, new in patches:
        count = src.count(old)
        if count != 1:
            raise RuntimeError(f"source patch matched {count} times in {script}: {old!r}")
        src = src.replace(old, new)
        applied.append({"old": old, "new": new})
    out_name = f"_patched_{script}"
    with open(os.path.join(upstream_root, "RealAnalysis", out_name), "w") as f:
        f.write(src)
    return out_name, applied


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

    import patch_procedure

    procedure_patch = patch_procedure.build(upstream_root)

    script = cfg["script"]
    argv = [str(a) for a in cfg.get("argv", [])]
    repeats = cfg.get("repeats")

    source_patches = []
    if cfg.get("source_patches"):
        script, source_patches = _apply_source_patches(
            upstream_root, script, [tuple(p) for p in cfg["source_patches"]]
        )
    entry = _patch_repeats(upstream_root, script, repeats) if repeats else script

    shard = cfg.get("shard")
    env = dict(os.environ)
    if shard:
        env["STCP_SHARD_LO"], env["STCP_SHARD_HI"] = str(shard[0]), str(shard[1])

    cmd = [sys.executable, "-u", os.path.join("RealAnalysis", entry)] + argv
    t0 = time.time()
    proc = _run_streaming(cmd, upstream_root, env)
    seconds = time.time() - t0
    if proc.returncode != 0:
        return {
            "kind": "real",
            "dataset": cfg["dataset"],
            "status": "FAILED",
            "source_patches": source_patches,
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

    sum_tab = load_sum_tab(upstream_root)

    with open(pkl, "rb") as f:
        res_dict = pickle.load(f)

    per_repeat_all = res_dict.get("_per_repeat")
    if shard:
        lo, hi = shard
        sliced = {}
        for name, d in (per_repeat_all or {}).items():
            if name.startswith("_"):
                continue
            sliced[name] = {
                k: [row[lo:hi] for row in np.asarray(v).tolist()]
                for k, v in d.items()
            }
        # The pickle is a valid run over this shard's repeats (summation_real is
        # sliced to them), so its per-key aggregates are correct statistics of
        # 10 repeats -- but lambda selection must happen once on all 50, so the
        # aggregates are shipped raw and reduced later rather than summarised here.
        agg = {
            k: [np.asarray(x).tolist() for x in v]
            for k, v in res_dict.items()
            if not k.startswith("_")
        }
        return {
            "kind": "real_shard",
            "status": "OK",
            "dataset": cfg["dataset"],
            "shard": [lo, hi],
            "command": " ".join(cmd),
            "n_reported": int(cfg["n_reported"]),
            "n_over_m": cfg.get("n_over_m"),
            "repeats": repeats or 50,
            "source_patches": source_patches,
            "procedure_patch": procedure_patch,
            "seconds": round(seconds, 1),
            "aggregates": agg,
            "per_repeat": sliced,
        }

    n_reported = int(cfg["n_reported"])
    summary = _summarise(res_dict, sum_tab, n_reported)

    per_repeat = res_dict.pop("_per_repeat", None)
    raw = {k: [np.asarray(a).tolist() for a in v] for k, v in res_dict.items()}
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"real_{cfg['dataset']}.json"), "w") as f:
        json.dump({"summary": summary, "raw": raw, "per_repeat": per_repeat}, f, indent=1)

    return {
        "kind": "real",
        "status": "OK",
        "dataset": cfg["dataset"],
        "command": " ".join(cmd),
        "result_pickle": os.path.relpath(pkl, upstream_root),
        "n_reported": n_reported,
        "n_over_m": cfg.get("n_over_m"),
        "repeats": repeats or 50,
        "source_patches": source_patches,
        "procedure_patch": procedure_patch,
        "seconds": round(seconds, 1),
        "summary": summary,
        "raw": raw,
        "per_repeat": per_repeat,
        "improvement_formula": "(a_ref - a_1) / (a_ref - a_0) * 100  [RealAnalysis/sum_tab.py:61, Appendix C.1]",
    }
