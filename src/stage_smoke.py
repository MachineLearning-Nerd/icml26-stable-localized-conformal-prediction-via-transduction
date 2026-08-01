"""Timing probe: measure the cost of one repeat of each upstream pipeline.

Nothing scientific is concluded here. Its only job is to size the full runs so
we pick a cpu-upgrade timeout that will not kill a real experiment.
"""

import os
import subprocess
import sys
import time


def _time_sim(upstream_root, dtype, n, m, repeats):
    sys.path.insert(0, os.path.join(upstream_root, "SimuAnalysis"))
    import config
    from core import run_experiment

    kwargs = config.common_run_kwargs()
    kwargs["repeats"] = repeats
    t0 = time.time()
    res = run_experiment(dtype=dtype, n=n, m=m, **kwargs)
    dt = time.time() - t0
    return {
        "dtype": dtype,
        "n": n,
        "m": m,
        "repeats": repeats,
        "seconds_total": round(dt, 2),
        "seconds_per_repeat": round(dt / repeats, 2),
        "projected_50_repeats_minutes": round(dt / repeats * 50 / 60, 1),
        "result_keys": sorted(res.keys()),
        "meta": res.get("meta"),
    }


def patched_entry(upstream_root, script, repeats):
    """Copy an upstream entry script, overriding only its hard-coded `repeats`.

    Everything else — data loading, target/source split, hyperparameters — stays
    byte-identical to the authors' script, so the probe measures their pipeline
    rather than a paraphrase of it.
    """
    src_path = os.path.join(upstream_root, "RealAnalysis", script)
    with open(src_path) as f:
        lines = f.readlines()
    hits = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("epoches, repeats, alpha") and "=" in stripped:
            lhs, rhs = stripped.split("=", 1)
            vals = [v.strip() for v in rhs.split(",")]
            vals[1] = str(repeats)
            indent = line[: len(line) - len(line.lstrip())]
            lines[i] = f"{indent}{lhs.strip()} = {', '.join(vals)}\n"
            hits.append(i)
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one repeats assignment in {script}, found {len(hits)}")
    out_name = f"_probe_{repeats}_{script}"
    out_path = os.path.join(upstream_root, "RealAnalysis", out_name)
    with open(out_path, "w") as f:
        f.writelines(lines)
    return out_name


def _time_real(upstream_root, script, argv, timeout_s, repeats=None):
    """Run a real-data entry script as a subprocess; it writes its own results."""
    if repeats is not None:
        script = patched_entry(upstream_root, script, repeats)
    cmd = [sys.executable, "-u", os.path.join("RealAnalysis", script)] + [str(a) for a in argv]
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd, cwd=upstream_root, capture_output=True, text=True, timeout=timeout_s
        )
        rc, out, err, killed = p.returncode, p.stdout, p.stderr, False
    except subprocess.TimeoutExpired as e:
        rc, out, err, killed = None, (e.stdout or b"").decode(errors="replace"), (e.stderr or b"").decode(errors="replace"), True
    return {
        "script": script,
        "argv": [str(a) for a in argv],
        "seconds": round(time.time() - t0, 2),
        "timed_out": killed,
        "returncode": rc,
        "stdout_tail": out[-4000:],
        "stderr_tail": err[-4000:],
    }


def run(cfg, upstream_root):
    out = {"probes": []}
    for probe in cfg.get("sim_probes", []):
        out["probes"].append(_time_sim(upstream_root, **probe))
    for probe in cfg.get("real_probes", []):
        out["probes"].append(_time_real(upstream_root, **probe))
    return out
