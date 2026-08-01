"""Run one repeat-shard of the paper's simulation and print its raw arrays.

`repeats` stays at the paper's 50 so the shared source-side model is seeded
identically to a single 50-repeat run; only the loop range is restricted. The
raw per-repeat arrays are printed to stdout because HF job filesystems do not
survive the job.
"""

import json
import os
import sys
import time

import patch_core


def run(cfg, upstream_root):
    lo, hi = int(cfg["shard"][0]), int(cfg["shard"][1])
    module_name, diff = patch_core.build(upstream_root, lo, hi)
    guard = patch_core.assert_science_unchanged(upstream_root)

    sys.path.insert(0, os.path.join(upstream_root, "SimuAnalysis"))
    import config

    mod = __import__(module_name)

    dtype, n, m = cfg["dtype"], int(cfg["n"]), int(cfg["m"])
    kwargs = config.common_run_kwargs()  # repeats stays 50: seeds must not move

    t0 = time.time()
    res = mod.run_experiment(dtype=dtype, n=n, m=m, **kwargs)
    seconds = time.time() - t0

    raw = res.pop("_raw")

    import shard_reduce

    pairs = {
        "base": ("cov_base", "size_base"),
        "oracle": ("cov_orac", "size_orac"),
        "StCP": ("cov_slcp", "size_slcp"),
        "StCP-sel": ("cov_sel", "size_sel"),
        "SDCP": ("cov_sdcp", "size_sdcp"),
        "PPI": ("cov_ppi", "size_ppi"),
        "NOAL": ("cov_noal", "size_noal"),
    }
    reduced = {
        key: shard_reduce.reduce_pair(raw[ck], raw[sk], lo, hi)
        for key, (ck, sk) in pairs.items()
        if raw.get(ck) is not None
    }
    reduced["selected_idx"] = [row[lo:hi] for row in raw["selected_idx"]]

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    stem = f"shard_{dtype}_n{n}_m{m}_{lo}_{hi}"
    with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
        json.dump(reduced, f)

    return {
        "kind": "sim_shard",
        "dtype": dtype,
        "n": n,
        "m": m,
        "shard": [lo, hi],
        "repeats_declared": kwargs["repeats"],
        "lambda_grid": kwargs["lbds"],
        "seconds": round(seconds, 1),
        "seconds_per_repeat": round(seconds / max(1, hi - lo), 1),
        "patch": diff,
        "patch_guard": guard,
        "meta": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in res["meta"].items()},
        "reduced": reduced,
    }
