"""Make the authors' simulation shardable by repeat, without changing its science.

`SimuAnalysis/core.run_experiment` seeds the shared source-side model once with
`setseed(repeats + 100)` and then seeds every repeat independently with
`setseed(1 + rep)`. Repeats are therefore fully deterministic and mutually
independent, so running repeats `[lo, hi)` in one process and `[hi, hi')` in
another yields exactly the numbers a single 50-repeat process would have
produced -- provided `repeats` stays 50 so the shared source model is unchanged.

This module rewrites `core.py` textually (three edits, all printed into the
evidence) to:

  1. iterate `range(SHARD_LO, SHARD_HI)` instead of `range(repeats)`;
  2. print a timestamped line per repeat, so a long job is observable;
  3. return the raw per-repeat coverage/size arrays, so shards can be merged
     before the authors' aggregation runs.

Nothing else is touched: the estimator, the lambda grid, the seeds and the
aggregation all stay byte-identical to upstream.
"""

import os
import re

MARKER_LOOP = "    for rep in range(repeats):\n"
MARKER_RETURN = "    return res\n"

RAW_CAPTURE = '''    res["_raw"] = {
        "shard": [SHARD_LO, SHARD_HI],
        "cov_base": cov_base.tolist(),
        "size_base": size_base.tolist(),
        "cov_orac": cov_orac.tolist(),
        "size_orac": size_orac.tolist(),
        "cov_slcp": cov_slcp.tolist(),
        "size_slcp": size_slcp.tolist(),
        "cov_sel": cov_sel.tolist(),
        "size_sel": size_sel.tolist(),
        "selected_idx": selected_idx.tolist(),
        "cov_sdcp": None if cov_sdcp is None else cov_sdcp.tolist(),
        "size_sdcp": None if size_sdcp is None else size_sdcp.tolist(),
        "cov_ppi": None if cov_ppi is None else cov_ppi.tolist(),
        "size_ppi": None if size_ppi is None else size_ppi.tolist(),
        "cov_noal": None if cov_noal is None else cov_noal.tolist(),
        "size_noal": None if size_noal is None else size_noal.tolist(),
    }
'''

LOOP_REPLACEMENT = """    import time as _time
    _t0 = _time.time()
    for rep in range(SHARD_LO, min(SHARD_HI, repeats)):
        print(f"[shard {SHARD_LO}:{SHARD_HI}] repeat {rep} start t+{_time.time()-_t0:.1f}s", flush=True)
"""

HEADER = "SHARD_LO = {lo}\nSHARD_HI = {hi}\n"


def build(upstream_root, lo, hi):
    """Write `SimuAnalysis/core_shard.py` and return (module_name, diff_summary)."""
    src_path = os.path.join(upstream_root, "SimuAnalysis", "core.py")
    with open(src_path) as f:
        src = f.read()

    if src.count(MARKER_LOOP) != 1:
        raise RuntimeError(f"expected exactly one repeat loop, found {src.count(MARKER_LOOP)}")
    if src.count(MARKER_RETURN) != 1:
        raise RuntimeError(f"expected exactly one `return res`, found {src.count(MARKER_RETURN)}")

    out = src.replace(MARKER_LOOP, LOOP_REPLACEMENT)
    out = out.replace(MARKER_RETURN, RAW_CAPTURE + MARKER_RETURN)

    # Insert the shard bounds after the last top-level import block.
    anchor = 'SUPPORTED_SIGMA_DTYPES = ["quad", "softplus", "logabs", "sqrtabs"]\n'
    if anchor not in out:
        raise RuntimeError("could not locate shard-constant anchor in core.py")
    out = out.replace(anchor, HEADER.format(lo=lo, hi=hi) + anchor, 1)

    dst_path = os.path.join(upstream_root, "SimuAnalysis", "core_shard.py")
    with open(dst_path, "w") as f:
        f.write(out)

    diff = {
        "edits": 3,
        "loop": f"for rep in range(repeats)  ->  for rep in range({lo}, min({hi}, repeats))",
        "progress": "per-repeat timestamped stdout line added",
        "raw": 'res["_raw"] added with per-repeat cov/size arrays',
        "src_lines": len(src.splitlines()),
        "dst_lines": len(out.splitlines()),
        "unchanged_body_lines": len(
            [l for l in src.splitlines() if l in set(out.splitlines())]
        ),
    }
    return "core_shard", diff


def assert_science_unchanged(upstream_root):
    """Guard: the shard copy must differ from upstream only in the three edits.

    Both directions are checked. Watching only for *removed* lines would miss the
    failure that actually matters -- a line ADDED into the repeat loop, a stray
    reseed or an altered argument, which changes the science while deleting
    nothing. Since `build` applies three fixed textual substitutions, every added
    line is known in advance and anything else is foreign.
    """
    with open(os.path.join(upstream_root, "SimuAnalysis", "core.py")) as f:
        a = f.read().splitlines()
    with open(os.path.join(upstream_root, "SimuAnalysis", "core_shard.py")) as f:
        b = f.read().splitlines()
    removed = [l for l in a if l not in set(b)]
    # Only the original `for rep in range(repeats):` line may disappear.
    unexpected = [l for l in removed if not re.match(r"\s*for rep in range\(repeats\):", l)]
    if unexpected:
        raise RuntimeError(f"shard patch removed unexpected lines: {unexpected[:5]}")

    allowed = set(LOOP_REPLACEMENT.splitlines()) | set(RAW_CAPTURE.splitlines())
    added = [l for l in b if l not in set(a)]
    foreign = [l for l in added
               if l not in allowed and not re.match(r"SHARD_(LO|HI) = \d+$", l.strip())]
    if foreign:
        raise RuntimeError(f"shard patch added unexpected lines: {foreign[:5]}")
    return {"removed_lines": removed, "added_lines": added, "ok": True}
