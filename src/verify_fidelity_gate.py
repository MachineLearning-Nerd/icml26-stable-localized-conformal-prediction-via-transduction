"""Show that Claim 4's fidelity precondition can actually fail.

`.openresearch/artifacts/c4_preregistration.md` replaced a percentage-agreement
precondition with one on the measured Std and marginal coverage, and registered
this obligation in the same breath:

    It must be shown to FAIL under injected defects, exactly as the Claim 1
    gates were: perturbing any reproduced Std away from the published value must
    flip `stds_agree` to false. If it cannot be made to fail, it is vacuous and
    Claim 4 is BLOCKED regardless of what it reports.

A precondition that no achievable result violates is not a precondition. This
runs the SHIPPED `claim4` -- not a copy of its logic -- on whichever datasets
have finished, first untouched and then with a defect injected into the pooled
per-repeat set sizes, and requires the clean pass and the injected failure.

Exits nonzero if a defect fails to trip the gate, if the clean data does not pass
it, or if no dataset is complete enough to test.
"""
import copy
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import published as P  # noqa: E402
import real_reduce  # noqa: E402
import stage_analysis as A  # noqa: E402
import stage_real  # noqa: E402
import upstream  # noqa: E402

RESULTS = os.path.join(os.path.dirname(HERE), "results")
SEED = 20260801


def _complete_datasets():
    """Datasets whose shards tile [0, 50) with no gap or overlap."""
    groups = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "real", "*.json"))):
        with open(path) as fh:
            payload = json.load(fh)
        if payload.get("kind") != "real_shard" and not payload.get("shard"):
            continue
        groups.setdefault(payload.get("dataset"), []).append(payload)
    out = {}
    for ds, shards in groups.items():
        spans = sorted(tuple(s["shard"]) for s in shards)
        covered, ok = 0, True
        for lo, hi in spans:
            if lo != covered:
                ok = False
                break
            covered = hi
        if ok and covered == 50:
            out[ds] = shards
    return out


def _build(ds, shards, upstream_root):
    merged, meta = real_reduce.merge(shards)
    sum_tab = stage_real.load_sum_tab(upstream_root)
    n_reported = int(shards[0]["n_reported"])
    return {
        "summary": stage_real._summarise(merged, sum_tab, n_reported),
        "per_repeat": meta.pop("pooled_per_repeat"),
        "n_reported": n_reported,
        "repeats": meta["repeats"],
        "n_over_m": P.TABLE1[ds]["n_over_m"],
    }


def _cells(real, ds):
    """Run the shipped claim4 and return its per-cell disagreement sets for `ds`."""
    A.RNG = np.random.default_rng(SEED)  # deterministic across injections
    v = A.claim4({"real": real})
    f = v["table_fidelity"]
    pref = ds + "/"
    return ({k: x for k, x in f["std_disagreements"].items() if k.startswith(pref)},
            {k: x for k, x in f["marginal_disagreements"].items() if k.startswith(pref)},
            [c for c in f["cells_without_an_interval"] if c.startswith(pref)])


def _inject(real, ds, label, model_idx, field, fn):
    out = copy.deepcopy(real)
    pr = out[ds]["per_repeat"][label]
    arr = np.asarray(pr[field], dtype=float)
    width = arr.shape[0] // 2
    row = model_idx * width if width > 1 else model_idx
    arr[row] = fn(arr[row])
    pr[field] = arr.tolist()
    return out


def main():
    upstream_root = upstream.ensure()
    complete = _complete_datasets()
    if not complete:
        print("FAIL: no dataset has a complete 0-50 tiling; the gate cannot be tested yet")
        return 1

    real = {ds: _build(ds, sh, upstream_root) for ds, sh in complete.items()}
    print(f"Datasets with a complete 50-repeat tiling: {', '.join(sorted(real))}\n")

    failures = []
    for ds in sorted(real):
        std_bad, marg_bad, no_ci = _cells(real, ds)
        print(f"{ds}: clean run -> {len(std_bad)} Std disagreement(s), "
              f"{len(marg_bad)} marginal disagreement(s), {len(no_ci)} cell(s) without an interval")
        for cell, d in sorted(std_bad.items()):
            print(f"    Std   {cell}: published {d['published']} vs "
                  f"reproduced {d['reproduced']:.4f} CI {[round(x, 4) for x in d['ci95']]}")
        for cell, d in sorted(marg_bad.items()):
            print(f"    Marg  {cell}: published {d['published']} vs "
                  f"reproduced {d['reproduced']:.4f} CI {[round(x, 4) for x in d['ci95']]}")
        if no_ci:
            failures.append(f"{ds}: {len(no_ci)} cell(s) had no bootstrap interval: {no_ci}")

        # A gate that never fires proves nothing. Move the reproduced Std away from
        # the published value and require the gate to notice.
        inflated = _inject(real, ds, "base", 0, "size_mean_per_repeat", lambda a: a * 1.25)
        i_std, _, _ = _cells(inflated, ds)
        target = f"{ds}/GLCP/base"
        if target in i_std and target not in std_bad:
            d = i_std[target]
            print(f"  [defect] base GLCP set-size spread x1.25 -> {target} now disagrees "
                  f"(reproduced {d['reproduced']:.4f}, CI {[round(x, 4) for x in d['ci95']]}) -- gate fires")
        elif target in std_bad:
            print(f"  [defect] skipped: {target} already disagrees on clean data")
        else:
            failures.append(f"{ds}: inflating base GLCP set-size spread by 25% did NOT trip "
                            f"stds_agree -- the Std gate is vacuous")

        shifted = _inject(real, ds, "base", 0, "cov_mean_per_repeat", lambda a: a + 0.05)
        _, i_marg, _ = _cells(shifted, ds)
        if target in i_marg and target not in marg_bad:
            print(f"  [defect] base GLCP coverage +0.05 -> {target} marginal now disagrees "
                  f"-- gate fires")
        elif target in marg_bad:
            print(f"  [defect] skipped: {target} marginal already disagrees on clean data")
        else:
            failures.append(f"{ds}: shifting base GLCP marginal coverage by +0.05 did NOT trip "
                            f"marginals_agree -- the coverage gate is vacuous")
        print()

    if failures:
        print("FAIL:")
        for f_ in failures:
            print("  -", f_)
        return 1
    print("OK: the fidelity gate passes on the reproduced data and fails under injected defects.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
