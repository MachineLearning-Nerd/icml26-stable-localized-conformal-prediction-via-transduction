"""Sharding must not change the answer.

Every dataset and simulation setting in this logbook was assembled from repeat
shards run as separate jobs, and the shards are not all the same width: the
Table 2 n=30 setting, for instance, ends up as two 10-repeat shards followed by
six 5-repeat ones. If merging were even slightly wrong -- averaging per-shard
means without weighting by repeat count is the obvious way to get it wrong --
every number downstream would be quietly off, and nothing else in the pipeline
would notice.

This reconstructs known data, reduces it under several shard layouts, and
requires every layout to reproduce the unsharded answer exactly. It also runs
the wrong merge on purpose and requires it to differ, so that "the layouts agree"
is a statement with content rather than a test that cannot fail.

Run: python src/verify_shard_merge.py   (exits nonzero on any disagreement)
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import shard_reduce as S  # noqa: E402

TOL = 1e-12
METRICS = ("marginal", "size", "std", "cond_miscoverage")


def _layouts():
    """(name, spans) covering the layouts this campaign actually produces."""
    return [
        ("uniform 10-repeat shards", [(i, i + 10) for i in range(0, 50, 10)]),
        ("uniform 5-repeat shards", [(i, i + 5) for i in range(0, 50, 5)]),
        # The real logabs-n30-m500 layout: two wide shards then six narrow ones.
        ("mixed 2x10 + 6x5", [(0, 10), (10, 20)] + [(i, i + 5) for i in range(20, 50, 5)]),
        # Order must not matter: every statistic is over the pooled repeat axis.
        ("mixed, shards out of order",
         [(20, 25), (0, 10), (45, 50), (10, 20), (25, 30), (35, 40), (30, 35), (40, 45)]),
    ]


def _worst(got, ref):
    return max(float(np.abs(np.asarray(got[k]) - np.asarray(ref[k])).max()) for k in METRICS)


def _check_patch_guard(failures):
    """The shard patch must be caught if it ever changes the science.

    `patch_core.build` applies three fixed textual edits to the authors' `core.py`.
    The guard that certifies "nothing else changed" is only worth its claim if it
    fires on a change, so two are injected here: a reseed added inside the repeat
    loop, and a quietly altered loop bound. Both add lines without removing any,
    which is the case a removal-only guard cannot see.
    """
    import shutil
    import tempfile

    import patch_core
    import upstream

    up_real = upstream.ensure()
    with tempfile.TemporaryDirectory() as tmp:
        # Only SimuAnalysis is needed: that is where core.py lives and where the
        # patched copy is written.
        shutil.copytree(os.path.join(up_real, "SimuAnalysis"),
                        os.path.join(tmp, "SimuAnalysis"))
        patch_core.build(tmp, 0, 10)
        core = os.path.join(tmp, "SimuAnalysis", "core_shard.py")
        clean = open(core).read()

        print("\nShard-patch guard (patch_core.assert_science_unchanged):")
        try:
            patch_core.assert_science_unchanged(tmp)
            print("  clean patch                      accepted")
        except RuntimeError as exc:
            failures.append(f"the guard rejects its own clean patch: {exc}")
            return

        injections = {
            "reseed added inside the repeat loop":
                clean.replace('        print(f"[shard', '        setseed(999)\n'
                              '        print(f"[shard', 1),
            "loop bound quietly altered":
                clean.replace("min(SHARD_HI, repeats)", "min(SHARD_HI, repeats-1)", 1),
        }
        for name, text in injections.items():
            if text == clean:
                failures.append(f"injection '{name}' did not modify the file")
                continue
            open(core, "w").write(text)
            try:
                patch_core.assert_science_unchanged(tmp)
                failures.append(f"the guard did NOT catch: {name}")
                print(f"  {name:32s} NOT CAUGHT")
            except RuntimeError:
                print(f"  {name:32s} caught")
            open(core, "w").write(clean)


def main():
    rng = np.random.default_rng(0)
    models, repeats, testn = 2, 50, 400
    cov = (rng.random((models, repeats, testn)) < 0.9).astype(float)
    size = 2.0 + rng.standard_normal((models, repeats, testn)) * 0.3

    def summarise(spans):
        return S.summarise([S.reduce_pair(cov, size, lo, hi) for lo, hi in spans])

    whole = summarise([(0, repeats)])
    print(f"Reference: {repeats} repeats reduced in one piece\n")

    failures = []
    for name, spans in _layouts():
        got = summarise(spans)
        worst = _worst(got, whole)
        ok = got["n_repeats"] == repeats and worst <= TOL
        print(f"  {name:32s} n={got['n_repeats']:3d}  max|diff| = {worst:.3e}  "
              f"{'ok' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"{name}: max|diff| {worst:.3e} over {METRICS}")

    # Negative control. Averaging each shard's own Std, rather than pooling the
    # repeats, is the natural wrong implementation. It must differ, or the checks
    # above would pass for a merge that is not doing anything.
    mixed = [(0, 10), (10, 20)] + [(i, i + 5) for i in range(20, 50, 5)]
    naive = np.mean([np.asarray(summarise([sp])["std"]) for sp in mixed], axis=0)
    gap = float(np.abs(naive - np.asarray(whole["std"])).max())
    print(f"\n  negative control: averaging per-shard Std instead of pooling "
          f"repeats differs by {gap:.4f}")
    if gap <= TOL:
        failures.append("the wrong merge produced the same answer as the right one, "
                        "so these checks cannot discriminate")

    _check_patch_guard(failures)

    if failures:
        print("\nFAIL:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nOK: every shard layout reproduces the unsharded result, the wrong merge "
          "does not, and the patch guard catches an injected science change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
