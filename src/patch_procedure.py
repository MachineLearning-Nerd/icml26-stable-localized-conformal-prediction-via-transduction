"""Make the real-data pipeline emit per-repeat set sizes, so Table 1 gets CIs.

`RealAnalysis/procedure.py` pickles only aggregates: for each method it stores
`[mar, mar_std, size, size_std, local_cov]`, where `size_std` is the "Std"
column of Table 1. With only that scalar there is no way to attach an
uncertainty to a reproduced percentage -- and a reproduction of a 0/2 claim that
cannot say how precise it is invites exactly the "is this really a match?"
objection.

Every quantity needed is already in memory: `SIZE1..SIZE7` have shape
`(method_slot, repeats, testN)`, and `Std` is the sample sd across repeats of
`SIZE.mean(-1)`. This patch appends those per-repeat means (and the matching
coverage means) to the pickled dict under `_per_repeat`, changing nothing that
is computed or reported.
"""

import os

LOOP = "    for rep in range(repeats):\n"

# `seed_rep = seed + 1 + rep` is set at the top of every iteration and all shared
# state (source predictor, base generator, test-group clustering) is fixed before
# the loop, so repeats are independent and reproducible one at a time. Restricting
# the loop to a slice therefore computes exactly the rows a full run would, which
# lets a 50-repeat dataset run as five jobs that each fit inside the job timeout.
# `repeats` itself is left at 50 so every array keeps its full shape and the
# per-repeat rows land at their true indices; unfilled rows are dropped on merge.
LOOP_SHARDED = ("    _lo = int(os.environ.get('STCP_SHARD_LO', 0))\n"
                "    _hi = int(os.environ.get('STCP_SHARD_HI', repeats))\n"
                "    for rep in range(_lo, min(_hi, repeats)):\n"
                "        print(f'[real shard {_lo}:{_hi}] repeat {rep} start "
                "t+{(datetime.datetime.now()-_T0).total_seconds():.1f}s', flush=True)\n")

T0 = "import datetime\n_T0 = datetime.datetime.now()\n"

# `summation_real` averages over every row of `cov`/`size`, but a sharded run
# fills only its own rows and leaves the rest at zero. Slicing here -- rather
# than at the call sites, of which there are fourteen -- makes each shard's
# pickle a valid run over its repeats, keyed exactly as the authors key it, so
# lambda selection in `sum_compare_result` behaves identically.
SUM_SIG = "def summation_real(cov, size, IND, fullIND, alpha=.1):\n"
SUM_SIG_CLS = "def summation_real_cls(cov, size, IND, fullIND, alpha=.1):\n"
SLICE = ("    _lo = int(os.environ.get('STCP_SHARD_LO', 0))\n"
         "    _hi = int(os.environ.get('STCP_SHARD_HI', len(IND)))\n"
         "    cov, size, IND = cov[_lo:_hi], size[_lo:_hi], IND[_lo:_hi]\n")

ANCHOR = "    with open(f'{SimRpath}/{SimName}.pkl', 'wb') as f:\n"

INJECT = '''    resDict['_per_repeat'] = {}
    for _nm, _COV, _SIZE in [('base', COV1, SIZE1), ('SLCP', COV2, SIZE2), ('SLCP-sel', COV3, SIZE3),
                             ('SDCP', COV4, SIZE4), ('PPI', COV5, SIZE5), ('ORCP', COV6, SIZE6),
                             ('NOAL', COV7, SIZE7)]:
        resDict['_per_repeat'][_nm] = {
            'cov_mean_per_repeat': np.asarray(_COV).mean(axis=-1).tolist(),
            'size_mean_per_repeat': np.asarray(_SIZE).mean(axis=-1).tolist(),
        }
    resDict['_per_repeat']['_note'] = ('shape (method_slot, repeats); for SLCP the slot axis is '
                                       'method-major over the lambda/param grid, matching COV2 indexing '
                                       'i + j*len(param_comb)')
'''


def build(upstream_root, script_dir="RealAnalysis", module="procedure.py"):
    src_path = os.path.join(upstream_root, script_dir, module)
    with open(src_path) as f:
        src = f.read()
    count = src.count(ANCHOR)
    if count != 2:
        raise RuntimeError(f"expected 2 pickle sites in {module} (reg + cls), found {count}")
    n_loops = src.count(LOOP)
    if n_loops != 2:
        raise RuntimeError(f"expected 2 repeat loops in {module} (reg + cls), found {n_loops}")
    if "import os" not in src.split("def ")[0]:
        raise RuntimeError("procedure.py does not import os at module level")

    for sig in (SUM_SIG, SUM_SIG_CLS):
        if src.count(sig) != 1:
            raise RuntimeError(f"expected exactly one {sig.strip()}")

    out = src.replace(ANCHOR, INJECT + ANCHOR).replace(LOOP, LOOP_SHARDED)
    out = out.replace(SUM_SIG, SUM_SIG + SLICE).replace(SUM_SIG_CLS, SUM_SIG_CLS + SLICE)
    out = T0 + out
    if out.count("STCP_SHARD_LO") != 4:
        raise RuntimeError("shard patch did not apply to both loops and both summations")
    with open(src_path, "w") as f:
        f.write(out)
    return {
        "patched": os.path.join(script_dir, module),
        "sites": count,
        "sharded_loops": n_loops,
        "shard_env": ["STCP_SHARD_LO", "STCP_SHARD_HI"],
        "sharded_summations": ["summation_real", "summation_real_cls"],
        "adds": "_per_repeat: per-repeat mean coverage and mean set size for all 7 methods",
        "changes_reported_numbers": False,
    }
