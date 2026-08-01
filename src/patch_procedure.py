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
    out = src.replace(ANCHOR, INJECT + ANCHOR)
    with open(src_path, "w") as f:
        f.write(out)
    return {
        "patched": os.path.join(script_dir, module),
        "sites": count,
        "adds": "_per_repeat: per-repeat mean coverage and mean set size for all 7 methods",
        "changes_reported_numbers": False,
    }
