"""Rebuild a 50-repeat Table 1 entry from equal-sized repeat shards.

A dataset's 50 repeats are independent given the shared source model, so they can
run as five jobs. What cannot be sharded is the *summary*: `sum_compare_result`
picks one lambda per method by comparing statistics across the whole grid, and
that choice has to be made once on all 50 repeats, not five times on ten.

So each shard ships its raw per-key aggregates and its per-repeat means, and this
module merges them into a dict shaped exactly like the authors' `resDict` before
handing it to their own `sum_compare_result`.

Exactness, per column of `[mar, mar_std, size, size_std, local_cov]`:

  mar, size   exact -- plain means over an equal number of repeats per shard, so
              the mean of shard means is the mean over all 50.
  size_std    exact -- recomputed from the pooled per-repeat means, which is the
              definition (`np.std(np.mean(size, axis=1))`).
  mar_std     exact -- same, from pooled per-repeat coverage means.
  local_cov   approximate -- a *weighted* mean whose per-group weights differ by
              shard, so the mean of shard values is not identical to the pooled
              value. It is reported for context only: it feeds no selection in
              `sum_compare_result` and no quantity in Claim 4.
"""

import numpy as np

SLOT_KEYS = ("SLCP",)


def _pool(per_repeat, name, field):
    """Concatenate a per-repeat field across shards in repeat order."""
    rows = [np.asarray(sh[name][field], dtype=float) for sh in per_repeat]
    return np.concatenate(rows, axis=-1)


def merge(shards):
    """shards: list of stage_real shard results, any order. Returns a resDict."""
    shards = sorted(shards, key=lambda s: s["shard"][0])
    spans = [tuple(s["shard"]) for s in shards]
    for a, b in zip(spans, spans[1:]):
        if a[1] != b[0]:
            raise ValueError(f"shards are not contiguous: {spans}")
    widths = {b - a for a, b in spans}
    if len(widths) != 1:
        raise ValueError(f"shards must be equal width to average unweighted: {spans}")

    aggs = [s["aggregates"] for s in shards]
    keys = set(aggs[0])
    for a in aggs[1:]:
        if set(a) != keys:
            raise ValueError("shards disagree on result keys")

    merged = {}
    for key in sorted(keys):
        mar = np.mean([np.asarray(a[key][0], dtype=float) for a in aggs], axis=0)
        size = np.mean([np.asarray(a[key][2], dtype=float) for a in aggs], axis=0)
        loc = np.mean([np.asarray(a[key][4], dtype=float) for a in aggs], axis=0)
        merged[key] = [mar, None, size, None, loc]

    # size_std / mar_std come from the pooled per-repeat means, never from the
    # shard-level standard deviations (a mean of standard deviations is not the
    # standard deviation of the pool).
    label_of = _label_map(keys)
    for key in merged:
        label = label_of[key]
        model_slot = 0 if key.split()[0] == "GLCP" else 1
        cov = _pool([s["per_repeat"] for s in shards], label, "cov_mean_per_repeat")
        siz = _pool([s["per_repeat"] for s in shards], label, "size_mean_per_repeat")
        cov, siz = _select_slots(cov, siz, label, model_slot, np.shape(merged[key][0]))
        merged[key][1] = np.std(cov, axis=-1)
        merged[key][3] = np.std(siz, axis=-1)

    n_repeats = sum(b - a for a, b in spans)
    return merged, {"shards": spans, "repeats": n_repeats}


def _label_map(keys):
    """Map the authors' pickle keys onto the `_per_repeat` labels."""
    out = {}
    for key in keys:
        parts = key.split(" ", 1)
        out[key] = parts[1] if len(parts) > 1 else "base"
    return out


def _select_slots(cov, siz, label, model_slot, target_shape):
    """Pick this key's rows out of the method-major per-repeat slot axis.

    `COV1/3..7` carry two slots, one per base method (`GLCP`, `SCC`), so the
    model index selects directly. `COV2` (SLCP) is laid out as
    `i + j*len(param_comb)`, i.e. the whole lambda grid for GLCP followed by the
    whole grid for SCC, so the model index selects a contiguous block.
    """
    cov, siz = np.asarray(cov, dtype=float), np.asarray(siz, dtype=float)
    n_slots = cov.shape[0]
    if not target_shape:  # scalar key: one slot per base method
        return cov[model_slot], siz[model_slot]
    width = n_slots // 2
    lo = model_slot * width
    return cov[lo:lo + width], siz[lo:lo + width]
