"""Pin BLAS/OpenMP/torch thread pools to the container's real CPU quota.

Must be imported before numpy or torch: both read these environment variables
at import time. Inside a container ``os.cpu_count()`` reports the *host* core
count while the cgroup grants only a few vCPUs, so unpinned pools spawn far
more threads than runnable cores and spin-contention dominates.
"""

import os


def _cgroup_quota():
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except OSError:
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        if quota > 0:
            return max(1, quota // period)
    except OSError:
        pass
    return None


def cpu_quota():
    n = _cgroup_quota()
    if n:
        return n
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def pin(n=None):
    # An explicit budget wins over the detected quota: when several shards run
    # side by side on one machine each must claim a slice, or they oversubscribe
    # the same cores and every one of them slows down.
    n = n or int(os.environ.get("STCP_THREADS") or 0) or cpu_quota()
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(n)
    return n


def pin_torch(n=None):
    """Pin PyTorch's own pools, which the environment variables do not fully cover.

    `OMP_NUM_THREADS` reaches the intra-op pool, but the *inter-op* pool defaults
    to the machine's core count regardless. With several shards on one host that
    multiplies: four slots at 8 inter-op threads each produced a load average of
    33 on 8 cores, which is the same oversubscription that makes this workload
    run an order of magnitude slow.
    """
    n = n or PINNED
    try:
        import torch
    except ImportError:
        return None
    torch.set_num_threads(int(n))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass       # already fixed by earlier parallel work; intra-op pin still applies
    return torch.get_num_threads(), torch.get_num_interop_threads()


PINNED = pin()
