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
    n = n or cpu_quota()
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(n)
    return n


PINNED = pin()
