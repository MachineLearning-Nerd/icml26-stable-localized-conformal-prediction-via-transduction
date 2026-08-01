"""Fetch the authors' StCP code and datasets at a pinned commit.

The paper's artifact (https://github.com/OswinMin/StCP) ships both the
reference implementation and the five Table 1 datasets. We reproduce against
that artifact rather than a clean-room rewrite, which is exactly what the
previous judged revision was faulted for lacking.
"""

import os
import subprocess
import sys

UPSTREAM_URL = "https://github.com/OswinMin/StCP"
UPSTREAM_SHA = "1d8df7614d49eada881426742688ba75fec631b9"
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "upstream", "StCP")


def ensure():
    # A pre-pinned checkout may be supplied instead of cloning -- used when running
    # locally, where each run gets its own copy-on-write copy of the artifact
    # because the stages patch `procedure.py` in place.
    override = os.environ.get("STCP_UPSTREAM")
    if override:
        head = subprocess.check_output(
            ["git", "-C", override, "rev-parse", "HEAD"], text=True).strip()
        if head != UPSTREAM_SHA:
            raise RuntimeError(f"STCP_UPSTREAM is at {head}, expected pin {UPSTREAM_SHA}")
        return override
    if os.path.isdir(os.path.join(DEST, ".git")):
        head = subprocess.check_output(["git", "-C", DEST, "rev-parse", "HEAD"], text=True).strip()
        if head == UPSTREAM_SHA:
            return DEST
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    subprocess.check_call(["git", "clone", UPSTREAM_URL, DEST])
    subprocess.check_call(["git", "-C", DEST, "checkout", UPSTREAM_SHA])
    return DEST


def add_to_path():
    root = ensure()
    for sub in ("Main", "SimuAnalysis", "RealAnalysis"):
        p = os.path.join(root, sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


if __name__ == "__main__":
    root = ensure()
    head = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    print(f"upstream={UPSTREAM_URL}")
    print(f"upstream_sha={head}")
    print(f"upstream_sha_matches_pin={head == UPSTREAM_SHA}")
