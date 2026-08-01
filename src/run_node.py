"""Fixed entrypoint for every experiment node.

The run command is identical on every node (`bash run.sh`). What a node does is
decided entirely by the committed `config/node.json`, never by environment
variables or a different command line.
"""

import threads  # noqa: F401  -- must precede numpy/torch

import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import upstream  # noqa: E402


def node_config():
    with open(os.path.join(ROOT, "config", "node.json")) as f:
        return json.load(f)


def environment_report(n_threads):
    import numpy
    import scipy
    import torch

    try:
        git_sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_sha = os.environ.get("GIT_SHA", "unknown")
    return {
        "git_sha": git_sha,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "os_cpu_count": os.cpu_count(),
        "cgroup_cpu_quota": threads.cpu_quota(),
        "threads_pinned": n_threads,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
    }


def emit(name, payload):
    print(f"===BEGIN {name}===", flush=True)
    print(json.dumps(payload, indent=1, default=str), flush=True)
    print(f"===END {name}===", flush=True)


def main():
    import torch

    n = threads.PINNED
    torch.set_num_threads(n)
    torch.set_num_interop_threads(1)

    cfg = node_config()
    upstream_root = upstream.add_to_path()
    env = environment_report(n)
    env["upstream_sha"] = subprocess.check_output(
        ["git", "-C", upstream_root, "rev-parse", "HEAD"], text=True
    ).strip()
    env["node_config"] = cfg
    emit("ENVIRONMENT", env)
    # `make_pages` reads the pinned environment from disk, not from the job log, so
    # it has to land there too: without this the published claim pages report the
    # Git SHA, Python and library versions as `None`, which is exactly the
    # provenance the evidence gate requires them to show.
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "environment.json"), "w") as fh:
        json.dump(env, fh, indent=1)

    stage = cfg["stage"]
    mod = __import__(f"stage_{stage}")
    t0 = time.time()
    result = mod.run(cfg, upstream_root)
    result["wall_seconds"] = round(time.time() - t0, 2)
    result["environment"] = env
    emit(f"RESULT_{stage.upper()}", result)

    # A verifier must fail loudly. Any stage may set `exit_code`; a nonzero one
    # propagates so the job itself is marked failed rather than quietly "done".
    code = int(result.get("exit_code", 0))
    if result.get("status") in ("FAILED", "MISSING_RESULT"):
        code = code or 1
    if code:
        print(f"VERIFIER FAILED: exit_code={code}", flush=True)
        sys.exit(code)


if __name__ == "__main__":
    main()
