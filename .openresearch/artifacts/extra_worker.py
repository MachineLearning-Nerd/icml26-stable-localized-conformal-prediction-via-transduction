"""Run a named list of nodes on a spare slot, alongside the main driver.

The driver snapshots `queue.json` at startup, so the m-sweep nodes appended
afterwards are invisible to it. That makes them safe to run here: there is no
shared work item and therefore no race. Results are written through
`drivelocal`'s own helpers so a node finished here is indistinguishable from one
the driver finished, and `_checkpoint` removes it from the queue so the chained
relaunch does not redo it.

Usage: extra_worker.py <slot> <branch> [<branch> ...]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import drivelocal  # noqa: E402
import runlocal  # noqa: E402


def main():
    slot = int(sys.argv[1])
    names = sys.argv[2:]
    os.environ["STCP_THREADS"] = os.environ.get("THREADS_PER_SLOT", "2")
    for name in names:
        try:
            code, out, err, secs = runlocal.run(name, slot)
        except Exception as exc:  # noqa: BLE001 - keep the slot alive for the rest
            print(f"ERROR {name}: {exc}", flush=True)
            continue
        payload = None
        for tag in drivelocal.STAGE_TAG.values():
            payload = runlocal._block(out, tag)
            if payload:
                break
        if code == 0 and payload and payload.get("status", "OK") == "OK":
            rel = drivelocal._write(name, payload)
            drivelocal._record(name, payload, secs)
            drivelocal._checkpoint(name)
            print(f"OK {name} -> {rel} ({secs:.0f}s)", flush=True)
        else:
            status = (payload or {}).get("status", f"exit={code}")
            print(f"FAIL {name}: {status} :: {err[-300:]}", flush=True)


if __name__ == "__main__":
    main()
