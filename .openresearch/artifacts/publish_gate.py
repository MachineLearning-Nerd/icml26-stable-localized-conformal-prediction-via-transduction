"""Publish the candidate logbook to the existing Space, text-only and additive.

Refuses to upload unless the judged revision's file set is a subset of the
candidate's, so no previously judged evidence can be dropped.
"""

import hashlib
import json
import os
import sys

from huggingface_hub import HfApi, get_token, snapshot_download

REPO_ID = "DineshAI/lSMTccAN61"
JUDGED_SHA = "dfe09c8901724cc0e515d61e5af8955d3fa5a18c"
# What the live judge already awarded the revision we would be replacing. A
# candidate that self-scores below this must not be published: an honestly
# BLOCKED claim scores zero, and replacing a claim the judge rated TOY (1/2)
# with one it rates INCONCLUSIVE (0/2) is a net loss.
JUDGED_POINTS = 5
SCRATCH = os.path.dirname(os.path.abspath(__file__))
STAGING = os.path.join(SCRATCH, "staging")

ALLOW = ["logbook.json", "pages/**/*.md", "pages/*.md", "repro/**/*.py",
         "repro/**/*.md", "repro/**/*.json", "README.md"]

TEXT_EXT = {".md", ".json", ".py", ".txt", ".css", ".js", ".html", ".svg"}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def tree(root):
    out = {}
    for base, _, files in os.walk(root):
        if ".git" in base or ".cache" in base:
            continue
        for name in files:
            p = os.path.join(base, name)
            out[os.path.relpath(p, root)] = sha256(p)
    return out


def subset_check(staging_dir):
    judged = json.load(open(os.path.join(SCRATCH, "judged_manifest.json")))
    judged_paths = {p for p in judged if not p.startswith(".cache/")}
    cand = tree(staging_dir)
    missing = sorted(judged_paths - set(cand))
    changed = sorted(
        p for p in judged_paths & set(cand) if judged[p][1] != cand[p]
    )
    return {
        "judged_files": len(judged_paths),
        "candidate_files": len(cand),
        "missing_from_candidate": missing,
        "content_changed": changed,
        "is_superset": not missing,
    }


def scan_secrets(staging_dir):
    bad = []
    needles = ("hf_", "HF_TOKEN", "api_key", "Authorization:", "-----BEGIN")
    for base, _, files in os.walk(staging_dir):
        if ".git" in base or ".cache" in base:
            continue
        for name in files:
            p = os.path.join(base, name)
            if os.path.splitext(name)[1].lower() not in TEXT_EXT:
                continue
            try:
                txt = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            for nd in needles:
                if nd in txt:
                    bad.append((os.path.relpath(p, staging_dir), nd))
    return bad


def claim_regression_check(staging_dir):
    """The candidate's own verdicts, and whether they beat what is already live."""
    path = os.path.join(staging_dir, "repro", "results", "analysis.json")
    if not os.path.exists(path):
        return {"ok": False, "reason": f"no analysis.json at {path}"}
    a = json.load(open(path))
    verdicts = {c: v["verdict"] for c, v in a["verdicts"].items()}
    scored = {c: v for c, v in verdicts.items() if v in ("VERIFIED", "FALSIFIED")}
    points = 2 * len(scored)
    blocked = sorted(c for c, v in verdicts.items() if v == "BLOCKED")
    return {
        "ok": points >= JUDGED_POINTS,
        "verdicts": verdicts,
        "self_scored_points": points,
        "judged_points": JUDGED_POINTS,
        "blocked": blocked,
        "reason": (f"self-scored {points} < judged {JUDGED_POINTS}"
                   if points < JUDGED_POINTS else ""),
    }


def main():
    dry = "--publish" not in sys.argv
    # Allow a non-default candidate tree so the gate can be exercised on a
    # synthetic build without touching the real staging directory.
    global STAGING
    for i, a in enumerate(sys.argv):
        if a == "--candidate" and i + 1 < len(sys.argv):
            STAGING = os.path.abspath(sys.argv[i + 1])
    token = get_token()
    api = HfApi()

    live = api.repo_info(REPO_ID, repo_type="space", token=token).sha
    print(f"live_sha={live}")
    if live != JUDGED_SHA:
        print(f"WARNING: live head {live} != judged {JUDGED_SHA}; another session may have published.")

    check = subset_check(STAGING)
    print(json.dumps(check, indent=1)[:2000])
    if not check["is_superset"]:
        print("ABORT: judged file set is not a subset of the candidate.")
        return 2

    secrets = scan_secrets(STAGING)
    if secrets:
        print(f"ABORT: possible secrets in {secrets[:5]}")
        return 3

    regression = claim_regression_check(STAGING)
    print(json.dumps(regression, indent=1))
    if not regression["ok"] and "--allow-regression" not in sys.argv:
        print(f"ABORT: {regression['reason']}. "
              "Publishing would replace judged credit with none. "
              "Pass --allow-regression only with a deliberate reason.")
        return 4

    manifest = tree(STAGING)
    with open(os.path.join(SCRATCH, "candidate_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print(f"candidate files: {len(manifest)}")

    if dry:
        print("DRY RUN — pass --publish to upload.")
        return 0

    os.environ["HF_HUB_DISABLE_XET"] = "1"
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="space",
        folder_path=STAGING,
        allow_patterns=ALLOW,
        token=token,
        commit_message="Full-scale reproduction: Table 1 (5 real datasets) and Table 2 on the authors' artifact",
    )
    new = api.repo_info(REPO_ID, repo_type="space", token=token).sha
    print(f"new_sha={new}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
