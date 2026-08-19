#!/usr/bin/env python3
"""Verify the published branch, history, dossier, and claim boundary."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = "MachineLearning-Nerd/icml26-stable-localized-conformal-prediction-via-transduction"
EXPECTED_URL = f"https://github.com/{REPOSITORY}"
EXPECTED_IDENTITY = "MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>"
OVERALL = "PARTIAL_CLAIMS_1_TO_2_VERIFIED_SCOPED_CLAIM_3_BLOCKED_CLAIM_4_SOURCE_BAND_FALSIFIED_CLAIM_5_GLCP_SCOPED_VERIFIED_CLAIM_6_REPAIRED_PROOF_SCOPED_VERIFIED"
BOUNDARY = "C3_INCOMPLETE_CALIBRATION_SIZES_C4_SOURCE_TABLE_CONTRADICTION_C6_PRINTED_ENDPOINT_REPAIRED_NO_FULL_PAPER_REPRODUCTION"
CLAIM_STATUSES = {
    "C1": "verified_scoped",
    "C2": "verified_scoped_proof_algebra",
    "C3": "blocked_empirical_rate_evidence",
    "C4": "falsified_source_table_band",
    "C5": "verified_scoped_glcp_contract",
    "C6": "verified_scoped_repaired_proof_audit",
}
ANALYSIS_VERDICTS = {"C1": "VERIFIED", "C2": "VERIFIED", "C3": "BLOCKED", "C4": "FALSIFIED", "C5": "VERIFIED", "C6": "VERIFIED"}


def git(*args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=not binary)
    return result.stdout if binary else result.stdout.strip()


def fail(message: str) -> None:
    print(f"FINAL_AUDIT=FAILED reason={message}", file=sys.stderr)
    raise SystemExit(1)


def expected_branches() -> set[str]:
    branches = {"main", "check/exchangeability", "check/invariants"}
    branches.update(f"control/noshift-s{i}" for i in range(5))
    branches.update(f"control/noshift-w{i}" for i in range(10))
    branches.update({"control/noshift-w7-metadata", "control/noshift-w8-metadata", "control/noshift-w9-metadata"})

    datasets = ("bio", "crime", "derma", "star", "tissue")
    for dataset in datasets:
        branches.add(f"real/{dataset}")
        branches.update(f"real/{dataset}-s{i}" for i in range(5))
        for i in range(10):
            if dataset == "crime" and i in (4, 5):
                continue
            branches.add(f"real/{dataset}-w{i}")

    settings = (
        ("30", "30", range(10)),
        ("30", "100", range(10)),
        ("30", "500", range(4, 10)),
        ("100", "500", range(10)),
        ("500", "500", range(10)),
    )
    for n, m, workers in settings:
        prefix = f"sim/logabs-n{n}-m{m}"
        branches.update(f"{prefix}-s{i}" for i in range(5))
        branches.update(f"{prefix}-w{i}" for i in workers)
    if len(branches) != 170:
        fail(f"expected_branch_generator_returned_{len(branches)}")
    return branches


def remote_branch_names() -> set[str]:
    output = git("ls-remote", "--heads", "origin")
    return {line.split("\t", 1)[1].removeprefix("refs/heads/") for line in output.splitlines() if line.strip()}


def local_branch_names() -> set[str]:
    output = git("for-each-ref", "--format=%(refname:strip=2)", "refs/heads")
    return {line for line in output.splitlines() if line}


def ref_for_branch(branch: str) -> str:
    for ref in (branch, f"origin/{branch}", f"refs/remotes/origin/{branch}"):
        result = subprocess.run(["git", "show-ref", "--verify", "--quiet", f"refs/heads/{ref}"], cwd=ROOT)
        if result.returncode == 0:
            return ref
        result = subprocess.run(["git", "show-ref", "--verify", "--quiet", f"refs/remotes/origin/{branch}"], cwd=ROOT)
        if result.returncode == 0:
            return f"origin/{branch}"
    fail(f"missing_branch_ref_{branch.replace('/', '_')}")
    raise AssertionError


def blob_sha256(branch: str, path: str) -> str:
    ref = ref_for_branch(branch)
    payload = git("show", f"{ref}:{path}", binary=True)
    assert isinstance(payload, bytes)
    return hashlib.sha256(payload).hexdigest()


def load_json(path: str) -> dict:
    try:
        return json.loads((ROOT / path).read_text())
    except Exception as exc:  # pragma: no cover - failure is rendered below
        fail(f"invalid_json_{path.replace('/', '_')}_{type(exc).__name__}")
        raise AssertionError


def main() -> None:
    expected = expected_branches()
    remote = remote_branch_names()
    if remote != expected:
        fail(f"remote_branches_expected_{len(expected)}_observed_{len(remote)}")
    if local_branch_names() - expected or "main" not in local_branch_names():
        fail("local_branch_refs_are_not_a_subset_with_main")
    symref = git("ls-remote", "--symref", "origin", "HEAD")
    if "refs/heads/main\tHEAD" not in symref:
        fail("remote_default_branch_is_not_main")

    remote_url = git("remote", "get-url", "origin").removesuffix(".git")
    if remote_url != EXPECTED_URL:
        fail("origin_url_mismatch")
    if git("status", "--porcelain"):
        fail("working_tree_not_clean")
    if git("for-each-ref", "--format=%(refname)", "refs/original"):
        fail("rewrite_backup_refs_present")

    identities = set(git("log", "--all", "--format=%an <%ae> | %cn <%ce>").splitlines())
    if identities != {EXPECTED_IDENTITY + " | " + EXPECTED_IDENTITY}:
        fail("non_canonical_commit_identity_present")
    commit_count = int(git("rev-list", "--all", "--count"))

    required_files = [
        "README.md",
        "branch-audit.md",
        "CLAIM_EVIDENCE.md",
        "SOURCE_AUDIT.md",
        "ENVIRONMENT.md",
        "REPORT.md",
        "CITATION.cff",
        "AUTHOR_THANK_YOU.md",
        "STATUS.md",
        "claims.json",
        "reproduction_verdicts.json",
        "AUTONOMOUS_STATE.json",
        "EVIDENCE_MANIFEST.json",
        "verify_final.py",
        "results/EVAL.md",
        "results/analysis.json",
        "results/compute_provenance.json",
        ".openresearch/artifacts/claim_contract.json",
        ".openresearch/artifacts/source_audit.md",
        "src/verify_transcription.py",
        "src/verify_thm42_certificate.py",
        "src/verify_thm46_certificate.py",
        "src/verify_thm47_certificate.py",
        "src/verify_claim4_band.py",
        "config/node.json",
    ]
    for path in required_files:
        if not (ROOT / path).is_file():
            fail(f"missing_required_file_{path.replace('/', '_')}")

    state = load_json("AUTONOMOUS_STATE.json")
    if state["repository"] != REPOSITORY or state["branch_layout"]["expected_count"] != 170:
        fail("autonomous_state_repository_or_branch_count_mismatch")
    if state["audit"]["overall_verdict"] != OVERALL or state["audit"]["publication_boundary"] != BOUNDARY:
        fail("autonomous_state_boundary_mismatch")
    if state["audit"]["publication_allowed"] or state["audit"]["score_claim"] or state["audit"]["official_author_endorsement"]:
        fail("autonomous_state_publication_flags_must_be_false")
    if state["audit"]["self_scored_points"] != 10 or state["audit"]["self_scored_total"] != 12:
        fail("autonomous_state_self_score_mismatch")
    if state["attribution"]["canonical_identity"] != EXPECTED_IDENTITY:
        fail("autonomous_state_attribution_mismatch")
    if state["history"]["expected_reachable_commits"] != commit_count:
        fail("autonomous_state_commit_count_mismatch")
    if state["history"]["published_main_parent"] != git("rev-parse", "main^"):
        fail("autonomous_state_main_parent_mismatch")

    claims = load_json("claims.json")
    verdicts = load_json("reproduction_verdicts.json")
    if claims["overall_verdict"] != OVERALL or verdicts["overall_verdict"] != OVERALL:
        fail("machine_readable_overall_verdict_mismatch")
    if claims["publication_boundary"] != BOUNDARY or verdicts["publication_boundary"] != BOUNDARY:
        fail("machine_readable_boundary_mismatch")
    for record in (claims, verdicts):
        if record["publication_allowed"] or record["score_claim"] or record["official_author_endorsement"]:
            fail("machine_readable_publication_flags_must_be_false")
        if record["self_scored_points"] != 10 or record["self_scored_total"] != 12:
            fail("machine_readable_self_score_mismatch")
    claim_records = {item["id"]: item["status"] for item in claims["claims"]}
    verdict_records = {key: value["status"] for key, value in verdicts["claims"].items()}
    if claim_records != CLAIM_STATUSES or verdict_records != CLAIM_STATUSES:
        fail("machine_readable_claim_status_mismatch")

    analysis = load_json("results/analysis.json")
    observed_analysis = {key: value["verdict"] for key, value in analysis["verdicts"].items()}
    if observed_analysis != ANALYSIS_VERDICTS or analysis["self_scored_points"] != 10:
        fail("analysis_verdict_or_score_mismatch")
    readme = (ROOT / "README.md").read_text()
    status = (ROOT / "STATUS.md").read_text()
    for marker in (OVERALL, BOUNDARY, "MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>"):
        if marker not in readme or marker not in status:
            fail("readme_or_status_marker_missing")

    manifest = load_json("EVIDENCE_MANIFEST.json")
    if manifest["repository"] != REPOSITORY or manifest["overall_verdict"] != OVERALL:
        fail("evidence_manifest_header_mismatch")
    entries = manifest["files"] + manifest["branch_evidence"]
    for entry in entries:
        if blob_sha256(entry["branch"], entry["path"]) != entry["sha256"]:
            fail(f"evidence_hash_mismatch_{entry['branch'].replace('/', '_')}_{entry['path'].replace('/', '_')}")

    print(
        "FINAL_AUDIT=VERIFIED "
        f"branches={len(remote)} commits={commit_count} "
        "claims=C1:verified_scoped,C2:verified_scoped_proof_algebra,"
        "C3:blocked_empirical_rate_evidence,C4:falsified_source_table_band,"
        "C5:verified_scoped_glcp_contract,C6:verified_scoped_repaired_proof_audit "
        "self_scored=10/12 publication_allowed=false"
    )


if __name__ == "__main__":
    main()
