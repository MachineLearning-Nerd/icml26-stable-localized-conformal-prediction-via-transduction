"""Traverse a candidate Space the way an evaluator would and report what is missing.

The rule this enforces: anything not reachable by following links from the
canonical entrypoint earns nothing, however good it is. So this starts at
`pages/index.md`, follows only links it finds, and refuses to use knowledge of
the repository layout to fill a gap. Whatever it cannot reach, an evaluator
cannot reach either.

Usage: redteam.py <candidate-dir>          (exits nonzero if any claim row fails)
"""

import json
import os
import re
import sys

CLAIMS = ["C1", "C2", "C3", "C4", "C5", "C6"]

# Each requirement is (label, predicate over the reachable text of a claim page).
# They are deliberately literal: the question is whether an evaluator reading
# only this page can find the thing, not whether it exists somewhere.
REQUIRED = [
    ("exact claim + source quantifiers",
     lambda t, links: "Exact claim under test" in t and "SHA-256" in t),
    ("assumptions audited numerically",
     lambda t, links: bool(re.search(r"Evidence integrity|integrity", t))),
    ("executable source code linked",
     lambda t, links: any(l.endswith(".py") for l in links)),
    ("fixed command + pinned environment",
     lambda t, links: "run.sh" in t or "run_node.py" in t),
    ("raw numbers inline",
     lambda t, links: t.count("|") > 40),
    ("downloadable raw JSON",
     lambda t, links: any(l.endswith(".json") for l in links)),
    ("independent checker named",
     lambda t, links: "stage_analysis.py" in t or "verify_transcription.py" in t),
    ("negative control reported",
     lambda t, links: bool(re.search(r"[Nn]egative control|control", t))),
    ("verifier exits nonzero on failure",
     lambda t, links: "nonzero" in t),
    ("git sha / seeds / compute",
     lambda t, links: "git_sha" in t or "Git SHA" in t or "seed" in t.lower()),
]


def read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def page_path(root, slug):
    for cand in (os.path.join(root, "pages", slug, "page.md"),
                 os.path.join(root, "pages", slug + ".md")):
        if os.path.exists(cand):
            return cand
    return None


def links_in(text):
    """(page slugs, file paths) referenced by a markdown page.

    Backticked `repro/...` paths count as file references too. They are not
    clickable, but they tell the evaluator where to look, and one that points at
    nothing is the same failure as a dead link -- evidence that is named but not
    reachable. Only `repro/` is checked: other backticked paths refer to the
    upstream artifact or to node branches, which are not part of this tree.
    """
    slugs = set(re.findall(r"\]\(#/([A-Za-z0-9\-_]+)\)", text))
    files = set(re.findall(r"\]\(((?!#|https?:)[^)\s]+)\)", text))
    files |= {p.rstrip(".,;:") for p in re.findall(r"`(repro/[^`\s]+)`", text)}
    return slugs, files


def traverse(root):
    """Breadth-first from the entrypoint. Returns reached pages and dead links."""
    start = os.path.join(root, "pages", "index.md")
    if not os.path.exists(start):
        return None, [("<entrypoint>", "pages/index.md")], {}
    reached, dead, texts = {}, [], {}
    queue = [("index", start)]
    seen = {"index"}
    while queue:
        slug, path = queue.pop(0)
        text = read(path)
        reached[slug] = path
        texts[slug] = text
        slugs, files = links_in(text)
        for f in files:
            if not os.path.exists(os.path.join(root, f)):
                dead.append((slug, f))
        for s in slugs:
            if s in seen:
                continue
            p = page_path(root, s)
            if p is None:
                dead.append((slug, f"#/{s}"))
                continue
            seen.add(s)
            queue.append((s, p))
    return reached, dead, texts


def claim_slug(texts, claim):
    """Find the claim's canonical page by how the index names it."""
    n = claim[1]
    for slug, text in texts.items():
        if slug == "index":
            continue
        if re.search(rf"^#\s*Claim {n}\b", text, re.M):
            return slug
    return None


HISTORICAL_LABEL = "Historical rejected baseline"


def navigation_checks(texts, claim_pages):
    """Reachability is not enough; the index itself has to read correctly.

    A claim page linked only from the visibility matrix is still reachable, but
    an evaluator scanning the index would not see it, and a rejected verifier
    listed above the current one reads as the current one.
    """
    index = texts["index"]
    problems = []
    lines = index.splitlines()
    order = [i for i, ln in enumerate(lines) if "](#/" in ln]

    for claim, slug in claim_pages.items():
        if slug and f"](#/{slug})" not in index:
            problems.append(f"{claim} page `{slug}` is not linked from the index")

    hist = [i for i in order if HISTORICAL_LABEL in lines[i]]
    current = [i for i in order if re.search(r"Current verification", lines[i])]
    if not current:
        problems.append("no 'Current verification' entry in the index")
    elif hist and min(current) > min(hist):
        problems.append("a historical page is listed above the current verification")
    for i in order:
        if re.search(r"[Hh]istorical|[Rr]ejected|[Ss]uperseded|[Oo]ld ", lines[i]) \
                and HISTORICAL_LABEL not in lines[i]:
            problems.append(f"index entry looks historical but is not labelled "
                            f"exactly '{HISTORICAL_LABEL}': {lines[i].strip()}")
    return problems


def main(root):
    reached, dead, texts = traverse(root)
    print(f"Red-team traversal of {root}")
    print(f"Entrypoint: pages/index.md\n")
    if reached is None:
        print("FAIL: no canonical entrypoint at pages/index.md")
        return 1
    print(f"Pages reachable from the entrypoint: {len(reached)}")
    print("  " + ", ".join(sorted(reached)) + "\n")

    orphans = []
    pages_dir = os.path.join(root, "pages")
    for name in sorted(os.listdir(pages_dir)):
        d = os.path.join(pages_dir, name)
        if os.path.isdir(d) and name not in reached:
            orphans.append(name)
    if orphans:
        print(f"Pages on disk but NOT reachable: {orphans}\n")

    rows, failures, claim_pages = [], [], {}
    for c in CLAIMS:
        slug = claim_slug(texts, c)
        claim_pages[c] = slug
        if slug is None:
            rows.append((c, "MISSING", ["no reachable page"]))
            failures.append(c)
            continue
        text = texts[slug]
        _, files = links_in(text)
        missing = [label for label, pred in REQUIRED if not pred(text, files)]
        rows.append((c, slug, missing))
        if missing:
            failures.append(c)

    print("| Claim | Canonical page | Missing from the traversal |")
    print("| --- | --- | --- |")
    for c, slug, missing in rows:
        print(f"| {c} | {slug} | {', '.join(missing) if missing else '—'} |")

    nav = navigation_checks(texts, claim_pages)
    print()
    print("Navigation: " + ("OK" if not nav else f"{len(nav)} problem(s)"))
    for p in nav:
        print("  -", p)

    if dead:
        print(f"\nDead links ({len(dead)}):")
        for src, target in dead[:25]:
            print(f"  {src} -> {target}")

    ok = not failures and not dead and not nav and not orphans
    print()
    print("RELEASE-READY" if ok else
          f"NOT RELEASE-READY: claims with gaps {failures}, dead links {len(dead)}, "
          f"navigation problems {len(nav)}, orphan pages {orphans}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "staging"))
