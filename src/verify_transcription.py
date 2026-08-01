"""Re-derive `published.py` from the paper text and fail if they disagree.

Claims 4 and 5 are adjudicated partly on the paper's own printed numbers -- the
claimed GLCP band is contradicted by TISSUE at 13.5%, and "largest gains at
n=30" is contradicted by the CQR row peaking at n=100. Both conclusions are only
as good as the transcription they rest on, so the transcription is not trusted:
this parses the archived table text and compares every Std cell and every
percentage annotation.

Run: python src/verify_transcription.py   (exits nonzero on any mismatch)
"""

import os
import re
import sys

import published as P

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   ".openresearch", "artifacts", "published_tables_source.txt")
DATASETS = ["CRIME", "BIO", "STAR", "DERMA", "TISSUE"]
HEADER = r"base SDCP PPI ours ours-sel oracle DP"


def std_block(text):
    """The Std block only.

    Both captions mention the words "Std", "Marginal" and "Size", so the blocks
    are located after the column header rather than by first occurrence.
    """
    head = re.search(HEADER, text)
    rest = text[head.end():]
    i = rest.index(" Std ") + len(" Std ")
    ends = [k for k in (rest.find(" Marginal ", i), rest.find(" Size ", i)) if k > 0]
    return rest[i:min(ends)]


def normalize(s):
    """Collapse ar5iv's double rendering of every numeric cell.

    Each cell appears once as readable text and once as the LaTeX that produced
    it. Dropping the LaTeX and then de-duplicating adjacent equal numbers leaves
    one token per cell, with the percentage annotation still attached.
    """
    s = s.replace("\u200b", " ")
    s = re.sub(r"\\[a-zA-Z]+\{[^{}]*\}(\^\{[^{}]*\})?(\\,\([^()]*\))?", " ", s)
    s = re.sub(r"\\[,%]", " ", s)
    s = re.sub(r"\(\s*([\d.]+)\s*%\s*\)", r"(\1%)", s)
    s = re.sub(r"\s+", " ", s)
    return re.sub(r"\b(\d+\.\d+) \1\b", r"\1", s)


def cells(s):
    """[(value, pct or None)] -- the superscript mark may sit between the two."""
    return [(float(v), float(p) if p else None)
            for v, p in re.findall(r"(\d+\.\d+)\s*[+\-]?\s*(?:\((\d+\.\d)%\))?", normalize(s))]


def compare(label, got, want, fails):
    if len(got) != 7:
        fails.append(f"{label}: parsed {len(got)} cells, expected 7")
        return
    for k, (value, _) in enumerate(got):
        if abs(value - want["std"][k]) > 1e-9:
            fails.append(f"{label} std[{k}]: paper {value} != published.py {want['std'][k]}")
    for slot, name in ((3, "ours"), (4, "ours-sel")):
        pct = got[slot][1]
        if pct is None:
            fails.append(f"{label}: no percentage annotation on the {name} cell")
        elif abs(pct - want["pct"][name]) > 1e-9:
            fails.append(f"{label} pct[{name}]: paper {pct} != published.py {want['pct'][name]}")


def check_table1(text, fails):
    block = std_block(text.split("=====")[0])
    seen = set()
    for row in re.split(r"(?=(?:%s) 30)" % "|".join(DATASETS), block)[1:]:
        ds = row.split()[0]
        seen.add(ds)
        # One row carries the GLCP-type half then the CQR-type half, unlabelled.
        got = cells(re.sub(r"^\S+ \d+ / \d+ \d+/\d+", "", row))
        compare(f"Table 1 {ds}/GLCP", got[:7], P.TABLE1[ds]["GLCP"], fails)
        compare(f"Table 1 {ds}/CQR", got[7:14], P.TABLE1[ds]["CQR"], fails)
    missing = set(DATASETS) - seen
    if missing:
        fails.append(f"Table 1: rows not found in the paper text: {sorted(missing)}")


def check_table2(text, fails):
    block = std_block(text.split("=====")[1])
    seen = set()
    for row in re.split(r"(?=LogAbs \d+)", block)[1:]:
        n = int(row.split()[1])
        seen.add(n)
        glcp, cqr = row.split("CQR")[0], row.split("CQR")[1]
        compare(f"Table 2 n={n}/GLCP", cells(glcp), P.TABLE2[n]["GLCP"], fails)
        compare(f"Table 2 n={n}/CQR", cells(cqr), P.TABLE2[n]["CQR"], fails)
    missing = set(P.TABLE2) - seen
    if missing:
        fails.append(f"Table 2: rows not found in the paper text: {sorted(missing)}")


def findings():
    """The two paper-internal facts the claims are adjudicated against."""
    glcp = {ds: P.TABLE1[ds]["GLCP"]["pct"]["ours"] for ds in DATASETS}
    cqr = {ds: P.TABLE1[ds]["CQR"]["pct"]["ours"] for ds in DATASETS}
    t2 = {n: P.TABLE2[n]["CQR"]["pct"]["ours"] for n in sorted(P.TABLE2)}
    lo, hi = P.CLAIM4_GLCP_BAND
    qlo, qhi = P.CLAIM4_CQR_BAND
    slack = 0.5
    return {
        "table1_glcp_pct": glcp,
        "table1_cqr_pct": cqr,
        "glcp_cells_below_claimed_floor": {k: v for k, v in glcp.items() if v < lo - slack},
        "glcp_cells_above_claimed_ceiling": {k: v for k, v in glcp.items() if v > hi + slack},
        "cqr_cells_outside_claimed_band": {k: v for k, v in cqr.items()
                                           if v < qlo - slack or v > qhi + slack},
        "table2_cqr_pct_by_n": t2,
        "table2_cqr_argmax_n": max(t2, key=t2.get),
    }


def main():
    if not os.path.exists(SRC):
        print(f"FAIL: archived paper text missing at {SRC}")
        return 1
    text = open(SRC, encoding="utf-8").read()
    fails = []
    check_table1(text, fails)
    check_table2(text, fails)
    f = findings()

    print("Transcription audit of published.py against the archived paper text\n")
    print(f"Table 1 GLCP `ours` reductions : {f['table1_glcp_pct']}")
    print(f"  claimed band {list(P.CLAIM4_GLCP_BAND)} (+/- 0.5 endpoint rounding)")
    print(f"  below floor  : {f['glcp_cells_below_claimed_floor'] or 'none'}")
    print(f"  above ceiling: {f['glcp_cells_above_claimed_ceiling'] or 'none'}")
    print(f"Table 1 CQR  `ours` reductions : {f['table1_cqr_pct']}")
    print(f"  claimed band {list(P.CLAIM4_CQR_BAND)}, outside: "
          f"{f['cqr_cells_outside_claimed_band'] or 'none'}")
    print(f"Table 2 CQR by n: {f['table2_cqr_pct_by_n']} -> argmax n={f['table2_cqr_argmax_n']}")

    if fails:
        print(f"\nFAIL: {len(fails)} mismatch(es) between published.py and the paper text")
        for line in fails:
            print("  -", line)
        return 1
    print("\nOK: every Table 1 and Table 2 Std cell and percentage in published.py "
          "matches the paper text.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
