"""Adjudicate all six claims against the committed results. Exits nonzero on failure.

Reads only files under `results/`, which are the verbatim outputs the compute
nodes printed. Every number reported on the candidate pages comes from here.
"""

import glob
import json
import os
import sys

import numpy as np

import published as P
import shard_reduce

MODELS = ["GLCP", "CQR"]
RNG = np.random.default_rng(20260801)
BOOT = 2000


def _root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- simulation


def _merge_setting(setting_dir):
    parts = {}
    for path in sorted(glob.glob(os.path.join(setting_dir, "*.json"))):
        for key, val in _load_json(path).items():
            if key != "selected_idx":
                parts.setdefault(key, []).append(val)
    return parts


def _sim_table(parts, n, alpha=0.1):
    """Rebuild the authors' aggregate arrays, then pick the `ours` row as they do."""
    res = {}
    n_rep = None
    for key, shards in parts.items():
        s = shard_reduce.summarise(shards, alpha=alpha)
        n_rep = s["n_repeats"]
        res[key] = [np.asarray(s[m]) for m in ("marginal", "size", "std", "cond_miscoverage")]
    # `ours` = smallest Std among lambdas whose marginal coverage is acceptable.
    mar, std = np.asarray(res["StCP"][0]), np.asarray(res["StCP"][2])
    lo, up = 0.9 - 0.01, 0.9 + 1.0 / (n + 1)
    out = {"n_repeats": n_rep, "models": {}}
    for mi, model in enumerate(MODELS):
        mask = (mar[mi] >= lo) & (mar[mi] <= up)
        idx = int(np.where(mask)[0][np.argmin(std[mi][mask])]) if mask.any() else int(
            np.argmin(np.abs(mar[mi] - 0.9))
        )
        row = {}
        for key, label in [("base", "base"), ("SDCP", "SDCP"), ("PPI", "PPI"),
                           ("StCP-sel", "ours-sel"), ("oracle", "oracle"), ("NOAL", "DP")]:
            row[label] = {
                "marginal": float(res[key][0][mi]),
                "size": float(res[key][1][mi]),
                "std": float(res[key][2][mi]),
            }
        row["ours"] = {
            "marginal": float(mar[mi][idx]),
            "size": float(res["StCP"][1][mi][idx]),
            "std": float(std[mi][idx]),
            "lambda_index": idx,
        }
        base_std = row["base"]["std"]
        for lab in ("ours", "ours-sel"):
            row[lab]["std_improvement_pct"] = (base_std - row[lab]["std"]) / base_std * 100.0
        out["models"][model] = {
            "row": row,
            "lambda_curve": {
                "marginal": np.asarray(mar[mi]).tolist(),
                "std": np.asarray(std[mi]).tolist(),
                "size": np.asarray(res["StCP"][1][mi]).tolist(),
            },
            "selected_lambda_index": idx,
        }
    return out


def _boot_pct_sim(parts, model_idx, lam_idx, n_boot=BOOT):
    """Bootstrap the Table 2 percentage over repeats (base and ours move together)."""
    base = np.concatenate(
        [np.asarray(p["size_mean_per_repeat"])[model_idx] for p in parts["base"]], axis=-1
    )
    stcp = np.concatenate(
        [np.asarray(p["size_mean_per_repeat"])[model_idx][lam_idx] for p in parts["StCP"]], axis=-1
    )
    r = len(base)
    pcts = []
    for _ in range(n_boot):
        idx = RNG.integers(0, r, r)
        b, s = base[idx].std(), stcp[idx].std()
        if b > 1e-12:
            pcts.append((b - s) / b * 100.0)
    pcts = np.array(pcts)
    return {
        "bootstrap_mean_pct": float(pcts.mean()),
        "ci95_pct": [float(np.percentile(pcts, 2.5)), float(np.percentile(pcts, 97.5))],
        "n_repeats": int(r),
        "n_boot": int(n_boot),
    }


# --------------------------------------------------------------------- claims


def claim1(results):
    inv = results.get("invariants")
    if not inv:
        return {"verdict": "BLOCKED", "reason": "invariants node produced no output"}
    l0, linf = inv["lambda0_identity"], inv["lambda_to_infinity"]
    checks = {
        "cdf_estimator_sees_source_only": inv["provenance"]["F_hat_S_given_X_trained_on"]["uses_target_labels"] is False,
        "lambda0_recovers_conformal_quantile": l0["max_gap_over_score_spacing"] <= 1.0,
        "lambda_to_infinity_shrinks_theta": (
            linf["fraction_non_increasing_steps"] >= 0.9 and linf["shrinkage_ratio"] < 1.0
        ),
    }
    return {
        "verdict": "VERIFIED" if all(checks.values()) else "FALSIFIED",
        "checks": checks,
        "evidence": {"lambda0": l0, "lambda_inf": linf, "provenance": inv["provenance"]},
    }


def claim2(results):
    """Thm 4.2: coverage stays valid across the whole lambda grid; DP does not."""
    per_setting, worst = {}, 0.0
    lo, up = P.TABLE_ANNOTATION_BAND
    for name, tab in results["sim"].items():
        for model in MODELS:
            cur = tab["models"][model]["lambda_curve"]["marginal"]
            dev = float(np.max(np.abs(np.array(cur) - 0.9)))
            inband = bool(np.all((np.array(cur) >= lo) & (np.array(cur) <= up)))
            per_setting[f"{name}/{model}"] = {
                "max_abs_deviation_over_lambda": dev,
                "all_lambda_in_annotation_band": inband,
                "n_lambda": len(cur),
            }
            worst = max(worst, dev)

    dp = {}
    for name, tab in results["sim"].items():
        for model in MODELS:
            v = tab["models"][model]["row"]["DP"]["marginal"]
            dp[f"{name}/{model}"] = {"marginal": v, "in_band": bool(lo <= v <= up)}
    for ds, tab in results["real"].items():
        for model in MODELS:
            v = tab["summary"][model]["DP"]["marginal"]
            dp[f"{ds}/{model}"] = {"marginal": v, "in_band": bool(lo <= v <= up)}

    dp_out = [k for k, v in dp.items() if not v["in_band"]]
    checks = {
        "stcp_coverage_valid_across_full_lambda_grid": all(
            v["all_lambda_in_annotation_band"] for v in per_setting.values()
        ),
        "negative_control_DP_leaves_band": len(dp_out) >= max(1, len(dp) // 2),
    }
    return {
        "verdict": "VERIFIED" if all(checks.values()) else "FALSIFIED",
        "checks": checks,
        "worst_deviation_over_all_lambda": worst,
        "per_setting": per_setting,
        "negative_control_DP": {"per_setting": dp, "out_of_band": dp_out},
    }


def claim3(results):
    """Thm 4.6: base variance ~ n^-1; StCP gains from lambda and from m >> n."""
    ns, base_var = [], []
    for n in (30, 100, 500):
        key = f"logabs-n{n}-m500"
        if key not in results["sim"]:
            continue
        ns.append(n)
        base_var.append(results["sim"][key]["models"]["GLCP"]["row"]["base"]["std"] ** 2)
    slope = ci = None
    if len(ns) >= 3:
        slope = float(np.polyfit(np.log(ns), np.log(base_var), 1)[0])
        boots = []
        for _ in range(500):
            jitter = np.array(base_var) * np.exp(RNG.normal(0, 1 / np.sqrt(2 * 49), len(ns)))
            boots.append(np.polyfit(np.log(ns), np.log(jitter), 1)[0])
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    lam_mono = {}
    for name, tab in results["sim"].items():
        for model in MODELS:
            curve = np.array(tab["models"][model]["lambda_curve"]["std"])
            mar = np.array(tab["models"][model]["lambda_curve"]["marginal"])
            lo, up = P.TABLE_ANNOTATION_BAND
            valid = (mar >= lo) & (mar <= up)
            v = curve[valid]
            lam_mono[f"{name}/{model}"] = {
                "std_at_smallest_valid_lambda": float(v[0]) if len(v) else None,
                "std_at_largest_valid_lambda": float(v[-1]) if len(v) else None,
                "decreases": bool(len(v) >= 2 and v[-1] < v[0]),
                "n_valid_lambda": int(valid.sum()),
            }

    m_trend = {}
    for m in (30, 100, 500):
        key = f"logabs-n30-m{m}"
        if key in results["sim"]:
            m_trend[m] = {
                model: results["sim"][key]["models"][model]["row"]["ours"]["std"]
                for model in MODELS
            }
    ms = sorted(m_trend)
    m_decreasing = {
        model: bool(all(m_trend[ms[i + 1]][model] <= m_trend[ms[i]][model] + 1e-9 for i in range(len(ms) - 1)))
        for model in MODELS
    } if len(ms) >= 2 else {}

    checks = {
        "base_variance_slope_consistent_with_minus_one": bool(ci and ci[0] <= -1.0 <= ci[1]),
        "stcp_std_decreases_with_lambda_in_valid_region": (
            sum(v["decreases"] for v in lam_mono.values()) >= 0.75 * len(lam_mono)
        ),
        "stcp_std_decreases_with_m_at_fixed_n": bool(m_decreasing and all(m_decreasing.values())),
    }
    return {
        "verdict": "VERIFIED" if all(checks.values()) else "FALSIFIED",
        "checks": checks,
        "base_variance_vs_n": {"n": ns, "variance": base_var, "loglog_slope": slope, "slope_ci95": ci},
        "std_vs_lambda": lam_mono,
        "ours_std_vs_m_at_n30": {"by_m": m_trend, "monotone_decreasing": m_decreasing},
    }


def claim4(results):
    """Table 1 across five real datasets, cell by cell against the published values."""
    lo, up = P.TABLE_ANNOTATION_BAND
    per_dataset, glcp_pcts, cqr_pcts = {}, [], []
    for ds, pub in P.TABLE1.items():
        got = results["real"].get(ds)
        if not got:
            per_dataset[ds] = {"status": "MISSING"}
            continue
        entry = {"n_over_m": pub["n_over_m"], "models": {}}
        for model in MODELS:
            g, p = got["summary"][model], pub[model]
            row = {}
            for i, meth in enumerate(P.METHODS):
                row[meth] = {
                    "std_reproduced": g[meth]["std"],
                    "std_published": p["std"][i],
                    "marginal_reproduced": g[meth]["marginal"],
                    "marginal_published": p["marginal"][i] if "marginal" in p else None,
                }
            for lab in ("ours", "ours-sel"):
                row[lab]["pct_reproduced"] = g[lab]["std_improvement_pct"]
                row[lab]["pct_published"] = p["pct"][lab]
            (glcp_pcts if model == "GLCP" else cqr_pcts).append(row["ours"]["pct_reproduced"])
            entry["models"][model] = {
                "rows": row,
                "reference": g["_reference"],
                "ours_beats_reference": g["ours"]["std"] < g["_reference"]["a_ref_std"],
                "ours_marginal_in_band": lo <= g["ours"]["marginal"] <= up,
                "ours_sel_marginal_in_band": lo <= g["ours-sel"]["marginal"] <= up,
            }
        per_dataset[ds] = entry

    ran_all = all(v.get("models") for v in per_dataset.values())
    beats = [
        v["models"][m]["ours_beats_reference"]
        for v in per_dataset.values() if v.get("models") for m in MODELS
    ]
    glcp_rng = [min(glcp_pcts), max(glcp_pcts)] if glcp_pcts else None
    cqr_rng = [min(cqr_pcts), max(cqr_pcts)] if cqr_pcts else None

    def band_holds(rng, band):
        return bool(rng and band[0] - 1e-9 <= rng[0] and rng[1] <= band[1] + 1e-9)

    checks = {
        "all_five_datasets_ran": ran_all,
        "ours_beats_reference_baseline_everywhere": bool(beats and all(beats)),
        "marginal_coverage_near_nominal": all(
            v["models"][m]["ours_marginal_in_band"]
            for v in per_dataset.values() if v.get("models") for m in MODELS
        ),
    }
    return {
        "verdict": "VERIFIED" if all(checks.values()) else "FALSIFIED",
        "checks": checks,
        "reproduced_glcp_pct_range": glcp_rng,
        "reproduced_cqr_pct_range": cqr_rng,
        "claimed_glcp_band": list(P.CLAIM4_GLCP_BAND),
        "claimed_cqr_band": list(P.CLAIM4_CQR_BAND),
        "claimed_glcp_band_covers_all_reproduced_cells": band_holds(glcp_rng, P.CLAIM4_GLCP_BAND),
        "claimed_cqr_band_covers_all_reproduced_cells": band_holds(cqr_rng, P.CLAIM4_CQR_BAND),
        "published_glcp_pct_range": [
            min(P.TABLE1[d][m]["pct"]["ours"] for d in P.TABLE1 for m in ["GLCP"]),
            max(P.TABLE1[d][m]["pct"]["ours"] for d in P.TABLE1 for m in ["GLCP"]),
        ],
        "per_dataset": per_dataset,
    }


def claim5(results):
    """Table 2: 31.2% (GLCP) and 16.3% (CQR) at n=30, and the ordering across n."""
    by_n, boots = {}, {}
    for n in (30, 100, 500):
        key = f"logabs-n{n}-m500"
        if key not in results["sim"]:
            continue
        by_n[n] = {
            model: {
                "std_base": results["sim"][key]["models"][model]["row"]["base"]["std"],
                "std_ours": results["sim"][key]["models"][model]["row"]["ours"]["std"],
                "pct": results["sim"][key]["models"][model]["row"]["ours"]["std_improvement_pct"],
                "pct_published": P.TABLE2[n][model]["pct"]["ours"],
            }
            for model in MODELS
        }
    if 30 in by_n:
        parts = results["_sim_parts"]["logabs-n30-m500"]
        for mi, model in enumerate(MODELS):
            lam = results["sim"]["logabs-n30-m500"]["models"][model]["selected_lambda_index"]
            boots[model] = _boot_pct_sim(parts, mi, lam)

    target_hit = {}
    for model in MODELS:
        if model in boots:
            t = P.CLAIM5_TARGETS[model]
            ci = boots[model]["ci95_pct"]
            target_hit[model] = bool(ci[0] <= t <= ci[1])

    largest_at_30 = {
        model: bool(by_n and by_n[30][model]["pct"] >= max(by_n[n][model]["pct"] for n in by_n))
        for model in MODELS
    } if 30 in by_n else {}

    noshift = results.get("noshift")
    control = None
    if noshift and 30 in by_n:
        control = {
            model: {
                "pct_with_shift": by_n[30][model]["pct"],
                "pct_no_shift": noshift["models"][model]["row"]["ours"]["std_improvement_pct"],
            }
            for model in MODELS
        }
        control["gain_shrinks_without_shift"] = all(
            control[m]["pct_no_shift"] < control[m]["pct_with_shift"] for m in MODELS
        )

    checks = {
        "n30_glcp_matches_31_2_within_ci": target_hit.get("GLCP", False),
        "n30_cqr_matches_16_3_within_ci": target_hit.get("CQR", False),
    }
    return {
        "verdict": "VERIFIED" if all(checks.values()) else "FALSIFIED",
        "checks": checks,
        "by_n": by_n,
        "bootstrap_at_n30": boots,
        "largest_gain_at_n30": largest_at_30,
        "published_cqr_pct_by_n": {n: P.TABLE2[n]["CQR"]["pct"]["ours"] for n in (30, 100, 500)},
        "negative_control_no_shift": control,
    }


def claim6(results):
    """Thm 4.7 band, plus the control that decides whether the band means anything."""
    lo, hi = P.THM47_BAND
    obs = {}
    for name, tab in results["sim"].items():
        for model in MODELS:
            v = tab["models"][model]["row"]["ours-sel"]["marginal"]
            obs[f"sim:{name}/{model}"] = {"coverage": v, "in_band": bool(lo <= v < hi)}
    for ds, tab in results["real"].items():
        for model in MODELS:
            v = tab["summary"][model]["ours-sel"]["marginal"]
            obs[f"real:{ds}/{model}"] = {"coverage": v, "in_band": bool(lo <= v < hi)}

    ctrl = results.get("control_exchangeability")
    informative = bool(ctrl and ctrl["control_is_informative"])
    checks = {
        "all_settings_inside_band": all(v["in_band"] for v in obs.values()),
        "control_makes_the_band_informative": informative,
    }
    verdict = "VERIFIED" if all(checks.values()) else ("BLOCKED" if not informative else "FALSIFIED")
    return {
        "verdict": verdict,
        "checks": checks,
        "band": [lo, hi],
        "observed": obs,
        "n_settings": len(obs),
        "negative_control": ctrl,
        "note": (
            "The band is 7.2 points wide; without a control that exits it, an in-band "
            "observation would be weak evidence and the claim would stay BLOCKED."
        ),
    }


# ----------------------------------------------------------------------- main


def run(cfg, upstream_root):
    root = _root()
    res = {"sim": {}, "real": {}, "_sim_parts": {}}

    shard_root = os.path.join(root, "results", "shards")
    for d in sorted(glob.glob(os.path.join(shard_root, "*"))):
        name = os.path.basename(d)
        parts = _merge_setting(d)
        if not parts:
            continue
        n = int(name.split("-n")[1].split("-")[0])
        table = _sim_table(parts, n)
        if name.startswith("noshift"):
            res["noshift"] = table
        else:
            res["sim"][name] = table
            res["_sim_parts"][name] = parts

    for path in sorted(glob.glob(os.path.join(root, "results", "real", "*.json"))):
        res["real"][os.path.basename(path)[:-5]] = _load_json(path)

    for name in ("invariants", "control_exchangeability"):
        path = os.path.join(root, "results", "checks", f"{name}.json")
        if os.path.exists(path):
            res[name] = _load_json(path)

    verdicts = {
        "C1": claim1(res), "C2": claim2(res), "C3": claim3(res),
        "C4": claim4(res), "C5": claim5(res), "C6": claim6(res),
    }
    full = {v["verdict"] in ("VERIFIED", "FALSIFIED") for v in verdicts.values()}
    points = sum(2 if v["verdict"] in ("VERIFIED", "FALSIFIED") else 0 for v in verdicts.values())
    failed = [k for k, v in verdicts.items() if v["verdict"] not in ("VERIFIED", "FALSIFIED")]

    out = {
        "kind": "analysis",
        "settings_merged": sorted(res["sim"]),
        "datasets": sorted(res["real"]),
        "verdicts": verdicts,
        "self_scored_points": points,
        "not_full_credit": failed,
        "exit_code": 0 if not failed else 1,
    }
    with open(os.path.join(root, "results", "analysis.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out
