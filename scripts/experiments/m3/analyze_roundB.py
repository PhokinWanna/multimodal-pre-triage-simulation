"""Round-B results: tables and figures from run_roundB.py (needs matplotlib).

    python scripts/experiments/m3/analyze_roundB.py [--run results/experiments/m3/roundB/run_main]

Writes results/experiments/m3/roundB/analysis/:
  report.md                 all tables in one readable file
  summary.json              every number, with 95 % bootstrap intervals (2,000 resamples over scenarios, seed 42)
  per_case.csv              one row per scenario: reference, round A and round B answers, camera, floor, changes
  fig_confusion.png/.pdf    confusion matrices, round A (fixed) and round B (main system)
  fig_rounds.png/.pdf       round A original / round A fixed / round B: accuracy, under- and over-triage
  fig_recall.png/.pdf       share of each reference level answered exactly, per round
  fig_weights.png/.pdf      independent vs posture:skin 50/50, 60/40, 40/60
  fig_modalities.png/.pdf   text only / + posture / + skin / all three
  fig_contribution.png/.pdf how much each input moves the answer (Shapley share)

Scope (protocol amendment 2026-11-14): reference ESI 1 = out of the kiosk's scope, scored as
"detected and sent to staff"; the main accuracy figures use the in-scope scenarios (ESI 2-5).
All-30 figures are also given so that round A and round B can be compared like for like.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np

import m3_metrics as M
from analyze_m3 import load_labels, setup_matplotlib
from m3_common import LABELS_FINAL, RESULTS_DIR, read_sources, write_csv
from roundb_common import ROUNDB_DIR, RUN_DIR, final_level, rel, utf8_console

BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#a5b4c8"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
WEIGHT_CONFIGS = [("IND", "Independent\n(no weights)"), ("W50", "Posture 50 :\nSkin 50"),
                  ("W60", "Posture 60 :\nSkin 40"), ("W40", "Posture 40 :\nSkin 60")]
MODALITY_CONFIGS = [("T", "Text only"), ("TP", "Text +\nposture"), ("TS", "Text +\nskin"), ("W50", "Text + posture\n+ skin")]
PLAYERS = ("text", "posture", "skin")
COALITION_CONFIG = {(): "NONE", ("text",): "T", ("posture",): "P", ("skin",): "S", ("text", "posture"): "TP",
                    ("text", "skin"): "TS", ("posture", "skin"): "PS", ("text", "posture", "skin"): "W50"}
HEADLINE = [("accuracy", "Exact accuracy", True), ("within_one", "Within one level", True),
            ("kappa_linear", "Weighted kappa (linear)", False), ("under_triage", "Under-triage", True),
            ("over_triage", "Over-triage", True)]


# ----------------------------------------------------------------------------- metrics
def scope_metrics(ref: list[int], pred: list) -> dict:
    """Point values: all scenarios, in-scope (reference ESI 2-5) and out-of-scope (reference ESI 1)."""
    ref = np.asarray(ref, dtype=int)
    p = np.array([int(x) if x is not None else 9 for x in pred])
    out = {"all": M.summary(ref, pred)}
    ins = ref >= 2
    if ins.any():
        s = M.summary(ref[ins], [pred[i] for i in np.where(ins)[0]])
        two, rest = ref[ins] == 2, ref[ins] >= 3
        pi = p[ins]
        s["esi2_sensitivity"] = float((pi[two] <= 2).mean()) if two.any() else float("nan")
        s["esi3to5_specificity"] = float((pi[rest] >= 3).mean()) if rest.any() else float("nan")
        s["esi5_recall"] = float((pi[ref[ins] == 5] == 5).mean()) if (ref[ins] == 5).any() else float("nan")
        out["in_scope"] = s
    one = ref == 1
    if one.any():
        out["out_of_scope"] = {"n": int(one.sum()), "detected_strict": float((p[one] == 1).mean()),
                               "detected_lenient": float((p[one] <= 2).mean())}
    return out


def with_ci(ref: list[int], pred: list) -> dict:
    """scope_metrics + bootstrap intervals for the scalar values."""
    ref_a = np.asarray(ref, dtype=int)
    point = scope_metrics(ref, pred)

    def collect(idx):
        return scope_metrics(list(ref_a[idx]), [pred[i] for i in idx])
    rng = np.random.default_rng(42)
    boots = [collect(rng.integers(0, len(ref), len(ref))) for _ in range(2000)]
    out = {}
    for group, vals in point.items():
        out[group] = {}
        for key, value in vals.items():
            if not isinstance(value, (int, float)) or key in ("n", "invalid_predictions"):
                out[group][key] = value
                continue
            series = [b[group][key] for b in boots if group in b and not np.isnan(b[group].get(key, np.nan))]
            lo, hi = (np.percentile(series, [2.5, 97.5]) if series else (np.nan, np.nan))
            out[group][key] = {"value": float(value), "ci_low": float(lo), "ci_high": float(hi)}
    return out


def cell(v, pct=True) -> str:
    if isinstance(v, dict):
        if v["value"] is None or np.isnan(v["value"]):
            return "n/a"
        f = (lambda x: f"{100 * x:.0f}%") if pct else (lambda x: f"{x + 0.0:.2f}".replace("-0.00", "0.00"))
        return f"{f(v['value'])} ({f(v['ci_low'])}–{f(v['ci_high'])})"
    return "n/a" if v is None else str(v)


# ----------------------------------------------------------------------------- loading
def load_records(run_dir: Path) -> dict:
    by = defaultdict(dict)
    sampled = defaultdict(lambda: defaultdict(list))
    with open(run_dir / "records.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["run"]:
                sampled[r["config"]][r["id"]].append(r)
            else:
                by[r["config"]][r["id"]] = r
    return {"by": by, "sampled": sampled}


def load_round_a(name: str) -> dict:
    path = RESULTS_DIR / f"analysis_{name}" / "per_case.csv"
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def int_or_none(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def llm_only(rec: dict):
    """What the LLM answered on its own (the rules' level when it gave no usable answer)."""
    return rec["esi_llm"] if rec["esi_llm"] is not None else rec["esi_rules"]


# ----------------------------------------------------------------------------- Shapley
def shapley(values: dict) -> dict:
    """Shapley value of each input for one scenario. values: {coalition tuple: urgency}."""
    n = len(PLAYERS)
    phi = {}
    for p in PLAYERS:
        others = [q for q in PLAYERS if q != p]
        total = 0.0
        for k in range(len(others) + 1):
            for subset in combinations(others, k):
                with_p = tuple(q for q in PLAYERS if q in subset or q == p)
                without = tuple(q for q in PLAYERS if q in subset)
                weight = factorial(k) * factorial(n - k - 1) / factorial(n)
                total += weight * (values[with_p] - values[without])
        phi[p] = total
    return phi


def contribution(by: dict, ids: list[str], getter) -> dict:
    per_case = {}
    for i in ids:
        vals = {c: 6 - getter(by[cfg][i]) for c, cfg in COALITION_CONFIG.items()}
        per_case[i] = shapley(vals)
    mean_abs = {p: float(np.mean([abs(per_case[i][p]) for i in ids])) for p in PLAYERS}
    total = sum(mean_abs.values()) or 1.0
    return {"mean_abs_levels": mean_abs, "share": {p: mean_abs[p] / total for p in PLAYERS},
            "mean_signed_levels": {p: float(np.mean([per_case[i][p] for i in ids])) for p in PLAYERS},
            "per_case": per_case}


# ----------------------------------------------------------------------------- figures
def save(fig, path: Path, plt):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, facecolor="white")
    plt.close(fig)


def style(ax, ylabel=None, ymax=100):
    ax.set_ylim(0, ymax * 1.12)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="y", length=0, colors=MUTED)
    ax.tick_params(axis="x", length=0, colors=INK)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)


def grouped_bars(ax, groups: list[str], series: list[tuple[str, str, list[float]]], fmt="{:.0f}"):
    x = np.arange(len(groups))
    width = 0.8 / len(series)
    for k, (label, colour, vals) in enumerate(series):
        pos = x - 0.4 + width * (k + 0.5)
        ax.bar(pos, vals, width * 0.9, color=colour, label=label)
        for xx, v in zip(pos, vals):
            if not np.isnan(v):
                ax.text(xx, v + 1.5, fmt.format(v), ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, groups)


def fig_confusion(panels: list[tuple[str, np.ndarray]], path: Path):
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.0), squeeze=False)
    for ax, (title, cm) in zip(axes[0], panels):
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.max()))
        for i in range(5):
            for j in range(5):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() * 0.55 else INK, fontsize=11)
        ax.set_xticks(range(5), [str(i) for i in range(1, 6)])
        ax.set_yticks(range(5), [str(i) for i in range(1, 6)])
        ax.set_xlabel("ESI level given by the system")
        ax.set_ylabel("Reference ESI level")
        ax.set_title(title, fontsize=10)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
    save(fig, path, plt)


def fig_bars(groups, series, ylabel, title, path: Path, figsize=(6.4, 3.4), note=None):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=figsize)
    grouped_bars(ax, groups, series)
    style(ax, ylabel)
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(frameon=False, ncol=len(series), loc="upper center", bbox_to_anchor=(0.5, -0.22 if "\n" in groups[0] else -0.14),
              fontsize=9)
    if note:
        fig.text(0.01, 0.005, note, fontsize=7.5, color=MUTED, ha="left", va="bottom")
    save(fig, path, plt)


def fig_contribution(shares: list[tuple[str, dict]], path: Path):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.2, 1.2 + 0.75 * len(shares)))
    colours = {"text": BLUE, "posture": ORANGE, "skin": AQUA}
    names = {"text": "Patient's words", "posture": "Posture", "skin": "Skin"}
    for row, (label, share) in enumerate(shares):
        left = 0.0
        for p in PLAYERS:
            w = 100 * share[p]
            ax.barh(row, w, left=left, color=colours[p], edgecolor="white", linewidth=2, height=0.55,
                    label=names[p] if row == 0 else None)
            if w >= 6:
                ax.text(left + w / 2, row, f"{w:.0f}%", ha="center", va="center", fontsize=9,
                        color="white" if p == "text" else INK)
            left += w
    ax.set_yticks(range(len(shares)), [s[0] for s in shares])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the change in the answer (Shapley, %)", color=MUTED)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=3, loc="lower center", fontsize=9)
    ax.set_title("How much each input moves the ESI answer", fontsize=10, loc="left")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, default=RUN_DIR)
    ap.add_argument("--labels", type=Path, default=LABELS_FINAL)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-round-a", action="store_true", help="leave the stored round-A results out")
    args = ap.parse_args(argv)

    labels = load_labels(args.labels)
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    vision = json.loads((args.run / "vision.json").read_text(encoding="utf-8"))["cases"]
    data = load_records(args.run)
    by = data["by"]
    primary = config.get("primary_config", "W50")
    ids = [i for i in by[primary] if i in labels]
    ref = [labels[i] for i in ids]
    out = args.out or (ROUNDB_DIR / "analysis")
    out.mkdir(parents=True, exist_ok=True)
    res = {"config": config, "n": len(ids), "n_in_scope": int(sum(r >= 2 for r in ref)),
           "n_out_of_scope": int(sum(r == 1 for r in ref))}
    md = [f"# M3 round B results\n", f"Run: `{rel(args.run)}` · prompt **{config['selection']['candidate']}** · "
          f"{config['llm_model']} · posture rules {config['posture_rules']} · skin pipeline {config['skin_pipeline']} · "
          f"n = {len(ids)} scenarios ({res['n_in_scope']} in scope, {res['n_out_of_scope']} reference ESI 1).\n",
          "Intervals are 95 % percentile bootstrap intervals over scenarios (2,000 resamples).\n"]

    def preds(cfg, key="esi_final"):
        return [by[cfg][i][key] if key != "llm_only" else llm_only(by[cfg][i]) for i in ids]

    # ---- 1. systems
    round_a = {} if args.no_round_a else {"A_original": load_round_a("original"), "A_fixed": load_round_a("fixed")}
    systems = []
    for key, label in (("A_original", "Round A, original pipeline"), ("A_fixed", "Round A, fixed pipeline")):
        rows = round_a.get(key) or {}
        if all(i in rows for i in ids):
            systems.append((key, label, [int_or_none(rows[i]["esi_pred_A"]) for i in ids]))
    systems += [("B_llm_only", "Round B, LLM answer before the safety floor", preds(primary, "llm_only")),
                ("B_independent", "Round B, independent camera parts", preds("IND")),
                ("B_main", "Round B, MAIN (weighted 50/50)", preds(primary))]
    res["systems"] = {k: with_ci(ref, p) for k, _, p in systems}
    labels_of = {k: lab for k, lab, _ in systems}

    def table(title, keys, rows_spec, note=None):
        md.append(f"\n## {title}\n")
        md.append("| | " + " | ".join(labels_of.get(k, k) for k in keys) + " |")
        md.append("| --- |" + " --- |" * len(keys))
        for group, metric, label, pct in rows_spec:
            md.append(f"| {label} | " + " | ".join(cell(res_lookup[k].get(group, {}).get(metric), pct) for k in keys) + " |")
        if note:
            md.append(f"\n{note}")

    res_lookup = dict(res["systems"])
    sys_keys = [k for k, _, _ in systems]
    in_rows = [("in_scope", m, lab, pct) for m, lab, pct in HEADLINE] + [
        ("in_scope", "esi2_sensitivity", "ESI 2 found (answered 1–2)", True),
        ("in_scope", "esi3to5_specificity", "ESI 3–5 not over-called (answered 3–5)", True),
        ("in_scope", "esi5_recall", "ESI 5 answered as 5", True)]
    table(f"1. In-scope scenarios (reference ESI 2–5, n = {res['n_in_scope']})", sys_keys, in_rows)
    table(f"2. Out-of-scope scenarios (reference ESI 1, n = {res['n_out_of_scope']})", sys_keys,
          [("out_of_scope", "detected_strict", "Answered ESI 1 (sent to staff at once)", True),
           ("out_of_scope", "detected_lenient", "Answered ESI 1 or 2 (urgent alert)", True)])
    table(f"3. All scenarios (n = {len(ids)}), for comparison with round A", sys_keys,
          [("all", m, lab, pct) for m, lab, pct in HEADLINE] + [
              ("all", "high_acuity_sensitivity", "High acuity (ESI 1–2) sensitivity", True),
              ("all", "high_acuity_specificity", "High acuity (ESI 1–2) specificity", True)])

    # ---- 2. weights
    weight_systems = {}
    for cfg, label in WEIGHT_CONFIGS:
        weight_systems[cfg] = with_ci(ref, preds(cfg))
        weight_systems[cfg + "_rules"] = with_ci(ref, preds(cfg + "_rules"))
        labels_of[cfg] = label.replace("\n", " ")
        labels_of[cfg + "_rules"] = label.replace("\n", " ") + " — rules only"
    res["weights"] = weight_systems
    res_lookup = weight_systems
    wk = [c for c, _ in WEIGHT_CONFIGS]
    w_rows = [("in_scope", m, lab + " (in scope)", pct) for m, lab, pct in HEADLINE] + [
        ("in_scope", "esi5_recall", "ESI 5 answered as 5", True),
        ("out_of_scope", "detected_lenient", "Reference ESI 1 answered 1–2", True),
        ("all", "accuracy", f"Exact accuracy (all {len(ids)})", True),
        ("all", "under_triage", f"Under-triage (all {len(ids)})", True),
        ("all", "over_triage", f"Over-triage (all {len(ids)})", True)]
    table("4. Posture : skin weights — LLM + safety floor", wk, w_rows)
    table("5. Posture : skin weights — rule-based path only (no LLM)", [c + "_rules" for c in wk], w_rows,
          "The rule path uses the stricter floor (MODERATE → at most ESI 3), so the weights show more clearly here.")
    changed = []
    for i, r in zip(ids, ref):
        finals = {c: by[c][i]["esi_final"] for c in wk}
        rules = {c: by[c + "_rules"][i]["esi_final"] for c in wk}
        if len(set(finals[c] for c in wk if c != "IND")) > 1 or len(set(rules[c] for c in wk if c != "IND")) > 1:
            changed.append({"id": i, "reference": r, **{f"final_{c}": finals[c] for c in wk},
                            **{f"rules_{c}": rules[c] for c in wk},
                            **{f"visual_{c}": f"{by[c][i]['visual_score']} {by[c][i]['visual_band']}" for c in wk if c != "IND"},
                            "camera": f"{vision[i]['posture_label']} / {vision[i]['skin_status']}"})
    res["weights_changed_cases"] = changed
    md.append(f"\nScenarios whose answer differs between the three weightings: **{len(changed)} of {len(ids)}**.\n")
    if changed:
        md.append("| Scenario | Reference | Camera (posture / skin) | Visual 50/50 | 60/40 | 40/60 | LLM+floor 50/50 · 60/40 · 40/60 | Rules 50/50 · 60/40 · 40/60 |")
        md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for c in changed:
            md.append(f"| {c['id']} | {c['reference']} | {c['camera']} | {c['visual_W50']} | {c['visual_W60']} | {c['visual_W40']} | "
                      f"{c['final_W50']} · {c['final_W60']} · {c['final_W40']} | {c['rules_W50']} · {c['rules_W60']} · {c['rules_W40']} |")

    # ---- 3. modalities and contribution
    mod = {}
    for cfg, label in MODALITY_CONFIGS:
        mod[cfg] = with_ci(ref, preds(cfg))
        mod[cfg + "_llm"] = with_ci(ref, preds(cfg, "llm_only"))
        labels_of[cfg] = label.replace("\n", " ")
    res["modalities"] = mod
    res_lookup = mod
    table("6. Which inputs are switched on — LLM + safety floor", [c for c, _ in MODALITY_CONFIGS], w_rows)
    contrib = {"llm_only": contribution(by, ids, llm_only), "final": contribution(by, ids, lambda r: r["esi_final"])}
    with_finding = [i for i in ids if vision[i]["posture_label"] != "none" or vision[i]["skin_status"] == "RED_FLUSHING"]
    if with_finding:
        contrib["llm_only_camera_finding"] = contribution(by, with_finding, llm_only)
        contrib["final_camera_finding"] = contribution(by, with_finding, lambda r: r["esi_final"])
    res["contribution"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_case"} for k, v in contrib.items()}
    res["contribution"]["n_with_camera_finding"] = len(with_finding)
    flips = {}
    for p, off in (("text", "PS"), ("posture", "TS"), ("skin", "TP")):
        flips[p] = {"llm_only": int(sum(llm_only(by[primary][i]) != llm_only(by[off][i]) for i in ids)),
                    "final": int(sum(by[primary][i]["esi_final"] != by[off][i]["esi_final"] for i in ids))}
    res["answer_changes_when_input_removed"] = flips
    md.append("\n## 7. How much each input moves the answer\n")
    md.append("Each scenario was answered with every combination of the three inputs switched on or off "
              "(8 combinations). The Shapley value gives each input its fair share of the change in the answer; "
              "the table shows the average size of that change in ESI levels and each input's share.\n")
    md.append("| | Patient's words | Posture | Skin |")
    md.append("| --- | --- | --- | --- |")
    for key, label in (("llm_only", "LLM alone — share"), ("final", "Whole system (LLM + floor) — share"),
                       ("llm_only_camera_finding", f"LLM alone, scenarios where the camera reported a finding (n = {len(with_finding)}) — share"),
                       ("final_camera_finding", f"Whole system, same scenarios — share")):
        if key in contrib:
            c = contrib[key]
            md.append(f"| {label} | " + " | ".join(f"{100 * c['share'][p]:.0f}% ({c['mean_abs_levels'][p]:.2f} levels)" for p in PLAYERS) + " |")
    md.append(f"| Scenarios whose answer changes when this input is removed (LLM alone / whole system) | "
              + " | ".join(f"{flips[p]['llm_only']} / {flips[p]['final']} of {len(ids)}" for p in PLAYERS) + " |")

    # ---- 4. floor policy
    policies = {}
    for policy in ("none", "high_only", "high_and_moderate"):
        for cfg in ("W50", "W60", "W40"):
            p = [final_level(llm_only(by[cfg][i]), by[cfg][i]["caps"], by[cfg][i]["visual_band"], policy) for i in ids]
            policies[f"{cfg}:{policy}"] = scope_metrics(ref, p)
    res["floor_policies"] = policies
    md.append("\n## 8. Sensitivity: how strict the visual safety floor is\n")
    md.append("| Floor | Weights | Accuracy in scope | Under-triage in scope | Over-triage in scope | ESI 5 answered 5 | Reference ESI 1 answered 1–2 |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    weight_names = {"W50": "50 : 50", "W60": "60 : 40", "W40": "40 : 60"}
    names = {"none": "No visual floor", "high_only": "HIGH → at most ESI 2 (main)", "high_and_moderate": "HIGH → 2 and MODERATE → 3"}
    for policy in ("none", "high_only", "high_and_moderate"):
        for cfg in ("W50", "W60", "W40"):
            s = policies[f"{cfg}:{policy}"]
            md.append(f"| {names[policy]} | {weight_names[cfg]} | {100 * s['in_scope']['accuracy']:.0f}% | "
                      f"{100 * s['in_scope']['under_triage']:.0f}% | {100 * s['in_scope']['over_triage']:.0f}% | "
                      f"{100 * s['in_scope']['esi5_recall']:.0f}% | {100 * s['out_of_scope']['detected_lenient']:.0f}% |")

    # ---- 5. camera parts
    truth = {r.file: r for r in read_sources() if r.set == "test"}
    cam_rows, reasons = [], Counter()
    for i in ids:
        v = vision[i]
        t_post = truth.get(v["image_file"])
        t_skin = truth.get(v["image_file_skin"] or v["image_file"])
        call = {"RED_FLUSHING": "yes", "NORMAL": "no"}.get(v["skin_status"], "unknown")
        if call == "unknown":
            reasons[v.get("skin_reason") or "no answer"] += 1
        cam_rows.append({"id": i, "true_posture": t_post.true_posture if t_post else "", "posture_B": v["posture_label"],
                         "true_flushing": t_skin.true_flushing if t_skin else "", "skin_B": call,
                         "skin_reason_B": v.get("skin_reason", ""), "skin_probability_B": v.get("skin_probability")})
    cam = {"round_B": {
        "posture_correct": int(sum(c["posture_B"] == c["true_posture"] for c in cam_rows)),
        "skin_answered": int(sum(c["skin_B"] != "unknown" for c in cam_rows)),
        "skin_correct_of_answered": int(sum(c["skin_B"] == c["true_flushing"] for c in cam_rows if c["skin_B"] != "unknown")),
        "true_flushing_n": int(sum(c["true_flushing"] == "yes" for c in cam_rows)),
        "true_flushing_called_yes": int(sum(c["skin_B"] == "yes" for c in cam_rows if c["true_flushing"] == "yes")),
        "true_normal_n": int(sum(c["true_flushing"] == "no" for c in cam_rows)),
        "true_normal_called_yes": int(sum(c["skin_B"] == "yes" for c in cam_rows if c["true_flushing"] == "no")),
        "skin_not_answered_reasons": dict(reasons)}}
    for key in ("A_original", "A_fixed"):
        rows = round_a.get(key) or {}
        if all(i in rows for i in ids):
            rr = [rows[i] for i in ids]
            cam[key] = {"posture_correct": int(sum(r["camera_posture"] == r["true_posture"] for r in rr)),
                        "skin_answered": int(sum(r["camera_skin"] != "unknown" for r in rr)),
                        "skin_correct_of_answered": int(sum(r["camera_skin"] == r["true_flushing"] for r in rr if r["camera_skin"] != "unknown")),
                        "true_flushing_called_yes": int(sum(r["camera_skin"] == "yes" for r in rr if r["true_flushing"] == "yes")),
                        "true_normal_called_yes": int(sum(r["camera_skin"] == "yes" for r in rr if r["true_flushing"] == "no"))}
    res["camera"] = cam
    n = len(ids)
    md.append("\n## 9. Camera parts on the test photos\n")
    md.append("| | Posture correct | Skin: gave an answer | Skin: correct when answered | Truly flushed called flushed | Truly normal called flushed |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    b = cam["round_B"]
    for key, label in (("A_original", "Round A original (rules v1, whole-face CNN)"), ("A_fixed", "Round A fixed (rules v2, patches)"),
                       ("round_B", "Round B (rules v3, patches + image check)")):
        if key in cam:
            c = cam[key]
            md.append(f"| {label} | {c['posture_correct']}/{n} | {c['skin_answered']}/{n} | "
                      f"{c['skin_correct_of_answered']}/{c['skin_answered']} | {c['true_flushing_called_yes']}/{b['true_flushing_n']} | "
                      f"{c['true_normal_called_yes']}/{b['true_normal_n']} |")
    if reasons:
        md.append("\nRound B skin check gave no answer because: " + "; ".join(f"{k} ({v})" for k, v in reasons.most_common()) + ".")

    # ---- 6. reliability
    main_recs = [by[primary][i] for i in ids]
    lat = [r["llm_latency_s"] for r in main_recs if r["llm_latency_s"] is not None]
    res["reliability"] = {
        "llm_invalid_outputs_main": int(sum(not r["llm_valid"] for r in main_recs)),
        "fallback_reasons_main": dict(Counter(r["fallback_reason"] for r in main_recs if r["fallback_reason"])),
        "floor_changed_answer_main": int(sum(r["esi_final"] != r["esi_before_floor"] for r in main_recs)),
        "latency_median_s": float(np.median(lat)) if lat else None, "latency_p90_s": float(np.percentile(lat, 90)) if lat else None,
        "latency_max_s": float(max(lat)) if lat else None,
        "level_field_vs_steps_disagree": int(sum(r["esi_llm_level_field"] is not None and r["esi_llm_from_steps"] is not None
                                                 and r["esi_llm_level_field"] != r["esi_llm_from_steps"] for r in main_recs))}
    rel_ = res["reliability"]
    md.append("\n## 10. Reliability of the main system\n")
    md.append(f"- LLM outputs that could not be used: {rel_['llm_invalid_outputs_main']} of {n} "
              f"({rel_['fallback_reasons_main'] or 'none'}).")
    md.append(f"- Safety floor changed the LLM's answer in {rel_['floor_changed_answer_main']} of {n} scenarios.")
    if lat:
        md.append(f"- LLM time per scenario: median {rel_['latency_median_s']:.1f} s, 90th percentile {rel_['latency_p90_s']:.1f} s, "
                  f"longest {rel_['latency_max_s']:.1f} s.")
    md.append(f"- The model's level field and its own step answers disagreed in {rel_['level_field_vs_steps_disagree']} of {n}.")
    samp = data["sampled"].get(f"{primary}_sampled")
    if samp:
        maj = [M.majority_vote([r["esi_final"] for r in samp[i]]) for i in ids]
        agree = [Counter(r["esi_final"] for r in samp[i]).most_common(1)[0][1] / len(samp[i]) for i in ids]
        res["consistency"] = {"majority": scope_metrics(ref, maj), "mean_agreement": float(np.mean(agree)),
                              "runs": len(next(iter(samp.values())))}
        md.append(f"- Sampled runs (temperature 0.8, {res['consistency']['runs']} per scenario): the most common answer was given in "
                  f"{100 * np.mean(agree):.0f}% of runs on average; majority-vote accuracy in scope "
                  f"{100 * res['consistency']['majority']['in_scope']['accuracy']:.0f}%.")

    # ---- per-case file
    per_case = []
    for i, r in zip(ids, ref):
        m = by[primary][i]
        row = {"id": i, "esi_reference": r, "scope": "out (ESI 1)" if r == 1 else "in"}
        for key in ("A_original", "A_fixed"):
            rows = round_a.get(key) or {}
            row[f"esi_{key}"] = int_or_none(rows[i]["esi_pred_A"]) if i in rows else None
        row.update({"esi_B_llm_only": llm_only(m), "esi_B_independent": by["IND"][i]["esi_final"], "esi_B_main": m["esi_final"],
                    "esi_B_W60": by["W60"][i]["esi_final"], "esi_B_W40": by["W40"][i]["esi_final"],
                    "esi_B_text_only": by["T"][i]["esi_final"], "esi_B_rules_W50": by["W50_rules"][i]["esi_final"],
                    "direction_B_main": "" if m["esi_final"] == r else ("under" if m["esi_final"] > r else "over"),
                    "camera_posture_B": vision[i]["posture_label"], "camera_skin_B": vision[i]["skin_status"],
                    "skin_reason_B": vision[i].get("skin_reason", ""), "visual_score_50_50": m["visual_score"],
                    "visual_band_50_50": m["visual_band"], "floor": "; ".join(c["reason"] for c in m["caps"]),
                    "step_a": m["steps"].get("step_a"), "step_b": m["steps"].get("step_b"), "resources": m["steps"].get("resources"),
                    "llm_latency_s": m["llm_latency_s"]})
        per_case.append(row)
    write_csv(out / "per_case.csv", per_case, list(per_case[0].keys()))

    # ---- figures
    try:
        setup_matplotlib()
    except ImportError:
        print("matplotlib is not installed: tables written, figures skipped.")
    else:
        panels = []
        if "A_fixed" in res["systems"]:
            panels.append(("Round A (fixed pipeline)", M.confusion(ref, next(p for k, _, p in systems if k == "A_fixed"))))
        panels.append(("Round B (main system)", M.confusion(ref, preds(primary))))
        fig_confusion(panels, out / "fig_confusion")

        round_keys = [k for k in ("A_original", "A_fixed", "B_main") if k in res["systems"]]
        short = {"A_original": "Round A\noriginal", "A_fixed": "Round A\nfixed", "B_main": "Round B\nmain"}
        fig_bars([short[k] for k in round_keys],
                 [(lab, col, [100 * res["systems"][k]["all"][m]["value"] for k in round_keys])
                  for (m, lab), col in zip((("accuracy", "Exact accuracy"), ("under_triage", "Under-triage"),
                                            ("over_triage", "Over-triage")), (BLUE, ORANGE, AQUA))],
                 f"% of the {len(ids)} scenarios", f"Round A and round B on the same {len(ids)} scenarios", out / "fig_rounds")
        colours = {"A_original": GREY, "A_fixed": ORANGE, "B_main": BLUE}
        fig_bars([f"ESI {lv}" for lv in M.LEVELS],
                 [(short[k].replace("\n", " "), colours[k],
                   [100 * res["systems"][k]["all"]["recall_per_level"][lv] for lv in M.LEVELS]) for k in round_keys],
                 "% answered exactly", "Share of each reference level answered exactly", out / "fig_recall",
                 note=f"n = {Counter(ref).most_common(1)[0][1]} scenarios per level")
        three = (("accuracy", "Exact accuracy"), ("under_triage", "Under-triage"), ("over_triage", "Over-triage"))
        fig_bars([lab for _, lab in WEIGHT_CONFIGS],
                 [(lab, col, [100 * weight_systems[c]["in_scope"][m]["value"] for c, _ in WEIGHT_CONFIGS])
                  for (m, lab), col in zip(three, (BLUE, ORANGE, AQUA))],
                 f"% of the {res['n_in_scope']} in-scope scenarios", "Posture : skin weights (LLM + safety floor)",
                 out / "fig_weights", figsize=(6.8, 3.6))
        fig_bars([lab for _, lab in WEIGHT_CONFIGS],
                 [(lab, col, [100 * weight_systems[c + "_rules"]["in_scope"][m]["value"] for c, _ in WEIGHT_CONFIGS])
                  for (m, lab), col in zip(three, (BLUE, ORANGE, AQUA))],
                 f"% of the {res['n_in_scope']} in-scope scenarios", "Posture : skin weights (rule-based path only, no LLM)",
                 out / "fig_weights_rules", figsize=(6.8, 3.6))
        fig_bars([lab for _, lab in MODALITY_CONFIGS],
                 [(lab, col, [100 * mod[c]["in_scope"][m]["value"] for c, _ in MODALITY_CONFIGS])
                  for (m, lab), col in zip(three, (BLUE, ORANGE, AQUA))],
                 f"% of the {res['n_in_scope']} in-scope scenarios", "Inputs switched on (LLM + safety floor)",
                 out / "fig_modalities", figsize=(6.8, 3.6))
        shares = [("LLM alone", contrib["llm_only"]["share"]), ("Whole system", contrib["final"]["share"])]
        if "llm_only_camera_finding" in contrib:
            shares += [(f"LLM alone, camera\nfound something (n={len(with_finding)})", contrib["llm_only_camera_finding"]["share"]),
                       (f"Whole system, camera\nfound something", contrib["final_camera_finding"]["share"])]
        fig_contribution(shares, out / "fig_contribution")

    (out / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    m = res["systems"]["B_main"]
    print(f"Round B main system ({config['selection']['candidate']}, weighted 50/50):")
    print(f"  in scope (n={res['n_in_scope']}): accuracy {cell(m['in_scope']['accuracy'])}, "
          f"under-triage {cell(m['in_scope']['under_triage'])}, over-triage {cell(m['in_scope']['over_triage'])}, "
          f"kappa {cell(m['in_scope']['kappa_linear'], False)}")
    if "out_of_scope" in m:
        print(f"  reference ESI 1 (n={res['n_out_of_scope']}): answered 1 {cell(m['out_of_scope']['detected_strict'])}, "
              f"answered 1-2 {cell(m['out_of_scope']['detected_lenient'])}")
    print(f"Saved {rel(out)}/report.md, summary.json, per_case.csv and figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
