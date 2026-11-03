"""Turn M3 run outputs into the results tables and figures.

CNN environment (needs matplotlib):

    python scripts/experiments/m3/analyze_m3.py --run-a results/experiments/m3/runs/<..._fixed_A> \
        [--run-b results/experiments/m3/runs/<..._fixed_B>] [--name fixed]

Needs docs/experiments/m3/labels_final.csv (columns: id, esi_reference) = the agreed reference
labels. Writes results/experiments/m3/analysis_<name>/:
  metrics_table.csv / .md     accuracy, within-1, weighted kappa (linear, quadratic), under/over-triage,
                              high-acuity (ESI 1-2) sensitivity/specificity, AUC — each with a 95% bootstrap CI
  per_case.csv                every scenario: reference, prediction(s), camera result vs truth, error source
  camera_components.md        posture and skin accuracy of the camera part
  confusion_matrix.png/.pdf, recall_per_level.png/.pdf, roc.png/.pdf
  summary.json                everything above in one file
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import m3_metrics as M
from m3_common import LABELS_FINAL, RESULTS_DIR, read_sources, write_csv

LEVEL_NAMES = {1: "1 Resuscitation", 2: "2 Emergency", 3: "3 Urgency", 4: "4 Semi-urgency", 5: "5 Non-urgency"}
METRIC_LABELS = [
    ("accuracy", "Exact accuracy"), ("within_one", "Within one level"),
    ("kappa_linear", "Weighted kappa (linear)"), ("kappa_quadratic", "Weighted kappa (quadratic)"),
    ("under_triage", "Under-triage rate"), ("over_triage", "Over-triage rate"),
    ("high_acuity_sensitivity", "High-acuity (ESI 1-2) sensitivity"),
    ("high_acuity_specificity", "High-acuity (ESI 1-2) specificity"),
    ("auc_high_acuity", "AUC, high acuity"),
]


def load_run(run_dir: Path) -> tuple[dict, dict[str, list[dict]], dict[str, dict]]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    calls: dict[str, list[dict]] = defaultdict(list)
    with open(run_dir / "cases.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                calls[rec["id"]].append(rec)
    with open(run_dir / "vision.csv", newline="", encoding="utf-8") as f:
        vision = {r["id"]: r for r in csv.DictReader(f)}
    return config, calls, vision


def load_labels(path: Path) -> dict[str, int]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("id")]
    open_rows = [r["id"] for r in rows if r.get("status", "agreed") not in ("agreed", "decided")
                 or not (r.get("esi_reference") or "").strip()]
    if open_rows:
        raise SystemExit(f"Reference labels not final for: {', '.join(open_rows)} (see compare_labels.py)")
    return {r["id"].strip(): int(r["esi_reference"]) for r in rows}


def metrics_with_ci(ref, pred, score=None) -> dict:
    ref = np.asarray(ref)
    pred = list(pred)
    point = M.summary(ref, pred, score)
    out = {}
    for key, _ in METRIC_LABELS:
        def stat(idx, key=key):
            sub_pred = [pred[i] for i in idx]
            sub_score = None if score is None else [score[i] for i in idx]
            return M.summary(ref[idx], sub_pred, sub_score)[key]
        lo, hi = M.bootstrap_ci(len(ref), stat)
        out[key] = {"value": point[key], "ci_low": lo, "ci_high": hi}
    out["recall_per_level"] = point["recall_per_level"]
    out["invalid_predictions"] = point["invalid_predictions"]
    out["n"] = point["n"]
    return out


def fmt(v: dict, pct: bool) -> str:
    if v["value"] is None or np.isnan(v["value"]):
        return "n/a"
    f = (lambda x: f"{100 * x:.1f}%") if pct else (lambda x: f"{x:.3f}")
    ci = "" if np.isnan(v["ci_low"]) else f" ({f(v['ci_low'])} – {f(v['ci_high'])})"
    return f(v["value"]) + ci


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Pagella", "Times New Roman", "DejaVu Serif"],
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def plot_confusion(cm: np.ndarray, title: str, path: Path):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.max()))
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.55 else "#1b1f24", fontsize=11)
    ax.set_xticks(range(5), [str(i) for i in range(1, 6)])
    ax.set_yticks(range(5), [str(i) for i in range(1, 6)])
    ax.set_xlabel("Predicted ESI (M3)")
    ax.set_ylabel("Reference ESI")
    ax.set_title(title, fontsize=10)
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def plot_recall(recall_a: dict, recall_b: dict | None, n_per_level: dict, path: Path):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    x = np.arange(5)
    width = 0.38 if recall_b else 0.6
    a = [recall_a[lv] * 100 for lv in M.LEVELS]
    ax.bar(x - (width / 2 if recall_b else 0), a, width, color="#2f62c9", label="Option A (temperature 0)")
    if recall_b:
        ax.bar(x + width / 2, [recall_b[lv] * 100 for lv in M.LEVELS], width, color="#a5b4c8",
               label="Option B (majority of 10 runs)")
    ax.set_xticks(x, [f"{LEVEL_NAMES[lv]}\n(n={n_per_level.get(lv, 0)})" for lv in M.LEVELS], fontsize=8)
    ax.set_ylabel("Correct (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    ax.set_title("M3: correctly triaged scenarios per reference ESI level", fontsize=10, pad=24)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def plot_roc(curves: list[tuple[str, np.ndarray, np.ndarray, float]], path: Path):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    ax.plot([0, 1], [0, 1], color="#c3c8cf", lw=1, ls="--")
    styles = (("#2f62c9", "-", 2.2), ("#d9861c", "--", 1.6))   # B dashed so identical curves stay visible
    for (label, fpr, tpr, auc), (colour, ls, lw) in zip(curves, styles):
        # straight segments between thresholds = ROC with tied scores (matches the Mann-Whitney AUC)
        ax.plot(fpr, tpr, color=colour, ls=ls, lw=lw, marker="o", ms=3, label=f"{label} (AUC {auc:.3f})")
    ax.set_xlabel("1 − specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("High acuity (reference ESI 1–2)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-a", type=Path, required=True)
    ap.add_argument("--run-b", type=Path, default=None)
    ap.add_argument("--labels", type=Path, default=LABELS_FINAL)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    labels = load_labels(args.labels)
    cfg_a, calls_a, vision = load_run(args.run_a)
    name = args.name or cfg_a["pipeline"]
    out = RESULTS_DIR / f"analysis_{name}"
    out.mkdir(parents=True, exist_ok=True)

    ids = [i for i in calls_a if i in labels]
    missing = sorted(set(calls_a) - set(labels))
    if missing:
        print(f"WARNING: no reference label for {missing}; they are left out.")
    ref = [labels[i] for i in ids]
    pred_a = [calls_a[i][0]["esi_pred"] for i in ids]
    res = {"config_a": cfg_a, "n": len(ids), "option_A": metrics_with_ci(ref, pred_a)}

    curves = []
    fpr, tpr = M.roc_points([int(r <= 2) for r in ref], [6 - p if p else 0 for p in pred_a])
    curves.append(("Option A", fpr, tpr, res["option_A"]["auc_high_acuity"]["value"]))

    per_case = []
    true_cam = {r.file: r for r in read_sources() if r.set == "test"}
    pred_b, score_b, consistency = {}, {}, {}
    if args.run_b:
        cfg_b, calls_b, _ = load_run(args.run_b)
        res["config_b"] = cfg_b
        for i in ids:
            preds = [c["esi_pred"] for c in calls_b.get(i, [])]
            pred_b[i] = M.majority_vote(preds)
            valid = [p for p in preds if p is not None]
            score_b[i] = (sum(p <= 2 for p in valid) / len(valid)) if valid else 0.0
            consistency[i] = (Counter(valid).most_common(1)[0][1] / len(valid)) if valid else 0.0
        res["option_B_majority"] = metrics_with_ci(ref, [pred_b[i] for i in ids], [score_b[i] for i in ids])
        res["option_B_mean_consistency"] = float(np.mean([consistency[i] for i in ids]))
        fpr, tpr = M.roc_points([int(r <= 2) for r in ref], [score_b[i] for i in ids])
        curves.append(("Option B", fpr, tpr, res["option_B_majority"]["auc_high_acuity"]["value"]))

    # Camera part and error source
    cam_post, cam_skin = [], []
    for i, r in zip(ids, ref):
        v = vision.get(i, {})
        truth = true_cam.get(v.get("image_file", ""))                               # posture truth
        skin_truth = true_cam.get(v.get("image_file_skin") or v.get("image_file", ""))  # skin truth (face photo if two)
        posture_ok = truth is not None and v.get("posture_label") == truth.true_posture
        skin_call = {"RED_FLUSHING": "yes", "NORMAL": "no"}.get(v.get("skin_status", ""), "unknown")
        skin_ok = skin_truth is not None and skin_call == skin_truth.true_flushing
        cam_post.append(posture_ok)
        cam_skin.append(skin_ok)
        a = calls_a[i][0]
        wrong = a["esi_pred"] != r
        source = "" if not wrong else ("camera: posture" if not posture_ok else "camera: skin" if not skin_ok else "reasoning")
        per_case.append({
            "id": i, "esi_reference": r, "esi_pred_A": a["esi_pred"],
            "esi_pred_B_majority": pred_b.get(i), "B_share_ESI_1_2": score_b.get(i), "B_consistency": consistency.get(i),
            "image_file": v.get("image_file"), "image_file_skin": v.get("image_file_skin", ""),
            "sharpness_posture_photo": v.get("sharpness_posture_photo"), "sharpness_skin_photo": v.get("sharpness_skin_photo"),
            "true_posture": truth.true_posture if truth else "",
            "camera_posture": v.get("posture_label"), "true_flushing": skin_truth.true_flushing if skin_truth else "",
            "camera_skin": skin_call, "skin_probability": v.get("skin_probability"),
            "correct_A": not wrong, "direction_A": "" if not wrong or a["esi_pred"] is None else ("under" if a["esi_pred"] > r else "over"),
            "error_source_A": source, "fallback_A": a["fallback"], "llm_latency_s_A": a["llm_latency_s"],
        })
    write_csv(out / "per_case.csv", per_case, list(per_case[0].keys()))

    lat = [c["llm_latency_s"] for i in ids for c in calls_a[i] if c.get("llm_latency_s") is not None]
    res["camera"] = {"posture_accuracy": float(np.mean(cam_post)), "skin_accuracy": float(np.mean(cam_skin))}

    def med(vals):
        vals = [float(x) for x in vals if x not in (None, "")]
        return float(np.median(vals)) if vals else None
    # Does blur go with camera mistakes? (median sharpness of correctly vs wrongly read photos)
    res["sharpness"] = {
        "posture_correct_median": med(p["sharpness_posture_photo"] for p, ok in zip(per_case, cam_post) if ok),
        "posture_wrong_median": med(p["sharpness_posture_photo"] for p, ok in zip(per_case, cam_post) if not ok),
        "skin_correct_median": med(p["sharpness_skin_photo"] for p, ok in zip(per_case, cam_skin) if ok),
        "skin_wrong_median": med(p["sharpness_skin_photo"] for p, ok in zip(per_case, cam_skin) if not ok),
    }
    res["errors_by_source_A"] = dict(Counter(p["error_source_A"] for p in per_case if p["error_source_A"]))
    res["latency_A"] = {"median_s": float(np.median(lat)) if lat else None, "max_s": float(max(lat)) if lat else None,
                        "calls_over_6s": int(sum(x > 6.0 for x in lat)), "fallbacks": int(sum(p["fallback_A"] for p in per_case))}

    # Tables
    rows, md = [], ["| Metric | Option A (temperature 0) |" + (" Option B (majority of 10) |" if args.run_b else ""),
                    "| --- | --- |" + (" --- |" if args.run_b else "")]
    for key, label in METRIC_LABELS:
        pct = key not in ("kappa_linear", "kappa_quadratic", "auc_high_acuity")
        a_txt = fmt(res["option_A"][key], pct)
        b_txt = fmt(res["option_B_majority"][key], pct) if args.run_b else ""
        rows.append({"metric": label, "option_A": a_txt, "option_B": b_txt})
        md.append(f"| {label} | {a_txt} |" + (f" {b_txt} |" if args.run_b else ""))
    md.append(f"\nn = {len(ids)} scenarios; 95% CI = percentile bootstrap (2,000 resamples). "
              f"Invalid LLM outputs (A): {res['option_A']['invalid_predictions']}.")
    write_csv(out / "metrics_table.csv", rows, ["metric", "option_A", "option_B"])
    (out / "metrics_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "camera_components.md").write_text(
        f"| Camera part | Correct |\n| --- | --- |\n"
        f"| Posture (vs true_posture) | {sum(cam_post)}/{len(cam_post)} ({100 * np.mean(cam_post):.1f}%) |\n"
        f"| Skin (vs true_flushing) | {sum(cam_skin)}/{len(cam_skin)} ({100 * np.mean(cam_skin):.1f}%) |\n\n"
        f"Wrong M3 answers by first failing part (A): {res['errors_by_source_A']}\n", encoding="utf-8")

    # Figures
    cm = M.confusion(ref, pred_a)
    plot_confusion(cm, f"M3 ({name} pipeline), option A, n={len(ids)}", out / "confusion_matrix")
    n_per = Counter(ref)
    plot_recall(res["option_A"]["recall_per_level"], res.get("option_B_majority", {}).get("recall_per_level"),
                n_per, out / "recall_per_level")
    plot_roc(curves, out / "roc")
    res["confusion_matrix_A"] = cm.tolist()
    (out / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("\n".join(md))
    print(f"\nCamera: posture {100 * np.mean(cam_post):.1f}% correct, skin {100 * np.mean(cam_skin):.1f}% correct")
    print(f"Latency A: {res['latency_A']}")
    print(f"\nSaved results in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
