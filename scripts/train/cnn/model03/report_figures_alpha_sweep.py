"""Report-quality and interactive figures for the focal-loss alpha sweep.

Reads ``focal_alpha_XXX/metrics.json`` and ``test_scores.csv`` from a sweep
folder (made by ``train_and_evaluate.py --alphas ...``) and writes to
``<results-dir>/report_figures/``:

Static figures for the paper/report (PNG at 300 dpi + vector PDF):
  * ``fig_alpha_confusion_matrices``: the five test confusion matrices side by
    side, coloured by row percentage on one shared scale.
  * ``fig_alpha_recall_comparison``: (a) sensitivity vs specificity per alpha,
    (b) missed Flushing patches (FN) vs false alarms (FP) per alpha.
  * ``fig_alpha_threshold_heatmap``: recall and F1 for every alpha x decision
    threshold (0.01-0.99), with threshold 0.50 and the best-F1 point marked.

Interactive file (open in any browser, works offline):
  * ``alpha_sweep_interactive.html``: rotatable 3D surface of F1 / recall /
    precision over alpha x threshold, plus interactive confusion matrices,
    comparison bars and metric lines (hover shows every value).

Usage::

    python report_figures_alpha_sweep.py --results-dir results_alpha_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)
PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}
SENS_COLOR, SPEC_COLOR = "#C0392B", "#2471A3"
FN_COLOR, FP_COLOR = "#C0392B", "#E67E22"


def load_runs(results_dir: Path) -> list[dict]:
    runs = []
    for metrics_path in sorted(results_dir.glob("focal_alpha_*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        labels, scores = [], []
        with (metrics_path.parent / "test_scores.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                labels.append(int(row["label"]))
                scores.append(float(row["probability"]))
        runs.append({
            "alpha": float(metrics["focal_alpha_positive_class"]),
            "at50": metrics["test"]["at_0.50"],
            "roc_auc": metrics["test"]["roc_auc"],
            "epochs": metrics.get("epochs_completed"),
            "labels": np.asarray(labels, dtype=int),
            "scores": np.asarray(scores, dtype=float),
        })
    if not runs:
        raise FileNotFoundError(f"No focal_alpha_*/metrics.json under {results_dir}")
    return sorted(runs, key=lambda run: run["alpha"])


def threshold_grid(runs: list[dict]) -> dict[str, np.ndarray]:
    """Recall, precision, F1 and specificity for every alpha (rows) x threshold (columns)."""
    shape = (len(runs), len(THRESHOLDS))
    out = {key: np.zeros(shape) for key in ("recall", "precision", "f1", "specificity")}
    for i, run in enumerate(runs):
        y, s = run["labels"], run["scores"]
        for j, t in enumerate(THRESHOLDS):
            pred = s >= t
            tp = int(np.sum(pred & (y == 1)))
            fp = int(np.sum(pred & (y == 0)))
            fn = int(np.sum(~pred & (y == 1)))
            tn = int(np.sum(~pred & (y == 0)))
            recall = tp / (tp + fn) if tp + fn else 0.0
            precision = tp / (tp + fp) if tp + fp else 0.0
            out["recall"][i, j] = recall
            out["precision"][i, j] = precision
            out["f1"][i, j] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            out["specificity"][i, j] = tn / (tn + fp) if tn + fp else 0.0
    return out


def save(fig, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png")
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def fig_confusion_matrices(runs: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(2.15 * len(runs) + 0.9, 2.9))
    norm = Normalize(0, 1)
    image = None
    for index, (ax, run) in enumerate(zip(np.atleast_1d(axes), runs)):
        m = run["at50"]
        counts = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        rates = counts / counts.sum(axis=1, keepdims=True)
        image = ax.imshow(rates, cmap="Blues", norm=norm)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{counts[r, c]:,}\n({100 * rates[r, c]:.1f}%)", ha="center", va="center",
                        fontsize=8.5, color="white" if rates[r, c] > 0.55 else "#1B1B1B")
        ax.set_xticks([0, 1], ["Normal", "Flushing"])
        ax.set_yticks([0, 1], ["Normal", "Flushing"] if index == 0 else ["", ""])
        ax.grid(False)
        ax.set_title(f"α = {run['alpha']:.2f}\n(Normal weight {1 - run['alpha']:.2f})")
        ax.set_xlabel(f"Predicted\nSens {100 * m['recall']:.1f}% · Spec {100 * m['specificity']:.1f}%", fontsize=8.5)
        for spine in ax.spines.values():
            spine.set_visible(True)
    np.atleast_1d(axes)[0].set_ylabel("Actual")
    cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Share of actual class")
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    save(fig, output_dir, "fig_alpha_confusion_matrices")


def fig_recall_comparison(runs: list[dict], output_dir: Path) -> None:
    alphas = [f"{run['alpha']:.2f}" for run in runs]
    x = np.arange(len(runs))
    width = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.4))
    sens = [100 * run["at50"]["recall"] for run in runs]
    spec = [100 * run["at50"]["specificity"] for run in runs]
    bars = ax1.bar(x - width / 2, sens, width, color=SENS_COLOR, label="Sensitivity (recall)")
    ax1.bar_label(bars, fmt="%.1f", fontsize=7.5, padding=2)
    bars = ax1.bar(x + width / 2, spec, width, color=SPEC_COLOR, label="Specificity")
    ax1.bar_label(bars, fmt="%.1f", fontsize=7.5, padding=2)
    ax1.set_xticks(x, alphas)
    ax1.set_ylim(0, 108)
    ax1.set_xlabel("Flushing-class weight α")
    ax1.set_ylabel("Test score at threshold 0.50 (%)")
    ax1.set_title("(a) Sensitivity vs specificity")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2, frameon=False)
    ax1.grid(axis="x", visible=False)

    fn = [run["at50"]["fn"] for run in runs]
    fp = [run["at50"]["fp"] for run in runs]
    bars = ax2.bar(x - width / 2, fn, width, color=FN_COLOR, label="Missed Flushing (FN, of 818)")
    ax2.bar_label(bars, labels=[f"{v:,}" for v in fn], fontsize=7.5, padding=2)
    bars = ax2.bar(x + width / 2, fp, width, color=FP_COLOR, label="False alarms (FP, of 2,930)")
    ax2.bar_label(bars, labels=[f"{v:,}" for v in fp], fontsize=7.5, padding=2)
    ax2.set_yscale("symlog", linthresh=10)
    ax2.set_ylim(0, 12000)
    ax2.set_yticks([0, 10, 100, 1000, 10000], ["0", "10", "100", "1,000", "10,000"])
    ax2.set_xticks(x, alphas)
    ax2.set_xlabel("Flushing-class weight α")
    ax2.set_ylabel("Test patches (log scale)")
    ax2.set_title("(b) Errors: missed cases vs false alarms")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2, frameon=False)
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, output_dir, "fig_alpha_recall_comparison")


def fig_threshold_heatmap(runs: list[dict], grid: dict[str, np.ndarray], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.1), sharey=True)
    j50 = int(np.argmin(np.abs(THRESHOLDS - 0.50)))
    for ax, key, title in ((axes[0], "recall", "(a) Sensitivity (recall)"), (axes[1], "f1", "(b) F1 score")):
        image = ax.imshow(grid[key], aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="lower",
                          extent=(THRESHOLDS[0] - 0.005, THRESHOLDS[-1] + 0.005, -0.5, len(runs) - 0.5))
        ax.axvline(0.50, color="white", linestyle="--", linewidth=1)
        for i, run in enumerate(runs):
            value = grid[key][i, j50]
            ax.text(0.505, i, f"{value:.2f}", color="black" if value > 0.6 else "white", fontsize=7.5, va="center")
            if key == "f1":
                jbest = int(np.argmax(grid["f1"][i]))
                ax.plot(THRESHOLDS[jbest], i, marker="*", color="#FF4D4D", markersize=9,
                        markeredgecolor="white", markeredgewidth=0.6)
        ax.set_yticks(range(len(runs)), [f"{run['alpha']:.2f}" for run in runs])
        ax.set_xlabel("Decision threshold")
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    axes[0].set_ylabel("Flushing-class weight α")
    fig.text(0.5, -0.04, "Dashed line: threshold 0.50 (value printed beside it).  Star: threshold with the highest test F1 for each α (post hoc, for illustration only).",
             ha="center", fontsize=8.5)
    fig.tight_layout()
    save(fig, output_dir, "fig_alpha_threshold_heatmap")


def interactive_html(runs: list[dict], grid: dict[str, np.ndarray], output_dir: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    alphas = [run["alpha"] for run in runs]
    alpha_text = [f"{a:.2f}" for a in alphas]
    figures = []

    # 1. 3D surface with buttons for F1 / recall / precision
    surface = go.Figure()
    for k, (key, name) in enumerate((("f1", "F1"), ("recall", "Sensitivity (recall)"), ("precision", "Precision"))):
        surface.add_trace(go.Surface(
            x=THRESHOLDS, y=alphas, z=grid[key], colorscale="Viridis", cmin=0, cmax=1, visible=(k == 0),
            name=name, colorbar={"title": name},
            hovertemplate="α = %{y:.2f}<br>threshold = %{x:.2f}<br>" + name + " = %{z:.3f}<extra></extra>"))
    surface.update_layout(
        title="3D view: how α and the decision threshold together set the operating point (test set)",
        scene={"xaxis_title": "Decision threshold", "yaxis_title": "Flushing weight α", "zaxis_title": "Score",
               "zaxis": {"range": [0, 1]}, "camera": {"eye": {"x": 1.6, "y": -1.6, "z": 0.9}}},
        updatemenus=[{"type": "buttons", "direction": "right", "x": 0.0, "y": 1.08, "xanchor": "left",
                      "buttons": [{"label": label, "method": "update",
                                   "args": [{"visible": [i == k for i in range(3)]}]}
                                  for k, label in enumerate(("F1", "Recall", "Precision"))]}],
        height=620, margin={"l": 0, "r": 0, "t": 90, "b": 0})
    figures.append(surface)

    # 2. Confusion matrices
    cm = make_subplots(rows=1, cols=len(runs), subplot_titles=[f"α = {a}" for a in alpha_text], horizontal_spacing=0.03)
    for c, run in enumerate(runs, start=1):
        m = run["at50"]
        counts = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        rates = counts / counts.sum(axis=1, keepdims=True)
        text = [[f"{counts[r, k]:,}<br>({100 * rates[r, k]:.1f}%)" for k in range(2)] for r in range(2)]
        cm.add_trace(go.Heatmap(z=rates, x=["Pred. Normal", "Pred. Flushing"], y=["Actual Normal", "Actual Flushing"],
                                text=text, texttemplate="%{text}", colorscale="Blues", zmin=0, zmax=1,
                                showscale=(c == len(runs)), hovertemplate="%{y} → %{x}<br>%{text}<extra></extra>"),
                     row=1, col=c)
        cm.update_yaxes(autorange="reversed", showticklabels=(c == 1), row=1, col=c)
    cm.update_layout(title="Test confusion matrices at threshold 0.50 (colour = share of the actual class)", height=360)
    figures.append(cm)

    # 3. Sensitivity / specificity bars and error counts
    bars = make_subplots(rows=1, cols=2, subplot_titles=("Sensitivity vs specificity (%)", "Missed Flushing vs false alarms"))
    bars.add_trace(go.Bar(x=alpha_text, y=[100 * r["at50"]["recall"] for r in runs], name="Sensitivity",
                          marker_color=SENS_COLOR, texttemplate="%{y:.1f}"), row=1, col=1)
    bars.add_trace(go.Bar(x=alpha_text, y=[100 * r["at50"]["specificity"] for r in runs], name="Specificity",
                          marker_color=SPEC_COLOR, texttemplate="%{y:.1f}"), row=1, col=1)
    bars.add_trace(go.Bar(x=alpha_text, y=[r["at50"]["fn"] for r in runs], name="Missed (FN)",
                          marker_color=FN_COLOR, texttemplate="%{y:,}"), row=1, col=2)
    bars.add_trace(go.Bar(x=alpha_text, y=[r["at50"]["fp"] for r in runs], name="False alarms (FP)",
                          marker_color=FP_COLOR, texttemplate="%{y:,}"), row=1, col=2)
    bars.update_xaxes(title_text="Flushing weight α")
    bars.update_layout(barmode="group", height=420, title="Threshold 0.50 comparison across α")
    figures.append(bars)

    # 4. Metric lines
    lines = go.Figure()
    for key, name in (("accuracy", "Accuracy"), ("recall", "Sensitivity"), ("specificity", "Specificity"),
                      ("precision", "Precision"), ("f1", "F1")):
        lines.add_trace(go.Scatter(x=alphas, y=[r["at50"][key] for r in runs], mode="lines+markers", name=name))
    lines.add_trace(go.Scatter(x=alphas, y=[r["roc_auc"] for r in runs], mode="lines+markers", name="ROC-AUC",
                               line={"color": "black", "dash": "dot"}))
    lines.update_layout(title="Test metrics against α (threshold 0.50)", xaxis_title="Flushing weight α",
                        yaxis_title="Score", yaxis_range=[-0.02, 1.05], height=440, hovermode="x unified")
    figures.append(lines)

    parts = [fig.to_html(full_html=False, include_plotlyjs=(i == 0)) for i, fig in enumerate(figures)]
    html = ("<!doctype html><html><head><meta charset='utf-8'><title>Focal-loss α sweep</title>"
            "<style>body{font-family:Georgia,serif;max-width:1150px;margin:24px auto;padding:0 16px;color:#222}"
            "h1{font-size:22px}p{color:#555}</style></head><body>"
            "<h1>Skin CNN: effect of the focal-loss class weight α</h1>"
            "<p>Same split, seed and settings for every run; only α changes (Flushing weight α, Normal weight 1 − α). "
            "Drag to rotate the 3D surface, hover for exact values, click legend items to hide series.</p>"
            + "".join(parts) + "</body></html>")
    (output_dir / "alpha_sweep_interactive.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--no-html", action="store_true", help="Skip the interactive Plotly file.")
    args = parser.parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = results_dir / "report_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(PAPER_STYLE)
    runs = load_runs(results_dir)
    grid = threshold_grid(runs)
    fig_confusion_matrices(runs, output_dir)
    fig_recall_comparison(runs, output_dir)
    fig_threshold_heatmap(runs, grid, output_dir)
    if not args.no_html:
        interactive_html(runs, grid, output_dir)
    for i, run in enumerate(runs):
        j = int(np.argmax(grid["f1"][i]))
        print(f"alpha={run['alpha']:.2f}: F1@0.50={run['at50']['f1']:.4f}  best F1={grid['f1'][i, j]:.4f} "
              f"at threshold {THRESHOLDS[j]:.2f} (recall {grid['recall'][i, j]:.4f}, precision {grid['precision'][i, j]:.4f})")
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
