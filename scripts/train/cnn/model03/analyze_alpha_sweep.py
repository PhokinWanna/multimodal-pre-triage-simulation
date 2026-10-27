"""Summarize a focal-loss alpha sweep as tables and line graphs.

Reads every ``focal_alpha_XXX/metrics.json`` (and ``test_scores.csv``) produced
by ``train_and_evaluate.py --alphas ...`` and writes, under
``<results-dir>/alpha_sweep/``:

* ``alpha_sweep_table.csv`` / ``alpha_sweep_table.md``: one row per alpha with
  the test confusion counts and metrics at threshold 0.50, plus the
  validation-selected operating point (precision floor 0.95).
* ``alpha_sweep_metrics.png``: line graph of each metric against alpha.
* ``alpha_sweep_confusion_counts.png``: TP/FN and TN/FP counts against alpha.
* ``alpha_sweep_score_shift.png``: how the predicted-probability distribution
  of each class moves as alpha changes.
* ``alpha_sweep_curves.png``: ROC and precision-recall curves for every alpha.
* ``alpha_sweep_learning_curves.png``: training vs validation loss and AUC per
  epoch for every alpha, to check for over- or underfitting.

Usage::

    python analyze_alpha_sweep.py --results-dir results_alpha_sweep
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
from sklearn.metrics import precision_recall_curve, roc_curve

PRECISION_FLOOR = "0.95"
METRIC_LINES = (
    ("accuracy", "Accuracy", "#4D4D4D", "o"),
    ("recall", "Sensitivity (recall)", "#D1495B", "s"),
    ("specificity", "Specificity", "#2E86AB", "^"),
    ("precision", "Precision", "#8C6BB1", "D"),
    ("f1", "F1", "#E07A24", "v"),
    ("balanced_accuracy", "Balanced accuracy", "#3A9D5D", "P"),
    ("roc_auc", "ROC-AUC", "#000000", "*"),
)


def load_runs(results_dir: Path, weights: str = "final") -> list[dict]:
    runs = []
    for metrics_path in sorted(results_dir.glob("focal_alpha_*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        scores_path = metrics_path.parent / "test_scores.csv"
        if weights == "best":
            if "best_checkpoint" not in metrics:
                raise KeyError(f"{metrics_path} has no best_checkpoint results; rerun training")
            metrics = {**metrics, "test": metrics["best_checkpoint"]["test"]}
            scores_path = metrics_path.parent / "test_scores_best_checkpoint.csv"
        labels, scores = [], []
        if scores_path.is_file():
            with scores_path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    labels.append(int(row["label"]))
                    scores.append(float(row["probability"]))
        runs.append({
            "alpha": float(metrics["focal_alpha_positive_class"]),
            "metrics": metrics,
            "labels": np.asarray(labels, dtype=int),
            "scores": np.asarray(scores, dtype=float),
        })
    if not runs:
        raise FileNotFoundError(f"No focal_alpha_*/metrics.json found under {results_dir}")
    return sorted(runs, key=lambda run: run["alpha"])


def table_rows(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        test = run["metrics"]["test"]
        at50 = test["at_0.50"]
        selected = (test.get("validation_selected_thresholds") or {}).get(PRECISION_FLOOR)
        rows.append({
            "alpha_flushing": run["alpha"],
            "weight_normal": round(1.0 - run["alpha"], 2),
            "epochs_completed": run["metrics"].get("epochs_completed"),
            "tp": at50["tp"],
            "fn": at50["fn"],
            "tn": at50["tn"],
            "fp": at50["fp"],
            "accuracy": at50["accuracy"],
            "recall": at50["recall"],
            "specificity": at50["specificity"],
            "precision": at50["precision"],
            "f1": at50["f1"],
            "balanced_accuracy": (at50["recall"] + at50["specificity"]) / 2,
            "roc_auc": test["roc_auc"],
            "average_precision": test["average_precision"],
            "positive_predictions": at50["tp"] + at50["fp"],
            "val_selected_threshold_p95": selected["threshold"] if selected else None,
            "test_recall_at_p95_threshold": selected["recall"] if selected else None,
            "test_precision_at_p95_threshold": selected["precision"] if selected else None,
        })
    return rows


def write_tables(rows: list[dict], output_dir: Path) -> None:
    with (output_dir / "alpha_sweep_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{100 * value:.2f}%"

    header = ("| α (Flushing) | Normal weight | TP / FN | TN / FP | Accuracy | Sensitivity | "
              "Specificity | Precision | F1 | Balanced acc. | ROC-AUC | AP |")
    lines = [header, "|" + " ---: |" * 12]
    for row in rows:
        precision = pct(row["precision"])
        if row["positive_predictions"] == 0:
            precision += " †"
        lines.append(
            f"| {row['alpha_flushing']:.2f} | {row['weight_normal']:.2f} | "
            f"{row['tp']:,} / {row['fn']:,} | {row['tn']:,} / {row['fp']:,} | "
            f"{pct(row['accuracy'])} | {pct(row['recall'])} | {pct(row['specificity'])} | "
            f"{precision} | {pct(row['f1'])} | {pct(row['balanced_accuracy'])} | "
            f"{row['roc_auc']:.3f} | {row['average_precision']:.3f} |"
        )
    lines += ["", "Validation-selected operating point (precision floor 0.95 on validation, applied to test):", "",
              "| α (Flushing) | Threshold | Test recall | Test precision |", "| ---: | ---: | ---: | ---: |"]
    for row in rows:
        threshold = row["val_selected_threshold_p95"]
        lines.append(
            f"| {row['alpha_flushing']:.2f} | "
            f"{'not reachable' if threshold is None else f'{threshold:.4f}'} | "
            f"{pct(row['test_recall_at_p95_threshold'])} | {pct(row['test_precision_at_p95_threshold'])} |"
        )
    if any(row["positive_predictions"] == 0 for row in rows):
        lines += ["", "† No patch was predicted as Flushing, so precision is undefined (reported as 0)."]
    (output_dir / "alpha_sweep_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_metrics(rows: list[dict], output_dir: Path) -> None:
    alphas = [row["alpha_flushing"] for row in rows]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    for key, label, color, marker in METRIC_LINES:
        values = [row[key] for row in rows]
        ax.plot(alphas, values, marker=marker, color=color, linewidth=2, markersize=7, label=label)
    ax.set_xticks(alphas, [f"{alpha:.2f}" for alpha in alphas])
    ax.set_xlabel("Focal-loss weight α of the Flushing (Abnormal) class   [Normal weight = 1 − α]")
    ax.set_ylabel("Test score at threshold 0.50")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.3)
    ax.set_title("Effect of focal-loss α on skin-classifier test metrics (same split, seed and settings)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(output_dir / "alpha_sweep_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_counts(rows: list[dict], output_dir: Path) -> None:
    alphas = [row["alpha_flushing"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    panels = (
        (axes[0], (("tp", "TP: Flushing detected", "#3A9D5D"), ("fn", "FN: Flushing missed", "#D1495B")),
         f"Flushing test patches (n = {rows[0]['tp'] + rows[0]['fn']:,})"),
        (axes[1], (("tn", "TN: Normal kept Normal", "#2E86AB"), ("fp", "FP: false alarm", "#E07A24")),
         f"Normal test patches (n = {rows[0]['tn'] + rows[0]['fp']:,})"),
    )
    for ax, series, title in panels:
        for key, label, color in series:
            values = [row[key] for row in rows]
            ax.plot(alphas, values, marker="o", color=color, linewidth=2, label=label)
            for alpha, value in zip(alphas, values):
                ax.annotate(f"{value:,}", (alpha, value), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=8, color=color)
        ax.set_xticks(alphas, [f"{alpha:.2f}" for alpha in alphas])
        ax.set_xlabel("α (Flushing weight)")
        ax.set_ylabel("Patches at threshold 0.50")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Confusion counts on the test set as α changes", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "alpha_sweep_confusion_counts.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_score_shift(runs: list[dict], output_dir: Path) -> None:
    runs = [run for run in runs if run["scores"].size]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(runs))
    for offset, label, color, name in ((-0.18, 0, "#2E86AB", "Normal patches"),
                                       (0.18, 1, "#D1495B", "Flushing patches")):
        data = [run["scores"][run["labels"] == label] for run in runs]
        parts = ax.boxplot(data, positions=positions + offset, widths=0.3, patch_artist=True,
                           showfliers=False, medianprops={"color": "black"})
        for box in parts["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.6)
        ax.plot([], [], color=color, linewidth=8, alpha=0.6, label=name)
    ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1, label="Threshold 0.50")
    ax.set_xticks(positions, [f"{run['alpha']:.2f}" for run in runs])
    ax.set_xlabel("α (Flushing weight)")
    ax.set_ylabel("Predicted probability of Flushing")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Predicted-probability distribution per class on the test set (box = 25th–75th percentile)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "alpha_sweep_score_shift.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_curves(runs: list[dict], output_dir: Path) -> None:
    runs = [run for run in runs if run["scores"].size]
    if not runs:
        return
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for index, run in enumerate(runs):
        color = cmap(index / max(len(runs) - 1, 1))
        auc_value = run["metrics"]["test"]["roc_auc"]
        ap_value = run["metrics"]["test"]["average_precision"]
        fpr, tpr, _ = roc_curve(run["labels"], run["scores"])
        axes[0].plot(fpr, tpr, color=color, linewidth=2, label=f"α={run['alpha']:.2f} (AUC {auc_value:.3f})")
        precision, recall, _ = precision_recall_curve(run["labels"], run["scores"])
        axes[1].plot(recall, precision, color=color, linewidth=2, label=f"α={run['alpha']:.2f} (AP {ap_value:.3f})")
        at50 = run["metrics"]["test"]["at_0.50"]
        axes[0].scatter(1 - at50["specificity"], at50["recall"], color=color, edgecolor="black", zorder=4)
        if at50["tp"] + at50["fp"]:
            axes[1].scatter(at50["recall"], at50["precision"], color=color, edgecolor="black", zorder=4)
    axes[0].plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    axes[0].set_xlabel("False positive rate (1 − specificity)")
    axes[0].set_ylabel("True positive rate (sensitivity)")
    axes[0].set_title("ROC curves")
    axes[1].set_xlabel("Recall (sensitivity)")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–recall curves")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left" if ax is axes[1] else "lower right")
    fig.suptitle("Test-set curves for each α (dots mark threshold 0.50)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "alpha_sweep_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(results_dir: Path, runs: list[dict], output_dir: Path) -> None:
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    drawn = False
    for index, run in enumerate(runs):
        history_path = results_dir / f"focal_alpha_{int(round(run['alpha'] * 100)):03d}" / "training_history.csv"
        if not history_path.is_file():
            continue
        rows = list(csv.DictReader(history_path.open(newline="", encoding="utf-8")))
        epochs = np.arange(1, len(rows) + 1)
        color = cmap(index / max(len(runs) - 1, 1))
        for ax, train_key, val_key in ((axes[0], "loss", "val_loss"), (axes[1], "auc", "val_auc")):
            ax.plot(epochs, [float(r[train_key]) for r in rows], color=color, linestyle="--", alpha=0.6)
            ax.plot(epochs, [float(r[val_key]) for r in rows], color=color, linewidth=2,
                    label=f"α={run['alpha']:.2f}")
        best = int(np.argmax([float(r["val_auc"]) for r in rows]))
        axes[1].scatter(best + 1, float(rows[best]["val_auc"]), color=color, edgecolor="black", zorder=4)
        drawn = True
    if not drawn:
        plt.close(fig)
        return
    axes[0].set_title("Focal loss (dashed = training, solid = validation)")
    axes[0].set_ylabel("Loss (lower is better)")
    axes[1].set_title("ROC-AUC (dot = best validation epoch)")
    axes[1].set_ylabel("AUC")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Learning curves per α: a widening train-validation gap would signal overfitting", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "alpha_sweep_learning_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Folder containing focal_alpha_XXX run folders.")
    parser.add_argument("--weights", choices=("final", "best"), default="final",
                        help="Evaluate the weights used at the end of training (final) or the "
                             "best validation-AUC checkpoint (best).")
    args = parser.parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = results_dir / ("alpha_sweep" if args.weights == "final" else "alpha_sweep_best_checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(results_dir, args.weights)
    rows = table_rows(runs)
    write_tables(rows, output_dir)
    plot_metrics(rows, output_dir)
    plot_counts(rows, output_dir)
    plot_score_shift(runs, output_dir)
    plot_curves(runs, output_dir)
    plot_learning_curves(results_dir, runs, output_dir)
    print((output_dir / "alpha_sweep_table.md").read_text(encoding="utf-8"))
    print(f"Wrote tables and figures to {output_dir}")


if __name__ == "__main__":
    main()
