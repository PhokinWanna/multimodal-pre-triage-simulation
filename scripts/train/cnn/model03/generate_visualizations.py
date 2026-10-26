"""Generate publication-ready plots for the Model-02/Model-03 comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
)


CLASS_NAMES = ("Normal", "Flushing")
RUNS = (("focal_alpha_025", "Model-03 alpha 0.25"), ("focal_alpha_075", "Model-03 alpha 0.75"))
COLORS = {
    "Model-02 original validation": "#777777",
    "Model-03 alpha 0.25": "#2878B5",
    "Model-03 alpha 0.75": "#E07A24",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    scores = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            labels.append(int(row["label"]))
            scores.append(float(row["probability"]))
    return np.asarray(labels), np.asarray(scores)


def load_model02_validation(
    model_path: Path, source_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    model = tf.keras.models.load_model(model_path, compile=False)
    images = []
    labels = []
    for class_name, label in (("Normal", 0), ("Flushing", 1)):
        class_dir = source_dir / "val" / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing Model-02 validation folder: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            image = cv2.imread(str(path))
            if image is None:
                raise OSError(f"Unable to read Model-02 validation image: {path}")
            image = cv2.resize(image, (64, 64))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            images.append(image.astype(np.float32) / 255.0)
            labels.append(label)
    if not images:
        raise ValueError(f"No Model-02 validation images found in {source_dir}")
    probabilities = model.predict(np.asarray(images), batch_size=64, verbose=0).reshape(-1)
    return np.asarray(labels, dtype=int), probabilities


def plot_training_curves(results_dir: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    metrics = (
        ("loss", "val_loss", "Focal loss", "Loss (lower is better)"),
        ("auc", "val_auc", "ROC-AUC metric", "AUC"),
        ("recall", "val_recall", "Recall at 0.50", "Recall"),
    )
    for ax, (train_key, val_key, title, ylabel) in zip(axes, metrics):
        for run, label in RUNS:
            history = list(csv.DictReader((results_dir / run / "training_history.csv").open()))
            epochs = np.arange(1, len(history) + 1)
            color = COLORS[label]
            ax.plot(epochs, [float(row[train_key]) for row in history], color=color, alpha=0.45, linestyle="--")
            ax.plot(epochs, [float(row[val_key]) for row in history], color=color, linewidth=2, label=f"{label} validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Model-03 training histories (same numeric-prefix-grouped split)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(
    model02: dict,
    model03: dict[str, dict],
    model03_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> None:
    metric_specs = (
        ("roc_auc", "ROC-AUC"),
        ("pr_auc_trapezoidal", "PR-AUC (trapezoidal)"),
        ("at_0.50", "Precision"),
        ("at_0.50", "Recall"),
        ("at_0.50", "Specificity"),
        ("at_0.50", "F1"),
    )
    labels = ["Model-02\noriginal val", "Model-03\nalpha 0.25", "Model-03\nalpha 0.75"]
    data = []
    for key, subkey in metric_specs:
        if key == "pr_auc_trapezoidal":
            values = [float(model02[key])]
            for run, _ in RUNS:
                y_true, scores = model03_scores[run]
                precision, recall, _ = precision_recall_curve(y_true, scores)
                values.append(float(auc(recall, precision)))
            data.append(values)
            continue
        if key == "at_0.50":
            values = [
                float(model02[subkey.lower() if subkey != "F1" else "f1"]),
                float(model03["focal_alpha_025"]["test"]["at_0.50"][subkey.lower() if subkey != "F1" else "f1"]),
                float(model03["focal_alpha_075"]["test"]["at_0.50"][subkey.lower() if subkey != "F1" else "f1"]),
            ]
        else:
            values = [
                float(model02[key]),
                float(model03["focal_alpha_025"]["test"][key]),
                float(model03["focal_alpha_075"]["test"][key]),
            ]
        data.append(values)
    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    x = np.arange(len(metric_specs))
    width = 0.24
    colors = [COLORS["Model-02 original validation"], COLORS["Model-03 alpha 0.25"], COLORS["Model-03 alpha 0.75"]]
    for i, (label, color) in enumerate(zip(labels, colors)):
        bars = ax.bar(x + (i - 1) * width, [row[i] for row in data], width, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, [name for _, name in metric_specs])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Reported Model-02 validation vs. Model-03 prefix-grouped test")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=3, loc="upper left")
    ax.text(
        0.01,
        -0.19,
        "Caution: Model-02 is historical validation; Model-03 uses a filename-prefix-grouped test split.\n"
        "Prefixes do not verify person identity. Compare alpha 0.25 vs 0.75 on the same split; historical bars are context.",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox={"facecolor": "#FFF4D6", "edgecolor": "#C49A36", "boxstyle": "round,pad=0.5"},
    )
    fig.tight_layout()
    fig.savefig(output_dir / "model_metric_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(
    model02: dict, model03: dict[str, dict], output_dir: Path
) -> None:
    matrices = [
        ("Model-02 original validation\nthreshold 0.50", [model02["tn"], model02["fp"], model02["fn"], model02["tp"]]),
        ("Model-03 alpha 0.25\nprefix-grouped test, 0.50", [model03["focal_alpha_025"]["test"]["at_0.50"][key] for key in ("tn", "fp", "fn", "tp")]),
        ("Model-03 alpha 0.75\nprefix-grouped test, 0.50", [model03["focal_alpha_075"]["test"]["at_0.50"][key] for key in ("tn", "fp", "fn", "tp")]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for ax, (title, values) in zip(axes, matrices):
        matrix = np.asarray(values).reshape(2, 2)
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks((0, 1), ("Normal", "Flushing"))
        ax.set_yticks((0, 1), ("Normal", "Flushing"))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title, fontsize=10)
        threshold = matrix.max() / 2
        for row in range(2):
            for col in range(2):
                ax.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center",
                        color="white" if matrix[row, col] > threshold else "black", fontsize=13, fontweight="bold")
    fig.suptitle("Confusion matrices: Normal=0, Flushing=1", fontsize=13)
    fig.text(
        0.5,
        -0.01,
        "Different split membership; Model-03 filename prefixes do not verify person identities.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_curves(
    model02_scores: tuple[np.ndarray, np.ndarray],
    model03_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    all_scores = [("Model-02 original validation", *model02_scores)]
    all_scores.extend((label, *model03_scores[run]) for run, label in RUNS)
    for label, labels, scores in all_scores:
        color = COLORS[label]
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color, linewidth=2, label=f"{label} (AUC={roc_auc:.3f})")
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)
        axes[1].plot(
            recall,
            precision,
            color=color,
            linewidth=2,
            label=f"{label} (trapezoidal PR-AUC={pr_auc:.3f})",
        )
    axes[0].plot((0, 1), (0, 1), "k--", linewidth=1, alpha=0.6)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curves")
    axes[1].set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision–recall curves (trapezoidal PR-AUC)",
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Ranking performance across thresholds (split contexts differ)", fontsize=13)
    fig.text(
        0.5,
        -0.02,
        "Model-02 uses original validation; Model-03 uses a filename-prefix-grouped test split. Person-level separation is unverified.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "roc_pr_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_tradeoff(model03_scores: dict[str, tuple[np.ndarray, np.ndarray]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for run, label in RUNS:
        labels, scores = model03_scores[run]
        precision, recall, _ = precision_recall_curve(labels, scores)
        ax.plot(recall, precision, color=COLORS[label], linewidth=2, label=label)
        ax.scatter(
            [read_json(output_dir.parent / run / "metrics.json")["test"]["at_0.50"]["recall"]],
            [read_json(output_dir.parent / run / "metrics.json")["test"]["at_0.50"]["precision"]],
            color=COLORS[label], edgecolor="black", marker="o", zorder=4, label=f"{label} threshold 0.50",
        )
        selected = read_json(output_dir.parent / run / "metrics.json")["test"]["validation_selected_thresholds"]
        choice = selected["0.95"]
        if choice is not None:
            ax.scatter([choice["recall"]], [choice["precision"]], color=COLORS[label],
                       edgecolor="black", marker="*", s=150, zorder=5,
                       label=f"{label} val-selected, precision floor 0.95")
    ax.set(
        xlabel="Test recall",
        ylabel="Test precision",
        title="Precision–recall operating points on prefix-grouped test",
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.text(
        0.1,
        0.02,
        "Stars: threshold chosen on validation, then applied unchanged to test. "
        "Person-level separation is unverified.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 0.76, 1))
    fig.savefig(output_dir / "threshold_tradeoff.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_split_distribution(split_counts_path: Path, output_dir: Path) -> None:
    rows = list(csv.DictReader(split_counts_path.open(newline="", encoding="utf-8")))
    splits = [row["split"].title() for row in rows]
    normal = np.asarray([int(row["normal_images"]) for row in rows])
    flushing = np.asarray([int(row["flushing_images"]) for row in rows])
    prefix_groups = np.asarray([int(row["prefix_groups"]) for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(splits))
    total = normal + flushing
    bars = axes[0].bar(x, prefix_groups, color="#4C78A8")
    axes[0].bar_label(bars, fmt="{:,.0f}", padding=3)
    axes[0].set_xticks(x, splits)
    axes[0].set_ylabel("Filename-prefix groups")
    axes[0].set_title("Prefix groups per split")
    axes[0].grid(axis="y", alpha=0.2)

    width = 0.34
    bars_normal = axes[1].bar(x - width / 2, normal, width, label="Normal", color="#4C9F70")
    bars_flushing = axes[1].bar(x + width / 2, flushing, width, label="Flushing", color="#D96B5F")
    axes[1].bar_label(bars_normal, fmt="{:,.0f}", padding=3, fontsize=8)
    axes[1].bar_label(bars_flushing, fmt="{:,.0f}", padding=3, fontsize=8)
    axes[1].set_xticks(x, [f"{split}\n{count:,} total" for split, count in zip(splits, total)])
    axes[1].set_ylabel("Image patches")
    axes[1].set_title("Patch counts by class")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].set_ylim(0, max(normal.max(), flushing.max()) * 1.16)
    fig.suptitle("Model-03 filename-prefix-grouped data split", fontsize=13)
    fig.text(
        0.5,
        0.01,
        "Filename prefixes are not verified person identities; person-level separation is unconfirmed.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "split_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Old split_dataset folder used to reconstruct Model-02 validation curves.")
    parser.add_argument("--model02-path", type=Path, required=True,
                        help="Path to the original best_skin_cnn.keras checkpoint.")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    output_dir = results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    model02_metrics = read_json(results_dir / "model02_reference_metrics.json")
    model03_metrics = {run: read_json(results_dir / run / "metrics.json") for run, _ in RUNS}
    model02_scores = load_model02_validation(
        args.model02_path.expanduser().resolve(),
        args.source_dir.expanduser().resolve(),
    )
    model03_scores = {
        run: load_scores(results_dir / run / "test_scores.csv")
        for run, _ in RUNS
    }

    plot_training_curves(results_dir, output_dir)
    plot_metric_comparison(model02_metrics, model03_metrics, model03_scores, output_dir)
    plot_confusion_matrices(model02_metrics, model03_metrics, output_dir)
    plot_curves(model02_scores, model03_scores, output_dir)
    plot_threshold_tradeoff(model03_scores, output_dir)
    plot_split_distribution(results_dir / "split_counts.csv", output_dir)
    print(f"Generated visualizations in {output_dir}")


if __name__ == "__main__":
    main()
