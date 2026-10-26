"""Generate CNN report charts from verified Model-02/Model-03 metric JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_auc_score


MODEL02_LABEL = "Model-02 original validation"
MODEL03_RUNS = (
    ("alpha_025", "Model-03 alpha 0.25"),
    ("alpha_075", "Model-03 alpha 0.75"),
)
METRICS = (
    ("roc_auc", "ROC-AUC"),
    ("pr_auc_trapezoidal", "PR-AUC\n(trapezoidal)"),
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("specificity", "Specificity"),
    ("f1", "F1"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model02-metrics", type=Path, required=True)
    parser.add_argument("--model03-alpha025-metrics", type=Path, required=True)
    parser.add_argument("--model03-alpha075-metrics", type=Path, required=True)
    parser.add_argument("--model03-alpha025-scores", type=Path, required=True)
    parser.add_argument("--model03-alpha075-scores", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory; it must not already exist.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Metrics JSON not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON metrics file: {path}") from exc


def check_probability_metric(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1; got {value}.")


def normalize_model02(metrics: dict) -> dict:
    result = {
        "name": MODEL02_LABEL,
        "split": metrics["split"],
        "threshold": float(metrics["threshold"]),
        "sample_count": int(metrics["validation_samples"]),
        "roc_auc": float(metrics["roc_auc"]),
        "pr_auc_trapezoidal": float(metrics["pr_auc_trapezoidal"]),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "specificity": float(metrics["specificity"]),
        "f1": float(metrics["f1"]),
        "tn": int(metrics["tn"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "tp": int(metrics["tp"]),
    }
    return validate_evaluation(result)


def load_score_csv(path: Path, expected_count: int) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Model-03 score CSV not found: {path}")
    labels: list[int] = []
    scores: list[float] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"label", "probability"}.issubset(reader.fieldnames or set()):
            raise ValueError(f"Score CSV must contain label and probability columns: {path}")
        for row in reader:
            label = int(row["label"])
            score = float(row["probability"])
            if label not in (0, 1) or not 0.0 <= score <= 1.0:
                raise ValueError(f"Invalid label/probability in score CSV: {path}")
            labels.append(label)
            scores.append(score)
    if len(labels) != expected_count:
        raise ValueError(
            f"Score count in {path} is {len(labels)}, expected {expected_count}."
        )
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)


def normalize_model03(
    metrics: dict,
    scores_path: Path,
    run_key: str,
    label: str,
) -> dict:
    run = metrics
    if "test" not in run or "focal_alpha_positive_class" not in run:
        raise ValueError(f"Unexpected Model-03 metrics schema for {label}.")
    test = run["test"]
    operating_point = test["at_0.50"]
    y_true, probabilities = load_score_csv(scores_path, int(test["sample_count"]))
    score_matrix = confusion_matrix(
        y_true,
        (probabilities >= float(operating_point["threshold"])).astype(int),
        labels=[0, 1],
    )
    tn, fp, fn, tp = (int(value) for value in score_matrix.ravel())
    if (tn, fp, fn, tp) != tuple(
        int(operating_point[key]) for key in ("tn", "fp", "fn", "tp")
    ):
        raise ValueError(
            f"Score CSV confusion counts do not match the recorded metrics for {label}."
        )
    roc_auc = float(roc_auc_score(y_true, probabilities))
    if not np.isclose(roc_auc, float(test["roc_auc"]), atol=1e-6):
        raise ValueError(
            f"Score CSV ROC-AUC does not match the recorded metric for {label}: "
            f"CSV={roc_auc:.8f}, JSON={float(test['roc_auc']):.8f}."
        )
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, probabilities)
    pr_auc_trapezoidal = float(auc(recall_curve, precision_curve))
    result = {
        "name": label,
        "split": "Filename-prefix-grouped test",
        "threshold": float(operating_point["threshold"]),
        "sample_count": int(test["sample_count"]),
        "roc_auc": roc_auc,
        "pr_auc_trapezoidal": pr_auc_trapezoidal,
        "average_precision": float(test["average_precision"]),
        "accuracy": float(operating_point["accuracy"]),
        "precision": float(operating_point["precision"]),
        "recall": float(operating_point["recall"]),
        "specificity": float(operating_point["specificity"]),
        "f1": float(operating_point["f1"]),
        "tn": int(operating_point["tn"]),
        "fp": int(operating_point["fp"]),
        "fn": int(operating_point["fn"]),
        "tp": int(operating_point["tp"]),
        "focal_alpha_positive_class": float(run["focal_alpha_positive_class"]),
        "run_key": run_key,
    }
    return validate_evaluation(result)


def validate_evaluation(result: dict) -> dict:
    if result["sample_count"] != result["tn"] + result["fp"] + result["fn"] + result["tp"]:
        raise ValueError(
            f"Confusion counts do not sum to sample count for {result['name']}."
        )
    for metric_name in (
        "roc_auc",
        "pr_auc_trapezoidal",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "threshold",
    ):
        check_probability_metric(f"{result['name']} {metric_name}", result[metric_name])
    if "average_precision" in result:
        check_probability_metric(
            f"{result['name']} average_precision", result["average_precision"]
        )
    denominator = result["tn"] + result["fp"] + result["fn"] + result["tp"]
    computed = {
        "accuracy": (result["tp"] + result["tn"]) / denominator,
        "precision": result["tp"] / (result["tp"] + result["fp"]) if result["tp"] + result["fp"] else 0.0,
        "recall": result["tp"] / (result["tp"] + result["fn"]) if result["tp"] + result["fn"] else 0.0,
        "specificity": result["tn"] / (result["tn"] + result["fp"]) if result["tn"] + result["fp"] else 0.0,
        "f1": (
            2 * result["tp"] / (2 * result["tp"] + result["fp"] + result["fn"])
            if 2 * result["tp"] + result["fp"] + result["fn"]
            else 0.0
        ),
    }
    for metric_name, value in computed.items():
        if not np.isclose(value, result[metric_name], atol=0.0015):
            raise ValueError(
                f"{result['name']} {metric_name} conflicts with its confusion "
                f"matrix: JSON={result[metric_name]:.6f}, recomputed={value:.6f}."
            )
    return result


def write_metrics_csv(path: Path, evaluations: list[dict]) -> None:
    fields = (
        "name", "split", "sample_count", "threshold", "roc_auc",
        "pr_auc_trapezoidal", "average_precision", "accuracy", "precision", "recall",
        "specificity", "f1", "tn", "fp", "fn", "tp",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for evaluation in evaluations:
            writer.writerow({field: evaluation.get(field, "") for field in fields})


def plot_metrics(output_path: Path, evaluations: list[dict]) -> None:
    metric_names = [name for name, _ in METRICS]
    labels = [
        "ROC-\nAUC",
        "PR-AUC\n(trapezoidal)",
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1",
    ]
    x = np.arange(len(metric_names))
    width = 0.25
    colors = ("#777777", "#2878B5", "#E07A24")
    fig, ax = plt.subplots(figsize=(14, 7))
    for index, (evaluation, color) in enumerate(zip(evaluations, colors)):
        values = [evaluation[name] for name in metric_names]
        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=f"{evaluation['name']}\n{evaluation['split']}",
            color=color,
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title("CNN metrics at recorded operating point")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncols=1, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.text(
        0.01,
        -0.17,
        "Model-02 and Model-03 use different splits. Model-03 is filename-prefix-grouped; "
        "person-level separation is unverified. Compare Model-03 runs for the controlled result.",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "#FFF4D6", "edgecolor": "#C49A36", "boxstyle": "round,pad=0.4"},
    )
    fig.tight_layout(rect=(0, 0.06, 0.79, 1))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(output_path: Path, evaluations: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(evaluations), figsize=(5 * len(evaluations), 4.8))
    for ax, evaluation in zip(axes, evaluations):
        matrix = np.asarray(
            [[evaluation["tn"], evaluation["fp"]], [evaluation["fn"], evaluation["tp"]]]
        )
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks((0, 1), ("Normal", "Flushing"))
        ax.set_yticks((0, 1), ("Normal", "Flushing"))
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        ax.set_title(
            f"{evaluation['name']}\n{evaluation['split']}\n"
            f"N={evaluation['sample_count']:,}, threshold={evaluation['threshold']:.2f}",
            fontsize=9,
        )
        cutoff = matrix.max() / 2
        for row in range(2):
            for col in range(2):
                ax.text(
                    col,
                    row,
                    f"{matrix[row, col]:,}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, col] > cutoff else "black",
                    fontsize=12,
                    fontweight="bold",
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Confusion matrices (Normal=0, Flushing=1)", fontsize=13)
    fig.text(
        0.5,
        0.01,
        "Split membership differs; Model-03 grouping uses filename prefixes, not verified person IDs. See report.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, evaluations: list[dict]) -> None:
    rows = "\n".join(
        f"| {item['name']} | {item['split']} | {item['sample_count']:,} | "
        f"{item['roc_auc']:.4f} | {item['pr_auc_trapezoidal']:.4f} | "
        f"{item['precision']:.4f} | {item['recall']:.4f} | "
        f"{item['specificity']:.4f} | {item['f1']:.4f} |"
        for item in evaluations
    )
    text = f"""# Reconciled CNN evaluation charts

Charts and the CSV table in this folder are generated from the supplied metric
JSON files and Model-03 test score CSVs. No values are hard-coded into the
plotting code. The script checks
that each confusion matrix sums to its reported sample count and that
accuracy/precision/recall/specificity/F1 agree with the confusion counts
(allowing for rounded summary values).

## Results

| Model | Evaluation split | N | ROC-AUC | PR-AUC (trapezoidal) | Precision | Recall | Specificity | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

The Model-02 row is the historical validation report. The Model-03 rows are
from a rebuilt filename-prefix-grouped test split. These are different splits
and must not be presented as a matched benchmark. Filename prefixes are not
verified person identities; person-level separation and absence of identity
leakage are not established. The controlled comparison is Model-03 alpha 0.25
versus Model-03 alpha 0.75, which share the same prefix-grouped split.

## Legacy chart audit

The former `generate_report_charts.py` contained a hard-coded 3,719-sample
matrix and hard-coded class metrics, with unsupported patient-level and
clinical labels. Those values do not match the verified Model-02 report
(3,791 validation samples; TN=2,955, FP=4, FN=612, TP=220) and are not
reproduced here. Treat charts produced by that old script as stale/unverified.
The existing Model-02 `training_report.png` matches the reported confusion
counts and AUC, but it is still a validation result at threshold 0.50 and
filename-prefix groups crossed boundaries in the old split. Neither the old
nor rebuilt split establishes independent person-level validation.

No new Model-02 training, threshold tuning, or test evaluation was performed
for this chart regeneration. Model-03 metrics are the previously recorded
experiment results.

The Model-02 text report's `PR-AUC` (0.9203) is the trapezoidal area under its
precision–recall curve. Model-03 also stores `average_precision`, a distinct
scikit-learn metric. For the cross-model chart, PR-AUC was recomputed from each
Model-03 test score CSV with the same trapezoidal integration convention as
Model-02. Average precision is retained in Model-03 run JSON but is not mixed
into this cross-model PR-AUC comparison.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite it: {output_dir}"
        )

    model02 = normalize_model02(load_json(args.model02_metrics.expanduser().resolve()))
    model03_025 = normalize_model03(
        load_json(args.model03_alpha025_metrics.expanduser().resolve()),
        args.model03_alpha025_scores.expanduser().resolve(),
        "alpha_025",
        "Model-03 alpha 0.25",
    )
    model03_075 = normalize_model03(
        load_json(args.model03_alpha075_metrics.expanduser().resolve()),
        args.model03_alpha075_scores.expanduser().resolve(),
        "alpha_075",
        "Model-03 alpha 0.75",
    )
    evaluations = [model02, model03_025, model03_075]

    output_dir.mkdir(parents=True, exist_ok=False)
    write_metrics_csv(output_dir / "metrics_comparison.csv", evaluations)
    plot_metrics(output_dir / "metrics_comparison.png", evaluations)
    plot_confusion_matrices(output_dir / "confusion_matrices.png", evaluations)
    write_report(output_dir / "REPORT.md", evaluations)
    (output_dir / "source_metrics.json").write_text(
        json.dumps(evaluations, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Reconciled metric charts and report saved to: {output_dir}")


if __name__ == "__main__":
    main()
