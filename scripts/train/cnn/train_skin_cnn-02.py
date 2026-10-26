"""Train and evaluate the binary Normal/Flushing CIELAB CNN."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


keras = tf.keras
layers = keras.layers

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 35
LEARNING_RATE = 1e-4
PATIENCE = 8
CLASS_MAPPING = {"Normal": 0, "Flushing": 1}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PRECISION_FLOORS = (0.95, 0.90, 0.80, 0.70)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "dataset-cnn" / "splits"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "cnn" / "model02" / "training_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing train/val/test class folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "New, non-existing directory for this run. Defaults to a timestamped "
            "directory under results/cnn/model02/training_runs/."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def make_output_dir(output_dir: Path | None) -> Path:
    if output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = DEFAULT_OUTPUT_ROOT / f"run_{timestamp}"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def prefix_group_from_name(filename: str) -> str:
    match = re.match(r"^(\d+)_", filename)
    if not match:
        raise ValueError(
            f"Cannot verify filename-prefix grouping; filename must start with "
            f"a numeric prefix followed by '_': {filename}"
        )
    return match.group(1)


def collect_split_records(
    data_dir: Path, split_name: str
) -> list[tuple[Path, int, str]]:
    split_dir = data_dir / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Required split directory is missing: {split_dir}")

    records: list[tuple[Path, int, str]] = []
    class_counts: Counter[str] = Counter()
    for class_name, label in CLASS_MAPPING.items():
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Required class directory is missing: {class_dir}"
            )
        class_files = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not class_files:
            raise ValueError(f"No supported images found in {class_dir}")
        for path in class_files:
                prefix_group = prefix_group_from_name(path.name)
                records.append((path, label, prefix_group))
        class_counts[class_name] = len(class_files)

    if len(class_counts) != len(CLASS_MAPPING) or any(
        count == 0 for count in class_counts.values()
    ):
        raise ValueError(f"{split_name} split must contain both target classes")
    return records


def validate_prefix_group_disjointness(
    splits: dict[str, list[tuple[Path, int, str]]]
) -> None:
    prefix_sets = {
        split_name: {prefix_group for _, _, prefix_group in records}
        for split_name, records in splits.items()
    }
    overlaps = []
    split_names = tuple(prefix_sets)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            shared = prefix_sets[left] & prefix_sets[right]
            if shared:
                overlaps.append((left, right, sorted(shared)[:10], len(shared)))
    if overlaps:
        details = "; ".join(
            f"{left}/{right}: {count} shared IDs (examples: {examples})"
            for left, right, examples, count in overlaps
        )
        raise ValueError(
            "Numeric filename prefixes occur in more than one split. Refusing "
            f"training/evaluation on a potentially leaky split. Rebuild it. {details}"
        )


class ImageSequence(keras.utils.Sequence):
    """Load image batches on demand so the entire image set stays out of RAM."""

    def __init__(
        self,
        records: list[tuple[Path, int, str]],
        batch_size: int,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        self.records = records
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(records))
        self.on_epoch_end()

    def __len__(self) -> int:
        return (len(self.records) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        start = batch_index * self.batch_size
        indices = self.indices[start : start + self.batch_size]
        images = np.empty((len(indices), *IMG_SIZE, 3), dtype=np.float32)
        labels = np.empty((len(indices), 1), dtype=np.float32)
        for output_index, record_index in enumerate(indices):
            path, label, _ = self.records[int(record_index)]
            image = cv2.imread(str(path))
            if image is None:
                raise OSError(f"OpenCV could not decode image: {path}")
            image = cv2.resize(image, IMG_SIZE)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            images[output_index] = image.astype(np.float32) / 255.0
            labels[output_index, 0] = label
        return images, labels

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)


class BinaryFocalLoss(keras.losses.Loss):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        name: str = "binary_focal_loss",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p_t = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        alpha_t = tf.where(
            tf.equal(y_true, 1.0), self.alpha, 1.0 - self.alpha
        )
        losses = (
            -alpha_t
            * tf.pow(1.0 - p_t, self.gamma)
            * tf.math.log(p_t)
        )
        return tf.reduce_mean(losses, axis=-1)

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "gamma": self.gamma,
            "alpha": self.alpha,
        }


def create_model() -> keras.Model:
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)
    x = layers.Conv2D(
        32, (3, 3), strides=2, padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.SeparableConv2D(
        64, (3, 3), padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SeparableConv2D(
        128, (3, 3), padding="same", activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="CIELAB_Skin_CNN")
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss=BinaryFocalLoss(),
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def labels_from_records(records: list[tuple[Path, int, str]]) -> np.ndarray:
    return np.asarray([label for _, label, _ in records], dtype=int)


def predict_sequence(model: keras.Model, sequence: ImageSequence) -> np.ndarray:
    predictions = []
    for index in range(len(sequence)):
        images, _ = sequence[index]
        predictions.append(model.predict(images, verbose=0).reshape(-1))
    return np.concatenate(predictions)


def metrics_at_threshold(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_validation_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | None]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    selected: dict[str, float | None] = {}
    for floor in PRECISION_FLOORS:
        eligible = np.flatnonzero(precision[:-1] >= floor)
        if not len(eligible):
            selected[f"{floor:.2f}"] = None
            continue
        best = eligible[np.argmax(recall[eligible])]
        selected[f"{floor:.2f}"] = float(thresholds[best])
    return selected


def evaluate_scores(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    selected_thresholds: dict[str, float | None] | None = None,
) -> dict:
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, probabilities)
    result = {
        "sample_count": int(len(y_true)),
        "positive_count": int(np.sum(y_true)),
        "negative_count": int(len(y_true) - np.sum(y_true)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "pr_auc_trapezoidal": float(auc(recall_curve, precision_curve)),
        "at_threshold_0.50": metrics_at_threshold(y_true, probabilities, 0.50),
    }
    if selected_thresholds is not None:
        result["validation_selected_thresholds"] = {
            floor: (
                metrics_at_threshold(y_true, probabilities, threshold)
                if threshold is not None
                else None
            )
            for floor, threshold in selected_thresholds.items()
        }
    return result


def export_quantized_tflite(
    model: keras.Model,
    validation_sequence: ImageSequence,
    export_path: Path,
) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        sample_count = 0
        for index in range(len(validation_sequence)):
            images, _ = validation_sequence[index]
            for image in images:
                yield [np.expand_dims(image, axis=0)]
                sample_count += 1
                if sample_count >= 150:
                    return

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()
    export_path.write_bytes(tflite_model)


def write_predictions(
    path: Path, labels: np.ndarray, probabilities: np.ndarray
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sample_index", "label", "probability"))
        writer.writerows(
            (index, int(label), f"{float(probability):.9f}")
            for index, (label, probability) in enumerate(zip(labels, probabilities))
        )


def save_evaluation_plot(
    output_path: Path,
    history: keras.callbacks.History,
    y_test: np.ndarray,
    test_probabilities: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    history_metrics = (
        ("loss", "val_loss", "Focal loss"),
        ("auc", "val_auc", "ROC-AUC"),
        ("recall", "val_recall", "Recall at threshold 0.50"),
    )
    for ax, (train_key, val_key, title) in zip(axes[0], history_metrics):
        ax.plot(history.history[train_key], label="Train")
        ax.plot(history.history[val_key], label="Validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend()

    fpr, tpr, _ = roc_curve(y_test, test_probabilities)
    test_auc = roc_auc_score(y_test, test_probabilities)
    axes[1, 0].plot(fpr, tpr, label=f"Test ROC-AUC={test_auc:.4f}")
    axes[1, 0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[1, 0].set_title("Held-out test ROC curve")
    axes[1, 0].set_xlabel("False positive rate")
    axes[1, 0].set_ylabel("Recall")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    precision, recall, _ = precision_recall_curve(y_test, test_probabilities)
    pr_auc = auc(recall, precision)
    axes[1, 1].plot(recall, precision, label=f"PR-AUC={pr_auc:.4f}")
    axes[1, 1].set_title("Held-out test precision–recall curve")
    axes[1, 1].set_xlabel("Recall")
    axes[1, 1].set_ylabel("Precision")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)

    test_metrics = metrics_at_threshold(y_test, test_probabilities, 0.50)
    matrix = np.asarray(
        [
            [test_metrics["tn"], test_metrics["fp"]],
            [test_metrics["fn"], test_metrics["tp"]],
        ]
    )
    axes[1, 2].imshow(matrix, cmap="Blues")
    axes[1, 2].set_xticks((0, 1), ("Normal", "Flushing"))
    axes[1, 2].set_yticks((0, 1), ("Normal", "Flushing"))
    axes[1, 2].set_xlabel("Predicted")
    axes[1, 2].set_ylabel("Actual")
    axes[1, 2].set_title("Test confusion matrix @ 0.50")
    for row in range(2):
        for col in range(2):
            axes[1, 2].text(
                col,
                row,
                f"{matrix[row, col]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, col] > matrix.max() / 2 else "black",
            )

    fig.suptitle("Normal vs Flushing CNN: training and held-out evaluation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_metrics_text(path: Path, metrics: dict) -> None:
    lines = [
        "SKIN CNN - HELD-OUT TEST METRICS",
        "=" * 42,
        f"ROC-AUC: {metrics['test']['roc_auc']:.4f}",
        f"Average precision: {metrics['test']['average_precision']:.4f}",
        f"PR-AUC (trapezoidal): {metrics['test']['pr_auc_trapezoidal']:.4f}",
        "",
        "Threshold 0.50",
    ]
    for key, value in metrics["test"]["at_threshold_0.50"].items():
        lines.append(f"{key}: {value}")
    lines.extend(
        [
            "",
            "Thresholds selected on validation and applied unchanged to test",
        ]
    )
    for precision_floor, selected in metrics["test"][
        "validation_selected_thresholds"
    ].items():
        lines.append(f"Validation precision floor {precision_floor}: {selected}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_training(args: argparse.Namespace) -> Path:
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}. Expected train/val/test "
            "folders with Normal and Flushing subfolders."
        )
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive integers")

    set_reproducibility(args.seed)

    splits = {
        split_name: collect_split_records(data_dir, split_name)
        for split_name in ("train", "val", "test")
    }
    validate_prefix_group_disjointness(splits)
    output_dir = make_output_dir(args.output_dir)
    run_started_utc = datetime.now(timezone.utc).isoformat()

    train_sequence = ImageSequence(
        splits["train"], args.batch_size, shuffle=True, seed=args.seed
    )
    val_sequence = ImageSequence(splits["val"], args.batch_size)
    test_sequence = ImageSequence(splits["test"], args.batch_size)

    model = create_model()
    history_path = output_dir / "training_history.csv"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=PATIENCE,
            mode="max",
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            str(output_dir / "best_skin_cnn.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
        keras.callbacks.CSVLogger(str(history_path)),
    ]

    print(f"Dataset: {data_dir}")
    print(f"Output:  {output_dir}")
    print(
        "Images per split:",
        {name: len(records) for name, records in splits.items()},
    )
    print("Numeric filename-prefix groups are disjoint across train/val/test.")
    history = model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    best_model_path = output_dir / "best_skin_cnn.keras"
    model = keras.models.load_model(
        best_model_path,
        custom_objects={"BinaryFocalLoss": BinaryFocalLoss},
        compile=False,
    )

    val_probabilities = predict_sequence(model, val_sequence)
    test_probabilities = predict_sequence(model, test_sequence)
    y_val = labels_from_records(splits["val"])
    y_test = labels_from_records(splits["test"])
    selected_thresholds = select_validation_thresholds(y_val, val_probabilities)

    test_metrics = evaluate_scores(
        y_test, test_probabilities, selected_thresholds
    )
    val_metrics = evaluate_scores(y_val, val_probabilities)

    export_quantized_tflite(
        model,
        val_sequence,
        output_dir / "skin_classifier.tflite",
    )

    report = classification_report(
        y_test,
        (test_probabilities >= 0.50).astype(int),
        labels=[0, 1],
        target_names=("Normal", "Flushing"),
        digits=4,
        zero_division=0,
    )
    print("\nHeld-out test classification report @ 0.50")
    print(report)

    metrics = {
        "class_mapping": CLASS_MAPPING,
        "validation": val_metrics,
        "test": test_metrics,
        "classification_report_test_at_0.50": report,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    write_metrics_text(output_dir / "metrics_summary.txt", metrics)
    write_predictions(
        output_dir / "validation_scores.csv", y_val, val_probabilities
    )
    write_predictions(output_dir / "test_scores.csv", y_test, test_probabilities)
    save_evaluation_plot(
        output_dir / "training_report.png",
        history,
        y_test,
        test_probabilities,
    )
    with (output_dir / "model_architecture.txt").open(
        "w", encoding="utf-8"
    ) as stream:
        model.summary(print_fn=lambda line: stream.write(line + "\n"))

    config = {
        "run_started_utc": run_started_utc,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history.history["loss"]),
        "batch_size": args.batch_size,
        "learning_rate": LEARNING_RATE,
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "early_stopping": {"monitor": "val_auc", "patience": PATIENCE, "mode": "max"},
        "image_size": list(IMG_SIZE),
        "class_mapping": CLASS_MAPPING,
        "preprocessing": "OpenCV resize 64x64; BGR to CIELAB; float32 / 255",
        "split_counts": {
            name: {
                "images": len(records),
                "prefix_groups": len(
                    {prefix_group for _, _, prefix_group in records}
                ),
                "Normal": sum(label == 0 for _, label, _ in records),
                "Flushing": sum(label == 1 for _, label, _ in records),
            }
            for name, records in splits.items()
        },
        "prefix_group_disjointness_verified": True,
        "person_disjointness_verified": False,
        "tensorflow_version": tf.__version__,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "scikit_learn_version": __import__("sklearn").__version__,
        "model_parameter_count": int(model.count_params()),
        "model_size_bytes": best_model_path.stat().st_size,
        "tflite_size_bytes": (output_dir / "skin_classifier.tflite").stat().st_size,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nCompleted. All run artifacts are in: {output_dir}")
    return output_dir


if __name__ == "__main__":
    run_training(parse_args())
