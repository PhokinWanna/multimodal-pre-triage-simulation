"""Train and compare CNN loss settings on a numeric-prefix-grouped split."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 35
SEED = 42
CLASSES = {"Normal": 0, "Flushing": 1}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PRECISION_FLOORS = (0.95, 0.90, 0.80, 0.70)


def prefix_group_from_name(filename: str) -> str:
    match = re.match(r"^(\d+)_", filename)
    if not match:
        raise ValueError(f"Filename does not start with a numeric source ID: {filename}")
    return match.group(1)


def collect_records(source_dir: Path) -> dict[str, list[tuple[Path, int]]]:
    records: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for split in ("train", "val", "test"):
        for class_name, label in CLASSES.items():
            class_dir = source_dir / split / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Required class directory is missing: {class_dir}")
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    records[prefix_group_from_name(path.name)].append((path, label))
    if not records:
        raise ValueError(f"No supported images found under {source_dir}")
    return dict(records)


def make_grouped_splits(
    records: dict[str, list[tuple[Path, int]]],
) -> dict[str, list[tuple[Path, int]]]:
    prefix_groups = sorted(records)
    strata = [
        "+".join(
            str(label)
            for label in sorted({label for _, label in records[prefix_group]})
        )
        for prefix_group in prefix_groups
    ]
    train_ids, remainder_ids = train_test_split(
        prefix_groups,
        test_size=0.30,
        random_state=SEED,
        stratify=strata,
    )
    remainder_strata = [
        "+".join(
            str(label)
            for label in sorted({label for _, label in records[prefix_group]})
        )
        for prefix_group in remainder_ids
    ]
    val_ids, test_ids = train_test_split(
        remainder_ids,
        test_size=0.50,
        random_state=SEED,
        stratify=remainder_strata,
    )
    groups = {"train": train_ids, "val": val_ids, "test": test_ids}
    if any(
        set(groups[left]) & set(groups[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise RuntimeError("Filename-prefix group overlap detected in generated split")
    return {
        split: sorted(
            (record for prefix_group in ids for record in records[prefix_group]),
            key=lambda record: (record[1], record[0].name),
        )
        for split, ids in groups.items()
    }


class ImageSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        records: list[tuple[Path, int]],
        batch_size: int = BATCH_SIZE,
        shuffle: bool = False,
        seed: int = SEED,
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
        indices = self.indices[batch_index * self.batch_size : (batch_index + 1) * self.batch_size]
        images = np.empty((len(indices), *IMAGE_SIZE, 3), dtype=np.float32)
        labels = np.empty((len(indices), 1), dtype=np.float32)
        for output_index, record_index in enumerate(indices):
            path, label = self.records[int(record_index)]
            image = cv2.imread(str(path))
            if image is None:
                raise OSError(f"Unable to read image: {path}")
            image = cv2.resize(image, IMAGE_SIZE)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            images[output_index] = image.astype(np.float32) / 255.0
            labels[output_index, 0] = label
        return images, labels

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)


class BinaryFocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha: float, gamma: float = 2.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p_t = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1.0), self.alpha, 1.0 - self.alpha)
        return tf.reduce_mean(
            -alpha_t * tf.pow(1.0 - p_t, self.gamma) * tf.math.log(p_t)
        )

    def get_config(self) -> dict:
        return {**super().get_config(), "alpha": self.alpha, "gamma": self.gamma}


def create_model(alpha: float) -> tf.keras.Model:
    layers = tf.keras.layers
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)
    x = layers.Conv2D(32, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.SeparableConv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SeparableConv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="CIELAB_Skin_CNN")
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4),
        loss=BinaryFocalLoss(
            alpha=alpha,
            name=f"binary_focal_loss_alpha_{alpha:.2f}",
        ),
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def as_arrays(sequence: ImageSequence) -> tuple[np.ndarray, np.ndarray]:
    image_batches, label_batches = zip(*(sequence[index] for index in range(len(sequence))))
    return np.concatenate(image_batches), np.concatenate(label_batches).reshape(-1).astype(int)


def metrics_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def thresholds_for_precision(
    labels: np.ndarray, scores: np.ndarray, precision_floors: tuple[float, ...]
) -> dict[str, float | None]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    selected: dict[str, float | None] = {}
    for floor in precision_floors:
        valid = np.flatnonzero(precision[:-1] >= floor)
        if not len(valid):
            selected[f"{floor:.2f}"] = None
            continue
        best = valid[np.argmax(recall[valid])]
        selected[f"{floor:.2f}"] = float(thresholds[best])
    return selected


def evaluate(
    labels: np.ndarray,
    scores: np.ndarray,
    selected_thresholds: dict[str, float | None] | None = None,
) -> dict:
    output = {
        "sample_count": int(len(labels)),
        "positive_count": int(np.sum(labels)),
        "negative_count": int(len(labels) - np.sum(labels)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "at_0.50": metrics_at_threshold(labels, scores, 0.50),
    }
    if selected_thresholds is not None:
        output["validation_selected_thresholds"] = {
            floor: (
                metrics_at_threshold(labels, scores, threshold)
                if threshold is not None
                else None
            )
            for floor, threshold in selected_thresholds.items()
        }
    return output


def write_predictions(
    path: Path, labels: np.ndarray, scores: np.ndarray
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sample_index", "label", "probability"))
        writer.writerows(
            (index, int(label), f"{float(score):.9f}")
            for index, (label, score) in enumerate(zip(labels, scores))
        )


def write_split_counts(
    path: Path, splits: dict[str, list[tuple[Path, int]]], records: dict[str, list[tuple[Path, int]]]
) -> None:
    owner = {}
    for split, samples in splits.items():
        sample_paths = {record[0] for record in samples}
        for prefix_group, group_records in records.items():
            if any(record[0] in sample_paths for record in group_records):
                owner[prefix_group] = split
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("split", "prefix_groups", "normal_images", "flushing_images"))
        for split in ("train", "val", "test"):
            samples = splits[split]
            writer.writerow(
                (
                    split,
                    sum(value == split for value in owner.values()),
                    sum(label == 0 for _, label in samples),
                    sum(label == 1 for _, label in samples),
                )
            )


def train_run(
    name: str,
    alpha: float,
    splits: dict[str, list[tuple[Path, int]]],
    output_dir: Path,
) -> dict:
    tf.keras.backend.clear_session()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_sequence = ImageSequence(splits["train"], shuffle=True)
    val_sequence = ImageSequence(splits["val"])
    test_sequence = ImageSequence(splits["test"])
    model = create_model(alpha)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=8, mode="max", restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(run_dir / "best_model.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.CSVLogger(run_dir / "training_history.csv"),
    ]
    history = model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

    val_images, val_labels = as_arrays(val_sequence)
    test_images, test_labels = as_arrays(test_sequence)
    val_scores = model.predict(val_images, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
    test_scores = model.predict(test_images, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
    selected_thresholds = thresholds_for_precision(
        val_labels, val_scores, PRECISION_FLOORS
    )
    result = {
        "name": name,
        "focal_alpha_positive_class": alpha,
        "epochs_completed": len(history.history["loss"]),
        "best_validation_auc_keras_approx": float(max(history.history["val_auc"])),
        "validation": evaluate(val_labels, val_scores, selected_thresholds),
        "test": evaluate(test_labels, test_scores, selected_thresholds),
        "thresholds_selected_on_validation": selected_thresholds,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    write_predictions(run_dir / "validation_scores.csv", val_labels, val_scores)
    write_predictions(run_dir / "test_scores.csv", test_labels, test_scores)
    print(f"\n{name}: val ROC-AUC={result['validation']['roc_auc']:.4f}, "
          f"test ROC-AUC={result['test']['roc_auc']:.4f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "results",
    )
    args = parser.parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    records = collect_records(source_dir)
    splits = make_grouped_splits(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_split_counts(output_dir / "split_counts.csv", splits, records)
    metadata = {
        "seed": SEED,
        "source_image_count": sum(map(len, records.values())),
        "unique_numeric_prefix_groups": len(records),
        "class_mapping": CLASSES,
        "split_method": "70/15/15 numeric-filename-prefix-grouped, stratified by class presence",
        "preprocessing": "OpenCV resize to 64x64, BGR to CIELAB, float32 / 255",
        "source_directory": source_dir.name,
        "warning": (
            "Leading numeric filename prefixes are used only as grouping keys, "
            "not verified person identities. Different prefixes may refer to "
            "the same person, so person-disjointness and absence of identity "
            "leakage are not established."
        ),
    }
    (output_dir / "experiment.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print("Split sizes:", {name: len(samples) for name, samples in splits.items()})

    results = [
        train_run("focal_alpha_025", 0.25, splits, output_dir),
        train_run("focal_alpha_075", 0.75, splits, output_dir),
    ]
    (output_dir / "comparison.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nSaved experiment results to {output_dir}")


if __name__ == "__main__":
    main()
