"""Create a safe, deterministic filename-prefix-grouped Normal/Flushing split."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "dataset-cnn"
DEFAULT_SOURCE_DIR = DATA_ROOT / "patches"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "splits"
TARGET_CLASSES = ("Normal", "Flushing")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Extracted patches directory containing Normal/ and Flushing/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="New output directory; it must not already exist.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def prefix_group_from_name(filename: str) -> str:
    match = re.match(r"^(\d+)_", filename)
    if not match:
        raise ValueError(
            "Expected each image filename to start with a numeric prefix and "
            f"'_', got {filename!r}"
        )
    return match.group(1)


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(not np.isfinite(ratio) or ratio <= 0 for ratio in ratios):
        raise ValueError("Train, validation, and test ratios must all be positive.")
    if not np.isclose(sum(ratios), 1.0, atol=1e-8):
        raise ValueError(
            "Train, validation, and test ratios must sum to 1.0; "
            f"got {sum(ratios):.8f}."
        )


def paths_overlap(source_dir: Path, output_dir: Path) -> bool:
    return (
        source_dir == output_dir
        or source_dir in output_dir.parents
        or output_dir in source_dir.parents
    )


def collect_prefix_group_files(
    source_dir: Path,
) -> dict[str, dict[str, list[Path]]]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Extracted dataset directory not found: {source_dir}")

    matched_class_dirs: dict[str, Path] = {}
    entries = [entry for entry in source_dir.iterdir() if entry.is_dir()]
    for class_name in TARGET_CLASSES:
        matches = [entry for entry in entries if entry.name.casefold() == class_name.casefold()]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {class_name!r} directory in {source_dir}; "
                f"found {len(matches)}."
            )
        matched_class_dirs[class_name] = matches[0]

    group_files: dict[str, dict[str, list[Path]]] = defaultdict(
        lambda: {class_name: [] for class_name in TARGET_CLASSES}
    )
    for class_name, class_dir in matched_class_dirs.items():
        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"No supported image patches found in {class_dir}")
        for path in image_paths:
            prefix_group = prefix_group_from_name(path.name)
            group_files[prefix_group][class_name].append(path)

    if not group_files:
        raise ValueError(f"No image patches found under {source_dir}")
    return dict(group_files)


def class_signature(
    prefix_group: str, group_files: dict[str, dict[str, list[Path]]]
) -> str:
    present = [
        class_name
        for class_name in TARGET_CLASSES
        if group_files[prefix_group][class_name]
    ]
    return "+".join(present)


def split_prefix_groups(
    group_files: dict[str, dict[str, list[Path]]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    validate_ratios(train_ratio, val_ratio, test_ratio)
    prefix_groups = sorted(group_files)
    signatures = [
        class_signature(prefix_group, group_files) for prefix_group in prefix_groups
    ]
    signature_counts = Counter(signatures)
    if any(count < 3 for count in signature_counts.values()):
        too_small = {
            signature: count
            for signature, count in signature_counts.items()
            if count < 3
        }
        raise ValueError(
            "Cannot stratify every prefix-group class-signature into train/val/test; "
            f"each signature needs at least 3 prefix groups. Counts: {too_small}"
        )

    try:
        train_ids, remainder_ids, _, remainder_signatures = train_test_split(
            prefix_groups,
            signatures,
            test_size=val_ratio + test_ratio,
            random_state=seed,
            stratify=signatures,
        )
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        val_ids, test_ids = train_test_split(
            remainder_ids,
            test_size=relative_test_ratio,
            random_state=seed,
            stratify=remainder_signatures,
        )
    except ValueError as exc:
        raise ValueError(
            "Could not create the requested stratified prefix-group split. "
            "The dataset may have too few prefix groups in one or more class "
            "signatures for the requested ratios. No output was written."
        ) from exc

    splits = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }
    validate_split_assignments(splits)
    return splits


def validate_split_assignments(splits: dict[str, list[str]]) -> None:
    if tuple(splits) != SPLIT_NAMES:
        raise ValueError(f"Split names must be exactly {SPLIT_NAMES}, got {tuple(splits)}")
    group_sets = {name: set(ids) for name, ids in splits.items()}
    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            overlap = group_sets[left] & group_sets[right]
            if overlap:
                raise RuntimeError(
                    f"Filename-prefix group overlap between {left} and {right}: "
                    f"{sorted(overlap)[:10]}"
                )
    all_groups = set().union(*group_sets.values())
    if any(not group_sets[name] for name in SPLIT_NAMES):
        raise ValueError("Each split must contain at least one prefix group.")
    if len(all_groups) != sum(len(ids) for ids in group_sets.values()):
        raise RuntimeError("A prefix group was assigned to more than one split.")


def build_summary(
    group_files: dict[str, dict[str, list[Path]]],
    splits: dict[str, list[str]],
    source_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict:
    split_counts = {}
    for split_name, prefix_groups in splits.items():
        class_counts = {
            class_name: sum(
                len(group_files[prefix_group][class_name])
                for prefix_group in prefix_groups
            )
            for class_name in TARGET_CLASSES
        }
        split_counts[split_name] = {
            "prefix_groups": len(prefix_groups),
            "class_images": class_counts,
            "total_images": sum(class_counts.values()),
        }
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory_name": source_dir.name,
        "seed": seed,
        "requested_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "class_names": list(TARGET_CLASSES),
        "grouping_key": "leading numeric filename prefix before first underscore",
        "prefix_group_disjointness_verified": True,
        "person_disjointness_verified": False,
        "split_counts": split_counts,
        "total_prefix_groups": len(group_files),
        "total_images": sum(
            len(paths)
            for by_class in group_files.values()
            for paths in by_class.values()
        ),
    }


def write_split(
    group_files: dict[str, dict[str, list[Path]]],
    splits: dict[str, list[str]],
    output_dir: Path,
    summary: dict,
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite it: {output_dir}"
        )

    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.staging-",
        dir=output_dir.parent,
    ) as temporary_root:
        staging_dir = Path(temporary_root) / "split"
        for split_name, prefix_groups in splits.items():
            for class_name in TARGET_CLASSES:
                target_dir = staging_dir / split_name / class_name
                target_dir.mkdir(parents=True, exist_ok=True)
                for prefix_group in prefix_groups:
                    for source_path in group_files[prefix_group][class_name]:
                        shutil.copy2(source_path, target_dir / source_path.name)

        (staging_dir / "split_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        output_dir.mkdir(exist_ok=False)
        os.replace(staging_dir, output_dir)


def run_split(args: argparse.Namespace) -> Path:
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if paths_overlap(source_dir, output_dir):
        raise ValueError(
            "Source and output directories must not be equal or nested within "
            "one another."
        )
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; it will not be deleted or "
            f"overwritten: {output_dir}"
        )

    group_files = collect_prefix_group_files(source_dir)
    splits = split_prefix_groups(
        group_files,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
    summary = build_summary(
        group_files,
        splits,
        source_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
    write_split(group_files, splits, output_dir, summary)

    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    for split_name, counts in summary["split_counts"].items():
        print(
            f"{split_name}: {counts['prefix_groups']} filename-prefix groups, "
            f"{counts['class_images']}, {counts['total_images']} images"
        )
    print(
        f"Created filename-prefix-disjoint split for "
        f"{summary['total_prefix_groups']} groups / {summary['total_images']} images. "
        "Person-level separation has not been verified."
    )
    return output_dir


if __name__ == "__main__":
    run_split(parse_args())
