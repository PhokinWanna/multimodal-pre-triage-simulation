"""Shared paths, photo categories and helpers for the M3 experiment scripts."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA_DIR = PROJECT_ROOT / "data" / "experiments" / "m3"
IMAGES_DIR = DATA_DIR / "images"
SOURCES_CSV = DATA_DIR / "image_sources.csv"
SCENARIOS_CSV = PROJECT_ROOT / "docs" / "experiments" / "m3" / "scenarios.csv"
DOCS_DIR = PROJECT_ROOT / "docs" / "experiments" / "m3"
LABELS_PHOKIN = DOCS_DIR / "labels_phokin.csv"
LABELS_CLAUDE = DOCS_DIR / "sealed" / "labels_claude_draft.csv"
LABELS_FINAL = DOCS_DIR / "labels_final.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / "m3"

# Category -> (number of CASES, true posture, true flushing), as in docs/experiments/m3/IMAGE_CHECKLIST.md.
# "both_*" cases use TWO photos (protocol change 2026-11-02): <category>_NN_body.jpg gives the posture,
# <category>_NN_face.jpg (a flushed face) gives the skin result. All other cases use one photo.
TEST_SPEC = {
    "neither": (8, "none", "no"),
    "chest": (3, "chest", "no"),
    "abdomen": (3, "abdomen", "no"),
    "head": (2, "head", "no"),
    "flush": (7, "none", "yes"),
    "both_head": (3, "head", "yes"),
    "both_chest": (2, "chest", "yes"),
    "both_abdomen": (2, "abdomen", "yes"),
}
DEV_SPEC = {
    "dev_chest": (2, "chest", "no"),
    "dev_abdomen": (2, "abdomen", "no"),
    "dev_head": (2, "head", "no"),
    "dev_thinking": (2, "none", "no"),
    "dev_lap": (2, "none", "no"),
    "dev_crossed": (1, "none", "no"),
    "dev_flush": (2, "none", "yes"),
    "dev_neither": (1, "none", "no"),
}
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
SOURCE_COLUMNS = ["file", "set", "category", "true_posture", "true_flushing", "source_site", "page_url",
                  "author", "license", "downloaded_on", "notes"]


@dataclass
class ImageRecord:
    file: str
    set: str
    category: str
    true_posture: str
    true_flushing: str
    row: dict

    @property
    def path(self) -> Path:
        return sys.modules[__name__].IMAGES_DIR / self.set / self.file


def read_sources(path: Path | None = None) -> list[ImageRecord]:
    path = path or sys.modules[__name__].SOURCES_CSV   # looked up at call time (tests can redirect it)
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
        if not r.get("file"):
            continue
        out.append(ImageRecord(r["file"], r.get("set", "").lower(), r.get("category", "").lower(),
                               r.get("true_posture", "").lower(), r.get("true_flushing", "").lower(), r))
    return out


TWO_PHOTO_PREFIX = "both_"
PHOTO_ROLES = ("body", "face")


def is_two_photo(category: str) -> bool:
    return category.startswith(TWO_PHOTO_PREFIX)


def photo_role(name: str) -> str:
    """'both_head_01_body.jpg' -> 'body'; 'both_head_01_face.jpg' -> 'face'; others -> 'single'."""
    stem = Path(name).stem
    last = stem.rsplit("_", 1)[-1]
    return last if last in PHOTO_ROLES else "single"


def case_key(name: str) -> str:
    """Photo name without its role: 'both_head_01_face.jpg' -> 'both_head_01'; 'chest_02.jpg' -> 'chest_02'."""
    stem = Path(name).stem
    return stem.rsplit("_", 1)[0] if photo_role(name) != "single" else stem


def category_from_filename(name: str) -> str:
    """'both_head_01_body.jpg' -> 'both_head'; 'dev_lap_02.png' -> 'dev_lap'; 'chest_03.jpg' -> 'chest'."""
    key = case_key(name)
    return key.rsplit("_", 1)[0] if "_" in key else key


def sharpness(path: Path, width: int = 512) -> float | None:
    """Image sharpness = variance of the Laplacian on a grey copy resized to a fixed width
    (higher = sharper). Measured, never used to change the image."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    img = cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    return round(float(cv2.Laplacian(img, cv2.CV_64F).var()), 1)


def sample_training_patches(cls: str, n: int, seed: int = 42, max_images: int = 3000):
    """Up to n real 64x64 patches of class cls ("Normal"/"Flushing"), cut from data/dataset-cnn/<cls>
    with the SAME extraction as CNN training (Face Mesh forehead/cheeks, skin filter, JPEG round trip).
    Uses ready-made patches from data/dataset-cnn/patches/<cls> when they exist."""
    import random

    import cv2
    ready = PROJECT_ROOT / "data" / "dataset-cnn" / "patches" / cls
    rng = random.Random(seed)
    if ready.is_dir():
        files = sorted(ready.glob("*.jpg"))
        rng.shuffle(files)
        return [cv2.resize(cv2.imread(str(f)), (64, 64)) for f in files[:n]], f"ready-made patches ({ready})"
    import mediapipe as mp
    from triage_kiosk.vision import skin
    files = sorted(p for p in (PROJECT_ROOT / "data" / "dataset-cnn" / cls).glob("*") if p.suffix.lower() in IMAGE_EXTS)
    rng.shuffle(files)
    out = []
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                         min_detection_confidence=0.5) as fm:
        for f in files[:max_images]:
            img = cv2.imread(str(f))
            if img is None:
                continue
            res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                continue
            h, w = img.shape[:2]
            for _, cx, cy in skin.patch_centres(res.multi_face_landmarks[0].landmark, w, h):
                patch = skin.extract_square_patch(img, cx, cy)
                if skin.is_valid_skin_color(patch):
                    out.append(skin.jpeg_roundtrip(patch))
            if len(out) >= n:
                break
    return out[:n], f"patches cut from data/dataset-cnn/{cls} with the training extraction"


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
