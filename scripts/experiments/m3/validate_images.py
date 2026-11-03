"""Check the M3 photos, image_sources.csv and the scenario -> photo links before anything else runs.

    python scripts/experiments/m3/validate_images.py

Checks: every photo on disk is listed (and every listed photo exists); every scenario points to the
photos its category needs (posture photo, skin photo, or both); no test photo is used twice;
true labels are filled and fit the category; image size; disallowed sources.
Exit code 0 = ready; 1 = something to fix (all problems are listed).
"""

from __future__ import annotations

import csv
import sys
from collections import Counter

from m3_common import IMAGE_EXTS, IMAGES_DIR, SCENARIOS_CSV, SOURCES_CSV, TEST_SPEC, read_sources

MIN_SIDE = 640
REQUIRED = ["file", "set", "category", "true_posture", "true_flushing", "source_site", "page_url", "license"]
BANNED_TRAINING_SOURCES = ("kaggle.com/datasets/arnaud58", "massive-skin-disease", "skin-problem-detection",
                           "fairface", "ffhq", "dermnet")


def main() -> int:
    problems, warnings = [], []
    try:
        from PIL import Image
    except ImportError:
        Image = None
        warnings.append("Pillow not installed: image sizes not checked (pip install pillow)")
    if not SOURCES_CSV.is_file():
        print(f"Missing {SOURCES_CSV}")
        return 1
    records = read_sources()
    by_file = {(r.set, r.file): r for r in records}

    # 1. CSV <-> disk
    for set_name in ("test", "dev"):
        folder = IMAGES_DIR / set_name
        on_disk = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS} \
            if folder.is_dir() else set()
        listed = {f for s, f in by_file if s == set_name}
        for f in sorted(on_disk - listed):
            problems.append(f"{set_name}/{f}: photo is not listed in image_sources.csv")
        for f in sorted(listed - on_disk):
            problems.append(f"{set_name}/{f}: listed in the CSV but the file is missing")
    for (s, f), n in Counter((r.set, r.file) for r in records).items():
        if n > 1:
            problems.append(f"{s}/{f}: listed more than once")

    for r in records:
        tag = f"{r.set}/{r.file}"
        for col in REQUIRED:
            if not r.row.get(col):
                problems.append(f"{tag}: column '{col}' is empty")
        if r.true_posture not in ("none", "chest", "abdomen", "head"):
            problems.append(f"{tag}: true_posture must be none/chest/abdomen/head")
        if r.true_flushing not in ("yes", "no"):
            problems.append(f"{tag}: true_flushing must be yes or no")
        src = (r.row.get("page_url", "") + " " + r.row.get("source_site", "")).lower()
        if any(b in src for b in BANNED_TRAINING_SOURCES):
            problems.append(f"{tag}: source looks like a CNN training dataset ({r.row.get('source_site')})")
        if Image is not None and r.path.is_file():
            try:
                with Image.open(r.path) as im:
                    if min(im.size) < MIN_SIDE:
                        warnings.append(f"{tag}: {im.size[0]}x{im.size[1]} px (shorter side < {MIN_SIDE})")
            except Exception as e:
                problems.append(f"{tag}: cannot open image ({e})")

    # 2. Scenario -> photo links
    with open(SCENARIOS_CSV, newline="", encoding="utf-8-sig") as f:
        scenarios = [r for r in csv.DictReader(f) if r.get("id")]
    used = Counter()
    for s in scenarios:
        sid, cat = s["id"], s["image_category"]
        body, skin = s.get("image_file", ""), s.get("image_file_skin", "")
        if not body:
            problems.append(f"{sid}: image_file is empty")
            continue
        for f in (body, skin):
            if f:
                used[f] += 1
                rec = by_file.get(("test", f))
                if rec is None:
                    problems.append(f"{sid}: {f} is not in image_sources.csv (set=test)")
                elif rec.category != cat:
                    problems.append(f"{sid}: {f} has category '{rec.category}' but the scenario is '{cat}'")
        spec = TEST_SPEC.get(cat)
        if spec is None:
            problems.append(f"{sid}: unknown category '{cat}'")
            continue
        _, want_posture, want_flush = spec
        b = by_file.get(("test", body))
        if cat.startswith("both_"):
            k = by_file.get(("test", skin)) if skin else None
            if k is None:
                problems.append(f"{sid}: 'both' scenario needs image_file_skin (a flushed-face photo)")
            elif k.true_flushing != "yes":
                problems.append(f"{sid}: skin photo {skin} is not marked true_flushing = yes")
            if b and b.true_posture != want_posture:
                problems.append(f"{sid}: body photo shows '{b.true_posture}', scenario needs '{want_posture}'")
        else:
            if skin:
                problems.append(f"{sid}: only 'both' scenarios use image_file_skin")
            if b and (b.true_posture != want_posture or b.true_flushing != want_flush):
                problems.append(f"{sid}: photo is {b.true_posture}/{b.true_flushing}, scenario needs "
                                f"{want_posture}/{want_flush}")
    for f, n in used.items():
        if n > 1:
            problems.append(f"test/{f}: used by {n} scenarios (each photo once)")
    unused = sorted(f for s, f in by_file if s == "test" and f not in used)
    for f in unused:
        warnings.append(f"test/{f}: listed but not used by any scenario")

    print(f"Scenarios: {len(scenarios)} | test photos used: {len(used)} | dev photos: "
          f"{sum(r.set == 'dev' for r in records)}")
    dev = Counter(r.true_posture for r in records if r.set == "dev")
    print("Dev photos by true posture:", dict(dev))
    if not any(dev.get(p) for p in ("chest", "abdomen", "head")):
        warnings.append("dev set has no chest/abdomen/head examples: the rules can only be checked for false alarms")
    print()
    for w in warnings:
        print("WARNING ", w)
    for p in problems:
        print("PROBLEM ", p)
    print(f"\n{len(problems)} problem(s), {len(warnings)} warning(s).",
          "Ready." if not problems else "Fix the problems, then run again.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
