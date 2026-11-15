"""Chest-posture rules on a large labelled DEVELOPMENT pool (round B, never the test photos).

    python scripts/experiments/m3/posture_pool_eval.py [--time-budget 150]

Pool: Roboflow "Medical Emergency Detection" v2 (CC BY 4.0), local copy in
data/dataset-posture/. Labels used: ClutchingChest (chest posture) vs NaturalBody (no posture).
* One image per SOURCE photo (the export holds 3 augmented copies of each).
* Every source photo that is also an M3 TEST photo is excluded (byte-identical match).
* The pool is split by a hash of the source name: 70% tuning, 30% hold-out. The v3 chest
  zone was fitted on the tuning part only; the hold-out part is reported here.

The run is resumable (landmarks are cached in results/experiments/m3/roundB/cache/).
Output: results/experiments/m3/roundB/posture_pool.json and a printed table.
The pool has no abdomen or head labels: those rules are only checked on the 20 dev photos.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
import time

import numpy as np

from m3_common import IMAGES_DIR, PROJECT_ROOT, RESULTS_DIR

POOL = PROJECT_ROOT / "data" / "dataset-posture" / "Medical Emergency Detection.v2i.multiclass"
OUT_DIR = RESULTS_DIR / "roundB"
CACHE = OUT_DIR / "cache" / "posture_pool_landmarks.pkl"


def md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def source_name(filename: str) -> str:
    return filename.split("_jpg.rf.")[0].split(".rf.")[0]


def is_tuning(source: str) -> bool:
    return int(hashlib.md5(source.encode()).hexdigest(), 16) % 10 < 7


def load_pool() -> tuple[dict, dict]:
    """{source: {file, split, chest, natural}} with one file per source, and {test photo: source}."""
    by_source, by_md5 = {}, {}
    for split in ("train", "valid"):
        with open(POOL / split / "_classes.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row = {k.strip(): v.strip() for k, v in row.items()}
                name = row["filename"]
                path = POOL / split / name
                if not path.is_file():
                    continue
                by_md5[md5(path)] = name
                rec = {"file": name, "split": split, "chest": row["ClutchingChest"] == "1",
                       "natural": row["NaturalBody"] == "1"}
                src = source_name(name)
                if src not in by_source or name < by_source[src]["file"]:
                    by_source[src] = rec
    test_matches = {}
    for sub in ("Posture", "skin"):
        folder = IMAGES_DIR / "test" / sub
        for p in sorted(folder.glob("*")) if folder.is_dir() else []:
            if p.is_file() and md5(p) in by_md5:
                test_matches[f"{sub}/{p.name}"] = source_name(by_md5[md5(p)])
    return by_source, test_matches


def landmarks(by_source: dict, budget: float) -> dict:
    import cv2
    import mediapipe as mp
    from triage_kiosk.vision.posture import landmarks_from_mediapipe

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = pickle.loads(CACHE.read_bytes()) if CACHE.is_file() else {}
    todo = [s for s in by_source if s not in cache]
    t0 = time.time()
    if todo:
        with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5) as pose:
            for src in todo:
                if time.time() - t0 > budget:
                    break
                rec = by_source[src]
                img = cv2.imread(str(POOL / rec["split"] / rec["file"]))
                if img is None:
                    cache[src] = None
                    continue
                h, w = img.shape[:2]
                res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                cache[src] = {"w": w, "h": h, "lm": landmarks_from_mediapipe(res.pose_landmarks, w, h)
                              if res.pose_landmarks else None}
        CACHE.write_bytes(pickle.dumps(cache))
    left = len([s for s in by_source if s not in cache])
    if left:
        print(f"{left} images still to analyse: run the same command again to continue.")
    return cache


def evaluate(items, cache, fn, rng) -> dict:
    pairs = []
    for src, rec in items:
        c = cache.get(src)
        pred = bool(c and c["lm"] is not None and fn(c["lm"], c["h"], c["w"]).chest)
        pairs.append((rec["chest"], pred))
    a = np.array(pairs, dtype=bool)
    sens = lambda x: float(x[x[:, 0]][:, 1].mean())
    spec = lambda x: float(1 - x[~x[:, 0]][:, 1].mean())
    boot = np.array([(sens(b), spec(b)) for b in (a[rng.integers(0, len(a), len(a))] for _ in range(2000))])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return {"n": len(a), "n_chest": int(a[:, 0].sum()), "sensitivity": round(sens(a), 4),
            "sensitivity_ci": [round(float(lo[0]), 4), round(float(hi[0]), 4)],
            "specificity": round(spec(a), 4), "specificity_ci": [round(float(lo[1]), 4), round(float(hi[1]), 4)]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--time-budget", type=float, default=150.0, help="seconds of image analysis per run")
    args = ap.parse_args(argv)
    if not POOL.is_dir():
        print(f"Development pool not found: {POOL}")
        return 1
    from triage_kiosk.vision import posture as P

    by_source, test_matches = load_pool()
    excluded = set(test_matches.values())
    cache = landmarks(by_source, args.time_budget)
    items = [(s, r) for s, r in sorted(by_source.items())
             if s not in excluded and r["chest"] != r["natural"] and s in cache]
    tune = [x for x in items if is_tuning(x[0])]
    hold = [x for x in items if not is_tuning(x[0])]
    report = {"pool": "Roboflow Medical Emergency Detection v2 (one image per source photo)",
              "sources_total": len(by_source), "excluded_test_sources": sorted(excluded),
              "test_photos_matched": test_matches, "n_tuning": len(tune), "n_holdout": len(hold), "rules": {}}
    print(f"Pool: {len(by_source)} source photos, {len(excluded)} excluded because they are M3 test photos; "
          f"tuning {len(tune)}, hold-out {len(hold)}\n")
    print(f"{'rules':<14}{'split':<10}{'chest sensitivity':<28}{'specificity':<26}")
    for name, fn in (("v1 original", P.classify_posture_v1), ("v2 round A", P.classify_posture_v2),
                     ("v3 round B", P.classify_posture_v3)):
        report["rules"][name] = {}
        for split, part in (("tuning", tune), ("hold-out", hold)):
            r = evaluate(part, cache, fn, np.random.default_rng(42))
            report["rules"][name][split] = r
            print(f"{name:<14}{split:<10}"
                  f"{r['sensitivity'] * 100:5.1f}% ({r['sensitivity_ci'][0] * 100:.1f}-{r['sensitivity_ci'][1] * 100:.1f})        "
                  f"{r['specificity'] * 100:5.1f}% ({r['specificity_ci'][0] * 100:.1f}-{r['specificity_ci'][1] * 100:.1f})")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "posture_pool.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {(OUT_DIR / 'posture_pool.json').relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
