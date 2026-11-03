"""Run the camera analysis on the 14 DEVELOPMENT photos (never the test photos).

    python scripts/experiments/m3/check_dev_images.py

Kiosk environment, after convert_skin_model.py. For each dev photo it reports:
  * posture found by the corrected rules (v2) and by the original rules (v1), vs true_posture
  * skin result from the corrected patch pipeline and from the original whole-face pipeline,
    vs true_flushing, with each patch's probability
  * the measured distances (torso lengths), which is what we tune if a rule misses
Outputs: results/experiments/m3/dev_check.csv and annotated photos in
data/experiments/m3/dev_check_images/ (git-ignored: they contain the third-party photos).
Tuning is allowed on these photos only. The test photos are first analysed in the real run.
"""

from __future__ import annotations

import json
import sys

import cv2

from m3_common import DATA_DIR, PROJECT_ROOT, RESULTS_DIR, read_sources, write_csv

LEGACY_MODEL = PROJECT_ROOT / "models" / "legacy" / "skin_classifier_original.tflite"
NEW_MODEL = PROJECT_ROOT / "models" / "skin_classifier.tflite"


def annotate(img, result_v2, skin_v2, path):
    out = img.copy()
    dbg = result_v2.debug if result_v2 else {}
    torso = float(dbg.get("torso_px", 0) or 0)
    for key, colour in (("sternum", (0, 0, 255)), ("navel", (0, 165, 255))):
        if key in dbg and torso:
            c = tuple(int(v) for v in dbg[key])
            cv2.circle(out, c, int(torso * 0.30), colour, 2)
            cv2.putText(out, key, (c[0] + 5, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    lines = [f"posture v2: {result_v2.label() if result_v2 else 'no person'}",
             f"skin: {skin_v2.status} p={skin_v2.probability if skin_v2.probability is None else round(skin_v2.probability, 3)}"]
    for i, t in enumerate(lines):
        cv2.putText(out, t, (10, 25 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(str(path), out)


def main() -> int:
    from triage_kiosk.vision.engine import VisionEngine

    if not NEW_MODEL.with_suffix(".json").is_file():
        print("Run convert_skin_model.py first (models/skin_classifier.json is missing).")
        return 1
    fixed = VisionEngine(static_image_mode=True, posture_version="v2", skin_pipeline="patch",
                         skin_model_path=str(NEW_MODEL), require_skin_model=True)
    original = VisionEngine(static_image_mode=True, posture_version="v1", skin_pipeline="legacy_whole_face",
                            skin_model_path=str(LEGACY_MODEL if LEGACY_MODEL.is_file() else NEW_MODEL),
                            require_skin_model=True)

    out_img_dir = DATA_DIR / "dev_check_images"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in [r for r in read_sources() if r.set == "dev"]:
        img = cv2.imread(str(rec.path))
        if img is None:
            print(f"cannot read {rec.path}")
            continue
        a, b = fixed.analyze_image(img), original.analyze_image(img)
        pv2 = a["posture"].label() if a["posture"] else "no person"
        pv1 = b["posture"].label() if b["posture"] else "no person"
        s2, s1 = a["skin"], b["skin"]
        flush_v2 = {"RED_FLUSHING": "yes", "NORMAL": "no"}.get(s2.status, "unknown")
        flush_v1 = {"RED_FLUSHING": "yes", "NORMAL": "no"}.get(s1.status, "unknown")
        rows.append({
            "file": rec.file, "category": rec.category,
            "true_posture": rec.true_posture, "posture_v2": pv2, "posture_v1": pv1,
            "posture_v2_ok": pv2 == rec.true_posture, "posture_v1_ok": pv1 == rec.true_posture,
            "true_flushing": rec.true_flushing, "skin_v2": flush_v2, "skin_v1": flush_v1,
            "skin_v2_ok": flush_v2 == rec.true_flushing, "skin_v1_ok": flush_v1 == rec.true_flushing,
            "skin_v2_prob": None if s2.probability is None else round(s2.probability, 4),
            "skin_v2_patches": json.dumps(s2.patch_probabilities),
            "skin_v1_prob": None if s1.probability is None else round(s1.probability, 4),
            "filters_v2": ",".join(a["posture"].filters) if a["posture"] else "",
            "debug_v2": json.dumps(a["posture"].debug if a["posture"] else {}, default=str),
        })
        annotate(img, a["posture"], s2, out_img_dir / rec.file)

    if not rows:
        print("No dev photos listed in image_sources.csv")
        return 1
    cols = list(rows[0].keys())
    write_csv(RESULTS_DIR / "dev_check.csv", rows, cols)
    print(f"{'file':<22}{'true':<9}{'v2':<10}{'v1':<10}{'flush':<7}{'v2':<9}{'v1':<9}{'p(v2)':>7}")
    for r in rows:
        print(f"{r['file']:<22}{r['true_posture']:<9}{r['posture_v2']:<10}{r['posture_v1']:<10}"
              f"{r['true_flushing']:<7}{r['skin_v2']:<9}{r['skin_v1']:<9}{str(r['skin_v2_prob']):>7}")
    n = len(rows)
    for key, label in (("posture_v2_ok", "posture, fixed rules"), ("posture_v1_ok", "posture, original rules"),
                       ("skin_v2_ok", "skin, fixed pipeline"), ("skin_v1_ok", "skin, original pipeline")):
        print(f"{label:<26} {sum(r[key] for r in rows)}/{n} correct")
    print(f"\nSaved results/experiments/m3/dev_check.csv and annotated photos in {out_img_dir.relative_to(PROJECT_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
