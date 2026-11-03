"""Does the skin CNN detect REDNESS, or just image smoothness/source? (shortcut diagnostic)

    python scripts/experiments/m3/skin_shortcut_check.py [--model models/skin_classifier.tflite]

Run in the kiosk or CNN environment after convert_skin_model.py. Uses real 64x64 patches
(data/dataset-cnn/splits/test/{Normal,Flushing} if present, otherwise data/dataset-cnn/patches).

For each patch it measures P(Flushing) before and after controlled changes:
  blur      Gaussian blur (sigma 1.5)        -> should NOT change a redness detector much
  grain     add mild noise (sd 4 grey levels) -> should NOT change a redness detector much
  redder    raise CIELAB a* by +12 (Normal patches)   -> SHOULD raise P(Flushing)
  less_red  lower CIELAB a* by -12 (Flushing patches) -> SHOULD lower P(Flushing)
  flat      replace the patch by its mean colour (no texture) -> isolates colour from texture

Prints a table and writes results/experiments/m3/skin_shortcut_check.json.
Reading the result: if blur/grain move the score more than the colour changes do, the model
relies on texture/source more than on redness, which must be reported as a limitation.
"""

from __future__ import annotations

import argparse
import json
import sys
import cv2
import numpy as np

from m3_common import PROJECT_ROOT, RESULTS_DIR


def shift_a(img_bgr: np.ndarray, delta: float) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 1] = np.clip(lab[..., 1] + delta, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=str(PROJECT_ROOT / "models" / "skin_classifier.tflite"))
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from triage_kiosk.vision.skin import SkinModel, to_model_input
    model = SkinModel.from_tflite(args.model)
    thr = model.info.threshold
    rng = np.random.default_rng(args.seed)

    transforms = {
        "original": lambda im: im,
        "blur": lambda im: cv2.GaussianBlur(im, (0, 0), 1.5),
        "grain": lambda im: np.clip(im.astype(np.float32) + rng.normal(0, 4, im.shape), 0, 255).astype(np.uint8),
        "flat": lambda im: np.full_like(im, im.reshape(-1, 3).mean(axis=0).astype(np.uint8)),
        "redder": lambda im: shift_a(im, +12),
        "less_red": lambda im: shift_a(im, -12),
    }
    report = {"model": model.info.name, "threshold": thr, "classes": {}}
    print(f"Model: {model.info.name}  (threshold {thr:.3f})\n")
    from m3_common import sample_training_patches
    for cls in ("Normal", "Flushing"):
        imgs, source = sample_training_patches(cls, args.per_class, args.seed + 1)
        if not imgs:
            print(f"No {cls} training images found under data/dataset-cnn/{cls}")
            return 1
        files = imgs
        report.setdefault("sources", {})[cls] = source
        base = model.predict(np.stack([to_model_input(i) for i in imgs]))
        rows = {}
        for name, fn in transforms.items():
            p = base if name == "original" else model.predict(np.stack([to_model_input(fn(i)) for i in imgs]))
            rows[name] = {"mean_prob": round(float(p.mean()), 4),
                          "mean_change": round(float((p - base).mean()), 4),
                          "pct_called_flushing": round(float((p >= thr).mean() * 100), 1),
                          "pct_decision_flipped": round(float(((p >= thr) != (base >= thr)).mean() * 100), 1)}
        report["classes"][cls] = {"n": len(files), "results": rows}
        print(f"{cls} patches (n={len(files)})")
        print(f"  {'change':<10}{'mean P(F)':>10}{'Δ mean':>9}{'% Flushing':>12}{'% flipped':>11}")
        for name, r in rows.items():
            print(f"  {name:<10}{r['mean_prob']:>10.3f}{r['mean_change']:>+9.3f}{r['pct_called_flushing']:>11.1f}%{r['pct_decision_flipped']:>10.1f}%")
        print()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "skin_shortcut_check.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
