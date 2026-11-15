"""Where the skin image-suitability limits come from (round B). Uses TRAINING data and the
development photos only - never the M3 test photos.

    python scripts/experiments/m3/skin_quality_study.py [--per-class 300]

Parts:
  A. Face width (pixels) and patch lightness in the CNN's training images, per class.
  B. Blur dose-response: Normal/Flushing training patches blurred step by step; how often the CNN
     says "Flushing" as patch sharpness (variance of the Laplacian) falls.
  C. The old pipeline ("patch", round A) against the new one ("patch_qc", round B) on whole training
     images and on the development photos (all development faces are normal skin).
  D. Face size: Normal training images enlarged 1.5x and 2x (so the face is wider than the Normal
     class ever was in training), and wide Flushing training images reduced to the training size.
     Shows why a wide face is answered only when the CNN agrees at both sizes.
     Run alone with --parts D (it merges into the existing JSON).

Writes results/experiments/m3/roundB/skin_quality_study.json and prints a summary.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
import random
import sys

import cv2
import numpy as np

from m3_common import IMAGE_EXTS, PROJECT_ROOT, RESULTS_DIR, read_sources

OUT = RESULTS_DIR / "roundB" / "skin_quality_study.json"
BINS = [(0, 15), (15, 30), (30, 50), (50, 100), (100, 1e9)]


def pct(values, qs=(1, 5, 25, 50, 75, 95, 99)):
    return {f"p{q}": round(float(np.percentile(values, q)), 1) for q in qs} if len(values) else {}


def part_d(skin, model, qc, fm, budget: float, seed: int = 11) -> dict:
    """Wide faces at two sizes. Enlarging a photo also softens it, so the enlarged Normal images are a
    stress test of size and softness together; the wide Flushing images are real, unmodified photos."""
    import time
    from collections import Counter

    def label(r):
        return r.status if r.status != skin.UNKNOWN else "no answer: " + r.reason

    out, t0 = {}, time.time()
    plan = (("Normal x2.0", "Normal", 2.0, 0.45), ("Normal x1.5", "Normal", 1.5, 0.25), ("Flushing wide", "Flushing", 1.0, 0.30))
    for name, cls, scale, share in plan:
        files = sorted(p for p in (PROJECT_ROOT / "data" / "dataset-cnn" / cls).glob("*") if p.suffix.lower() in IMAGE_EXTS)
        random.Random(seed).shuffle(files)
        pairs, old, new, t1 = Counter(), Counter(), Counter(), time.time()
        for f in files:
            if time.time() - t1 > budget * share:
                break
            img = cv2.imread(str(f))
            if img is None:
                continue
            if scale != 1.0:
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                continue
            lm = res.multi_face_landmarks[0].landmark
            fw = skin.face_width_px(lm, img.shape[1])
            if fw <= qc.max_native_face_width_px:
                continue
            native = skin._classify_at_scale(img, lm, model, qc, 1.0, True)
            reduced = skin._classify_at_scale(img, lm, model, qc, qc.target_face_width_px / fw, True)
            pairs[f"{label(native)} | {label(reduced)}"] += 1
            old[skin.classify_patches(img, lm, model).status] += 1
            new[skin.classify_patches_qc(img, lm, model, qc).status] += 1
        n = sum(new.values())
        answered = n - new[skin.UNKNOWN]
        out[name] = {"n": n, "round_a_flushing": f"{old[skin.FLUSHING]}/{n - old[skin.UNKNOWN]}",
                     "round_b_answered": f"{answered}/{n}", "round_b_flushing": f"{new[skin.FLUSHING]}/{answered}",
                     "native_vs_reduced": dict(pairs.most_common())}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-class", type=int, default=300, help="training images per class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--parts", default="ABC", help="which parts to run: ABC (default), D, or ABCD")
    ap.add_argument("--time-budget", type=float, default=140.0, help="seconds for part D")
    args = ap.parse_args(argv)
    parts = set(args.parts.upper())

    import mediapipe as mp
    from triage_kiosk.vision import skin

    model = skin.SkinModel.from_tflite(str(PROJECT_ROOT / "models" / "skin_classifier.tflite"))
    thr = model.info.threshold
    qc = skin.SkinQualityConfig()
    qc_gate_only = replace(qc, max_native_face_width_px=1e9)      # sensitivity analysis: no size reduction
    report = {"model": model.info.name, "threshold": thr, "limits": qc.__dict__, "training": {}, "blur": {}, "pipelines": {}}
    if OUT.is_file() and not {"A", "B", "C"} <= parts:
        report = json.loads(OUT.read_text(encoding="utf-8"))
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                         min_detection_confidence=0.5)

    def analyse(img):
        res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        return (lm, skin.classify_patches(img, lm, model), skin.classify_patches_qc(img, lm, model, qc),
                skin.classify_patches_qc(img, lm, model, qc_gate_only))

    if parts & {"A", "B", "C"}:
        patches = {}
        for cls in ("Normal", "Flushing"):
            files = sorted(p for p in (PROJECT_ROOT / "data" / "dataset-cnn" / cls).glob("*") if p.suffix.lower() in IMAGE_EXTS)
            random.Random(args.seed).shuffle(files)
            widths, light, old, new, gate, reasons, cls_patches, n_faces = [], [], [], [], [], {}, [], 0
            for f in files:
                if n_faces >= args.per_class:
                    break
                img = cv2.imread(str(f))
                out = analyse(img) if img is not None else None
                if out is None:
                    continue
                lm, r_old, r_new, r_gate = out
                if r_old.status == skin.UNKNOWN:
                    continue
                n_faces += 1
                h, w = img.shape[:2]
                widths.append(skin.face_width_px(lm, w))
                for _, cx, cy in skin.patch_centres(lm, w, h):
                    p = skin.extract_square_patch(img, cx, cy)
                    if skin.is_valid_skin_color(p):
                        p = skin.jpeg_roundtrip(p)
                        cls_patches.append(p)
                        light.append(float(cv2.cvtColor(p, cv2.COLOR_BGR2LAB)[..., 0].mean()))
                old.append(r_old.status == skin.FLUSHING)
                new.append(r_new.status)
                gate.append(r_gate.status)
                if r_new.status == skin.UNKNOWN:
                    reasons[r_new.reason] = reasons.get(r_new.reason, 0) + 1
            patches[cls] = cls_patches
            answered = [s for s in new if s != skin.UNKNOWN]
            gate_ans = [s for s in gate if s != skin.UNKNOWN]
            report["training"][cls] = {"images_with_face": n_faces, "face_width_px": pct(widths), "patch_lightness": pct(light)}
            report["pipelines"][f"training {cls}"] = {
                "n": n_faces, "old_called_flushing_pct": round(100 * float(np.mean(old)), 1),
                "new_answered_pct": round(100 * len(answered) / max(1, n_faces), 1),
                "new_called_flushing_pct_of_answered": round(100 * float(np.mean([s == skin.FLUSHING for s in answered])), 1) if answered else None,
                "new_unknown_reasons": reasons,
                "gate_only_answered_pct": round(100 * len(gate_ans) / max(1, n_faces), 1),
                "gate_only_called_flushing_pct_of_answered": round(100 * float(np.mean([s == skin.FLUSHING for s in gate_ans])), 1) if gate_ans else None}

        # B. blur dose-response on patches
        for cls in ("Normal", "Flushing"):
            rows = []
            for p in patches[cls][:600]:
                for sigma in (0, 0.5, 0.8, 1.0, 1.5, 2.0):
                    q = p if sigma == 0 else cv2.GaussianBlur(p, (0, 0), sigma)
                    rows.append((sigma, skin.patch_sharpness(q), float(model.predict(skin.to_model_input(q)[None])[0])))
            a = np.array(rows)
            report["blur"][cls] = {
                "by_sigma": {str(s): {"median_sharpness": round(float(np.median(a[a[:, 0] == s][:, 1])), 1),
                                      "called_flushing_pct": round(100 * float(np.mean(a[a[:, 0] == s][:, 2] >= thr)), 1)}
                             for s in (0, 0.5, 0.8, 1.0, 1.5, 2.0)},
                "by_sharpness": {f"{lo}-{'inf' if hi > 1e8 else hi}": {
                    "n": int(((a[:, 1] >= lo) & (a[:, 1] < hi)).sum()),
                    "called_flushing_pct": round(100 * float(np.mean(a[(a[:, 1] >= lo) & (a[:, 1] < hi)][:, 2] >= thr)), 1)}
                    for lo, hi in BINS if ((a[:, 1] >= lo) & (a[:, 1] < hi)).any()}}

        # C. development photos (all normal skin)
        dev_rows = []
        for rec in [r for r in read_sources() if r.set == "dev"]:
            img = cv2.imread(str(rec.path))
            out = analyse(img) if img is not None else None
            if out is None:
                dev_rows.append({"file": rec.file, "face": False})
                continue
            lm, r_old, r_new, _ = out
            dev_rows.append({"file": rec.file, "face": True, "face_width_px": round(skin.face_width_px(lm, img.shape[1])),
                             "old": r_old.status, "old_prob": None if r_old.probability is None else round(r_old.probability, 3),
                             "new": r_new.status, "new_prob": None if r_new.probability is None else round(r_new.probability, 3),
                             "new_reason": r_new.reason})
        faces = [d for d in dev_rows if d["face"]]
        old_ans = [d for d in faces if d["old"] != skin.UNKNOWN]
        new_ans = [d for d in faces if d["new"] != skin.UNKNOWN]
        report["pipelines"]["development photos (true: all normal)"] = {
            "n_faces": len(faces),
            "old_answered": len(old_ans), "old_false_flushing": sum(d["old"] == skin.FLUSHING for d in old_ans),
            "new_answered": len(new_ans), "new_false_flushing": sum(d["new"] == skin.FLUSHING for d in new_ans),
            "rows": dev_rows}

    if "D" in parts:
        report["face_size"] = part_d(skin, model, qc, fm, args.time_budget)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    print(f"Model {model.info.name}, threshold {thr:.3f}\n")
    if parts & {"A", "B", "C"}:
        print("A. Training images")
        for cls, d in report["training"].items():
            print(f"   {cls:<9} face width px {d['face_width_px']}  patch lightness {d['patch_lightness']}")
        print("\nB. Blur: share of patches called Flushing, by patch sharpness")
        for cls, d in report["blur"].items():
            print(f"   {cls:<9} " + "  ".join(f"[{k}] {v['called_flushing_pct']}%" for k, v in d["by_sharpness"].items()))
        print("\nC. Old pipeline (round A) vs new pipeline with the suitability check (round B)")
        for name, d in report["pipelines"].items():
            if "rows" in d:
                print(f"   {name}: old answered {d['old_answered']}/{d['n_faces']}, false Flushing {d['old_false_flushing']}; "
                      f"new answered {d['new_answered']}/{d['n_faces']}, false Flushing {d['new_false_flushing']}")
            else:
                print(f"   {name}: old called Flushing {d['old_called_flushing_pct']}%; new answered {d['new_answered_pct']}%, "
                      f"of those called Flushing {d['new_called_flushing_pct_of_answered']}%  (not answered: {d['new_unknown_reasons']}); "
                      f"without the size reduction: answered {d['gate_only_answered_pct']}%, called Flushing {d['gate_only_called_flushing_pct_of_answered']}%")
    if "face_size" in report:
        print("\nD. Face size (native = original size, reduced = training size)")
        for name, d in report["face_size"].items():
            print(f"   {name}: n={d['n']}; round-A pipeline called Flushing {d['round_a_flushing']}; "
                  f"round-B answered {d['round_b_answered']} (Flushing {d['round_b_flushing']}); pairs {d['native_vs_reduced']}")
    print(f"\nSaved {OUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
