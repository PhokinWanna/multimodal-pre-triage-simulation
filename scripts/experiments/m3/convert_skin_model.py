"""Convert SkinCNN-FL-α0.75 (best checkpoint) to TFLite for the kiosk and prove both give the same answers.

Run in the CNN environment (TensorFlow 2.15, requirements-cnn.txt):

    python scripts/experiments/m3/convert_skin_model.py

What it does
1. Loads the Keras model (default: results_alpha_sweep/focal_alpha_075/best_model.keras = best
   checkpoint, epoch 33) and its metrics.json.
2. Takes the decision threshold selected ON THE VALIDATION SET for precision >= 0.95 at the best
   checkpoint (not a test-set number).
3. Builds the inference graph without the training-only augmentation layers (RandomFlip,
   RandomRotation are identity at inference) and converts it to float32 TFLite (no quantisation).
4. Match test: runs the Keras model and the TFLite model on real 64x64 training patches
   (data/dataset-cnn/patches/**, up to --samples) — or on random inputs if no patches exist — and
   reports the largest probability difference and decision agreement. PASS = max difference
   <= 1e-4 and 100 % agreement.
5. Only on PASS: keeps the original kiosk model as models/legacy/skin_classifier_original.tflite
   (needed for the appendix), writes models/skin_classifier.tflite and models/skin_classifier.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from m3_common import PROJECT_ROOT, RESULTS_DIR

DEFAULT_RUN = PROJECT_ROOT / "scripts" / "train" / "cnn" / "model03" / "results_alpha_sweep" / "focal_alpha_075"
MODELS = PROJECT_ROOT / "models"


def load_patches(n: int, seed: int = 42) -> tuple[np.ndarray, str]:
    import cv2
    from m3_common import sample_training_patches
    half = max(1, n // 2)
    batch, notes = [], []
    for cls in ("Normal", "Flushing"):
        patches, note = sample_training_patches(cls, half, seed)
        batch += [cv2.cvtColor(p, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0 for p in patches]
        notes.append(f"{len(patches)} {cls}")
    if not batch:
        rng = np.random.default_rng(seed)
        return rng.random((n, 64, 64, 3), dtype=np.float32), "random inputs (no training images found)"
    return np.stack(batch), "real patches: " + ", ".join(notes) + f" ({note})"


def inference_model(model):
    import tensorflow as tf
    skip = (tf.keras.layers.RandomFlip, tf.keras.layers.RandomRotation)
    inp = tf.keras.Input(shape=model.input_shape[1:])
    x = inp
    for layer in model.layers[1:]:
        if isinstance(layer, skip):
            continue
        x = layer(x, training=False) if "training" in layer.call.__code__.co_varnames else layer(x)
    return tf.keras.Model(inp, x, name=model.name + "_inference")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--precision-target", default="0.95", help="validation-selected threshold key in metrics.json")
    ap.add_argument("--name", default="SkinCNN-FL-α0.75 (best checkpoint)")
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--out-dir", type=Path, default=MODELS, help="where to write skin_classifier.tflite/.json")
    args = ap.parse_args()

    import tensorflow as tf

    keras_path = args.run_dir / "best_model.keras"
    metrics = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8"))
    best = metrics["best_checkpoint"]
    threshold = float(best["validation"]["validation_selected_thresholds"][args.precision_target]["threshold"])
    print(f"Model: {keras_path.relative_to(PROJECT_ROOT)} (best epoch {best.get('epoch')})")
    print(f"Threshold (validation, precision >= {args.precision_target}): {threshold:.4f}")

    model = tf.keras.models.load_model(keras_path, compile=False)
    infer = inference_model(model)
    converter = tf.lite.TFLiteConverter.from_keras_model(infer)
    tflite_bytes = converter.convert()

    x, source = load_patches(args.samples)
    k = model.predict(x, verbose=0).reshape(-1)
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    i_det, o_det = interp.get_input_details()[0], interp.get_output_details()[0]
    t = []
    for row in x:
        interp.set_tensor(i_det["index"], row[None].astype(np.float32))
        interp.invoke()
        t.append(float(interp.get_tensor(o_det["index"]).reshape(-1)[0]))
    t = np.array(t)
    diff = np.abs(k - t)
    agree = float(np.mean((k >= threshold) == (t >= threshold)))
    passed = bool(diff.max() <= 1e-4 and agree == 1.0)
    report = {
        "model_name": args.name, "keras_model": str(keras_path.relative_to(PROJECT_ROOT)),
        "keras_sha256": hashlib.sha256(keras_path.read_bytes()).hexdigest(), "best_epoch": best.get("epoch"),
        "threshold": threshold, "threshold_rule": f"validation-selected, precision >= {args.precision_target}, best checkpoint",
        "check_inputs": source, "max_abs_difference": float(diff.max()), "mean_abs_difference": float(diff.mean()),
        "decision_agreement": agree, "passed": passed, "tensorflow": tf.__version__,
        "created": dt.datetime.now().isoformat(timespec="seconds"), "tflite_bytes": len(tflite_bytes),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "skin_model_conversion.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Match test on {source}: max difference {diff.max():.2e}, mean {diff.mean():.2e}, "
          f"agreement {agree * 100:.2f}% -> {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("Not installing the model. See results/experiments/m3/skin_model_conversion.json")
        return 1

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    original = MODELS / "skin_classifier.tflite"
    legacy = MODELS / "legacy" / "skin_classifier_original.tflite"
    if original.is_file() and not legacy.exists() and not (MODELS / "skin_classifier.json").exists():
        legacy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, legacy)
        print(f"Original kiosk model kept as {legacy.relative_to(PROJECT_ROOT)} (for the appendix run)")
    (out / "skin_classifier.tflite").write_bytes(tflite_bytes)
    meta = {"name": args.name, "threshold": threshold, "combine": "mean", "source": report["keras_model"],
            "keras_sha256": report["keras_sha256"], "threshold_rule": report["threshold_rule"],
            "input": "64x64 forehead/cheek patch, BGR->CIELAB, /255", "created": report["created"]}
    (out / "skin_classifier.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Installed {out / 'skin_classifier.tflite'} and skin_classifier.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
