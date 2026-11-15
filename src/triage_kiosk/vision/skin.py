"""Facial skin check (Normal vs Flushing) that feeds the CNN the SAME input it was trained on.

Training (scripts/data/dataset_extraction-CNN.py) used 64x64 pixel patches cut at the
forehead and both cheekbones (MediaPipe Face Mesh landmarks 10, 117, 346), skipped a
cheek when the head was turned, kept only patches that look like skin, and saved them
as JPEG. The trainer then converted BGR -> CIELAB and divided by 255.

The ORIGINAL kiosk instead squeezed the whole face (eyes, lips, hair) into 64x64.
That path is kept as ``legacy_whole_face`` only for the M3 appendix.

Image-level decision: mean of the valid patch probabilities >= threshold, where the
threshold comes from the model's metadata (selected on the validation set).

Round B adds an image-suitability check (``classify_patches_qc``, pipeline "patch_qc"):
the CNN answers only when the face looks like what it was trained on, otherwise the
result is UNKNOWN with a reason. See ``SkinQualityConfig`` for where each limit comes from.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---- identical to scripts/data/dataset_extraction-CNN.py ----
PATCH_SIZE = 64
MIN_SKIN_RATIO = 0.70
MAX_YAW_THRESHOLD = 0.25
NOSE_ID = 1
FOREHEAD_ID = 10
CHEEK_L_EDGE_ID = 234
CHEEK_R_EDGE_ID = 454
CHEEK_L_ID = 117
CHEEK_R_ID = 346
TRAINING_JPEG_QUALITY = 95   # cv2.imwrite default used when the training patches were saved

NORMAL, FLUSHING, UNKNOWN = "NORMAL", "RED_FLUSHING", "UNKNOWN"


def is_valid_skin_color(patch_bgr: Optional[np.ndarray]) -> bool:
    """Same skin filter as the training patch extractor."""
    if patch_bgr is None:
        return False
    h_patch, w_patch = patch_bgr.shape[:2]
    if h_patch != PATCH_SIZE or w_patch != PATCH_SIZE:
        return False
    b_chan, g_chan, r_chan = cv2.split(patch_bgr)
    r_mean, g_mean, b_mean = float(np.mean(r_chan)), float(np.mean(g_chan)), float(np.mean(b_chan))
    if b_mean >= r_mean:
        return False
    if r_mean < 40 or (r_mean > 245 and g_mean > 245 and b_mean > 245):
        return False
    ycrcb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2YCrCb)
    _, cr_chan, cb_chan = cv2.split(ycrcb)
    skin_mask = (cr_chan >= 133) & (cr_chan <= 173) & (cb_chan >= 77) & (cb_chan <= 127)
    return float(np.sum(skin_mask)) / float(PATCH_SIZE * PATCH_SIZE) >= MIN_SKIN_RATIO


def extract_square_patch(image_bgr: np.ndarray, center_x: float, center_y: float,
                         patch_size: int = PATCH_SIZE) -> Optional[np.ndarray]:
    """Same square crop (no resizing) as the training patch extractor."""
    h, w = image_bgr.shape[:2]
    half = patch_size // 2
    x1, y1 = int(center_x - half), int(center_y - half)
    x2, y2 = x1 + patch_size, y1 + patch_size
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return None
    return image_bgr[y1:y2, x1:x2].copy()


def patch_centres(face_landmarks: Sequence, frame_w: int, frame_h: int) -> list[tuple[str, float, float]]:
    """Forehead always; each cheek unless the head is turned away from it (training rule)."""
    nose_x = face_landmarks[NOSE_ID].x
    dist_left = abs(nose_x - face_landmarks[CHEEK_L_EDGE_ID].x)
    dist_right = abs(face_landmarks[CHEEK_R_EDGE_ID].x - nose_x)
    total = dist_left + dist_right
    skip_left = total > 0 and dist_left / total < MAX_YAW_THRESHOLD
    skip_right = total > 0 and dist_right / total < MAX_YAW_THRESHOLD
    zones = [("Forehead", face_landmarks[FOREHEAD_ID].x * frame_w, face_landmarks[FOREHEAD_ID].y * frame_h)]
    if not skip_left:
        zones.append(("Zone_L", face_landmarks[CHEEK_L_ID].x * frame_w, face_landmarks[CHEEK_L_ID].y * frame_h))
    if not skip_right:
        zones.append(("Zone_R", face_landmarks[CHEEK_R_ID].x * frame_w, face_landmarks[CHEEK_R_ID].y * frame_h))
    return zones


def jpeg_roundtrip(patch_bgr: np.ndarray, quality: int = TRAINING_JPEG_QUALITY) -> np.ndarray:
    """Training patches were stored as JPEG; reproduce that compression step."""
    ok, buf = cv2.imencode(".jpg", patch_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else patch_bgr


def to_model_input(patch_bgr: np.ndarray) -> np.ndarray:
    """Trainer preprocessing: resize to 64x64 (no-op for patches), BGR -> CIELAB, / 255."""
    if patch_bgr.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
        patch_bgr = cv2.resize(patch_bgr, (PATCH_SIZE, PATCH_SIZE))
    lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB)
    return lab.astype(np.float32) / 255.0


# ---------------------------------------------------------------- model wrapper
@dataclass
class SkinModelInfo:
    name: str = "unknown"
    threshold: float = 0.65          # original kiosk default, used only when no metadata exists
    combine: str = "mean"
    source: str = ""
    from_metadata: bool = False      # True when threshold etc. came from <model>.json


class SkinModel:
    """TFLite (kiosk) or Keras (experiments) skin classifier returning P(Flushing) per patch."""

    def __init__(self, predict_fn, info: SkinModelInfo):
        self._predict = predict_fn
        self.info = info

    def predict(self, batch: np.ndarray) -> np.ndarray:
        return np.asarray(self._predict(batch), dtype=np.float64).reshape(-1)

    @staticmethod
    def load_info(model_path: str) -> SkinModelInfo:
        meta = Path(model_path).with_suffix(".json")
        if meta.is_file():
            data = json.loads(meta.read_text(encoding="utf-8"))
            return SkinModelInfo(name=data.get("name", "unknown"), threshold=float(data["threshold"]),
                                 combine=data.get("combine", "mean"), source=data.get("source", ""),
                                 from_metadata=True)
        return SkinModelInfo(name=Path(model_path).name)

    @classmethod
    def from_tflite(cls, model_path: str, threshold: Optional[float] = None) -> "SkinModel":
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:  # older environments
            from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
        interp = Interpreter(model_path=str(model_path))
        interp.allocate_tensors()
        inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
        shape = tuple(int(v) for v in inp["shape"])
        if len(shape) != 4 or shape[0] != 1 or shape[1:] != (PATCH_SIZE, PATCH_SIZE, 3):
            raise ValueError(f"Unsupported skin model input shape {shape}; expected (1, 64, 64, 3)")

        def predict(batch: np.ndarray) -> np.ndarray:
            probs = []
            for x in batch:
                interp.set_tensor(inp["index"], x[None, ...].astype(np.float32))
                interp.invoke()
                probs.append(float(np.asarray(interp.get_tensor(out["index"])).reshape(-1)[0]))
            return np.array(probs)

        info = cls.load_info(model_path)
        if threshold is not None:
            info.threshold = threshold
        return cls(predict, info)

    @classmethod
    def from_keras(cls, model_path: str, threshold: float, name: str = "") -> "SkinModel":
        import tensorflow as tf  # only needed for experiments
        model = tf.keras.models.load_model(str(model_path), compile=False)
        return cls(lambda b: model.predict(b, verbose=0),
                   SkinModelInfo(name=name or Path(model_path).name, threshold=threshold,
                                 source=str(model_path), from_metadata=True))


# ---------------------------------------------------------------- classification
@dataclass
class SkinResult:
    status: str = UNKNOWN
    probability: Optional[float] = None
    patch_probabilities: dict = field(default_factory=dict)
    patches_used: int = 0
    method: str = "patch_cnn"
    threshold: Optional[float] = None
    reason: str = ""                                   # why the result is UNKNOWN (quality check)
    quality: dict = field(default_factory=dict)        # measured values used by the quality check


@dataclass(frozen=True)
class SkinQualityConfig:
    """Limits of the image-suitability check. All come from the CNN's TRAINING data or from
    controlled changes to training patches (scripts/experiments/m3/skin_quality_study.py);
    none was chosen on the M3 test photos.

    * Face width. Patches are fixed 64x64 pixel crops, so the skin area inside a patch depends
      on how large the face is in the image. Training faces: Normal class 240-300 px wide (all
      close to 276 px), Flushing class about 200-630 px. Faces narrower than the 1st percentile
      of training (about 194 px) are refused: enlarging them would blur them, and blur alone
      makes the CNN answer "Flushing". For faces wider than the Normal class ever was, the CNN
      must give the same answer at the original size and reduced to the training size.
    * Sharpness. Variance of the Laplacian of the grey patch. Blurring Normal training patches
      until this value fell below 30 made the CNN call 39-91% of them "Flushing" (2% when sharp).
    * Lightness. Mean CIELAB L of the patch outside the 1st-99th percentile of training patches.
    """
    min_face_width_px: float = 200.0
    max_native_face_width_px: float = 300.0
    target_face_width_px: float = 276.0
    min_patch_sharpness: float = 30.0
    min_patch_lightness: float = 55.0     # OpenCV 8-bit L (0-255)
    max_patch_lightness: float = 228.0


REASON_NO_FACE = "no face found"
REASON_FACE_SMALL = "face too small in the image"
REASON_BLURRY = "image too blurry"
REASON_DARK = "face too dark"
REASON_BRIGHT = "face over-exposed"
REASON_NO_SKIN = "no usable skin area"
REASON_SCALE = "result changes with image size"


def face_width_px(face_landmarks: Sequence, frame_w: int) -> float:
    return abs(face_landmarks[CHEEK_R_EDGE_ID].x - face_landmarks[CHEEK_L_EDGE_ID].x) * frame_w


def patch_sharpness(patch_bgr: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def _classify_at_scale(image_bgr: np.ndarray, face_landmarks: Sequence, model: SkinModel, qc: "SkinQualityConfig",
                       scale: float, jpeg_like_training: bool) -> SkinResult:
    """Cut the patches at one image scale, apply the sharpness/lightness limits, run the CNN."""
    if scale != 1.0:
        image_bgr = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = image_bgr.shape[:2]
    result = SkinResult(method="patch_cnn_qc", threshold=model.info.threshold, quality={"scale": round(scale, 3)})
    names, inputs, dropped, measured = [], [], {}, {}
    for zone, cx, cy in patch_centres(face_landmarks, w, h):
        patch = extract_square_patch(image_bgr, cx, cy)
        if not is_valid_skin_color(patch):
            dropped[zone] = REASON_NO_SKIN
            continue
        if jpeg_like_training:
            patch = jpeg_roundtrip(patch)
        sharp = patch_sharpness(patch)
        light = float(cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)[..., 0].mean())
        measured[zone] = {"sharpness": round(sharp, 1), "lightness": round(light, 1)}
        if sharp < qc.min_patch_sharpness:
            dropped[zone] = REASON_BLURRY
        elif light < qc.min_patch_lightness:
            dropped[zone] = REASON_DARK
        elif light > qc.max_patch_lightness:
            dropped[zone] = REASON_BRIGHT
        else:
            names.append(zone)
            inputs.append(to_model_input(patch))
    result.quality.update({"patches": measured, "dropped": dropped})
    if not inputs:
        reasons = list(dropped.values())
        result.reason = next((r for r in (REASON_BLURRY, REASON_DARK, REASON_BRIGHT) if r in reasons), REASON_NO_SKIN)
        return result
    probs = model.predict(np.stack(inputs))
    result.patch_probabilities = {n: round(float(p), 4) for n, p in zip(names, probs)}
    result.patches_used = len(inputs)
    result.probability = float(np.max(probs) if model.info.combine == "max" else np.mean(probs))
    result.status = FLUSHING if result.probability >= model.info.threshold else NORMAL
    return result


def classify_patches_qc(image_bgr: np.ndarray, face_landmarks: Sequence, model: SkinModel,
                        qc: SkinQualityConfig = SkinQualityConfig(),
                        jpeg_like_training: bool = True) -> SkinResult:
    """Patch CNN with the image-suitability check (round B). UNKNOWN + reason when unsuitable.

    * Face narrower than ``min_face_width_px``: UNKNOWN.
    * Face within the training range: patches at the original size.
    * Face wider than the Normal class was ever trained on: the CNN is run twice, at the original
      size and reduced to the training size, and answers only when both agree. A result that
      changes with the size of the image is not a skin finding.
    """
    fw = face_width_px(face_landmarks, image_bgr.shape[1])
    if fw < qc.min_face_width_px:
        return SkinResult(method="patch_cnn_qc", threshold=model.info.threshold, reason=REASON_FACE_SMALL,
                          quality={"face_width_px": round(fw, 1)})
    native = _classify_at_scale(image_bgr, face_landmarks, model, qc, 1.0, jpeg_like_training)
    native.quality["face_width_px"] = round(fw, 1)
    if fw <= qc.max_native_face_width_px:
        return native
    reduced = _classify_at_scale(image_bgr, face_landmarks, model, qc, qc.target_face_width_px / fw, jpeg_like_training)
    quality = {"face_width_px": round(fw, 1), "original_size": {"status": native.status, "probability": native.probability,
                                                              "reason": native.reason, **native.quality},
               "training_size": {"status": reduced.status, "probability": reduced.probability,
                                 "reason": reduced.reason, **reduced.quality}}
    out = SkinResult(method="patch_cnn_qc", threshold=model.info.threshold, quality=quality)
    if UNKNOWN in (native.status, reduced.status):
        out.reason = reduced.reason or native.reason
    elif native.status != reduced.status:
        out.reason = REASON_SCALE
    else:
        out.status = reduced.status
        out.probability = float((native.probability + reduced.probability) / 2.0)
        out.patch_probabilities = reduced.patch_probabilities
        out.patches_used = reduced.patches_used
    return out


def classify_patches(image_bgr: np.ndarray, face_landmarks: Sequence, model: SkinModel,
                     jpeg_like_training: bool = True) -> SkinResult:
    h, w = image_bgr.shape[:2]
    names, inputs = [], []
    for zone, cx, cy in patch_centres(face_landmarks, w, h):
        patch = extract_square_patch(image_bgr, cx, cy)
        if not is_valid_skin_color(patch):
            continue
        if jpeg_like_training:
            patch = jpeg_roundtrip(patch)
        names.append(zone)
        inputs.append(to_model_input(patch))
    result = SkinResult(threshold=model.info.threshold)
    if not inputs:
        return result                                    # UNKNOWN: no usable skin patch
    probs = model.predict(np.stack(inputs))
    result.patch_probabilities = {n: round(float(p), 4) for n, p in zip(names, probs)}
    result.patches_used = len(inputs)
    result.probability = float(np.max(probs) if model.info.combine == "max" else np.mean(probs))
    result.status = FLUSHING if result.probability >= model.info.threshold else NORMAL
    return result


def classify_whole_face_legacy(face_roi_bgr: np.ndarray, model: SkinModel) -> SkinResult:
    """ORIGINAL kiosk behaviour (appendix only): whole face squeezed to 64x64."""
    result = SkinResult(method="legacy_whole_face", threshold=model.info.threshold)
    if face_roi_bgr is None or face_roi_bgr.size == 0:
        return result
    prob = float(model.predict(to_model_input(face_roi_bgr)[None, ...])[0])
    result.probability, result.patches_used = prob, 1
    result.status = FLUSHING if prob >= model.info.threshold else NORMAL
    return result


def face_bbox_crop(image_bgr: np.ndarray, face_landmarks: Sequence) -> np.ndarray:
    """Bounding box of all Face Mesh landmarks (what the original kiosk cropped)."""
    h, w = image_bgr.shape[:2]
    xs = [int(p.x * w) for p in face_landmarks]
    ys = [int(p.y * h) for p in face_landmarks]
    x1, x2 = max(0, min(xs)), min(w, max(xs))
    y1, y2 = max(0, min(ys)), min(h, max(ys))
    return image_bgr[y1:y2, x1:x2]


def color_fallback(face_roi_bgr: np.ndarray) -> str:
    """Original HSV colour rule used when no model is available (NOT a CNN result)."""
    if face_roi_bgr is None or face_roi_bgr.size == 0:
        return UNKNOWN
    hsv = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2HSV)
    mean_hue, mean_sat = float(np.mean(hsv[:, :, 0])), float(np.mean(hsv[:, :, 1]))
    return FLUSHING if (mean_hue < 12 or mean_hue > 168) and mean_sat > 110 else NORMAL
