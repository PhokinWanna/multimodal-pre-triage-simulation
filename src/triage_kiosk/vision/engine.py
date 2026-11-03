"""Vision engine: pose, face, and skin analysis from camera frames.

Uses MediaPipe Pose and Face Mesh. Posture rules live in ``posture.py`` (corrected
"v2" rules by default; the original "v1" rules are kept for comparison) and the skin
check lives in ``skin.py`` (forehead/cheek patches exactly as in CNN training).

Live kiosk: gestures must be held (1.5 s active, 3 s sustained) and the skin CNN runs
every 7 s. Experiments: ``analyze_image`` evaluates one still image with no timers.
"""

import logging
import time
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from ..config import SystemConfig
from . import skin as skin_mod
from .posture import PostureConfig, classify_posture, landmarks_from_mediapipe

logger = logging.getLogger(__name__)

GESTURES = ("chest_clutching", "abdominal_clutching", "head_clutching")


def posture_config_from_system() -> PostureConfig:
    c = SystemConfig
    return PostureConfig(
        chest_ratio_v1=getattr(c, "CHEST_PAIN_TORSO_RATIO", 0.35),
        abdomen_ratio_v1=getattr(c, "ABDOMEN_PAIN_TORSO_RATIO", 0.40),
        head_ratio_v1=getattr(c, "HEADACHE_HEAD_RATIO", 0.28),
        elbow_max_deg=getattr(c, "ELBOW_FLEXION_MAX_ANGLE", 115.0),
        crossed_ratio=getattr(c, "CROSSED_ARMS_WRIST_RATIO", 0.45),
        thinking_forearm_min_deg=getattr(c, "THINKING_FOREARM_MIN_ANGLE", 65.0),
        **getattr(c, "POSTURE_V2_PARAMS", {}),
    )


class VisionEngine:
    def __init__(self, static_image_mode: bool = False, posture_version: Optional[str] = None,
                 skin_pipeline: Optional[str] = None, skin_model_path: Optional[str] = None,
                 require_skin_model: Optional[bool] = None, skin_model: Optional[skin_mod.SkinModel] = None):
        self.static_image_mode = static_image_mode
        self.posture_version = posture_version or getattr(SystemConfig, "POSTURE_RULES_VERSION", "v2")
        self.skin_pipeline = skin_pipeline or getattr(SystemConfig, "SKIN_PIPELINE", "patch")
        self.posture_cfg = posture_config_from_system()

        # 1. MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        # 2. MediaPipe Face Mesh (refined landmarks, as in the CNN patch extractor)
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5 if static_image_mode else 0.6,
            min_tracking_confidence=0.6,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # 3. Gesture timers (live mode)
        self.gesture_start_times = {k: None for k in GESTURES}
        self.presence_counter = 0

        # 4. Skin CNN (throttled in live mode)
        self.last_skin_inference_time = 0.0
        self.cached_skin_status = skin_mod.UNKNOWN
        self.cached_skin_probability: Optional[float] = None

        # 5. Skin model
        if require_skin_model is None:
            require_skin_model = bool(getattr(SystemConfig, "SKIN_REQUIRE_MODEL", False))
        self.skin_model = skin_model or self._load_skin_model(
            skin_model_path or getattr(SystemConfig, "TFLITE_SKIN_MODEL_PATH", ""), require_skin_model)

    # ------------------------------------------------------------------ setup
    def _load_skin_model(self, model_path: str, required: bool) -> Optional[skin_mod.SkinModel]:
        try:
            model = skin_mod.SkinModel.from_tflite(model_path)
            # A model without metadata is the original kiosk model: keep its original threshold.
            if not model.info.from_metadata:
                model.info.threshold = getattr(SystemConfig, "SKIN_CONFIDENCE_THRESHOLD", 0.65)
            logger.info(f"Skin CNN loaded: {model.info.name} (threshold {model.info.threshold:.3f})")
            return model
        except Exception as exc:  # missing file, missing runtime, corrupt model
            if required:
                raise RuntimeError(f"Skin model is required but could not be loaded from {model_path}: {exc}")
            logger.error(f"Could not load skin model ({exc}); using the colour fallback (NOT CNN inference).")
            return None

    # ------------------------------------------------------------------ shared analysis
    def _posture(self, pose_landmarks, w: int, h: int):
        lm = landmarks_from_mediapipe(pose_landmarks, w, h)
        return classify_posture(lm, h, w, version=self.posture_version, cfg=self.posture_cfg)

    def _skin(self, frame: np.ndarray, face_landmarks) -> skin_mod.SkinResult:
        if self.skin_model is None:
            roi = skin_mod.face_bbox_crop(frame, face_landmarks)
            return skin_mod.SkinResult(status=skin_mod.color_fallback(roi), method="color_fallback")
        if self.skin_pipeline == "legacy_whole_face":
            return skin_mod.classify_whole_face_legacy(skin_mod.face_bbox_crop(frame, face_landmarks), self.skin_model)
        return skin_mod.classify_patches(frame, face_landmarks, self.skin_model)

    # ------------------------------------------------------------------ experiments
    def analyze_image(self, image_bgr: np.ndarray) -> dict:
        """One still image, no timers: posture flags + skin result (used by the M3 replay)."""
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pose_res = self.pose.process(rgb)
        face_res = self.face_mesh.process(rgb)
        posture = self._posture(pose_res.pose_landmarks, w, h) if pose_res.pose_landmarks else None
        skin_res = (self._skin(image_bgr, face_res.multi_face_landmarks[0].landmark)
                    if face_res.multi_face_landmarks else skin_mod.SkinResult(method="no_face"))
        return {
            "person_found": bool(posture and posture.person_found),
            "face_found": bool(face_res.multi_face_landmarks),
            "posture": posture,
            "visual_signs": {
                "chest_clutching": bool(posture and posture.chest),
                "abdominal_clutching": bool(posture and posture.abdomen),
                "head_clutching": bool(posture and posture.head),
            },
            "skin": skin_res,
            "skin_status": skin_res.status,
        }

    # ------------------------------------------------------------------ live kiosk
    def process_frame(self, frame: np.ndarray) -> dict:
        """Processes a camera frame with 3.0 s sustained-gesture evaluation."""
        h, w, _ = frame.shape
        clean_view = frame.copy()
        skeleton_view = frame.copy()
        now = time.time()

        telemetry = {
            "patient_present": False,
            "chest_clutching": False,
            "abdominal_clutching": False,
            "head_clutching": False,
            "sustained_gestures": {k: False for k in GESTURES},
            "dominant_current_gesture": "NONE",
            "skin_status": self.cached_skin_status,
            "skin_probability": self.cached_skin_probability,
            "clean_view": clean_view,
            "skeleton_view": skeleton_view,
            "active_filters": [],
        }

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = self.pose.process(rgb_frame)
        face_res = self.face_mesh.process(rgb_frame)

        if not pose_res.pose_landmarks:
            self.presence_counter = max(0, self.presence_counter - 1)
            for k in self.gesture_start_times:
                self.gesture_start_times[k] = None
            return telemetry

        self.presence_counter += 1
        if self.presence_counter >= getattr(SystemConfig, "PRESENCE_FRAMES_TRIGGER", 15):
            telemetry["patient_present"] = True

        posture = self._posture(pose_res.pose_landmarks, w, h)
        telemetry["active_filters"] = list(posture.filters)
        raw = {"chest_clutching": posture.chest, "abdominal_clutching": posture.abdomen,
               "head_clutching": posture.head}

        hold_time = getattr(SystemConfig, "GESTURE_HOLD_SECONDS", 1.5)
        sustained_time = getattr(SystemConfig, "GESTURE_SUSTAINED_SECONDS", 3.0)

        def _evaluate_duration(key: str, active: bool) -> tuple[bool, bool, float]:
            if active:
                if self.gesture_start_times[key] is None:
                    self.gesture_start_times[key] = now
                    return False, False, 0.0
                elapsed = now - self.gesture_start_times[key]
                return elapsed >= hold_time, elapsed >= sustained_time, elapsed
            self.gesture_start_times[key] = None
            return False, False, 0.0

        durations = {}
        for key in GESTURES:
            act, sus, dur = _evaluate_duration(key, raw[key])
            telemetry[key] = act
            telemetry["sustained_gestures"][key] = sus
            durations[key] = dur

        c_dur, a_dur, h_dur = durations["chest_clutching"], durations["abdominal_clutching"], durations["head_clutching"]
        if telemetry["chest_clutching"] and c_dur >= max(a_dur, h_dur):
            telemetry["dominant_current_gesture"] = "CHEST_PAIN"
        elif telemetry["abdominal_clutching"] and a_dur >= max(c_dur, h_dur):
            telemetry["dominant_current_gesture"] = "ABDOMINAL_PAIN"
        elif telemetry["head_clutching"] and h_dur >= max(c_dur, a_dur):
            telemetry["dominant_current_gesture"] = "HEAD_PAIN"

        # Throttled skin check (every SKIN_DUTY_CYCLE_SECONDS)
        duty_cycle = getattr(SystemConfig, "SKIN_DUTY_CYCLE_SECONDS", 7.0)
        if (now - self.last_skin_inference_time) >= duty_cycle and face_res.multi_face_landmarks:
            result = self._skin(frame, face_res.multi_face_landmarks[0].landmark)
            self.last_skin_inference_time = now
            if result.status != skin_mod.UNKNOWN:          # keep the last good reading otherwise
                self.cached_skin_status = result.status
                self.cached_skin_probability = result.probability
            telemetry["skin_status"] = self.cached_skin_status
            telemetry["skin_probability"] = self.cached_skin_probability

        # Diagnostics overlay
        self.mp_draw.draw_landmarks(skeleton_view, pose_res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        torso = float(posture.debug.get("torso_px", 100.0))
        if telemetry["chest_clutching"]:
            c = posture.debug.get("sternum") or posture.debug.get("chest_center")
            cv2.circle(skeleton_view, tuple(int(v) for v in c), int(torso * 0.30), (0, 0, 255), 3)
            cv2.putText(skeleton_view, "SIGN: CHEST PAIN", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif telemetry["abdominal_clutching"]:
            c = posture.debug.get("navel") or posture.debug.get("navel_center")
            cv2.circle(skeleton_view, tuple(int(v) for v in c), int(torso * 0.30), (0, 165, 255), 3)
            cv2.putText(skeleton_view, "SIGN: ABDOMINAL GUARDING", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        elif telemetry["head_clutching"]:
            cv2.putText(skeleton_view, "SIGN: HEAD DISTRESS", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        if telemetry["skin_status"] == skin_mod.FLUSHING:
            cv2.putText(skeleton_view, "CNN SKIN: ERYTHEMA / FLUSHING", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if telemetry["active_filters"]:
            cv2.putText(skeleton_view, "FILTER ACTIVE: " + ", ".join(telemetry["active_filters"]), (30, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        telemetry["clean_view"] = clean_view
        telemetry["skeleton_view"] = skeleton_view
        return telemetry

    def release(self):
        self.pose.close()
        self.face_mesh.close()
