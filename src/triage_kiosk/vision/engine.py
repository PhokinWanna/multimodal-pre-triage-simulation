"""Vision engine: pose, face, and skin analysis from camera frames.

Uses MediaPipe Pose and Face Mesh to detect sustained gestures (3-second
windows) and runs a throttled TFLite CNN on facial skin patches to classify
Normal vs Flushing, with a color-based fallback if the model is unavailable.
"""

import logging
import os
import cv2
import time
import numpy as np
import mediapipe as mp
from ..config import SystemConfig

logger = logging.getLogger(__name__)

class VisionEngine:
    def __init__(self):
        # 1. MediaPipe Pose Subsystem
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # 2. MediaPipe Face Mesh Subsystem
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils

        # 3. Dynamic Gesture Duration Trackers (Continuous Timers)
        self.gesture_start_times = {
            "chest_clutching": None,
            "abdominal_clutching": None,
            "head_clutching": None
        }
        self.presence_counter = 0

        # 4. Throttled Face Skin CNN (7.0s Duty Cycle)
        self.last_skin_inference_time = 0.0
        self.cached_skin_status = "NORMAL"

        # 5. TFLite Interpreter Bridge
        self.tflite_interpreter = None
        self._init_tflite_model()

    def _init_tflite_model(self):
        """Loads skin_classifier.tflite if present; otherwise runs mock CNN scaffold."""
        model_path = getattr(SystemConfig, "TFLITE_SKIN_MODEL_PATH", "")
        if os.path.exists(model_path):
            try:
                from ai_edge_litert.interpreter import Interpreter
                self.tflite_interpreter = Interpreter(model_path=model_path)
                self.tflite_interpreter.allocate_tensors()
                logger.info(f"Armed TFLite Skin CNN via LiteRT: {model_path}")
                return
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                logger.error(f"Could not initialize LiteRT model at {model_path}: {exc}")

        else:
            logger.warning(f"TFLite skin model is missing: {model_path}")
        logger.warning("Using the experimental color-based skin fallback; this is not CNN inference.")

    @staticmethod
    def _prepare_skin_tensor(
        face_roi: np.ndarray, input_shape: np.ndarray
    ) -> np.ndarray:
        """Match the trainer's square resize and BGR-to-CIELAB input pipeline."""
        if len(input_shape) != 4 or int(input_shape[0]) != 1 or int(input_shape[3]) != 3:
            raise ValueError(f"Unsupported TFLite input shape: {input_shape}")
        input_height, input_width = int(input_shape[1]), int(input_shape[2])
        resized = cv2.resize(face_roi, (input_width, input_height))
        lab_image = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        return np.expand_dims(lab_image.astype(np.float32) / 255.0, axis=0)

    @staticmethod
    def _calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Calculates 2D joint angle at vertex b in degrees."""
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    def _infer_skin_cnn(self, face_roi: np.ndarray) -> str:
        """Classifies facial ROI: 'NORMAL' vs 'RED_FLUSHING'."""
        if face_roi is None or face_roi.size == 0:
            return self.cached_skin_status

        if self.tflite_interpreter is not None:
            try:
                input_details = self.tflite_interpreter.get_input_details()
                output_details = self.tflite_interpreter.get_output_details()
                tensor = self._prepare_skin_tensor(
                    face_roi, input_details[0]["shape"]
                )

                self.tflite_interpreter.set_tensor(input_details[0]['index'], tensor)
                self.tflite_interpreter.invoke()
                predictions = self.tflite_interpreter.get_tensor(
                    output_details[0]["index"]
                )
                score = float(np.asarray(predictions).reshape(-1)[0])
                if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError(f"Invalid TFLite probability output: {score}")

                thresh = getattr(SystemConfig, "SKIN_CONFIDENCE_THRESHOLD", 0.65)
                status = "RED_FLUSHING" if score >= thresh else "NORMAL"
                self.cached_skin_status = status
                return status
            except (IndexError, TypeError, ValueError, RuntimeError, cv2.error) as exc:
                logger.error(f"TFLite inference failed; using color fallback: {exc}")

        # Colorimetry Scaffold Fallback
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        mean_hue = np.mean(hsv[:, :, 0])
        mean_sat = np.mean(hsv[:, :, 1])

        if (mean_hue < 12 or mean_hue > 168) and mean_sat > 110:
            self.cached_skin_status = "RED_FLUSHING"
        else:
            self.cached_skin_status = "NORMAL"

        return self.cached_skin_status

    def process_frame(self, frame: np.ndarray) -> dict:
        """Processes frame at ~30 FPS with scale-invariance and 3.0s sustained evaluation."""
        h, w, _ = frame.shape
        clean_view = frame.copy()
        skeleton_view = frame.copy()
        now = time.time()

        telemetry = {
            "patient_present": False,
            "chest_clutching": False,
            "abdominal_clutching": False,
            "head_clutching": False,
            "sustained_gestures": {
                "chest_clutching": False,
                "abdominal_clutching": False,
                "head_clutching": False
            },
            "dominant_current_gesture": "NONE",
            "skin_status": self.cached_skin_status,
            "clean_view": clean_view,
            "skeleton_view": skeleton_view,
            "active_filters": []
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

        landmarks = pose_res.pose_landmarks.landmark
        get_pt = lambda lm: np.array([float(lm.x * w), float(lm.y * h)])

        # Joint Coordinates
        l_sh = get_pt(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
        r_sh = get_pt(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])
        l_hip = get_pt(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP])
        r_hip = get_pt(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP])
        l_el = get_pt(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW])
        r_el = get_pt(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW])
        l_wr = get_pt(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST])
        r_wr = get_pt(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST])
        nose = get_pt(landmarks[self.mp_pose.PoseLandmark.NOSE])

        chest_center = (l_sh + r_sh) / 2.0
        navel_center = (l_hip + r_hip) / 2.0

        # Scale Factor (Handles sitting patients with cut-off hips)
        shoulder_width = float(np.linalg.norm(l_sh - r_sh)) + 1e-6
        raw_torso = float(np.linalg.norm(chest_center - navel_center))
        
        if raw_torso < (shoulder_width * 0.8) or navel_center[-1] >= (h - 25):
            torso_length = shoulder_width * 1.3
        else:
            torso_length = raw_torso + 1e-6

        l_elbow_ang = self._calculate_angle(l_sh, l_el, l_wr)
        r_elbow_ang = self._calculate_angle(r_sh, r_el, r_wr)

        # -------------------------------------------------------------
        # FILTER 1: Crossed Arms Filter
        # -------------------------------------------------------------
        cross_ratio = getattr(SystemConfig, "CROSSED_ARMS_WRIST_RATIO", 0.45)
        l_tuck = float(np.linalg.norm(l_wr - r_el) / torso_length) < cross_ratio
        r_tuck = float(np.linalg.norm(r_wr - l_el) / torso_length) < cross_ratio
        is_crossed_arms = bool(l_tuck and r_tuck)

        if is_crossed_arms:
            telemetry["active_filters"].append("CROSSED_ARMS_FILTER")

        # -------------------------------------------------------------
        # FILTER 2: Thinking Pose Filter
        # -------------------------------------------------------------
        l_diff = np.abs(l_wr - l_el)
        r_diff = np.abs(r_wr - r_el)
        l_forearm_ang = float(np.degrees(np.arctan2(l_diff[-1], l_diff[0] + 1e-6)))
        r_forearm_ang = float(np.degrees(np.arctan2(r_diff[-1], r_diff[0] + 1e-6)))

        l_chin_dist = float(np.linalg.norm(l_wr - nose) / torso_length)
        r_chin_dist = float(np.linalg.norm(r_wr - nose) / torso_length)

        think_ang = getattr(SystemConfig, "THINKING_FOREARM_MIN_ANGLE", 65.0)
        is_thinking = bool(
            (l_chin_dist < 0.35 and l_forearm_ang >= think_ang) or
            (r_chin_dist < 0.35 and r_forearm_ang >= think_ang)
        )

        if is_thinking:
            telemetry["active_filters"].append("THINKING_POSE_FILTER")

        # -------------------------------------------------------------
        # CLINICAL CHEST PAIN ZONE (Below Clavicle, Above Navel)
        # -------------------------------------------------------------
        l_wrist_y = l_wr[-1]
        r_wrist_y = r_wr[-1]
        shoulder_y = chest_center[-1]
        navel_y = navel_center[-1]

        l_is_below_clavicle = (l_wrist_y > (shoulder_y + 10)) and (l_wrist_y < navel_y)
        r_is_below_clavicle = (r_wrist_y > (shoulder_y + 10)) and (r_wrist_y < navel_y)

        l_wrist_chest = float(np.linalg.norm(l_wr - chest_center) / torso_length)
        r_wrist_chest = float(np.linalg.norm(r_wr - chest_center) / torso_length)
        chest_ratio = getattr(SystemConfig, "CHEST_PAIN_TORSO_RATIO", 0.35)
        elbow_max = getattr(SystemConfig, "ELBOW_FLEXION_MAX_ANGLE", 115.0)

        raw_chest = (not is_crossed_arms) and (not is_thinking) and bool(
            (l_is_below_clavicle and l_wrist_chest <= chest_ratio and l_elbow_ang <= elbow_max) or
            (r_is_below_clavicle and r_wrist_chest <= chest_ratio and r_elbow_ang <= elbow_max)
        )

        # -------------------------------------------------------------
        # CLINICAL ABDOMINAL GUARDING ZONE
        # -------------------------------------------------------------
        l_is_at_navel = (l_wrist_y >= (navel_y - (torso_length * 0.2))) and (l_wrist_y <= (navel_y + (torso_length * 0.4)))
        r_is_at_navel = (r_wrist_y >= (navel_y - (torso_length * 0.2))) and (r_wrist_y <= (navel_y + (torso_length * 0.4)))
        abdo_ratio = getattr(SystemConfig, "ABDOMEN_PAIN_TORSO_RATIO", 0.40)

        raw_abdo = (not is_crossed_arms) and bool(
            (l_is_at_navel and float(np.linalg.norm(l_wr - navel_center) / torso_length) <= abdo_ratio) or
            (r_is_at_navel and float(np.linalg.norm(r_wr - navel_center) / torso_length) <= abdo_ratio)
        )

        # -------------------------------------------------------------
        # CRANIAL DISTRESS (HEADACHE) - FOREHEAD / TEMPLES ONLY
        # -------------------------------------------------------------
        nose_y = nose[-1]
        l_is_at_head = (l_wrist_y <= (nose_y + 15)) and (l_wrist_y >= (nose_y - (torso_length * 0.5)))
        r_is_at_head = (r_wrist_y <= (nose_y + 15)) and (r_wrist_y >= (nose_y - (torso_length * 0.5)))
        head_ratio = getattr(SystemConfig, "HEADACHE_HEAD_RATIO", 0.28)

        raw_head = (not is_thinking) and bool(
            (l_is_at_head and l_chin_dist <= head_ratio) or
            (r_is_at_head and r_chin_dist <= head_ratio)
        )

        # -------------------------------------------------------------
        # TIME-WINDOW EVALUATION (Eliminates Latching Bug)
        # -------------------------------------------------------------
        hold_time = getattr(SystemConfig, "GESTURE_HOLD_SECONDS", 1.5)
        sustained_time = getattr(SystemConfig, "GESTURE_SUSTAINED_THRESHOLD", 3.0)

        def _evaluate_duration(key: str, active: bool) -> tuple[bool, bool, float]:
            if active:
                if self.gesture_start_times[key] is None:
                    self.gesture_start_times[key] = now
                    return False, False, 0.0
                elapsed = now - self.gesture_start_times[key]
                is_active_held = elapsed >= hold_time
                is_sustained = elapsed >= sustained_time
                return is_active_held, is_sustained, elapsed
            else:
                self.gesture_start_times[key] = None
                return False, False, 0.0

        c_act, c_sus, c_dur = _evaluate_duration("chest_clutching", raw_chest)
        a_act, a_sus, a_dur = _evaluate_duration("abdominal_clutching", raw_abdo)
        h_act, h_sus, h_dur = _evaluate_duration("head_clutching", raw_head)

        telemetry["chest_clutching"] = c_act
        telemetry["abdominal_clutching"] = a_act
        telemetry["head_clutching"] = h_act

        telemetry["sustained_gestures"]["chest_clutching"] = c_sus
        telemetry["sustained_gestures"]["abdominal_clutching"] = a_sus
        telemetry["sustained_gestures"]["head_clutching"] = h_sus

        # Determine Dominant Active Gesture for Proactive Inquiries
        if c_act and c_dur >= max(a_dur, h_dur):
            telemetry["dominant_current_gesture"] = "CHEST_PAIN"
        elif a_act and a_dur >= max(c_dur, h_dur):
            telemetry["dominant_current_gesture"] = "ABDOMINAL_PAIN"
        elif h_act and h_dur >= max(c_dur, a_dur):
            telemetry["dominant_current_gesture"] = "HEAD_PAIN"
        else:
            telemetry["dominant_current_gesture"] = "NONE"

        # -------------------------------------------------------------
        # Throttled Skin CNN Duty Cycle (Every 7.0 seconds)
        # -------------------------------------------------------------
        duty_cycle = getattr(SystemConfig, "SKIN_DUTY_CYCLE_SECONDS", 7.0)
        if (now - self.last_skin_inference_time) >= duty_cycle:
            if face_res.multi_face_landmarks:
                flms = face_res.multi_face_landmarks[0].landmark
                xs = [int(p.x * w) for p in flms]
                ys = [int(p.y * h) for p in flms]
                x1, x2 = max(0, min(xs)), min(w, max(xs))
                y1, y2 = max(0, min(ys)), min(h, max(ys))
                face_crop = frame[y1:y2, x1:x2]

                telemetry["skin_status"] = self._infer_skin_cnn(face_crop)
                self.last_skin_inference_time = now

        # Draw Diagnostics
        self.mp_draw.draw_landmarks(skeleton_view, pose_res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        if telemetry["chest_clutching"]:
            cv2.circle(skeleton_view, tuple(chest_center.astype(int)), int(torso_length * 0.35), (0, 0, 255), 3)
            cv2.putText(skeleton_view, "SIGN: CHEST PAIN", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif telemetry["abdominal_clutching"]:
            cv2.circle(skeleton_view, tuple(navel_center.astype(int)), int(torso_length * 0.35), (0, 165, 255), 3)
            cv2.putText(skeleton_view, "SIGN: ABDOMINAL GUARDING", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        elif telemetry["head_clutching"]:
            cv2.circle(skeleton_view, tuple(nose.astype(int)), int(torso_length * 0.25), (255, 0, 255), 3)
            cv2.putText(skeleton_view, "SIGN: HEAD DISTRESS", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        if telemetry["skin_status"] == "RED_FLUSHING":
            cv2.putText(skeleton_view, "CNN SKIN: ERYTHEMA / FLUSHING", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if telemetry["active_filters"]:
            f_label = ", ".join(telemetry["active_filters"])
            cv2.putText(skeleton_view, f"FILTER ACTIVE: {f_label}", (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        telemetry["clean_view"] = clean_view
        telemetry["skeleton_view"] = skeleton_view
        return telemetry

    def release(self):
        self.pose.close()
        self.face_mesh.close()