"""Clinical posture rules (chest, abdomen, head) from MediaPipe Pose landmarks.

Pure geometry: no MediaPipe/OpenCV imports, so the rules can be unit-tested with
hand-made landmark positions.

Two rule sets are kept side by side:

* ``classify_posture_v1`` — the ORIGINAL kiosk rules, copied unchanged. Kept only so
  the M3 experiment can report the original version in an appendix.
* ``classify_posture_v2`` — the corrected rules (default). Fixes found in the
  M3 pre-check review:
    1. headache vs "thinking": thinking = hand at the CHIN (below the nose);
       headache = hand at the FOREHEAD/TEMPLE (at or above eye level). The old
       thinking filter had no vertical check and removed real headache poses.
    2. abdomen zone is centred on the estimated NAVEL (not the hip joints), and
       hands resting at/below the hip line (e.g. on the lap) are excluded.
    3. chest zone is centred on the mid-sternum (not the collarbones) and uses the
       hand centre (wrist + index + pinky), not only the wrist.
    4. a hand is used only when its landmarks are visible; hips only when visible
       and inside the frame.

Coordinates are pixels with y pointing DOWN (image convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np

# MediaPipe Pose landmark indices used here.
NOSE, L_EYE, R_EYE, MOUTH_L, MOUTH_R = 0, 2, 5, 9, 10
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 11, 12, 13, 14, 15, 16
L_PINKY, R_PINKY, L_INDEX, R_INDEX = 17, 18, 19, 20
L_HIP, R_HIP = 23, 24

# A landmark: (x_px, y_px, visibility 0..1)
Landmarks = Mapping[int, tuple[float, float, float]]


@dataclass(frozen=True)
class PostureConfig:
    # --- v1 (original) values, read from SystemConfig names ---
    chest_ratio_v1: float = 0.35
    abdomen_ratio_v1: float = 0.40
    head_ratio_v1: float = 0.28
    elbow_max_deg: float = 115.0
    crossed_ratio: float = 0.45
    thinking_forearm_min_deg: float = 65.0
    # --- v2 (corrected) values: INITIAL, to be tuned on development images only ---
    visibility_min: float = 0.5
    sternum_pos: float = 0.30        # mid-sternum: 30% of the way from shoulder line to hip line
    navel_pos: float = 0.72          # navel: 72% of the way (hip-joint landmarks sit below the navel)
    chest_radius: float = 0.24       # hand-centre distance to mid-sternum (tuned on dev photos: clutching 0.17-0.21, hands in front 0.25-0.28)
    abdomen_radius: float = 0.30     # hand-centre distance to navel, in torso lengths
    lap_margin: float = 0.05         # hands lower than (hip line - margin) are "resting", not holding the stomach
    head_radius: float = 0.30        # hand-centre distance to forehead point, in torso lengths
    head_max_below_eyes: float = 0.05  # head pose: hand centre no lower than this below eye level
    chin_radius: float = 0.20        # thinking pose: hand-centre distance to chin point
    head_to_torso: float = 2.2       # torso ≈ 2.2 x (shoulder line - nose) when the hips are not visible


@dataclass
class PostureResult:
    chest: bool = False
    abdomen: bool = False
    head: bool = False
    filters: list[str] = field(default_factory=list)
    person_found: bool = False
    debug: dict = field(default_factory=dict)  # distances etc., for tuning on dev images

    @property
    def any(self) -> bool:
        return self.chest or self.abdomen or self.head

    def label(self) -> str:
        """Single label for reporting: chest > abdomen > head > none."""
        if self.chest:
            return "chest"
        if self.abdomen:
            return "abdomen"
        if self.head:
            return "head"
        return "none"


# ---------------------------------------------------------------- helpers
def _pt(lm: Landmarks, idx: int) -> Optional[np.ndarray]:
    v = lm.get(idx)
    return None if v is None else np.array([float(v[0]), float(v[1])])


def _vis(lm: Landmarks, idx: int) -> float:
    v = lm.get(idx)
    return 0.0 if v is None else float(v[2])


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------- v1: original rules (unchanged)
def classify_posture_v1(lm: Landmarks, frame_h: int, frame_w: int,
                        cfg: PostureConfig = PostureConfig()) -> PostureResult:
    """The original kiosk rules, reproduced exactly (incl. their known problems)."""
    res = PostureResult()
    needed = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, NOSE]
    if any(i not in lm for i in needed):
        return res
    res.person_found = True
    l_sh, r_sh, l_hip, r_hip = (_pt(lm, i) for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP))
    l_el, r_el, l_wr, r_wr, nose = (_pt(lm, i) for i in (L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, NOSE))

    chest_center = (l_sh + r_sh) / 2.0
    navel_center = (l_hip + r_hip) / 2.0
    shoulder_width = _dist(l_sh, r_sh) + 1e-6
    raw_torso = _dist(chest_center, navel_center)
    if raw_torso < (shoulder_width * 0.8) or navel_center[-1] >= (frame_h - 25):
        torso = shoulder_width * 1.3
    else:
        torso = raw_torso + 1e-6

    l_elbow_ang, r_elbow_ang = _angle(l_sh, l_el, l_wr), _angle(r_sh, r_el, r_wr)

    l_tuck = _dist(l_wr, r_el) / torso < cfg.crossed_ratio
    r_tuck = _dist(r_wr, l_el) / torso < cfg.crossed_ratio
    crossed = bool(l_tuck and r_tuck)
    if crossed:
        res.filters.append("CROSSED_ARMS_FILTER")

    l_diff, r_diff = np.abs(l_wr - l_el), np.abs(r_wr - r_el)
    l_forearm = float(np.degrees(np.arctan2(l_diff[-1], l_diff[0] + 1e-6)))
    r_forearm = float(np.degrees(np.arctan2(r_diff[-1], r_diff[0] + 1e-6)))
    l_chin, r_chin = _dist(l_wr, nose) / torso, _dist(r_wr, nose) / torso
    thinking = bool((l_chin < 0.35 and l_forearm >= cfg.thinking_forearm_min_deg) or
                    (r_chin < 0.35 and r_forearm >= cfg.thinking_forearm_min_deg))
    if thinking:
        res.filters.append("THINKING_POSE_FILTER")

    sy, ny = chest_center[-1], navel_center[-1]
    l_below = (l_wr[-1] > sy + 10) and (l_wr[-1] < ny)
    r_below = (r_wr[-1] > sy + 10) and (r_wr[-1] < ny)
    res.chest = (not crossed) and (not thinking) and bool(
        (l_below and _dist(l_wr, chest_center) / torso <= cfg.chest_ratio_v1 and l_elbow_ang <= cfg.elbow_max_deg) or
        (r_below and _dist(r_wr, chest_center) / torso <= cfg.chest_ratio_v1 and r_elbow_ang <= cfg.elbow_max_deg))

    l_navel = ny - torso * 0.2 <= l_wr[-1] <= ny + torso * 0.4
    r_navel = ny - torso * 0.2 <= r_wr[-1] <= ny + torso * 0.4
    res.abdomen = (not crossed) and bool(
        (l_navel and _dist(l_wr, navel_center) / torso <= cfg.abdomen_ratio_v1) or
        (r_navel and _dist(r_wr, navel_center) / torso <= cfg.abdomen_ratio_v1))

    nyy = nose[-1]
    l_head = (l_wr[-1] <= nyy + 15) and (l_wr[-1] >= nyy - torso * 0.5)
    r_head = (r_wr[-1] <= nyy + 15) and (r_wr[-1] >= nyy - torso * 0.5)
    res.head = (not thinking) and bool((l_head and l_chin <= cfg.head_ratio_v1) or
                                       (r_head and r_chin <= cfg.head_ratio_v1))
    res.debug = {"torso_px": round(torso, 1), "chest_center": chest_center.round(1).tolist(),
                 "navel_center": navel_center.round(1).tolist()}
    return res


# ---------------------------------------------------------------- v2: corrected rules
def _hand_centre(lm: Landmarks, wrist: int, index: int, pinky: int, vis_min: float) -> Optional[np.ndarray]:
    """Palm centre ≈ mean of wrist, index and pinky (whichever are visible); needs a visible wrist."""
    if _vis(lm, wrist) < vis_min or wrist not in lm:
        return None
    pts = [_pt(lm, wrist)] + [_pt(lm, i) for i in (index, pinky) if i in lm and _vis(lm, i) >= vis_min]
    return np.mean(pts, axis=0)


def classify_posture_v2(lm: Landmarks, frame_h: int, frame_w: int,
                        cfg: PostureConfig = PostureConfig()) -> PostureResult:
    res = PostureResult()
    vm = cfg.visibility_min
    if any(i not in lm or _vis(lm, i) < vm for i in (L_SHOULDER, R_SHOULDER)):
        return res
    res.person_found = True

    l_sh, r_sh = _pt(lm, L_SHOULDER), _pt(lm, R_SHOULDER)
    sh_mid = (l_sh + r_sh) / 2.0
    shoulder_width = _dist(l_sh, r_sh) + 1e-6

    # Torso length and "down" direction: from shoulders to hips when the hips are reliable.
    hips_ok = all(i in lm and _vis(lm, i) >= vm and lm[i][1] < frame_h - 5 for i in (L_HIP, R_HIP))
    if hips_ok:
        hip_mid = (_pt(lm, L_HIP) + _pt(lm, R_HIP)) / 2.0
        torso = _dist(sh_mid, hip_mid)
        if torso < shoulder_width * 0.8:   # leaning/odd angle: fall back to a body-proportion estimate
            hips_ok = False
    if not hips_ok:
        # Body-proportion estimate. A turned body has narrow shoulders in the image, so also use the
        # head-to-shoulder height (dev tuning 2026-11-03) and take the larger estimate.
        torso = shoulder_width * 1.3
        if NOSE in lm and _vis(lm, NOSE) >= vm and lm[NOSE][1] < sh_mid[1]:
            torso = max(torso, cfg.head_to_torso * (sh_mid[1] - lm[NOSE][1]))
        hip_mid = sh_mid + np.array([0.0, torso])
    down = (hip_mid - sh_mid) / (np.linalg.norm(hip_mid - sh_mid) + 1e-6)

    sternum = sh_mid + down * torso * cfg.sternum_pos
    navel = sh_mid + down * torso * cfg.navel_pos
    hip_y = hip_mid[1]

    face_ok = NOSE in lm and _vis(lm, NOSE) >= vm
    if face_ok:
        nose = _pt(lm, NOSE)
        eyes = [_pt(lm, i) for i in (L_EYE, R_EYE) if i in lm and _vis(lm, i) >= vm]
        eye_mid = np.mean(eyes, axis=0) if eyes else nose - np.array([0.0, 0.06 * torso])
        forehead = eye_mid - np.array([0.0, 0.08 * torso])
        mouth = [_pt(lm, i) for i in (MOUTH_L, MOUTH_R) if i in lm and _vis(lm, i) >= vm]
        mouth_mid = np.mean(mouth, axis=0) if mouth else nose + np.array([0.0, 0.06 * torso])
        chin = mouth_mid + np.array([0.0, 0.07 * torso])

    l_el, r_el = _pt(lm, L_ELBOW), _pt(lm, R_ELBOW)
    hands = {
        "left": (_hand_centre(lm, L_WRIST, L_INDEX, L_PINKY, vm), l_el, _vis(lm, L_ELBOW) >= vm),
        "right": (_hand_centre(lm, R_WRIST, R_INDEX, R_PINKY, vm), r_el, _vis(lm, R_ELBOW) >= vm),
    }

    # Crossed arms: each wrist near the OTHER elbow, or (when the elbows are cut off) both wrists have
    # crossed the body midline at chest/abdomen height (dev tuning 2026-11-03).
    crossed = False
    if hands["left"][0] is not None and hands["right"][0] is not None and r_el is not None and l_el is not None:
        crossed = (_dist(_pt(lm, L_WRIST), r_el) / torso < cfg.crossed_ratio and
                   _dist(_pt(lm, R_WRIST), l_el) / torso < cfg.crossed_ratio)
        lw, rw = _pt(lm, L_WRIST), _pt(lm, R_WRIST)
        left_side = 1.0 if l_sh[0] > r_sh[0] else -1.0          # image x direction of the person's left
        lw_cross = (lw[0] - sh_mid[0]) * left_side < -0.05 * shoulder_width
        rw_cross = (rw[0] - sh_mid[0]) * left_side > 0.05 * shoulder_width
        at_torso = all(sh_mid[1] < w[1] < sh_mid[1] + 0.9 * torso for w in (lw, rw))
        crossed = crossed or (lw_cross and rw_cross and at_torso)
    if crossed:
        res.filters.append("CROSSED_ARMS_FILTER")

    debug = {"torso_px": round(torso, 1), "hips_reliable": hips_ok,
             "sternum": sternum.round(1).tolist(), "navel": navel.round(1).tolist()}
    thinking = False
    for side, (hand, elbow, elbow_vis) in hands.items():
        if hand is None:
            debug[side] = "hand not visible"
            continue
        d = {}
        d["chest_dist"] = round(_dist(hand, sternum) / torso, 3)
        d["navel_dist"] = round(_dist(hand, navel) / torso, 3)
        shoulder = l_sh if side == "left" else r_sh
        elbow_ang = _angle(shoulder, elbow, _pt(lm, L_WRIST if side == "left" else R_WRIST)) \
            if (elbow is not None and elbow_vis) else None
        d["elbow_deg"] = None if elbow_ang is None else round(elbow_ang, 1)

        # Chest: hand centre near mid-sternum, between shoulder line and upper abdomen, elbow bent.
        in_chest_band = sh_mid[1] <= hand[1] <= sh_mid[1] + 0.55 * torso
        bent = elbow_ang is None or elbow_ang <= cfg.elbow_max_deg
        side_chest = in_chest_band and d["chest_dist"] <= cfg.chest_radius and bent

        # Abdomen: hand centre near the navel and ABOVE the hip line (not resting on the lap).
        resting = hand[1] >= hip_y - cfg.lap_margin * torso
        side_abdo = (d["navel_dist"] <= cfg.abdomen_radius) and not resting and \
            hand[1] >= sh_mid[1] + 0.45 * torso
        d["resting_at_hips"] = bool(resting)

        side_head = side_think = False
        if face_ok:
            d["forehead_dist"] = round(_dist(hand, forehead) / torso, 3)
            d["chin_dist"] = round(_dist(hand, chin) / torso, 3)
            # Head: hand at forehead/temple height (not lower than just below the eyes).
            side_head = hand[1] <= eye_mid[1] + cfg.head_max_below_eyes * torso and \
                d["forehead_dist"] <= cfg.head_radius
            # Thinking: hand at the chin (below the nose), not at the forehead.
            side_think = (not side_head) and hand[1] > nose[1] and d["chin_dist"] <= cfg.chin_radius

        res.chest |= side_chest
        res.abdomen |= side_abdo
        res.head |= side_head
        thinking |= side_think
        debug[side] = d

    if thinking:
        res.filters.append("THINKING_POSE_FILTER")
    if crossed:
        res.chest = res.abdomen = False
    if thinking:
        res.chest = False          # a hand at the chin is never chest pain
    res.debug = debug
    return res


def landmarks_from_mediapipe(pose_landmarks, frame_w: int, frame_h: int) -> dict[int, tuple[float, float, float]]:
    """Convert a MediaPipe NormalizedLandmarkList to {index: (x_px, y_px, visibility)}."""
    return {i: (lm.x * frame_w, lm.y * frame_h, float(getattr(lm, "visibility", 1.0)))
            for i, lm in enumerate(pose_landmarks.landmark)}


def classify_posture(lm: Landmarks, frame_h: int, frame_w: int, version: str = "v2",
                     cfg: PostureConfig = PostureConfig()) -> PostureResult:
    if version == "v1":
        return classify_posture_v1(lm, frame_h, frame_w, cfg)
    return classify_posture_v2(lm, frame_h, frame_w, cfg)
