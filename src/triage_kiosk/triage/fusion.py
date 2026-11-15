"""Weighted fusion of the two camera results into one visual risk score (round B).

Until round B the posture rules and the skin CNN reached the reasoning step as two
independent findings. Here each is turned into a 0-1 score and combined:

    visual score = w_posture x posture score + w_skin x skin score        (w_posture + w_skin = 1)

* Posture score: the most serious posture seen. Chest 1.0 (possible cardiac pain), abdomen 0.7,
  head 0.6, none 0. The order follows the kiosk's own priority (chest > abdomen > head).
* Skin score: the CNN's probability of flushing (0-1).
* A part that could not be ASSESSED (skin UNKNOWN because the face is too small or blurred, no
  person found) adds no risk: its score is 0 and its weight is NOT given to the other part
  (``missing="zero"``, the default). Reason: with the weight moved to the other part, "skin
  unknown" would rank a patient higher than "skin checked and normal". ``missing="renormalise"``
  keeps the other behaviour for a sensitivity analysis.
* A part that is switched OFF in an experiment (posture off, skin off) is treated the same way:
  no evidence from that part.
* Bands, fixed before the test run: HIGH >= 0.50, MODERATE >= 0.25, otherwise LOW. With equal
  weights the strongest single sign (hand on chest) reaches HIGH and any other single sign
  reaches MODERATE.
* Safety floor (``esi_cap``): HIGH means the final level cannot be less urgent than ESI 2.
  MODERATE does not force a level on its own (a flushed face after sunburn, or a hand on the
  head with a mild headache, must be able to stay ESI 4-5); it is shown to the reasoning step
  as supporting evidence. ``policy="high_and_moderate"`` (MODERATE -> at most ESI 3) and
  ``policy="none"`` exist for the sensitivity analysis.

Pure functions, no I/O: unit-tested in tests/test_fusion.py.
"""

from __future__ import annotations

from typing import Optional

POSTURE_SCORES = {"chest_clutching": 1.0, "abdominal_clutching": 0.7, "head_clutching": 0.6}
POSTURE_NAMES = {"chest_clutching": "hand held on the chest", "abdominal_clutching": "hand held on the abdomen",
                 "head_clutching": "hand held on the head"}
BAND_HIGH, BAND_MODERATE = 0.50, 0.25
HIGH, MODERATE, LOW, NOT_ASSESSED = "HIGH", "MODERATE", "LOW", "NOT_ASSESSED"
FLUSHING_NAMES = {"RED_FLUSHING", "FLUSHING_(RED)", "PALLOR_(PALE)"}
NO_SKIN_RESULT = {"", "UNKNOWN", "NOT_ASSESSED", "NONE"}
CAP_POLICIES = ("none", "high_only", "high_and_moderate")
WEIGHT_PRESETS = {"50/50": (0.5, 0.5), "60/40": (0.6, 0.4), "40/60": (0.4, 0.6)}


def normalise_status(skin_status: Optional[str]) -> str:
    return (skin_status or "").strip().upper().replace(" ", "_")


def posture_score(visual_signs: Optional[dict]) -> Optional[float]:
    """None when posture was not assessed; otherwise the highest score among the postures seen."""
    if visual_signs is None:
        return None
    return max([score for key, score in POSTURE_SCORES.items() if visual_signs.get(key)], default=0.0)


def skin_score(skin_status: Optional[str], skin_probability: Optional[float]) -> Optional[float]:
    """None when the skin check gave no answer; otherwise P(flushing)."""
    status = normalise_status(skin_status)
    if status in NO_SKIN_RESULT:
        return None
    if skin_probability is not None:
        return float(min(1.0, max(0.0, skin_probability)))
    return 1.0 if status in FLUSHING_NAMES else 0.0


def band_of(score: Optional[float]) -> str:
    if score is None:
        return NOT_ASSESSED
    score = round(score, 6)          # 0.5 * 1.0 must be HIGH whatever the floating-point rounding
    return HIGH if score >= BAND_HIGH else MODERATE if score >= BAND_MODERATE else LOW


def visual_score(visual_signs: Optional[dict], skin_status: Optional[str], skin_probability: Optional[float],
                 w_posture: float = 0.5, w_skin: float = 0.5, missing: str = "zero") -> dict:
    """Combine the two camera results. Returns score (0-1, or None when nothing was assessed),
    band and the parts used. ``visual_signs=None`` = posture not assessed / switched off;
    skin status UNKNOWN / NOT_ASSESSED = skin not assessed / switched off."""
    if w_posture < 0 or w_skin < 0 or w_posture + w_skin <= 0:
        raise ValueError("weights must be non-negative and not both zero")
    if missing not in ("zero", "renormalise"):
        raise ValueError("missing must be 'zero' or 'renormalise'")
    total = w_posture + w_skin
    w_posture, w_skin = w_posture / total, w_skin / total
    p, s = posture_score(visual_signs), skin_score(skin_status, skin_probability)
    used = {"posture": w_posture if p is not None else 0.0, "skin": w_skin if s is not None else 0.0}
    if p is None and s is None:
        score = None
    elif missing == "renormalise" and (p is None or s is None):
        score = p if s is None else s
        used = {"posture": 1.0 if s is None else 0.0, "skin": 0.0 if s is None else 1.0}
    else:
        score = w_posture * (p or 0.0) + w_skin * (s or 0.0)
    return {"score": None if score is None else round(score, 3), "band": band_of(score),
            "posture_score": p, "skin_score": None if s is None else round(s, 3),
            "posture_assessed": p is not None, "skin_assessed": s is not None,
            "weights": {"posture": round(w_posture, 3), "skin": round(w_skin, 3)},
            "weights_used": {k: round(v, 3) for k, v in used.items()}, "missing": missing}


def esi_cap(band: str, policy: str = "high_only") -> Optional[int]:
    """Highest (least urgent) ESI level allowed for a visual band under a floor policy."""
    if policy not in CAP_POLICIES:
        raise ValueError(f"unknown cap policy: {policy}")
    if policy == "none":
        return None
    if band == HIGH:
        return 2
    if band == MODERATE and policy == "high_and_moderate":
        return 3
    return None


def describe_posture(visual_signs: Optional[dict]) -> str:
    if visual_signs is None:
        return "not assessed"
    seen = [name for key, name in POSTURE_NAMES.items() if visual_signs.get(key)]
    return ", ".join(seen) if seen else "no pain posture"


def describe_skin(skin_status: Optional[str], skin_probability: Optional[float] = None, reason: str = "") -> str:
    status = normalise_status(skin_status)
    if status in NO_SKIN_RESULT:
        return "not assessed" + (f" ({reason})" if reason else "")
    word = "flushed (red)" if status in FLUSHING_NAMES else "normal colour"
    return word if skin_probability is None else f"{word}, flushing probability {skin_probability:.2f}"
