"""Round-B vision: posture rules v3 and the skin image-suitability check (no camera, no model file)."""
import unittest

import numpy as np

from test_posture import H, W, body
from test_vision_preprocessing import FakeModel, face_landmarks
from triage_kiosk.vision import posture as P
from triage_kiosk.vision import skin


def v3(lm):
    return P.classify_posture_v3(lm, H, W)


def with_ears(lm):
    lm[P.L_EAR], lm[P.R_EAR] = (352, 84, 1), (288, 84, 1)
    return lm


class PostureV3Tests(unittest.TestCase):
    def test_default_version_is_v3(self):
        r = P.classify_posture(body(), H, W)
        self.assertEqual(r.debug.get("rules"), "v3")
        self.assertEqual(P.classify_posture(body(), H, W, version="v2").debug.get("rules"), None)

    def test_arms_down_and_lap_are_no_posture(self):
        self.assertEqual(v3(body()).label(), "none")
        lap = body(left_hand=(338, 300), right_hand=(302, 300), left_elbow=(380, 250), right_elbow=(260, 250))
        self.assertEqual(v3(lap).label(), "none")

    def test_hand_on_left_lower_chest_is_found_by_v3_but_not_v2(self):
        lm = body(right_hand=(345, 222), right_elbow=(262, 245))
        self.assertTrue(v3(lm).chest)
        self.assertFalse(P.classify_posture_v2(lm, H, W).chest)

    def test_one_hand_is_never_chest_and_abdomen_together(self):
        for y in range(190, 275, 5):
            r = v3(body(right_hand=(318, y), right_elbow=(262, 245)))
            self.assertFalse(r.chest and r.abdomen, y)
        self.assertTrue(v3(body(right_hand=(318, 262), right_elbow=(262, 250))).abdomen)

    def test_hand_on_temple_is_head(self):
        lm = with_ears(body(right_hand=(280, 80), right_elbow=(245, 150)))
        self.assertTrue(v3(lm).head)

    def test_chin_is_thinking_not_head(self):
        r = v3(with_ears(body(right_hand=(318, 122), right_elbow=(290, 200))))
        self.assertFalse(r.head)
        self.assertIn("THINKING_POSE_FILTER", r.filters)

    def test_crossed_arms_filter_still_blocks_chest(self):
        lm = body(left_hand=(262, 215), right_hand=(378, 215), left_elbow=(385, 225), right_elbow=(255, 225))
        r = v3(lm)
        self.assertIn("CROSSED_ARMS_FILTER", r.filters)
        self.assertFalse(r.chest)

    def test_hips_out_of_frame_use_the_extrapolated_torso(self):
        r = v3(body(hips_visible=False, right_hand=(318, 205), right_elbow=(265, 235)))
        self.assertEqual(r.debug["torso_source"], "extrapolated hips")
        self.assertTrue(r.chest)


def textured_skin(w, h, seed=0, bgr=(120, 150, 205), noise=12):
    rng = np.random.default_rng(seed)
    img = np.clip(np.array(bgr, np.float32) + rng.normal(0, noise, (h, w, 3)), 0, 255).astype(np.uint8)
    return img


class SkinQualityCheckTests(unittest.TestCase):
    def test_small_face_is_not_assessed(self):
        model = FakeModel([])
        r = skin.classify_patches_qc(textured_skin(400, 400), face_landmarks(400, 400), model)   # face 160 px wide
        self.assertEqual((r.status, r.reason), (skin.UNKNOWN, skin.REASON_FACE_SMALL))
        self.assertEqual(model.calls, [])

    def test_face_at_training_size_is_answered_once(self):
        model = FakeModel([0.9, 0.8, 0.7])
        r = skin.classify_patches_qc(textured_skin(650, 650), face_landmarks(650, 650), model)   # face 260 px
        self.assertEqual((r.status, r.patches_used, r.reason), (skin.FLUSHING, 3, ""))
        self.assertAlmostEqual(r.probability, 0.8)
        self.assertEqual(len(model.calls), 1)

    def test_smooth_image_is_too_blurry(self):
        flat = textured_skin(650, 650, noise=0)
        r = skin.classify_patches_qc(flat, face_landmarks(650, 650), FakeModel([]))
        self.assertEqual((r.status, r.reason), (skin.UNKNOWN, skin.REASON_BLURRY))

    def test_wide_face_needs_the_same_answer_at_both_sizes(self):
        img, lm = textured_skin(1200, 1200), face_landmarks(1200, 1200)                          # face 480 px
        agree = FakeModel([0.9, 0.9, 0.9, 0.7, 0.7, 0.7])
        r = skin.classify_patches_qc(img, lm, agree)
        self.assertEqual(r.status, skin.FLUSHING)
        self.assertAlmostEqual(r.probability, 0.8)
        self.assertEqual(len(agree.calls), 2)
        differ = FakeModel([0.9, 0.9, 0.9, 0.2, 0.2, 0.2])
        r = skin.classify_patches_qc(img, lm, differ)
        self.assertEqual((r.status, r.reason), (skin.UNKNOWN, skin.REASON_SCALE))
        self.assertEqual(r.quality["original_size"]["status"], skin.FLUSHING)
        self.assertEqual(r.quality["training_size"]["status"], skin.NORMAL)

    def test_round_a_pipeline_is_unchanged(self):
        r = skin.classify_patches(textured_skin(400, 400), face_landmarks(400, 400), FakeModel([0.9, 0.6, 0.3]))
        self.assertEqual(r.status, skin.FLUSHING)                                               # no size check


if __name__ == "__main__":
    unittest.main()
