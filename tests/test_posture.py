"""Posture rules on a hand-built body (no camera or MediaPipe needed).

Body: shoulders y=150 (x 270/370), hips y=290, nose (320, 90), eyes y=78, mouth y=106.
Torso length = 140 px. Each test moves the hands to a clinically meaningful place.
"""

import unittest

from triage_kiosk.vision import posture as P

H, W = 480, 640


def body(left_hand=None, right_hand=None, left_elbow=None, right_elbow=None, hips_visible=True, **vis):
    """Return MediaPipe-style landmarks {idx: (x, y, visibility)}; hands default to hanging at the sides."""
    lm = {
        P.NOSE: (320, 90, 1), P.L_EYE: (335, 78, 1), P.R_EYE: (305, 78, 1),
        P.MOUTH_L: (330, 106, 1), P.MOUTH_R: (310, 106, 1),
        P.L_SHOULDER: (370, 150, 1), P.R_SHOULDER: (270, 150, 1),
        P.L_HIP: (350, 290, 1 if hips_visible else 0.1), P.R_HIP: (290, 290, 1 if hips_visible else 0.1),
    }
    lh = left_hand or (395, 300)
    rh = right_hand or (245, 300)
    lm[P.L_ELBOW] = (*(left_elbow or (392, 225)), 1)
    lm[P.R_ELBOW] = (*(right_elbow or (248, 225)), 1)
    for wrist, index, pinky, (x, y) in ((P.L_WRIST, P.L_INDEX, P.L_PINKY, lh), (P.R_WRIST, P.R_INDEX, P.R_PINKY, rh)):
        lm[wrist] = (x, y, vis.get(f"wrist{wrist}", 1))
        lm[index] = (x, y - 10, 1)
        lm[pinky] = (x + 4, y - 8, 1)
    return lm


def v2(lm):
    return P.classify_posture_v2(lm, H, W)


def v1(lm):
    return P.classify_posture_v1(lm, H, W)


class V2RuleTests(unittest.TestCase):
    def test_arms_down_is_no_posture(self):
        r = v2(body())
        self.assertTrue(r.person_found)
        self.assertEqual(r.label(), "none")

    def test_hand_on_mid_chest(self):
        r = v2(body(right_hand=(318, 200), right_elbow=(265, 235)))
        self.assertTrue(r.chest)
        self.assertEqual(r.label(), "chest")

    def test_hand_on_upper_stomach(self):
        r = v2(body(right_hand=(318, 255), right_elbow=(262, 250)))
        self.assertTrue(r.abdomen)
        self.assertFalse(r.chest)

    def test_hands_resting_on_lap_are_not_abdominal_pain(self):
        lm = body(left_hand=(338, 300), right_hand=(302, 300), left_elbow=(380, 250), right_elbow=(260, 250))
        self.assertFalse(v2(lm).abdomen)
        self.assertTrue(v1(lm).abdomen, "original rules flag resting hands (the bug that v2 fixes)")

    def test_hand_on_forehead_is_headache_even_with_vertical_forearm(self):
        lm = body(right_hand=(318, 66), right_elbow=(300, 160))   # forearm almost vertical
        r = v2(lm)
        self.assertTrue(r.head)
        self.assertNotIn("THINKING_POSE_FILTER", r.filters)
        self.assertFalse(v1(lm).head, "original thinking filter removes this headache pose (the bug)")

    def test_hand_on_temple(self):
        self.assertTrue(v2(body(left_hand=(352, 76), left_elbow=(395, 150))).head)

    def test_hand_at_chin_is_thinking_not_headache(self):
        r = v2(body(right_hand=(320, 125), right_elbow=(305, 215)))
        self.assertFalse(r.head)
        self.assertFalse(r.chest)
        self.assertIn("THINKING_POSE_FILTER", r.filters)

    def test_crossed_arms_are_not_chest_pain(self):
        lm = body(left_hand=(262, 220), right_hand=(378, 220), left_elbow=(385, 222), right_elbow=(255, 222))
        r = v2(lm)
        self.assertIn("CROSSED_ARMS_FILTER", r.filters)
        self.assertFalse(r.chest or r.abdomen)

    def test_invisible_hand_is_ignored(self):
        lm = body(right_hand=(318, 200), right_elbow=(265, 235), **{f"wrist{P.R_WRIST}": 0.1})
        self.assertFalse(v2(lm).chest)

    def test_hidden_hips_use_body_proportions(self):
        r = v2(body(right_hand=(318, 200), right_elbow=(265, 235), hips_visible=False))
        self.assertFalse(r.debug["hips_reliable"])
        self.assertTrue(r.chest)

    def test_no_shoulders_means_no_person(self):
        lm = body()
        lm[P.L_SHOULDER] = (370, 150, 0.1)
        self.assertFalse(v2(lm).person_found)

    def test_dispatch_by_version(self):
        lm = body(right_hand=(318, 66), right_elbow=(300, 160))
        self.assertTrue(P.classify_posture(lm, H, W, version="v2").head)
        self.assertFalse(P.classify_posture(lm, H, W, version="v1").head)


if __name__ == "__main__":
    unittest.main()
