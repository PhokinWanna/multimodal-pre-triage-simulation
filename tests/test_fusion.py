import unittest

from triage_kiosk.triage import fusion as F

PRESETS = ((0.5, 0.5), (0.6, 0.4), (0.4, 0.6))


class VisualScoreTests(unittest.TestCase):
    def test_parts(self):
        self.assertEqual(F.posture_score({"chest_clutching": True, "head_clutching": True}), 1.0)
        self.assertEqual(F.posture_score({"abdominal_clutching": True}), 0.7)
        self.assertEqual(F.posture_score({}), 0.0)
        self.assertIsNone(F.posture_score(None))
        self.assertIsNone(F.skin_score("UNKNOWN", 0.9))
        self.assertEqual(F.skin_score("NORMAL", 0.2), 0.2)
        self.assertEqual(F.skin_score("RED_FLUSHING", None), 1.0)

    def test_three_weightings_differ(self):
        chest_only = [F.visual_score({"chest_clutching": True}, "NORMAL", 0.1, wp, ws)["score"] for wp, ws in PRESETS]
        self.assertEqual(chest_only, [0.55, 0.64, 0.46])
        flush_only = [F.visual_score({}, "RED_FLUSHING", 0.9, wp, ws) for wp, ws in PRESETS]
        self.assertEqual([v["band"] for v in flush_only], ["MODERATE", "MODERATE", "HIGH"])

    def test_part_not_assessed_adds_no_risk_and_keeps_the_weights(self):
        # skin unknown must never rank a patient higher than skin checked and normal
        unknown = F.visual_score({"abdominal_clutching": True}, "UNKNOWN", None, 0.5, 0.5)
        normal = F.visual_score({"abdominal_clutching": True}, "NORMAL", 0.0, 0.5, 0.5)
        self.assertEqual((unknown["score"], unknown["band"]), (normal["score"], "MODERATE"))
        self.assertEqual(unknown["weights_used"], {"posture": 0.5, "skin": 0.0})
        chest = [F.visual_score({"chest_clutching": True}, "UNKNOWN", None, wp, ws)["band"] for wp, ws in PRESETS]
        self.assertEqual(chest, ["HIGH", "HIGH", "MODERATE"])
        self.assertEqual(F.visual_score(None, "RED_FLUSHING", 0.8, 0.6, 0.4)["score"], 0.32)

    def test_nothing_assessed(self):
        v = F.visual_score(None, "NOT_ASSESSED", None)
        self.assertEqual((v["score"], v["band"]), (None, "NOT_ASSESSED"))
        self.assertEqual(F.visual_score({}, "UNKNOWN", None)["band"], "LOW")

    def test_renormalise_option_moves_the_weight(self):
        v = F.visual_score({"head_clutching": True}, "UNKNOWN", None, 0.4, 0.6, missing="renormalise")
        self.assertEqual((v["score"], v["weights_used"]), (0.6, {"posture": 1.0, "skin": 0.0}))

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            F.visual_score({}, "NORMAL", 0.1, 0, 0)
        with self.assertRaises(ValueError):
            F.visual_score({}, "NORMAL", 0.1, missing="guess")
        with self.assertRaises(ValueError):
            F.esi_cap("HIGH", "sometimes")

    def test_bands_and_caps(self):
        self.assertEqual([F.band_of(x) for x in (0.0, 0.24, 0.25, 0.49, 0.5, 1.0)],
                         ["LOW", "LOW", "MODERATE", "MODERATE", "HIGH", "HIGH"])
        bands = ("HIGH", "MODERATE", "LOW", "NOT_ASSESSED")
        self.assertEqual([F.esi_cap(b) for b in bands], [2, None, None, None])
        self.assertEqual([F.esi_cap(b, "high_and_moderate") for b in bands], [2, 3, None, None])
        self.assertEqual([F.esi_cap(b, "none") for b in bands], [None] * 4)

    def test_descriptions(self):
        self.assertEqual(F.describe_posture({"chest_clutching": True}), "hand held on the chest")
        self.assertEqual(F.describe_posture({}), "no pain posture")
        self.assertEqual(F.describe_skin("UNKNOWN", None, "face too small in the image"),
                         "not assessed (face too small in the image)")
        self.assertEqual(F.describe_skin("RED_FLUSHING", 0.9), "flushed (red), flushing probability 0.90")
