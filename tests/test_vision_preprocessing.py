"""Skin check: same patches and preprocessing as CNN training (no model file needed)."""

import types
import unittest

import cv2
import numpy as np

from triage_kiosk.vision import skin


class FakeModel(skin.SkinModel):
    def __init__(self, probs, threshold=0.55, combine="mean"):
        self.calls = []
        it = iter(probs)

        def predict(batch):
            self.calls.append(batch.shape)
            return [next(it) for _ in range(len(batch))]

        super().__init__(predict, skin.SkinModelInfo(name="fake", threshold=threshold, combine=combine,
                                                     from_metadata=True))


def face_landmarks(w, h, yaw=0.0):
    """Minimal Face Mesh stand-in: only the landmark ids the skin check uses."""
    pts = [types.SimpleNamespace(x=0.5, y=0.5) for _ in range(468)]
    pts[skin.FOREHEAD_ID] = types.SimpleNamespace(x=0.5, y=0.25)
    pts[skin.CHEEK_L_ID] = types.SimpleNamespace(x=0.38, y=0.55)
    pts[skin.CHEEK_R_ID] = types.SimpleNamespace(x=0.62, y=0.55)
    pts[skin.NOSE_ID] = types.SimpleNamespace(x=0.5 + yaw, y=0.5)
    pts[skin.CHEEK_L_EDGE_ID] = types.SimpleNamespace(x=0.30, y=0.5)
    pts[skin.CHEEK_R_EDGE_ID] = types.SimpleNamespace(x=0.70, y=0.5)
    return pts


def skin_image(w=400, h=400, bgr=(120, 150, 205)):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = bgr
    return img


class PreprocessingTests(unittest.TestCase):
    def test_model_input_is_cielab_scaled(self):
        bgr = np.zeros((64, 64, 3), np.uint8)
        bgr[:] = (0, 0, 255)
        x = skin.to_model_input(bgr)
        expected = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        np.testing.assert_allclose(x, expected)
        self.assertEqual(x.shape, (64, 64, 3))
        self.assertEqual(x.dtype, np.float32)

    def test_non_64_input_is_resized_like_the_trainer(self):
        self.assertEqual(skin.to_model_input(np.zeros((24, 40, 3), np.uint8)).shape, (64, 64, 3))


class PatchPipelineTests(unittest.TestCase):
    def test_three_patches_mean_probability(self):
        model = FakeModel([0.9, 0.6, 0.3], threshold=0.55)
        r = skin.classify_patches(skin_image(), face_landmarks(400, 400), model)
        self.assertEqual(r.patches_used, 3)
        self.assertAlmostEqual(r.probability, 0.6)
        self.assertEqual(r.status, skin.FLUSHING)
        self.assertEqual(set(r.patch_probabilities), {"Forehead", "Zone_L", "Zone_R"})
        self.assertEqual(model.calls[0], (3, 64, 64, 3))

    def test_below_threshold_is_normal(self):
        r = skin.classify_patches(skin_image(), face_landmarks(400, 400), FakeModel([0.2, 0.3, 0.4]))
        self.assertEqual(r.status, skin.NORMAL)

    def test_turned_head_skips_one_cheek(self):
        r = skin.classify_patches(skin_image(), face_landmarks(400, 400, yaw=-0.17), FakeModel([0.1, 0.1]))
        self.assertEqual(r.patches_used, 2)

    def test_non_skin_image_gives_unknown(self):
        r = skin.classify_patches(skin_image(bgr=(200, 120, 40)), face_landmarks(400, 400), FakeModel([]))
        self.assertEqual(r.status, skin.UNKNOWN)
        self.assertIsNone(r.probability)

    def test_legacy_whole_face_uses_one_squeezed_image(self):
        model = FakeModel([0.7], threshold=0.65)
        r = skin.classify_whole_face_legacy(skin_image(300, 200), model)
        self.assertEqual((r.method, r.status, r.patches_used), ("legacy_whole_face", skin.FLUSHING, 1))

    def test_jpeg_roundtrip_keeps_shape(self):
        self.assertEqual(skin.jpeg_roundtrip(skin_image(64, 64)).shape, (64, 64, 3))


if __name__ == "__main__":
    unittest.main()
