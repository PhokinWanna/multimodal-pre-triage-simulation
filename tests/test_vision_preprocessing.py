import unittest

import cv2
import numpy as np

from triage_kiosk.vision.engine import VisionEngine


class SkinModelPreprocessingTests(unittest.TestCase):
    def test_resizes_converts_bgr_to_lab_and_scales(self):
        bgr = np.zeros((24, 40, 3), dtype=np.uint8)
        bgr[:, :] = (0, 0, 255)
        tensor = VisionEngine._prepare_skin_tensor(
            bgr, np.asarray((1, 64, 64, 3), dtype=np.int32)
        )

        expected = cv2.cvtColor(cv2.resize(bgr, (64, 64)), cv2.COLOR_BGR2LAB)
        np.testing.assert_allclose(tensor, expected[np.newaxis].astype(np.float32) / 255.0)
        self.assertEqual(tensor.shape, (1, 64, 64, 3))
        self.assertEqual(tensor.dtype, np.float32)

    def test_rejects_unsupported_model_shape(self):
        with self.assertRaises(ValueError):
            VisionEngine._prepare_skin_tensor(
                np.zeros((64, 64, 3), dtype=np.uint8),
                np.asarray((1, 64, 64, 1), dtype=np.int32),
            )


if __name__ == "__main__":
    unittest.main()
