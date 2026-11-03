"""M3 metric functions: hand-checked values, plus scikit-learn cross-checks when available."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "m3"))
import m3_metrics as M  # noqa: E402


class MetricTests(unittest.TestCase):
    def test_perfect_agreement(self):
        ref = [1, 2, 3, 4, 5, 3]
        s = M.summary(ref, ref)
        self.assertEqual(s["accuracy"], 1.0)
        self.assertAlmostEqual(s["kappa_linear"], 1.0)
        self.assertAlmostEqual(s["kappa_quadratic"], 1.0)
        self.assertEqual(s["under_triage"], 0.0)
        self.assertEqual(s["auc_high_acuity"], 1.0)

    def test_under_and_over_triage_direction(self):
        ref = [2, 2, 4, 4]
        pred = [3, 2, 3, 4]          # one less urgent (under), one more urgent (over)
        s = M.summary(ref, pred)
        self.assertEqual(s["under_triage"], 0.25)
        self.assertEqual(s["over_triage"], 0.25)
        self.assertEqual(s["high_acuity_sensitivity"], 0.5)   # one of two ESI-2 patients missed
        self.assertEqual(s["high_acuity_specificity"], 1.0)

    def test_invalid_prediction_counts_as_wrong(self):
        s = M.summary([1, 2, 3], [1, None, 3])
        self.assertEqual(s["invalid_predictions"], 1)
        self.assertAlmostEqual(s["accuracy"], 2 / 3)

    def test_confusion_orientation(self):
        m = M.confusion([2], [3])
        self.assertEqual(m[1, 2], 1)            # row = reference 2, column = predicted 3

    def test_auc_with_ties(self):
        self.assertEqual(M.auc_mann_whitney([1, 0], [1.0, 1.0]), 0.5)
        self.assertEqual(M.auc_mann_whitney([1, 1, 0, 0], [0.9, 0.8, 0.3, 0.1]), 1.0)

    def test_majority_vote_ties_go_to_more_urgent(self):
        self.assertEqual(M.majority_vote([3, 2, 3, 2]), 2)
        self.assertEqual(M.majority_vote([4, 4, 3]), 4)
        self.assertIsNone(M.majority_vote([None, None]))

    def test_bootstrap_ci_contains_point_estimate(self):
        ref = np.array([1, 2, 3, 4, 5] * 6)
        pred = np.array([1, 2, 3, 4, 4] * 6)
        lo, hi = M.bootstrap_ci(len(ref), lambda idx: float((ref[idx] == pred[idx]).mean()), reps=500)
        self.assertLessEqual(lo, 0.8)
        self.assertGreaterEqual(hi, 0.8)

    def test_roc_points_span_unit_square(self):
        fpr, tpr = M.roc_points([1, 0, 1, 0], [5, 4, 3, 1])
        self.assertEqual((fpr[0], tpr[0], fpr[-1], tpr[-1]), (0.0, 0.0, 1.0, 1.0))


class SklearnCrossCheck(unittest.TestCase):
    def setUp(self):
        try:
            import sklearn.metrics  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn not installed")
        rng = np.random.default_rng(0)
        self.ref = rng.integers(1, 6, 200)
        self.pred = np.clip(self.ref + rng.integers(-2, 3, 200), 1, 5)

    def test_weighted_kappa_matches_sklearn(self):
        from sklearn.metrics import cohen_kappa_score
        for w in ("linear", "quadratic"):
            self.assertAlmostEqual(M.weighted_kappa(self.ref, self.pred, w),
                                   cohen_kappa_score(self.ref, self.pred, weights=w, labels=[1, 2, 3, 4, 5]), places=10)

    def test_auc_matches_sklearn(self):
        from sklearn.metrics import roc_auc_score
        y = (self.ref <= 2).astype(int)
        score = 6 - self.pred
        self.assertAlmostEqual(M.auc_mann_whitney(y, score), roc_auc_score(y, score), places=10)


if __name__ == "__main__":
    unittest.main()
