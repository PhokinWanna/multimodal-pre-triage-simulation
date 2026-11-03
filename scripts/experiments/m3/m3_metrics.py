"""Metrics for the M3 experiment (numpy only, so they can be unit-tested anywhere).

ESI: 1 = most urgent ... 5 = least urgent.
  under-triage = prediction LESS urgent than the reference (pred > ref)  <- the safety risk
  over-triage  = prediction MORE urgent than the reference (pred < ref)
High acuity = ESI 1-2 (positive class for sensitivity/specificity and AUC).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

LEVELS = (1, 2, 3, 4, 5)


def confusion(ref: Sequence[int], pred: Sequence[Optional[int]]) -> np.ndarray:
    """5x5 counts, rows = reference level, columns = predicted level (invalid predictions excluded)."""
    m = np.zeros((5, 5), dtype=int)
    for r, p in zip(ref, pred):
        if p is not None:
            m[int(r) - 1, int(p) - 1] += 1
    return m


def weighted_kappa(ref: Sequence[int], pred: Sequence[int], weights: str = "linear") -> float:
    """Cohen's weighted kappa over the 5 ESI levels (linear or quadratic disagreement weights)."""
    o = confusion(ref, pred).astype(float)
    n = o.sum()
    if n == 0:
        return float("nan")
    i, j = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
    w = np.abs(i - j) / 4.0 if weights == "linear" else ((i - j) / 4.0) ** 2
    e = np.outer(o.sum(axis=1), o.sum(axis=0)) / n
    denom = (w * e).sum()
    return float(1.0 - (w * o).sum() / denom) if denom > 0 else float("nan")


def auc_mann_whitney(labels: Sequence[int], scores: Sequence[float]) -> float:
    """ROC AUC = P(score_pos > score_neg) + 0.5 P(tie). Works with step (ordinal) scores."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * ties) / (len(pos) * len(neg)))


def roc_points(labels: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """(FPR, TPR) for every distinct threshold, from (0,0) to (1,1)."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    p, n = max(1, (labels == 1).sum()), max(1, (labels == 0).sum())
    fpr, tpr = [0.0], [0.0]
    for t in sorted(set(scores), reverse=True):
        pred = scores >= t
        tpr.append(float((pred & (labels == 1)).sum() / p))
        fpr.append(float((pred & (labels == 0)).sum() / n))
    if fpr[-1] != 1.0 or tpr[-1] != 1.0:
        fpr.append(1.0), tpr.append(1.0)
    return np.array(fpr), np.array(tpr)


def summary(ref: Sequence[int], pred: Sequence[Optional[int]], score: Optional[Sequence[float]] = None) -> dict:
    """All headline numbers. Invalid predictions (None) count as wrong and are reported separately."""
    ref = np.asarray(ref, dtype=int)
    valid = np.array([p is not None for p in pred])
    p = np.array([int(x) if x is not None else 0 for x in pred])
    n = len(ref)
    correct = valid & (p == ref)
    hi_ref = ref <= 2
    hi_pred = valid & (p <= 2)
    out = {
        "n": n,
        "invalid_predictions": int((~valid).sum()),
        "accuracy": float(correct.mean()) if n else float("nan"),
        "within_one": float((valid & (np.abs(p - ref) <= 1)).mean()) if n else float("nan"),
        "kappa_linear": weighted_kappa(ref[valid], p[valid], "linear"),
        "kappa_quadratic": weighted_kappa(ref[valid], p[valid], "quadratic"),
        "under_triage": float((valid & (p > ref)).mean()) if n else float("nan"),
        "over_triage": float((valid & (p < ref)).mean()) if n else float("nan"),
        "high_acuity_sensitivity": float((hi_pred & hi_ref).sum() / hi_ref.sum()) if hi_ref.sum() else float("nan"),
        "high_acuity_specificity": float((~hi_pred & ~hi_ref).sum() / (~hi_ref).sum()) if (~hi_ref).sum() else float("nan"),
        "recall_per_level": {lv: (float((correct & (ref == lv)).sum() / (ref == lv).sum()) if (ref == lv).sum() else float("nan"))
                             for lv in LEVELS},
    }
    if score is None:  # option A: the predicted level itself is the score (1 = most urgent -> highest score)
        score = [6 - x if x is not None else 0 for x in pred]
    out["auc_high_acuity"] = auc_mann_whitney(hi_ref.astype(int), score)
    return out


def bootstrap_ci(n: int, stat: Callable[[np.ndarray], float], reps: int = 2000, seed: int = 42,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap over scenarios. ``stat`` receives an index array."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        v = stat(rng.integers(0, n, n))
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def majority_vote(preds: Sequence[Optional[int]]) -> Optional[int]:
    """Most frequent valid level; ties go to the MORE urgent level (safer, decided in advance)."""
    vals = [p for p in preds if p is not None]
    if not vals:
        return None
    counts = {lv: vals.count(lv) for lv in set(vals)}
    best = max(counts.values())
    return min(lv for lv, c in counts.items() if c == best)
