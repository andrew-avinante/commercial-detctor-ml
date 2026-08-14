"""Evaluation metrics. Pure numpy -- no sklearn dependency.

Accuracy was the wrong metric for this problem. Only 7.4% of frames are inside
a fade, so a model that predicts "no fade" everywhere scores 92.6% and is
useless. Everything here is either threshold-free (average precision) or
reported alongside the threshold that produced it.

Event-level scoring is the one that actually reflects the product: what matters
is whether each real break is found once, at roughly the right time -- not the
per-frame agreement rate along its edges.
"""
from __future__ import annotations

import numpy as np


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve, computed exactly by ranking."""
    scores = np.asarray(scores, np.float64).ravel()
    labels = np.asarray(labels, np.float64).ravel()
    n_pos = labels.sum()
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / n_pos
    # Sum precision at each positive; equivalent to the step-wise PR integral.
    return float((precision * y).sum() / n_pos)


def best_f1(scores: np.ndarray, labels: np.ndarray,
            n_thresholds: int = 201) -> tuple[float, float, float, float]:
    """Sweep thresholds -> (best_f1, threshold, precision, recall)."""
    scores = np.asarray(scores, np.float64).ravel()
    labels = np.asarray(labels, np.float64).ravel()
    if labels.sum() == 0:
        return 0.0, 0.5, 0.0, 0.0

    best = (0.0, 0.5, 0.0, 0.0)
    for thr in np.linspace(0.02, 0.98, n_thresholds):
        pred = scores >= thr
        tp = float((pred & (labels > 0)).sum())
        fp = float((pred & (labels == 0)).sum())
        fn = float((~pred & (labels > 0)).sum())
        if tp == 0:
            continue
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
        if f1 > best[0]:
            best = (f1, float(thr), p, r)
    return best


def to_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Boolean mask -> list of half-open [start, end) runs of True."""
    mask = np.asarray(mask).astype(bool).ravel()
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with edge padding, for de-flickering scores."""
    if window <= 1:
        return np.asarray(x, np.float64).ravel()
    x = np.asarray(x, np.float64).ravel()
    pad = window // 2
    padded = np.pad(x, (pad, window - 1 - pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def hysteresis(scores: np.ndarray, high: float, low: float,
               min_frames: int = 0, smooth: int = 5) -> np.ndarray:
    """Two-threshold smoothing: seed a run above `high`, extend it down to `low`.

    A single per-frame `argmax` -- what `run.py` and `scan.py` do today --
    fragments one fade into several flickering events. Smoothing, then
    hysteresis, then a minimum duration turns the score trace into stable
    segments. The smoothing pass matters because without it a single low-scoring
    frame in the middle of a real fade splits it into two events, and both
    halves may then fall below `min_frames` and vanish entirely.
    """
    scores = moving_average(scores, smooth)
    above_low = scores >= low
    out = np.zeros(scores.size, bool)
    for a, b in to_segments(above_low):
        if scores[a:b].max() >= high:
            out[a:b] = True
    if min_frames > 0:
        for a, b in to_segments(out):
            if b - a < min_frames:
                out[a:b] = False
    return out


def event_scores(pred_mask: np.ndarray, true_mask: np.ndarray,
                 iou_threshold: float = 0.5) -> dict:
    """Match predicted segments to true segments greedily by IoU."""
    preds = to_segments(pred_mask)
    trues = to_segments(true_mask)
    if not preds and not trues:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                "tp": 0, "fp": 0, "fn": 0, "n_true": 0, "n_pred": 0}

    pairs = []
    for i, (pa, pb) in enumerate(preds):
        for j, (ta, tb) in enumerate(trues):
            inter = max(0, min(pb, tb) - max(pa, ta))
            if inter == 0:
                continue
            union = (pb - pa) + (tb - ta) - inter
            iou = inter / union
            if iou >= iou_threshold:
                pairs.append((iou, i, j))

    pairs.sort(reverse=True)
    return _greedy_match(pairs, preds, trues)


def _greedy_match(pairs, preds, trues) -> dict:
    used_p: set[int] = set()
    used_t: set[int] = set()
    for _, i, j in pairs:
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)

    tp = len(used_t)
    fp = len(preds) - len(used_p)
    fn = len(trues) - len(used_t)
    p = tp / max(tp + fp, 1e-12)
    r = tp / max(tp + fn, 1e-12)
    f1 = 2 * p * r / max(p + r, 1e-12)
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp,
            "fn": fn, "n_true": len(trues), "n_pred": len(preds)}


def evaluate(scores: np.ndarray, labels: np.ndarray, cfg=None,
             threshold: float | None = None) -> dict:
    """Full report for a (N, T) score matrix against (N, T) labels."""
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels, np.float64)

    ap = average_precision(scores, labels)
    f1, thr, prec, rec = best_f1(scores, labels)
    if threshold is not None:
        thr = threshold

    high = thr if cfg is None else max(thr, cfg.hyst_low)
    low = thr * 0.6 if cfg is None else cfg.hyst_low
    min_frames = 12 if cfg is None else cfg.min_event_frames
    smooth = 5 if cfg is None else cfg.smooth_frames

    agg = {"tp": 0, "fp": 0, "fn": 0, "n_true": 0, "n_pred": 0}
    for s, y in zip(scores, labels):
        ev = event_scores(hysteresis(s, high, low, min_frames, smooth), y > 0.5)
        for k in agg:
            agg[k] += ev[k]

    p = agg["tp"] / max(agg["tp"] + agg["fp"], 1e-12)
    r = agg["tp"] / max(agg["tp"] + agg["fn"], 1e-12)
    return {
        "ap": ap,
        "frame_f1": f1, "frame_precision": prec, "frame_recall": rec,
        "threshold": thr,
        "event_precision": p, "event_recall": r,
        "event_f1": 2 * p * r / max(p + r, 1e-12),
        **{f"event_{k}": v for k, v in agg.items()},
    }
