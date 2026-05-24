from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix for boxes in xyxy. a: (N,4), b: (M,4)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    import torch
    from torchvision.ops import box_iou

    ta = torch.from_numpy(a).float()
    tb = torch.from_numpy(b).float()
    return box_iou(ta, tb).numpy()


@dataclass
class MatchStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou_sum: float = 0.0
    iou_count: int = 0


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return float(p), float(r), float(f1)


def greedy_match(
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    *,
    iou_thres: float,
    conf_thres: float,
) -> Tuple[MatchStats, List[Tuple[int, int, float]], List[int], List[int]]:
    """Greedy one-to-one matching.

    Returns:
    - stats (tp/fp/fn and matched IoU sum)
    - matches: list of (gt_idx, pred_idx, iou)
    - unmatched_gt indices
    - unmatched_pred indices (after conf filtering)

    Matching rule:
    - Consider only preds with score >= conf_thres.
    - Find highest IoU pair iteratively, match if IoU>=iou_thres.
    - Class label does NOT constrain matching for IoU computation (needed for confusion).
      For TP/FP/FN and IoU mean, we count TP only if labels also match.
    """

    stats = MatchStats()

    keep = pred_scores >= conf_thres
    pred_boxes_k = pred_boxes[keep]
    pred_labels_k = pred_labels[keep]
    pred_scores_k = pred_scores[keep]

    if pred_boxes_k.size == 0 and gt_boxes.size == 0:
        return stats, [], [], []

    if pred_boxes_k.size == 0:
        stats.fn = int(gt_boxes.shape[0])
        return stats, [], list(range(int(gt_boxes.shape[0]))), []

    if gt_boxes.size == 0:
        stats.fp = int(pred_boxes_k.shape[0])
        return stats, [], [], list(range(int(pred_boxes_k.shape[0])))

    ious = box_iou_xyxy(pred_boxes_k.astype(np.float32), gt_boxes.astype(np.float32)).T  # (G,P)
    used_g = set()
    used_p = set()
    matches: List[Tuple[int, int, float]] = []

    while True:
        # pick best remaining pair
        g_idx, p_idx = np.unravel_index(np.argmax(ious), ious.shape)
        best = float(ious[g_idx, p_idx])
        if best < iou_thres:
            break
        if g_idx in used_g or p_idx in used_p:
            ious[g_idx, p_idx] = -1.0
            continue

        used_g.add(int(g_idx))
        used_p.add(int(p_idx))
        matches.append((int(g_idx), int(p_idx), best))
        ious[g_idx, :] = -1.0
        ious[:, p_idx] = -1.0

    unmatched_gt = [i for i in range(int(gt_boxes.shape[0])) if i not in used_g]
    unmatched_pred = [i for i in range(int(pred_boxes_k.shape[0])) if i not in used_p]

    # TP if class matches
    for gi, pi, iou in matches:
        if int(gt_labels[gi]) == int(pred_labels_k[pi]):
            stats.tp += 1
            stats.iou_sum += float(iou)
            stats.iou_count += 1
        else:
            stats.fp += 1
            stats.fn += 1

    stats.fp += len(unmatched_pred)
    stats.fn += len(unmatched_gt)

    return stats, matches, unmatched_gt, unmatched_pred


def update_confusion_matrix(
    cm: np.ndarray,
    *,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    gt_labels: np.ndarray,
    matches: List[Tuple[int, int, float]],
    unmatched_gt: List[int],
    unmatched_pred: List[int],
    conf_thres: float,
    background_index: int,
) -> None:
    """Update cm in-place.

    cm shape is (nc+1, nc+1): rows=gt, cols=pred, last index is background.
    """
    # preds were already filtered in greedy_match, but unmatched_pred indices are in that filtered space.
    keep = pred_scores >= conf_thres
    pred_labels_k = pred_labels[keep]

    # matched
    for gi, pi, _ in matches:
        gt = int(gt_labels[gi])
        pr = int(pred_labels_k[pi])
        cm[gt, pr] += 1

    # fn: gt matched to background
    for gi in unmatched_gt:
        gt = int(gt_labels[gi])
        cm[gt, background_index] += 1

    # fp: pred matched to background
    for pi in unmatched_pred:
        pr = int(pred_labels_k[pi])
        cm[background_index, pr] += 1


def mean_iou(stats: MatchStats) -> float:
    if stats.iou_count <= 0:
        return 0.0
    return float(stats.iou_sum / stats.iou_count)


def merge_stats(stats_list: List[MatchStats]) -> MatchStats:
    out = MatchStats()
    for s in stats_list:
        out.tp += int(s.tp)
        out.fp += int(s.fp)
        out.fn += int(s.fn)
        out.iou_sum += float(s.iou_sum)
        out.iou_count += int(s.iou_count)
    return out


def per_class_stats_from_matches(
    *,
    nc: int,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    matches: List[Tuple[int, int, float]],
    unmatched_gt: List[int],
    unmatched_pred: List[int],
    conf_thres: float,
) -> Dict[int, MatchStats]:
    """Compute per-class TP/FP/FN and IoU mean using the same greedy matching output."""

    keep = pred_scores >= conf_thres
    pred_labels_k = pred_labels[keep]

    out: Dict[int, MatchStats] = {i: MatchStats() for i in range(nc)}

    matched_gt = set()
    matched_pred = set()

    for gi, pi, iou in matches:
        matched_gt.add(gi)
        matched_pred.add(pi)
        gt = int(gt_labels[gi])
        pr = int(pred_labels_k[pi])
        if gt == pr:
            out[gt].tp += 1
            out[gt].iou_sum += float(iou)
            out[gt].iou_count += 1
        else:
            out[gt].fn += 1
            out[pr].fp += 1

    for gi in unmatched_gt:
        gt = int(gt_labels[gi])
        if 0 <= gt < nc:
            out[gt].fn += 1

    for pi in unmatched_pred:
        pr = int(pred_labels_k[pi])
        if 0 <= pr < nc:
            out[pr].fp += 1

    return out
