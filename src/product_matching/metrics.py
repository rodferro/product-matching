"""Evaluation metrics. Kept small and explicit so they can be unit-tested by hand."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score

from .config import SEED

# ---------------------------------------------------------------- classification


def precision_recall_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def pr_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the precision-recall curve (average precision)."""
    return float(average_precision_score(np.asarray(y_true), np.asarray(scores)))


def brier(y_true: Sequence[int], prob: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    prob = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    return float(np.mean((prob - y_true) ** 2))


def reliability(y_true: Sequence[int], prob: Sequence[float], n_bins: int = 10) -> dict:
    """Equal-width bins of predicted probability: mean prediction vs observed rate.

    Also returns the expected calibration error (ECE): the bin-size-weighted mean
    gap between predicted and observed rates.
    """
    y_true = np.asarray(y_true, dtype=float)
    prob = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1], right=False), 0, n_bins - 1)
    bins = []
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        mean_pred = float(prob[mask].mean())
        observed = float(y_true[mask].mean())
        ece += n / len(prob) * abs(mean_pred - observed)
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": n,
                     "mean_pred": mean_pred, "observed": observed})  # fmt: skip
    return {"bins": bins, "ece": float(ece)}


def best_f1_threshold(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Threshold on `scores` that maximizes F1 (ties -> highest threshold)."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(scores)[::-1]:
        f1 = precision_recall_f1(y_true, scores >= t)["f1"]
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


# ---------------------------------------------------------------- extraction


def abstaining_scores(gold: Sequence[str | None], pred: Sequence[str | None]) -> dict[str, float]:
    """For fields where the gold label always exists (tipo, categoria, marca).

    A prediction of None is an abstention.
    - coverage: share of items with an answer
    - accuracy_answered: accuracy among answered items (precision)
    - accuracy_all: correct answers / all items (abstentions count as wrong)
    """
    n = len(gold)
    answered = [p is not None for p in pred]
    correct = [p is not None and p == g for g, p in zip(gold, pred, strict=True)]
    n_answered = sum(answered)
    return {
        "n": n,
        "coverage": n_answered / n if n else 0.0,
        "accuracy_answered": sum(correct) / n_answered if n_answered else 0.0,
        "accuracy_all": sum(correct) / n if n else 0.0,
    }


def presence_scores(gold: Sequence[str | None], pred: Sequence[str | None]) -> dict[str, float]:
    """For fields where None is a valid answer (cor, modelo read from the name).

    - accuracy: exact match, None == None counts as correct
    - precision: correct non-None predictions / non-None predictions
    - recall: correct non-None predictions / non-None gold values
    - coverage: share of items with a non-None prediction
    """
    n = len(gold)
    exact = sum(g == p for g, p in zip(gold, pred, strict=True))
    pred_present = sum(p is not None for p in pred)
    gold_present = sum(g is not None for g in gold)
    hit = sum(p is not None and g == p for g, p in zip(gold, pred, strict=True))
    return {
        "n": n,
        "accuracy": exact / n if n else 0.0,
        "precision": hit / pred_present if pred_present else 0.0,
        "recall": hit / gold_present if gold_present else 0.0,
        "coverage": pred_present / n if n else 0.0,
    }


# ---------------------------------------------------------------- ranking


def dcg(gains: Sequence[float]) -> float:
    gains = np.asarray(gains, dtype=float)
    if gains.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, gains.size + 2))
    return float(np.sum((2**gains - 1) / discounts))


def ndcg_at_k(ranked: Sequence[int], relevance: Mapping[int, int], k: int = 10) -> float:
    """nDCG@k with graded relevance (unjudged items count as 0)."""
    gains = [relevance.get(doc, 0) for doc in list(ranked)[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(
    ranked: Sequence[int], relevance: Mapping[int, int], k: int = 10, min_grade: int = 1
) -> float:
    """Relevant items found in the top k / min(k, number of relevant items).

    Capping the denominator at k keeps broad queries (hundreds of relevant
    listings) on the same 0-1 scale as narrow ones.
    """
    relevant = {d for d, g in relevance.items() if g >= min_grade}
    if not relevant:
        return 0.0
    found = sum(1 for d in list(ranked)[:k] if d in relevant)
    return found / min(k, len(relevant))


# ---------------------------------------------------------------- uncertainty


def bootstrap_ci(
    stat: Callable[..., float],
    *arrays: Sequence,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI of `stat(*arrays)`, resampling items with replacement."""
    rng = np.random.default_rng(seed)
    arrays = tuple(np.asarray(a, dtype=object) for a in arrays)
    n = len(arrays[0])
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            values.append(stat(*(a[idx] for a in arrays)))
        except ValueError:  # e.g. a resample with a single class
            continue
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
