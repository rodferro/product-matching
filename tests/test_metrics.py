import math

import pytest

from product_matching.metrics import (
    abstaining_scores,
    best_f1_threshold,
    bootstrap_ci,
    brier,
    ndcg_at_k,
    pr_auc,
    precision_recall_f1,
    presence_scores,
    recall_at_k,
    reliability,
)


def test_precision_recall_f1_by_hand():
    # tp=2, fp=1, fn=1 -> p=2/3, r=2/3, f1=2/3
    m = precision_recall_f1([1, 1, 1, 0, 0], [1, 1, 0, 1, 0])
    assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 1
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_precision_recall_f1_no_predictions():
    m = precision_recall_f1([1, 0], [0, 0])
    assert m == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 1}


def test_pr_auc_perfect_and_inverted():
    assert pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    # positives ranked last: AP = mean(1/3, 2/4) = 5/12
    assert pr_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(5 / 12)


def test_brier():
    assert brier([1, 0], [1.0, 0.0]) == 0.0
    assert brier([1, 0], [0.5, 0.5]) == pytest.approx(0.25)


def test_reliability_perfectly_calibrated_bins():
    # bin [0.2,0.3): 5 items predicted 0.2, 1 positive -> observed 0.2
    y = [1, 0, 0, 0, 0] + [1, 1, 1, 1, 0]
    p = [0.2] * 5 + [0.8] * 5
    r = reliability(y, p, n_bins=10)
    assert r["ece"] == pytest.approx(0.0)
    assert [b["n"] for b in r["bins"]] == [5, 5]


def test_reliability_ece_by_hand():
    # all predicted 0.9, half positive -> gap 0.4
    r = reliability([1, 0, 1, 0], [0.9] * 4)
    assert r["ece"] == pytest.approx(0.4)


def test_best_f1_threshold():
    assert best_f1_threshold([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.35)


def test_abstaining_scores():
    m = abstaining_scores(["a", "b", "c", "d"], ["a", "x", None, None])
    assert m["coverage"] == 0.5
    assert m["accuracy_answered"] == 0.5
    assert m["accuracy_all"] == 0.25


def test_presence_scores():
    gold = ["Preto", None, "Azul", None]
    pred = ["Preto", "Branco", None, None]
    m = presence_scores(gold, pred)
    assert m["accuracy"] == 0.5  # Preto==Preto and None==None
    assert m["precision"] == 0.5  # 1 of 2 non-None predictions right
    assert m["recall"] == 0.5  # 1 of 2 gold values found
    assert m["coverage"] == 0.5


def test_ndcg_by_hand():
    rel = {1: 2, 2: 1}
    assert ndcg_at_k([1, 2, 3], rel, k=3) == pytest.approx(1.0)
    # swapped order: DCG = 1/log2(2) + 3/log2(3); IDCG = 3 + 1/log2(3)
    expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
    assert ndcg_at_k([2, 1, 3], rel, k=3) == pytest.approx(expected)
    assert ndcg_at_k([3, 4], rel, k=2) == 0.0
    assert ndcg_at_k([1], {}, k=10) == 0.0


def test_recall_at_k_caps_denominator():
    rel = {i: 1 for i in range(100)}
    assert recall_at_k(list(range(10)), rel, k=10) == 1.0
    assert recall_at_k([0, 200, 1], {0: 2, 1: 1, 2: 1}, k=3) == pytest.approx(2 / 3)
    assert recall_at_k([0, 1], {0: 1, 1: 0}, k=2) == 1.0  # grade 0 is not relevant


def test_bootstrap_ci_contains_point_estimate():
    y = [1, 0] * 50
    lo, hi = bootstrap_ci(lambda a: float(sum(a)) / len(a), y, n_boot=200)
    assert lo <= 0.5 <= hi
