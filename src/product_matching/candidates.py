"""Candidate generation (blocking).

Comparing every pair of 26k listings means ~344M pairs. Instead we only score
pairs that some cheap signal says might match:

1. they share a model-code token (e.g. "un40c6900"),
2. one is among the other's top-k neighbours by character n-gram TF-IDF,
3. one is among the other's top-k neighbours by sentence embedding.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .attributes import extract_codes

MAX_CODE_GROUP = 20  # codes shared by more listings than this are too generic (e.g. CPU models)


def char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, dtype=np.float32
    )


def code_pairs(norm_names: pd.Series, max_group: int = MAX_CODE_GROUP) -> set[tuple[int, int]]:
    """Index pairs (positions) that share at least one model code."""
    groups: dict[str, list[int]] = defaultdict(list)
    for pos, name in enumerate(norm_names):
        for code in extract_codes(name):
            groups[code].append(pos)
    pairs: set[tuple[int, int]] = set()
    for members in groups.values():
        if 2 <= len(members) <= max_group:
            pairs.update(combinations(sorted(set(members)), 2))
    return pairs


def top_k_neighbours(matrix, k: int, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Top-k cosine neighbours (excluding self) for L2-normalized rows (dense or sparse)."""
    n = matrix.shape[0]
    idx = np.zeros((n, k), dtype=np.int64)
    sim = np.zeros((n, k), dtype=np.float32)
    is_sparse = sparse.issparse(matrix)
    for start in range(0, n, chunk):
        block = matrix[start : start + chunk]
        s = block @ matrix.T
        s = s.toarray() if is_sparse else np.asarray(s)
        rows = np.arange(s.shape[0])
        s[rows, rows + start] = -np.inf
        part = np.argpartition(-s, k, axis=1)[:, :k]
        part_sim = np.take_along_axis(s, part, axis=1)
        order = np.argsort(-part_sim, axis=1)
        idx[start : start + chunk] = np.take_along_axis(part, order, axis=1)
        sim[start : start + chunk] = np.take_along_axis(part_sim, order, axis=1)
    return idx, sim


def _pairs_from_neighbours(idx: np.ndarray) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for i, row in enumerate(idx):
        for j in row:
            j = int(j)
            out.add((i, j) if i < j else (j, i))
    return out


def generate_candidates(
    norm_names: pd.Series,
    embeddings: np.ndarray | None,
    k: int = 10,
    char_matrix=None,
) -> pd.DataFrame:
    """Union of the three blocking rules. Returns positions `a < b` and which rules fired."""
    if char_matrix is None:
        char_matrix = char_vectorizer().fit_transform(norm_names)
    by_code = code_pairs(norm_names)
    by_char = _pairs_from_neighbours(top_k_neighbours(char_matrix, k)[0])
    by_emb = (
        _pairs_from_neighbours(top_k_neighbours(embeddings, k)[0])
        if embeddings is not None
        else set()
    )
    all_pairs = sorted(by_code | by_char | by_emb)
    frame = pd.DataFrame(all_pairs, columns=["a", "b"])
    keys = list(zip(frame["a"], frame["b"], strict=True))
    frame["by_code"] = [p in by_code for p in keys]
    frame["by_char"] = [p in by_char for p in keys]
    frame["by_emb"] = [p in by_emb for p in keys]
    return frame
