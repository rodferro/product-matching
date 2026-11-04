"""Grouped train/test splits.

Listings that are near-copies of each other (same model code, same normalized
name, or linked by a labelled pair) are put in the same group, and whole
groups go to either train or test. Otherwise a model could "memorize" a
listing in train and get credit for its twin in test.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .candidates import code_pairs
from .config import SEED


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def listing_groups(
    norm_names: pd.Series, extra_edges: Iterable[tuple[int, int]] = ()
) -> np.ndarray:
    """Group id per listing position."""
    uf = UnionFind(len(norm_names))
    for a, b in code_pairs(norm_names):
        uf.union(a, b)
    first_seen: dict[str, int] = {}
    for pos, name in enumerate(norm_names):
        if name in first_seen:
            uf.union(first_seen[name], pos)
        else:
            first_seen[name] = pos
    for a, b in extra_edges:
        uf.union(int(a), int(b))
    return np.array([uf.find(i) for i in range(len(norm_names))])


def is_test_group(
    group_ids: np.ndarray, test_fraction: float, salt: str = "", seed: int = SEED
) -> np.ndarray:
    """Deterministic group-level assignment: hash(group) < test_fraction."""

    def u(g: int) -> float:
        h = hashlib.sha256(f"{seed}:{salt}:{g}".encode()).hexdigest()
        return int(h[:12], 16) / 16**12

    cache: dict[int, bool] = {}
    out = np.empty(len(group_ids), dtype=bool)
    for i, g in enumerate(group_ids):
        g = int(g)
        if g not in cache:
            cache[g] = u(g) < test_fraction
        out[i] = cache[g]
    return out


EXTRACTION_TEST_FRACTION = 0.2
PAIR_TEST_FRACTION = 0.375  # ~300 of ~800 labelled pairs


def extraction_test_mask(norm_names: pd.Series) -> np.ndarray:
    """Listing-level test split for attribute extraction (~20% of listings, grouped)."""
    return is_test_group(listing_groups(norm_names), EXTRACTION_TEST_FRACTION, salt="extraction")


def pair_test_mask(norm_names: pd.Series, pos_a: np.ndarray, pos_b: np.ndarray) -> np.ndarray:
    """Test split for labelled pairs. Pairs that share a listing (or a group) stay together."""
    groups = listing_groups(norm_names, extra_edges=zip(pos_a, pos_b, strict=True))
    listing_is_test = is_test_group(groups, PAIR_TEST_FRACTION, salt="pairs")
    return listing_is_test[np.asarray(pos_a)]
