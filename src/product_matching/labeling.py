"""Draw the samples that get hand-labelled, and read the labelled CSVs back.

Label files live in labels/ as plain CSV so they can be reviewed and edited:

- pairs.csv             : label 1 = same product, 0 = different (see README for the rule)
- extraction_check.csv  : hand-corrected `cor` and `modelo` read from the name (empty = None)
- queries.csv           : search queries
- judgments.csv         : graded relevance 0/1/2 of (query, listing)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import LABELS_DIR, SEED

PAIR_COLUMNS = [
    "pair_id", "id_a", "id_b", "source", "nome_a", "nome_b", "marca_a", "marca_b",
    "label", "unsure", "note",
]  # fmt: skip


def sample_pairs(
    df: pd.DataFrame,
    candidates: pd.DataFrame,
    n_code: int = 350,
    n_neighbour: int = 300,
    n_random: int = 150,
    seed: int = SEED,
) -> pd.DataFrame:
    """Stratified sample of listing pairs to label.

    - code: share a model code and the same brand (most likely duplicates)
    - neighbour: near neighbours by text/embedding without a shared code (hard cases)
    - random: random pairs with the same `tipo` (mostly easy negatives)
    """
    rng = np.random.default_rng(seed)
    brand = df["marca"].to_numpy()
    cand = candidates.copy()
    cand["same_brand"] = brand[cand["a"]] == brand[cand["b"]]

    code_pool = cand[cand["by_code"] & cand["same_brand"]]
    nn_pool = cand[~cand["by_code"] & (cand["by_char"] | cand["by_emb"])]
    code = code_pool.iloc[rng.choice(len(code_pool), min(n_code, len(code_pool)), replace=False)]
    nn = nn_pool.iloc[rng.choice(len(nn_pool), min(n_neighbour, len(nn_pool)), replace=False)]

    taken = set(zip(code["a"], code["b"], strict=True)) | set(zip(nn["a"], nn["b"], strict=True))
    by_tipo = df[df["tipo"].notna()].groupby("tipo").indices
    tipos = [t for t, idx in by_tipo.items() if len(idx) >= 2]
    weights = np.array([len(by_tipo[t]) for t in tipos], dtype=float)
    weights /= weights.sum()
    random_pairs: list[tuple[int, int]] = []
    while len(random_pairs) < n_random:
        t = tipos[rng.choice(len(tipos), p=weights)]
        a, b = sorted(int(x) for x in rng.choice(by_tipo[t], 2, replace=False))
        if (a, b) not in taken:
            taken.add((a, b))
            random_pairs.append((a, b))

    rows = (
        [(a, b, "code") for a, b in zip(code["a"], code["b"], strict=True)]
        + [(a, b, "neighbour") for a, b in zip(nn["a"], nn["b"], strict=True)]
        + [(a, b, "random") for a, b in random_pairs]
    )
    order = rng.permutation(len(rows))
    rows = [rows[i] for i in order]
    ids = df["id"].to_numpy()
    out = pd.DataFrame(
        {
            "pair_id": range(1, len(rows) + 1),
            "id_a": [ids[a] for a, _, _ in rows],
            "id_b": [ids[b] for _, b, _ in rows],
            "source": [s for _, _, s in rows],
            "nome_a": [df["nome"].iat[a] for a, _, _ in rows],
            "nome_b": [df["nome"].iat[b] for _, b, _ in rows],
            "marca_a": [df["marca"].iat[a] for a, _, _ in rows],
            "marca_b": [df["marca"].iat[b] for _, b, _ in rows],
        }
    )
    out["label"] = ""
    out["unsure"] = ""
    out["note"] = ""
    return out[PAIR_COLUMNS]


def sample_extraction_check(
    df: pd.DataFrame, test_mask: np.ndarray, n: int = 200, seed: int = SEED
):
    """Random test-split listings whose color/model will be read off the name by hand."""
    pool = df[test_mask]
    sample = pool.sample(n=n, random_state=seed).sort_values("id")
    return pd.DataFrame(
        {
            "id": sample["id"],
            "nome": sample["nome"],
            "cor_dataset": sample["cor"],
            "modelo_dataset": sample["modelo"],
            "cor": "",
            "modelo": "",
            "unsure": "",
            "note": "",
        }
    )


def read_labels(name: str, labels_dir: Path = LABELS_DIR) -> pd.DataFrame:
    return pd.read_csv(labels_dir / name, dtype=str, keep_default_na=False)


def labelled_pairs(labels_dir: Path = LABELS_DIR) -> pd.DataFrame:
    pairs = read_labels("pairs.csv", labels_dir)
    pairs = pairs[pairs["label"].isin(["0", "1"])].copy()
    for col in ("pair_id", "id_a", "id_b", "label"):
        pairs[col] = pairs[col].astype(int)
    return pairs
