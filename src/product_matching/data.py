"""Load the product CSV into a clean DataFrame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_PATH, FIELDS
from .normalize import fix_encoding, normalize_text

COLUMNS = ["nome", *FIELDS]


def load_products(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Read the tab-separated product file.

    - Empty fields arrive as the string "None"; they become real missing values (None).
    - Names with broken encoding (e.g. "SÃ©rie") are repaired.
    - Adds `id` (row number in the original file) and `nome_norm` (normalized name).
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = df[COLUMNS].copy()
    df.insert(0, "id", range(len(df)))
    for col in COLUMNS:
        df[col] = df[col].map(lambda v: fix_encoding(v.strip()) if isinstance(v, str) else v)
        df[col] = df[col].astype(object).where(~df[col].isin(["None", ""]), None)
    df = df[df["nome"].notna()].reset_index(drop=True)
    df["nome_norm"] = df["nome"].map(normalize_text)
    return df
