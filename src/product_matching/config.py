"""Paths, seeds and model names shared by the whole pipeline."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("PM_ROOT", Path(__file__).resolve().parents[2]))
DATA_PATH = Path(os.environ.get("PM_DATA", ROOT / "data" / "raw" / "base_info_produtos.csv"))
LABELS_DIR = Path(os.environ.get("PM_LABELS", ROOT / "labels"))
ARTIFACTS_DIR = Path(os.environ.get("PM_ARTIFACTS", ROOT / "artifacts"))
REPORTS_DIR = Path(os.environ.get("PM_REPORTS", ROOT / "reports"))

SEED = 42

# Embedding models. e5 models expect "query: " / "passage: " prefixes.
EMBEDDING_MODEL = os.environ.get("PM_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
EMBEDDING_MODEL_LARGE = "intfloat/multilingual-e5-base"  # used only as an ablation in search eval

# Fields of a listing, with their original Portuguese column names.
FIELDS = ["tipo", "categoria", "marca", "cor", "modelo"]


def set_seeds(seed: int = SEED) -> None:
    """Fix every random number generator the pipeline touches."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover - torch is a hard dependency, but keep tests light
        pass
