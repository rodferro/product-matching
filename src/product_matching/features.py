"""Pairwise features for "are these two listings the same product?"."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .attributes import conflicts, extract_codes, extract_colors, extract_specs
from .candidates import char_vectorizer
from .normalize import normalize_text, tokenize

FEATURES = [
    "char_cos",        # character 3-5-gram TF-IDF cosine
    "word_cos",        # word TF-IDF cosine
    "emb_cos",         # sentence-embedding cosine
    "token_jaccard",   # overlap of token sets
    "number_jaccard",  # overlap of tokens that contain digits
    "shared_code",     # share a model-code token
    "shared_code_idf", # rarity of the rarest shared code (0 if none)
    "code_conflict",   # both have codes, none shared
    "same_brand",      # 1 same, 0 different, 0.5 unknown
    "spec_conflicts",  # number of conflicting specs (capacity, voltage, size, ...)
    "capacity_conflict",
    "voltage_conflict",
    "color_conflict",
    "length_ratio",    # shorter name / longer name, in tokens
]  # fmt: skip


@dataclass
class ListingInfo:
    """Everything the features need about one listing."""

    norm: str
    tokens: frozenset[str]
    numbers: frozenset[str]
    codes: frozenset[str]
    specs: dict[str, frozenset[str]]
    colors: frozenset[str]
    brand: str | None
    n_tokens: int


def brand_key(brand: str | None) -> str | None:
    if not brand:
        return None
    key = " ".join(tokenize(normalize_text(brand)))
    return key or None


def listing_info(name: str, brand: str | None = None, norm: str | None = None) -> ListingInfo:
    norm = norm if norm is not None else normalize_text(name)
    toks = tokenize(norm)
    return ListingInfo(
        norm=norm,
        tokens=frozenset(toks),
        numbers=frozenset(t for t in toks if any(c.isdigit() for c in t)),
        codes=frozenset(extract_codes(norm)),
        specs=extract_specs(norm),
        colors=frozenset(extract_colors(norm)),
        brand=brand_key(brand),
        n_tokens=max(len(toks), 1),
    )


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class PairFeaturizer:
    """Fitted on the catalogue: vectorizers, code rarity, brand dictionary."""

    char_vec: TfidfVectorizer = field(default_factory=char_vectorizer)
    word_vec: TfidfVectorizer = field(
        default_factory=lambda: TfidfVectorizer(
            tokenizer=str.split,
            token_pattern=None,
            lowercase=False,
            sublinear_tf=True,
            dtype=np.float32,
        )  # fmt: skip
    )
    code_idf: dict[str, float] = field(default_factory=dict)
    max_idf: float = 0.0
    brands: dict[str, str] = field(default_factory=dict)  # brand key -> display name

    def fit(self, catalogue: pd.DataFrame) -> PairFeaturizer:
        norms = catalogue["nome_norm"]
        self.char_vec.fit(norms)
        self.word_vec.fit(norms.map(lambda s: " ".join(tokenize(s))))
        df_counts = Counter(c for n in norms for c in set(extract_codes(n)))
        n_docs = len(norms)
        self.code_idf = {c: math.log(n_docs / k) for c, k in df_counts.items()}
        self.max_idf = math.log(n_docs)
        for b in catalogue["marca"].dropna().unique():
            key = brand_key(b)
            if key and len(key) >= 2:
                self.brands.setdefault(key, b)
        return self

    def detect_brand(self, norm: str) -> str | None:
        """Longest known brand that appears as a whole-token run in the name."""
        text = f" {' '.join(tokenize(norm))} "
        best = None
        for key in self.brands:
            if len(key) >= 3 and f" {key} " in text and (best is None or len(key) > len(best)):
                best = key
        return best

    def features(
        self,
        a: list[ListingInfo],
        b: list[ListingInfo],
        emb_a: np.ndarray | None = None,
        emb_b: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Feature table for aligned lists of listings a[i] vs b[i]."""
        ca = self.char_vec.transform([x.norm for x in a])
        cb = self.char_vec.transform([x.norm for x in b])
        wa = self.word_vec.transform([" ".join(sorted(x.tokens)) for x in a])
        wb = self.word_vec.transform([" ".join(sorted(x.tokens)) for x in b])
        char_cos = np.asarray(ca.multiply(cb).sum(axis=1)).ravel()
        word_cos = np.asarray(wa.multiply(wb).sum(axis=1)).ravel()
        if emb_a is not None and emb_b is not None:
            emb_cos = np.sum(emb_a * emb_b, axis=1)
        else:
            emb_cos = np.zeros(len(a))
        rows = []
        for x, y in zip(a, b, strict=True):
            shared = x.codes & y.codes
            spec_conf = {k: conflicts(x.specs[k], y.specs[k]) for k in x.specs}
            bx = x.brand or self.detect_brand(x.norm)
            by = y.brand or self.detect_brand(y.norm)
            rows.append(
                {
                    "token_jaccard": _jaccard(x.tokens, y.tokens),
                    "number_jaccard": _jaccard(x.numbers, y.numbers),
                    "shared_code": float(bool(shared)),
                    "shared_code_idf": max(
                        (self.code_idf.get(c, self.max_idf) for c in shared), default=0.0
                    )
                    / self.max_idf,
                    "code_conflict": float(bool(x.codes) and bool(y.codes) and not shared),
                    "same_brand": 0.5 if not (bx and by) else float(bx == by),
                    "spec_conflicts": float(sum(spec_conf.values())),
                    "capacity_conflict": float(spec_conf["capacity"]),
                    "voltage_conflict": float(spec_conf["voltage"]),
                    "color_conflict": float(conflicts(x.colors, y.colors)),
                    "length_ratio": min(x.n_tokens, y.n_tokens) / max(x.n_tokens, y.n_tokens),
                }
            )
        out = pd.DataFrame(rows)
        out["char_cos"] = char_cos
        out["word_cos"] = word_cos
        out["emb_cos"] = emb_cos
        return out[FEATURES]
