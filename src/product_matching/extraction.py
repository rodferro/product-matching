"""(b) Recover type, category, brand, color and model from a product name.

Every field returns a value and a confidence, or None when it is not sure:

- tipo, categoria: linear classifier (logistic loss) on word + character
  TF-IDF of the normalized name.
- marca: a brand dictionary (brands that appear in names reliably), falling
  back to the classifier.
- cor: color words found in the name (rule-based).
- modelo: candidate code tokens from the name, scored by a small gradient
  boosting model; the best candidate is returned if it scores high enough.

Abstention thresholds are chosen on a validation split carved out of the
training data, never on the test data: the lowest confidence at which
validation precision stays at or above `TARGET_PRECISION`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from .attributes import extract_colors, is_spec
from .config import SEED
from .features import brand_key
from .normalize import normalize_text, tokenize
from .splits import is_test_group, listing_groups

TARGET_PRECISION = {"tipo": 0.95, "categoria": 0.95, "marca": 0.95, "modelo": 0.90}
VALIDATION_FRACTION = 0.2


def alnum(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower()) if s else ""


def threshold_for_precision(conf: np.ndarray, correct: np.ndarray, target: float) -> float:
    """Lowest threshold t such that precision of {conf >= t} is >= target.

    Scans thresholds from high to low and keeps the last one that still meets
    the target, so the answer favours coverage. Returns 1.01 (always abstain)
    if no threshold reaches the target.
    """
    order = np.argsort(-conf, kind="stable")
    conf_sorted, correct_sorted = conf[order], correct[order].astype(float)
    cum_precision = np.cumsum(correct_sorted) / np.arange(1, len(conf) + 1)
    best = 1.01
    for i in range(len(conf)):
        # only cut between distinct confidence values
        if i + 1 < len(conf) and conf_sorted[i + 1] == conf_sorted[i]:
            continue
        if cum_precision[i] >= target and i >= 19:  # need >= 20 answers to trust the estimate
            best = float(conf_sorted[i])
    return best


def _validation_mask(norm_names: pd.Series) -> np.ndarray:
    return is_test_group(listing_groups(norm_names), VALIDATION_FRACTION, salt="validation")


# ------------------------------------------------------------------ classifiers


def _text_features():
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, dtype=np.float32)
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
        max_features=100_000, dtype=np.float32,
    )  # fmt: skip
    return word, char


@dataclass
class FieldClassifier:
    field_name: str
    threshold: float = 0.0
    word: TfidfVectorizer | None = None
    char: TfidfVectorizer | None = None
    model: SGDClassifier | None = None

    def _x(self, norms: pd.Series, fit: bool = False):
        if fit:
            self.word, self.char = _text_features()
            return hstack([self.word.fit_transform(norms), self.char.fit_transform(norms)]).tocsr()
        return hstack([self.word.transform(norms), self.char.transform(norms)]).tocsr()

    def _fit_model(self, norms: pd.Series, labels: pd.Series) -> None:
        x = self._x(norms, fit=True)
        # Elastic-net keeps ~0.3% of the (classes x features) weights non-zero, so the
        # brand model is ~2 MB instead of ~600 MB, at the same validation accuracy.
        self.model = SGDClassifier(
            loss="log_loss", penalty="elasticnet", alpha=1e-5, l1_ratio=0.5,
            max_iter=30, tol=None, random_state=SEED, n_jobs=-1,
        ).fit(x, labels)  # fmt: skip
        self.model.sparsify()

    def predict_with_confidence(self, norms: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        proba = self.model.predict_proba(self._x(pd.Series(list(norms))))
        best = proba.argmax(axis=1)
        return self.model.classes_[best], proba[np.arange(len(best)), best]

    def fit(self, norms: pd.Series, labels: pd.Series) -> FieldClassifier:
        """Fit on fit-split, pick threshold on validation split, refit on everything."""
        keep = labels.notna().to_numpy()
        norms, labels = norms[keep].reset_index(drop=True), labels[keep].reset_index(drop=True)
        val = _validation_mask(norms)
        self._fit_model(norms[~val], labels[~val])
        pred, conf = self.predict_with_confidence(norms[val])
        correct = pred == labels[val].to_numpy()
        self.threshold = threshold_for_precision(conf, correct, TARGET_PRECISION[self.field_name])
        self._fit_model(norms, labels)
        return self


@dataclass
class BrandExtractor:
    """Dictionary lookup first, classifier second."""

    classifier: FieldClassifier = field(default_factory=lambda: FieldClassifier("marca"))
    aliases: dict[str, tuple[str, float]] = field(default_factory=dict)  # key -> (brand, precision)
    threshold: float = 0.0

    def _build_aliases(self, norms: pd.Series, brands: pd.Series) -> dict[str, tuple[str, float]]:
        token_names = [f" {' '.join(tokenize(n))} " for n in norms]
        by_key: dict[str, Counter] = {}
        for b in brands.dropna().unique():
            key = brand_key(b)
            if key and len(key) >= 3:
                by_key.setdefault(key, Counter())
        for name, brand in zip(token_names, brands, strict=True):
            for key, counter in by_key.items():
                if f" {key} " in name:
                    counter[brand] += 1
        aliases = {}
        for key, counter in by_key.items():
            total = sum(counter.values())
            if total < 2:
                continue
            brand, hits = counter.most_common(1)[0]
            precision = (hits + 1) / (total + 2)  # Laplace-smoothed
            if precision >= 0.8:
                aliases[key] = (brand, precision)
        return aliases

    def lookup(self, norm: str) -> tuple[str | None, float]:
        text = f" {' '.join(tokenize(norm))} "
        best_key = None
        for key in self.aliases:
            if f" {key} " in text and (best_key is None or len(key) > len(best_key)):
                best_key = key
        if best_key is None:
            return None, 0.0
        return self.aliases[best_key]

    def predict_with_confidence(self, norms: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        norms = pd.Series(list(norms))
        pred, conf = self.classifier.predict_with_confidence(norms)
        pred, conf = pred.astype(object), conf.copy()
        for i, n in enumerate(norms):
            brand, precision = self.lookup(n)
            if brand is not None:
                pred[i], conf[i] = brand, precision
        return pred, conf

    def fit(self, norms: pd.Series, brands: pd.Series) -> BrandExtractor:
        keep = brands.notna().to_numpy()
        norms, brands = norms[keep].reset_index(drop=True), brands[keep].reset_index(drop=True)
        val = _validation_mask(norms)
        self.aliases = self._build_aliases(norms[~val], brands[~val])
        self.classifier._fit_model(norms[~val], brands[~val])
        pred, conf = self.predict_with_confidence(norms[val])
        correct = pred == brands[val].to_numpy()
        self.threshold = threshold_for_precision(conf, correct, TARGET_PRECISION["marca"])
        self.aliases = self._build_aliases(norms, brands)
        self.classifier._fit_model(norms, brands)
        return self


# ------------------------------------------------------------------ model codes

CANDIDATE_FEATURES = [
    "length", "n_digits", "n_alpha", "frac_digit", "is_numeric", "mixed",
    "is_bigram", "pos_rel", "is_first", "is_last", "log_df", "n_candidates", "n_tokens",
]  # fmt: skip


def model_candidates(norm: str) -> list[tuple[str, dict]]:
    """Code-like tokens (and short-prefix + number bigrams like 'fl 729') with features."""
    toks = norm.split()
    n = len(toks)
    out: list[tuple[str, int, bool]] = []
    for i, t in enumerate(toks):
        t = t.replace(".", "")
        if any(c.isdigit() for c in t) and len(t) >= 3 and not is_spec(t):
            out.append((t, i, False))
        if (
            i + 1 < n
            and t.isalpha()
            and len(t) <= 4
            and any(c.isdigit() for c in toks[i + 1])
            and not is_spec(toks[i + 1])
        ):
            out.append((t + toks[i + 1].replace(".", ""), i, True))
    seen = set()
    result = []
    for cand, i, bigram in out:
        if cand in seen:
            continue
        seen.add(cand)
        digits = sum(c.isdigit() for c in cand)
        alpha = sum(c.isalpha() for c in cand)
        result.append(
            (
                cand,
                {
                    "length": len(cand),
                    "n_digits": digits,
                    "n_alpha": alpha,
                    "frac_digit": digits / len(cand),
                    "is_numeric": float(alpha == 0),
                    "mixed": float(alpha > 0 and digits > 0),
                    "is_bigram": float(bigram),
                    "pos_rel": i / max(n - 1, 1),
                    "is_first": float(i == 0),
                    "is_last": float(i == n - 1),
                },
            )
        )
    for _, feats in result:
        feats["n_candidates"] = len(result)
        feats["n_tokens"] = n
    return result


@dataclass
class ModelCodeExtractor:
    threshold: float = 0.5
    doc_freq: dict[str, int] = field(default_factory=dict)
    n_docs: int = 1
    model: HistGradientBoostingClassifier | None = None

    def _rows(self, norms: pd.Series, labels: pd.Series | None = None):
        rows, groups, cands, targets = [], [], [], []
        for gi, norm in enumerate(norms):
            label = alnum(labels.iloc[gi]) if labels is not None else None
            for cand, feats in model_candidates(norm):
                feats = dict(feats)
                feats["log_df"] = math.log1p(self.doc_freq.get(cand, 0))
                rows.append(feats)
                groups.append(gi)
                cands.append(cand)
                targets.append(int(label == cand) if label is not None else 0)
        x = pd.DataFrame(rows, columns=CANDIDATE_FEATURES)
        return x, np.array(groups), cands, np.array(targets)

    def _fit_model(self, norms: pd.Series, labels: pd.Series) -> None:
        x, _, _, y = self._rows(norms, labels)
        self.model = HistGradientBoostingClassifier(random_state=SEED, max_iter=200).fit(x, y)

    def predict_with_confidence(self, norms: pd.Series) -> tuple[list[str | None], np.ndarray]:
        norms = pd.Series(list(norms))
        pred: list[str | None] = [None] * len(norms)
        conf = np.zeros(len(norms))
        x, groups, cands, _ = self._rows(norms)
        if len(x) == 0:
            return pred, conf
        proba = self.model.predict_proba(x)[:, 1]
        for p, g, c in zip(proba, groups, cands, strict=True):
            if p > conf[g]:
                conf[g], pred[g] = p, c
        return pred, conf

    def fit(self, norms: pd.Series, labels: pd.Series) -> ModelCodeExtractor:
        """Train only on listings whose `modelo` label appears among the name's candidates.

        Many `modelo` values in the data are missing or belong to another product,
        so only the rows where the label can be found in the name are trusted.
        """
        counts: Counter = Counter()
        for n in norms:
            counts.update({c for c, _ in model_candidates(n)})
        self.doc_freq, self.n_docs = dict(counts), len(norms)
        usable = [
            isinstance(lab, str) and alnum(lab) in {c for c, _ in model_candidates(n)}
            for n, lab in zip(norms, labels, strict=True)
        ]
        usable = np.array(usable)
        norms, labels = norms[usable].reset_index(drop=True), labels[usable].reset_index(drop=True)
        val = _validation_mask(norms)
        self._fit_model(norms[~val], labels[~val])
        pred, conf = self.predict_with_confidence(norms[val])
        correct = np.array([p == alnum(lab) for p, lab in zip(pred, labels[val], strict=True)])
        self.threshold = threshold_for_precision(conf, correct, TARGET_PRECISION["modelo"])
        self._fit_model(norms, labels)
        self.n_training_rows = int(len(norms))
        return self


# ------------------------------------------------------------------ all fields


@dataclass
class Extractor:
    tipo: FieldClassifier = field(default_factory=lambda: FieldClassifier("tipo"))
    categoria: FieldClassifier = field(default_factory=lambda: FieldClassifier("categoria"))
    marca: BrandExtractor = field(default_factory=BrandExtractor)
    modelo: ModelCodeExtractor = field(default_factory=ModelCodeExtractor)

    def fit(self, train: pd.DataFrame) -> Extractor:
        norms = train["nome_norm"].reset_index(drop=True)
        self.tipo.fit(norms, train["tipo"].reset_index(drop=True))
        self.categoria.fit(norms, train["categoria"].reset_index(drop=True))
        self.marca.fit(norms, train["marca"].reset_index(drop=True))
        self.modelo.fit(norms, train["modelo"].reset_index(drop=True))
        return self

    def thresholds(self) -> dict[str, float]:
        return {
            "tipo": self.tipo.threshold,
            "categoria": self.categoria.threshold,
            "marca": self.marca.threshold,
            "modelo": self.modelo.threshold,
        }

    def extract_many(self, names: list[str]) -> list[dict[str, dict]]:
        norms = pd.Series([normalize_text(n) for n in names])
        out: list[dict[str, dict]] = [{} for _ in names]
        for fname, extractor in (
            ("tipo", self.tipo),
            ("categoria", self.categoria),
            ("marca", self.marca),
            ("modelo", self.modelo),
        ):
            preds, confs = extractor.predict_with_confidence(norms)
            for i, (p, c) in enumerate(zip(preds, confs, strict=True)):
                keep = p is not None and c >= extractor.threshold
                out[i][fname] = {
                    "value": str(p) if keep else None,
                    "confidence": round(float(c), 4),
                }
        for i, n in enumerate(norms):
            colors = extract_colors(n)
            out[i]["cor"] = {"value": colors[0] if colors else None, "confidence": None}
        return out

    def extract(self, name: str) -> dict[str, dict]:
        return self.extract_many([name])[0]
