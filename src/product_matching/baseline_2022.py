"""The 2022 solution (legacy/challenge_v1.ipynb), ported so it can be scored.

Kept as close to the notebook as possible: same preprocessing, same TF-IDF
settings, XGBoost with default parameters, top-20 classes + "Outros" for
type/brand, and the same query construction. Two changes, both needed to
evaluate it fairly:

- models are fitted on our training split only (not on all rows), and
- the name vectorizer is fitted once on that split (the notebook re-used a
  vectorizer fitted on a different split, which leaks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

from .config import FIELDS, SEED

LEGACY_COLUMNS = ["nome", *FIELDS]


def legacy_preprocess(series: pd.Series) -> pd.Series:
    """lowercase -> strip accents -> non-alphanumerics to space -> drop 1-letter words -> stopwords."""
    s = series.fillna("None").astype(str).str.lower()
    s = s.str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")
    s = s.replace("[^A-Za-z0-9]+", " ", regex=True)
    s = s.replace(r"\b[a-z]\b", "", regex=True).str.replace(r"\s+", " ", regex=True)
    return s.replace(r"\b(de|com|para|em)\b\s*", "", regex=True)


def _legacy_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: legacy_preprocess(df[c]) for c in LEGACY_COLUMNS})


def _name_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(max_features=1_000, ngram_range=(1, 2))


def _full_transformer() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("nome_vect", TfidfVectorizer(max_features=1_000, ngram_range=(1, 2)), "nome"),
            ("tipo_vect", TfidfVectorizer(max_features=1_000, ngram_range=(1, 1)), "tipo"),
            ("marca_vect", TfidfVectorizer(max_features=1_000, ngram_range=(1, 1)), "marca"),
            (
                "categoria_vect",
                TfidfVectorizer(max_features=1_000, ngram_range=(1, 2)),
                "categoria",
            ),
            ("cor_vect", TfidfVectorizer(max_features=1_000, ngram_range=(1, 1)), "cor"),
            ("modelo_vect", TfidfVectorizer(max_features=1_000, ngram_range=(1, 1)), "modelo"),
        ]
    )


def _bucket_labels(df: pd.DataFrame, field: str) -> pd.Series:
    """The notebook's label grouping. Missing values are the literal class 'None'."""
    y = df[field].fillna("None")
    if field == "categoria":
        return y.replace(["None", "Instrumentos Musicais", "Pet Shop", "Joalheria"], "Outros")
    top = y.value_counts().index[:20]
    return y.where(y.isin(top), "Outros")


class LegacyMatcher:
    """(a) in 2022: cosine similarity of name TF-IDF vectors, read as a 'probability'."""

    def fit(self, names: pd.Series) -> LegacyMatcher:
        self.vectorizer = _name_vectorizer().fit(legacy_preprocess(names))
        return self

    def score(self, names_a: pd.Series, names_b: pd.Series) -> np.ndarray:
        a = self.vectorizer.transform(legacy_preprocess(pd.Series(list(names_a))))
        b = self.vectorizer.transform(legacy_preprocess(pd.Series(list(names_b))))
        return np.asarray(a.multiply(b).sum(axis=1)).ravel()


class LegacyClassifiers:
    """(b) in 2022: one XGBoost per field (categoria, tipo, marca) on name TF-IDF."""

    fields = ("categoria", "tipo", "marca")

    def fit(self, train: pd.DataFrame) -> LegacyClassifiers:
        from xgboost import XGBClassifier

        self.vectorizer = _name_vectorizer().fit(legacy_preprocess(train["nome"]))
        x = self.vectorizer.transform(legacy_preprocess(train["nome"]))
        self.encoders: dict[str, LabelEncoder] = {}
        self.models = {}
        for field in self.fields:
            enc = LabelEncoder().fit(_bucket_labels(train, field))
            model = XGBClassifier(random_state=SEED, n_jobs=-1, verbosity=0)
            model.fit(x, enc.transform(_bucket_labels(train, field)))
            self.encoders[field], self.models[field] = enc, model
        return self

    def predict_raw(self, names: pd.Series, field: str) -> np.ndarray:
        """Class names as the 2022 code returned them (including 'Outros' and 'None')."""
        x = self.vectorizer.transform(legacy_preprocess(pd.Series(list(names))))
        return self.encoders[field].inverse_transform(self.models[field].predict(x))

    def predict(self, names: pd.Series, field: str) -> list[str | None]:
        """'Outros' and 'None' carry no information, so they are scored as abstentions."""
        return [None if v in ("Outros", "None") else v for v in self.predict_raw(names, field)]


class LegacySearch:
    """(c) in 2022: kNN over 6-field TF-IDF; the query's category/brand/type are predicted.

    Also used for the 2022 answer to (b) for color and model: copy them from the
    nearest neighbour.
    """

    def fit(self, catalogue: pd.DataFrame, classifiers: LegacyClassifiers) -> LegacySearch:
        self.catalogue = catalogue.reset_index(drop=True)
        self.classifiers = classifiers
        self.transformer = _full_transformer()
        x = self.transformer.fit_transform(_legacy_frame(self.catalogue))
        self.nn = NearestNeighbors(metric="cosine").fit(x)
        return self

    def _query_frame(self, names: list[str]) -> pd.DataFrame:
        q = pd.DataFrame({c: ["None"] * len(names) for c in LEGACY_COLUMNS})
        q["nome"] = names
        q = _legacy_frame(q)
        for field in ("categoria", "marca", "tipo"):
            q[field] = list(self.classifiers.predict_raw(pd.Series(names), field))
        return q

    def search(self, query: str, k: int = 10) -> list[int]:
        """Positions in the catalogue, best first."""
        x = self.transformer.transform(self._query_frame([query]))
        _, idx = self.nn.kneighbors(x, n_neighbors=k)
        return [int(i) for i in idx[0]]

    def nearest_fields(self, names: list[str], fields: tuple[str, ...] = ("cor", "modelo")):
        x = self.transformer.transform(self._query_frame(names))
        _, idx = self.nn.kneighbors(x, n_neighbors=1)
        rows = self.catalogue.iloc[idx[:, 0]]
        return {f: [v if isinstance(v, str) else None for v in rows[f]] for f in fields}


# ---------------------------------------------------------------- process isolation
#
# On macOS, torch (embeddings) and xgboost (this baseline) each bundle an OpenMP
# runtime, and running both in one process crashes or deadlocks. The baseline
# only needs plain data in and out, so it runs in a child process that never
# imports torch.


def run_isolated(func, *args):
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(1) as pool:
        return pool.apply(func, args)


def extraction_outputs(train: pd.DataFrame, held_names: list[str], check_names: list[str]) -> dict:
    """2022 answers to (b): classifier fields for held-out names, nearest-neighbour color/model."""
    clf = LegacyClassifiers().fit(train)
    search = LegacySearch().fit(train, clf)
    held = pd.Series(held_names)
    return {
        "fields": {f: clf.predict(held, f) for f in LegacyClassifiers.fields},
        "nearest": search.nearest_fields(check_names),
    }


def search_outputs(catalogue: pd.DataFrame, queries: list[str], k: int) -> dict[str, list[int]]:
    """2022 answers to (c): top-k positions for each query."""
    search = LegacySearch().fit(catalogue, LegacyClassifiers().fit(catalogue))
    return {q: search.search(q, k) for q in queries}


class PrecomputedSearch:
    """Serves search results computed ahead of time (used for the isolated baseline)."""

    def __init__(self, results: dict[str, list[int]]):
        self.results = results

    def search(self, query: str, k: int = 10) -> list[int]:
        return self.results[query][:k]
