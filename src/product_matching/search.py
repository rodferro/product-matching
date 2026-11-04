"""(c) Return the most relevant listings for a short customer query.

Retrievers (all over the same normalized names):
- BM25: classic keyword ranking (rare words count more, long names count less).
- CharNgram: TF-IDF over 3-5 character pieces, tolerant to typos and plurals.
- Embedding: multilingual sentence embeddings, which understand that
  "geladeira" and "refrigerador" mean the same thing.
- Hybrid: reciprocal rank fusion (RRF) of BM25 and embeddings, with the
  standard constant k=60 (not tuned on the judged queries).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from .candidates import char_vectorizer
from .config import LABELS_DIR
from .embeddings import Encoder
from .normalize import normalize_text, tokenize


def _search_tokens(text: str) -> list[str]:
    return tokenize(normalize_text(text, keep_code_parts=True))


RRF_K = 60
FUSION_DEPTH = 100


def _top_k(scores: np.ndarray, k: int) -> list[int]:
    k = min(k, len(scores))
    part = np.argpartition(-scores, k - 1)[:k]
    return [int(i) for i in part[np.argsort(-scores[part], kind="stable")]]


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75

    def fit(self, names: pd.Series) -> BM25Index:
        """`names` are raw listing names (normalized here, keeping code parts)."""
        self.vectorizer = CountVectorizer(
            tokenizer=_search_tokens, token_pattern=None, lowercase=False, binary=False
        )
        tf = self.vectorizer.fit_transform(names).tocsr().astype(np.float32)
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * doc_len / doc_len.mean())
        tf = tf.tocoo()
        weights = tf.data * (self.k1 + 1) / (tf.data + norm[tf.row])
        self.matrix = sparse.csr_matrix((weights, (tf.row, tf.col)), shape=tf.shape).tocsc()
        return self

    def scores(self, query: str) -> np.ndarray:
        vocab = self.vectorizer.vocabulary_
        cols = [vocab[t] for t in set(_search_tokens(query)) if t in vocab]
        if not cols:
            return np.zeros(self.matrix.shape[0], dtype=np.float32)
        sub = self.matrix[:, cols] @ self.idf[cols]
        return np.asarray(sub).ravel()


@dataclass
class CharNgramIndex:
    def fit(self, norm_names: pd.Series) -> CharNgramIndex:
        self.vectorizer = char_vectorizer()
        self.matrix = self.vectorizer.fit_transform(norm_names)
        return self

    def scores(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([normalize_text(query)])
        return np.asarray((self.matrix @ q.T).todense()).ravel()


@dataclass
class EmbeddingIndex:
    encoder: Encoder = field(default_factory=Encoder)
    vectors: np.ndarray | None = None

    def fit(self, names: pd.Series, vectors: np.ndarray | None = None) -> EmbeddingIndex:
        self.vectors = vectors if vectors is not None else self.encoder.encode_catalog(list(names))
        return self

    def scores(self, query: str) -> np.ndarray:
        q = self.encoder.encode([query], kind="query")[0]
        return self.vectors @ q


@dataclass
class HybridIndex:
    """Reciprocal rank fusion: score = sum over retrievers of 1 / (k + rank)."""

    retrievers: list
    k: int = RRF_K
    depth: int = FUSION_DEPTH

    def scores(self, query: str) -> np.ndarray:
        n = None
        fused: dict[int, float] = {}
        for r in self.retrievers:
            s = r.scores(query)
            n = len(s)
            for rank, doc in enumerate(_top_k(s, self.depth), start=1):
                if s[doc] > 0 or isinstance(r, EmbeddingIndex):
                    fused[doc] = fused.get(doc, 0.0) + 1.0 / (self.k + rank)
        out = np.zeros(n or 0, dtype=np.float32)
        for doc, v in fused.items():
            out[doc] = v
        return out


def search(index, query: str, k: int = 10) -> list[int]:
    scores = index.scores(query)
    top = _top_k(scores, k)
    return [d for d in top if scores[d] > 0] if not isinstance(index, EmbeddingIndex) else top


# ------------------------------------------------------------------ judgment pool


def read_queries(labels_dir=LABELS_DIR) -> pd.DataFrame:
    return pd.read_csv(labels_dir / "queries.csv", dtype=str, keep_default_na=False)


def read_judgments(labels_dir=LABELS_DIR) -> dict[str, dict[int, int]]:
    j = pd.read_csv(labels_dir / "judgments.csv", dtype=str, keep_default_na=False)
    j = j[j["grade"].isin(["0", "1", "2"])]
    out: dict[str, dict[int, int]] = {}
    for qid, doc, grade in zip(
        j["query_id"], j["id"].astype(int), j["grade"].astype(int), strict=True
    ):
        out.setdefault(qid, {})[doc] = grade
    return out


POOL_DEPTH = 10
POOL_REGEX_CAP = 20


def build_systems(
    df: pd.DataFrame,
    emb_small: np.ndarray,
    emb_large: np.ndarray | None = None,
    queries: list[str] | None = None,
):
    """All retrieval systems that are evaluated (and that feed the judgment pool).

    The 2022 baseline runs in a separate process (see baseline_2022.run_isolated),
    so its results are precomputed for `queries`.
    """
    from .baseline_2022 import PrecomputedSearch, run_isolated, search_outputs
    from .config import EMBEDDING_MODEL_LARGE

    bm25 = BM25Index().fit(df["nome"])
    char = CharNgramIndex().fit(df["nome_norm"])
    emb = EmbeddingIndex().fit(df["nome"], emb_small)
    systems = {
        "baseline_2022": PrecomputedSearch(
            run_isolated(search_outputs, df, list(queries or []), FUSION_DEPTH)
        ),
        "bm25": bm25,
        "char_ngram": char,
        "embedding_e5_small": emb,
        "hybrid_bm25_e5_small": HybridIndex([bm25, emb]),
    }
    if emb_large is not None:
        systems["embedding_e5_base"] = EmbeddingIndex(Encoder(EMBEDDING_MODEL_LARGE)).fit(
            df["nome"], emb_large
        )
    return systems


def run_system(system, query: str, k: int = 10) -> list[int]:
    from .baseline_2022 import PrecomputedSearch

    if isinstance(system, PrecomputedSearch):
        return system.search(query, k)
    return search(system, query, k)


def write_judgment_pool(force: bool = False) -> None:
    """Pool = top-10 of every system + up to 20 listings matching the query's regex."""
    import logging

    from .config import EMBEDDING_MODEL_LARGE, SEED
    from .data import load_products

    log = logging.getLogger(__name__)
    path = LABELS_DIR / "judgments.csv"
    existing = pd.DataFrame()
    if path.exists():
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
        if force:
            existing = existing[existing["grade"] != ""]
    df = load_products()
    emb_small = Encoder().encode_catalog(df["nome"].tolist())
    emb_large = Encoder(EMBEDDING_MODEL_LARGE).encode_catalog(df["nome"].tolist())
    queries = read_queries()
    systems = build_systems(df, emb_small, emb_large, queries["query"].tolist())
    rng = np.random.default_rng(SEED)
    rows = []
    for q in queries.itertuples():
        pool: dict[int, str] = {}
        for name, system in systems.items():
            for doc in run_system(system, q.query, POOL_DEPTH):
                pool.setdefault(doc, name)
        if q.pool_regex:
            hits = np.flatnonzero(df["nome_norm"].str.contains(q.pool_regex, regex=True).to_numpy())
            rng.shuffle(hits)
            for doc in hits[:POOL_REGEX_CAP]:
                pool.setdefault(int(doc), "regex")
        for doc, source in sorted(pool.items()):
            rows.append({"query_id": q.query_id, "query": q.query, "id": doc,
                         "nome": df["nome"].iat[doc], "first_source": source, "grade": ""})  # fmt: skip
    pool_df = pd.DataFrame(rows)
    if not existing.empty:
        keep = existing.set_index(["query_id", "id"])["grade"].to_dict()
        pool_df["grade"] = [
            keep.get((qid, str(d)), "")
            for qid, d in zip(pool_df["query_id"], pool_df["id"], strict=True)
        ]
    pool_df.to_csv(path, index=False)
    log.info("wrote %s (%d rows, %d ungraded)", path, len(pool_df), (pool_df["grade"] == "").sum())
