"""Everything the API needs at serving time, built by `pm build` and loaded from artifacts/."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR, DATA_PATH, LABELS_DIR, set_seeds
from .embeddings import Encoder
from .extraction import Extractor
from .features import PairFeaturizer, listing_info
from .matching import PairMatcher
from .search import BM25Index, EmbeddingIndex, HybridIndex, search

log = logging.getLogger(__name__)
BUNDLE = "service.joblib"


@dataclass
class Service:
    catalogue: pd.DataFrame  # id, nome, tipo, categoria, marca, cor, modelo
    featurizer: PairFeaturizer
    matcher: PairMatcher
    extractor: Extractor
    bm25: BM25Index
    embedding_model: str
    embeddings: np.ndarray

    # ---------------------------------------------------------------- build / load

    @classmethod
    def build(
        cls,
        data_path: Path = DATA_PATH,
        labels_dir: Path = LABELS_DIR,
        cache_dir: Path | None = None,
    ) -> Service:
        """Fit every component on all available data and labels (evaluation used splits)."""
        from .data import load_products
        from .labeling import labelled_pairs

        set_seeds()
        df = load_products(data_path)
        encoder = Encoder(cache_dir=cache_dir)
        emb = encoder.encode_catalog(df["nome"].tolist())
        featurizer = PairFeaturizer().fit(df)
        pairs = labelled_pairs(labels_dir)
        infos = [
            listing_info(n, b, norm=nn)
            for n, b, nn in zip(df["nome"], df["marca"], df["nome_norm"], strict=True)
        ]
        a, b = pairs["id_a"].to_numpy(), pairs["id_b"].to_numpy()
        x = featurizer.features([infos[i] for i in a], [infos[i] for i in b], emb[a], emb[b])
        matcher = PairMatcher().fit(x, pairs["label"].to_numpy())
        extractor = Extractor().fit(df)
        bm25 = BM25Index().fit(df["nome"])
        catalogue = df[["id", "nome", "tipo", "categoria", "marca", "cor", "modelo"]].copy()
        return cls(catalogue, featurizer, matcher, extractor, bm25, encoder.model_name, emb)

    def save(self, directory: Path = ARTIFACTS_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / BUNDLE
        joblib.dump(self, path, compress=3)
        return path

    @classmethod
    def load(cls, directory: Path = ARTIFACTS_DIR) -> Service:
        return joblib.load(directory / BUNDLE)

    # ---------------------------------------------------------------- operations

    @property
    def encoder(self) -> Encoder:
        return Encoder(self.embedding_model)

    def match(self, a: dict, b: dict) -> dict:
        """Probability that listings a and b are the same product."""
        info_a = listing_info(a["nome"], a.get("marca"))
        info_b = listing_info(b["nome"], b.get("marca"))
        emb = self.encoder.encode([a["nome"], b["nome"]], kind="passage")
        x = self.featurizer.features([info_a], [info_b], emb[:1], emb[1:])
        prob = float(self.matcher.predict_proba(x)[0])
        return {
            "probability": round(prob, 4),
            "same_product": prob >= self.matcher.threshold,
            "threshold": round(self.matcher.threshold, 4),
        }

    def extract(self, nome: str) -> dict:
        return self.extractor.extract(nome)

    def search(self, query: str, k: int = 10) -> list[dict]:
        index = HybridIndex([self.bm25, EmbeddingIndex(self.encoder, self.embeddings)])
        scores = index.scores(query)
        rows = []
        for pos in search(index, query, k):
            r = self.catalogue.iloc[pos]
            rows.append(
                {
                    "id": int(r["id"]),
                    "nome": r["nome"],
                    "tipo": r["tipo"],
                    "marca": r["marca"],
                    "score": round(float(scores[pos]), 5),
                }
            )
        return rows


def build_all(directory: Path = ARTIFACTS_DIR) -> Path:
    service = Service.build()
    path = service.save(directory)
    log.info("wrote %s (%.0f MB)", path, path.stat().st_size / 1e6)
    return path
