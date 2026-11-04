"""Score the 2022 baseline and the new system on held-out labels.

`pm evaluate` runs everything and writes:
- reports/metrics.json  : every number, machine-readable
- reports/results.md    : the tables used in the README
- reports/figures/*.png : reliability plot, coverage curves
Nothing in the README is typed by hand; it is copied from results.md.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from .candidates import char_vectorizer, generate_candidates
from .config import DATA_PATH, LABELS_DIR, REPORTS_DIR, set_seeds
from .data import load_products
from .embeddings import Encoder
from .features import PairFeaturizer, listing_info
from .labeling import labelled_pairs
from .metrics import (
    best_f1_threshold,
    bootstrap_ci,
    brier,
    pr_auc,
    precision_recall_f1,
    reliability,
)
from .splits import pair_test_mask

log = logging.getLogger(__name__)


@dataclass
class World:
    """Data and expensive shared objects, built once per evaluation run."""

    df: pd.DataFrame
    emb: np.ndarray

    @classmethod
    def load(cls) -> World:
        df = load_products(DATA_PATH)
        assert (df["id"].to_numpy() == np.arange(len(df))).all(), "ids must equal positions"
        emb = Encoder().encode_catalog(df["nome"].tolist())
        return cls(df=df, emb=emb)

    @cached_property
    def candidates(self) -> pd.DataFrame:
        cm = char_vectorizer().fit_transform(self.df["nome_norm"])
        return generate_candidates(self.df["nome_norm"], self.emb, char_matrix=cm)

    @cached_property
    def infos(self):
        return [
            listing_info(n, b, norm=nn)
            for n, b, nn in zip(
                self.df["nome"], self.df["marca"], self.df["nome_norm"], strict=True
            )
        ]

    @cached_property
    def featurizer(self) -> PairFeaturizer:
        return PairFeaturizer().fit(self.df)

    def pair_features(self, pos_a: np.ndarray, pos_b: np.ndarray) -> pd.DataFrame:
        return self.featurizer.features(
            [self.infos[i] for i in pos_a],
            [self.infos[i] for i in pos_b],
            self.emb[pos_a],
            self.emb[pos_b],
        )


def _ci(stat, *arrays) -> list[float]:
    return [round(v, 4) for v in bootstrap_ci(stat, *arrays, n_boot=1000)]


def _score_block(y: np.ndarray, prob: np.ndarray, threshold: float, is_prob: bool) -> dict:
    prf = precision_recall_f1(y, prob >= threshold)
    out = {
        "threshold": round(float(threshold), 4),
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in prf.items()},
        "f1_ci95": _ci(
            lambda yy, pp: precision_recall_f1(yy.astype(int), pp >= threshold)["f1"], y, prob
        ),
        "pr_auc": round(pr_auc(y, prob), 4),
        "pr_auc_ci95": _ci(lambda yy, pp: pr_auc(yy.astype(int), pp.astype(float)), y, prob),
        "brier": round(brier(y, prob), 4),
        "ece": round(reliability(y, prob)["ece"], 4),
        "is_probability": is_prob,
    }
    return out


def evaluate_matching(world: World) -> dict:
    from .baseline_2022 import LegacyMatcher
    from .matching import PairMatcher

    pairs = labelled_pairs(LABELS_DIR)
    a, b = pairs["id_a"].to_numpy(), pairs["id_b"].to_numpy()
    y = pairs["label"].to_numpy()
    test = pair_test_mask(world.df["nome_norm"], a, b)
    tr, te = ~test, test
    x = world.pair_features(a, b)

    # new model
    model = PairMatcher().fit(x[tr], y[tr])
    prob_new = model.predict_proba(x[te])

    # 2022 baseline: name TF-IDF cosine, threshold picked on the training pairs
    legacy = LegacyMatcher().fit(world.df["nome"])
    cos_legacy = legacy.score(world.df["nome"].iloc[a], world.df["nome"].iloc[b])
    t_legacy = best_f1_threshold(y[tr], cos_legacy[tr])

    # ablation: embedding cosine alone
    emb_cos = x["emb_cos"].to_numpy()
    t_emb = best_f1_threshold(y[tr], emb_cos[tr])

    # blocking: how many labelled duplicates the candidate generator keeps
    cands = world.candidates
    cand_set = set(zip(cands["a"], cands["b"], strict=True))
    pos_pairs = [(min(i, j), max(i, j)) for i, j, lab in zip(a, b, y, strict=True) if lab == 1]
    kept = [p in cand_set for p in pos_pairs]
    by_source = pairs.assign(kept=[
        (min(i, j), max(i, j)) in cand_set for i, j in zip(a, b, strict=True)
    ]).query("label == 1").groupby("source")["kept"].agg(["sum", "count"])  # fmt: skip
    n = len(world.df)

    rel = reliability(y[te], prob_new)
    rel_legacy = reliability(y[te], np.clip(cos_legacy[te], 0, 1))
    _plot_reliability(rel, rel_legacy)

    return {
        "n_pairs": {"train": int(tr.sum()), "test": int(te.sum())},
        "n_positive": {"train": int(y[tr].sum()), "test": int(y[te].sum())},
        "test_by_source": pairs[te].groupby("source")["label"].agg(["count", "sum"])
        .rename(columns={"sum": "positives"}).to_dict(orient="index"),
        "baseline_2022": _score_block(y[te], cos_legacy[te], t_legacy, is_prob=False),
        "embedding_cosine": _score_block(y[te], emb_cos[te], t_emb, is_prob=False),
        "new": _score_block(y[te], prob_new, model.threshold, is_prob=True),
        "model": {**model.cv_report, "coefficients": {
            k: round(v, 3) for k, v in model.coefficients().items()}},
        "reliability_new": rel,
        "blocking": {
            "candidate_pairs": len(cands),
            "all_pairs": n * (n - 1) // 2,
            "reduction_ratio": round(1 - len(cands) / (n * (n - 1) / 2), 6),
            "labelled_positives_kept": int(sum(kept)),
            "labelled_positives": len(kept),
            "kept_by_source": {s: {"kept": int(r["sum"]), "of": int(r["count"])}
                               for s, r in by_source.iterrows()},
        },
    }  # fmt: skip


def _plot_reliability(rel_new: dict, rel_legacy: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = REPORTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", label="perfect calibration")
    for rel, label, color in (
        (rel_legacy, "2022: cosine similarity", "#c05a28"),
        (rel_new, "new: calibrated model", "#2a6fb0"),
    ):
        xs = [b["mean_pred"] for b in rel["bins"]]
        ys = [b["observed"] for b in rel["bins"]]
        sizes = [20 + 3 * b["n"] for b in rel["bins"]]
        ax.plot(xs, ys, color=color, lw=1.5, label=f"{label} (ECE {rel['ece']:.3f})")
        ax.scatter(xs, ys, s=sizes, color=color, alpha=0.7)
    ax.set_xlabel("predicted probability (bin mean)")
    ax.set_ylabel("observed share of true matches")
    ax.set_title("Reliability on held-out pairs")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "reliability.png", dpi=150)
    plt.close(fig)


def run_all(sections: tuple[str, ...] = ("matching", "extraction", "search")) -> dict:
    set_seeds()
    t0 = time.time()
    world = World.load()
    metrics: dict = {}
    path = REPORTS_DIR / "metrics.json"
    if path.exists():
        metrics = json.loads(path.read_text())
    runners = {"matching": evaluate_matching}
    for name in sections:
        if name in runners:
            log.info("evaluating %s", name)
            metrics[name] = runners[name](world)
    metrics["runtime_seconds"] = round(time.time() - t0, 1)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_default))
    return metrics


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(type(o))
