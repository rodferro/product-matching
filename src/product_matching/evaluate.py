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


def _round(d: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def evaluate_extraction(world: World) -> dict:
    from .baseline_2022 import extraction_outputs, run_isolated
    from .extraction import Extractor, alnum
    from .labeling import read_labels
    from .metrics import abstaining_scores, presence_scores
    from .splits import extraction_test_mask

    df = world.df
    test = extraction_test_mask(df["nome_norm"])
    train, held = df[~test].reset_index(drop=True), df[test].reset_index(drop=True)

    t0 = time.time()
    extractor = Extractor().fit(train)
    fit_seconds = time.time() - t0
    check = read_labels("extraction_check.csv").assign(id=lambda d: d["id"].astype(int))
    assert test[check["id"]].all(), "hand-checked names must come from the test split"
    check_names = df.loc[check["id"], "nome"].tolist()
    legacy = run_isolated(extraction_outputs, train, held["nome"].tolist(), check_names)

    results: dict = {
        "n_listings": {"train": len(train), "test": len(held)},
        "thresholds": {k: round(v, 4) for k, v in extractor.thresholds().items()},
        "fit_seconds": round(fit_seconds, 1),
        "fields": {},
    }

    # tipo / categoria / marca: scored against the dataset labels on the test split
    preds = extractor.extract_many(held["nome"].tolist())
    curves = {}
    for fname in ("tipo", "categoria", "marca"):
        has = held[fname].notna().to_numpy()
        gold = held.loc[has, fname].tolist()
        new = [p[fname]["value"] for p, h in zip(preds, has, strict=True) if h]
        conf = np.array([p[fname]["confidence"] for p, h in zip(preds, has, strict=True) if h])
        raw = extractor_raw(extractor, fname, held.loc[has, "nome_norm"])
        majority = train[fname].dropna().value_counts().index[0]
        results["fields"][fname] = {
            "n_test": int(has.sum()),
            "n_classes_train": int(train[fname].nunique()),
            "majority_class": majority,
            "baseline_majority": _round(abstaining_scores(gold, [majority] * len(gold))),
            "baseline_2022": _round(
                abstaining_scores(
                    gold, [p for p, h in zip(legacy["fields"][fname], has, strict=True) if h]
                )
            ),
            "new_no_abstention": _round(abstaining_scores(gold, raw)),
            "new": _round(abstaining_scores(gold, new)),
        }
        curves[fname] = _coverage_curve(gold, raw, conf)
    _plot_coverage(curves, extractor.thresholds())

    # cor / modelo: scored against the 200 hand-checked names (None is a valid answer)
    chk_preds = extractor.extract_many(check_names)
    nearest = legacy["nearest"]
    gold_cor = [v or None for v in check["cor"]]
    gold_mod = [alnum(v) or None for v in check["modelo"]]
    majority_cor = train["cor"].dropna().value_counts().index[0]
    majority_mod = alnum(train["modelo"].dropna().value_counts().index[0])
    new_mod = [alnum(p["modelo"]["value"]) or None for p in chk_preds]
    results["fields"]["cor"] = {
        "n_test": len(check),
        "gold_present": sum(g is not None for g in gold_cor),
        "baseline_always_none": _round(presence_scores(gold_cor, [None] * len(check))),
        "baseline_majority": _round(presence_scores(gold_cor, [majority_cor] * len(check))),
        "baseline_2022": _round(presence_scores(gold_cor, nearest["cor"])),
        "new": _round(presence_scores(gold_cor, [p["cor"]["value"] for p in chk_preds])),
        "dataset_field": _round(
            presence_scores(gold_cor, [v or None for v in check["cor_dataset"]])
        ),
    }
    results["fields"]["modelo"] = {
        "n_test": len(check),
        "gold_present": sum(g is not None for g in gold_mod),
        "baseline_always_none": _round(presence_scores(gold_mod, [None] * len(check))),
        "baseline_majority": _round(presence_scores(gold_mod, [majority_mod] * len(check))),
        "baseline_2022": _round(
            presence_scores(gold_mod, [alnum(v) or None for v in nearest["modelo"]])
        ),
        "new": _round(presence_scores(gold_mod, new_mod)),
        "dataset_field": _round(
            presence_scores(gold_mod, [alnum(v) or None for v in check["modelo_dataset"]])
        ),
        "training_rows_with_label_in_name": extractor.modelo.n_training_rows,
    }
    return results


def extractor_raw(extractor, fname: str, norms: pd.Series) -> list:
    """Top prediction with no abstention (for the coverage curve)."""
    part = getattr(extractor, fname)
    pred, _ = part.predict_with_confidence(norms)
    return list(pred)


def _coverage_curve(gold: list, pred: list, conf: np.ndarray) -> list[dict]:
    order = np.argsort(-conf, kind="stable")
    correct = np.array([p == g for p, g in zip(pred, gold, strict=True)])[order]
    n = len(gold)
    points = []
    for cov in np.linspace(0.05, 1.0, 20):
        k = max(1, int(round(cov * n)))
        points.append(
            {"coverage": round(k / n, 4), "accuracy": round(float(correct[:k].mean()), 4)}
        )
    return points


def _plot_coverage(curves: dict, thresholds: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = REPORTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"tipo": "#2a6fb0", "categoria": "#3a9a5b", "marca": "#c05a28"}
    for fname, pts in curves.items():
        ax.plot([p["coverage"] for p in pts], [p["accuracy"] for p in pts],
                color=colors[fname], lw=1.8, label=fname)  # fmt: skip
    ax.axhline(0.95, color="#999999", lw=1, ls="--", label="target precision 0.95")
    ax.set_xlabel("coverage (share of listings answered, most confident first)")
    ax.set_ylabel("accuracy of the answers given")
    ax.set_title("Accuracy vs coverage on held-out listings")
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(fig_dir / "coverage.png", dpi=150)
    plt.close(fig)


def evaluate_search(world: World) -> dict:
    from .config import EMBEDDING_MODEL_LARGE
    from .metrics import ndcg_at_k, recall_at_k
    from .search import build_systems, read_judgments, read_queries, run_system

    queries = read_queries()
    judgments = read_judgments()
    emb_large = Encoder(EMBEDDING_MODEL_LARGE).encode_catalog(world.df["nome"].tolist())
    systems = build_systems(world.df, world.emb, emb_large, queries["query"].tolist())

    per_query: dict[str, dict] = {}
    summary: dict[str, dict] = {}
    for name, system in systems.items():
        ndcgs, recalls, strict, unjudged, latency = [], [], [], 0, []
        for q in queries.itertuples():
            rel = judgments.get(q.query_id, {})
            t = time.perf_counter()
            ranked = run_system(system, q.query, 10)
            latency.append((time.perf_counter() - t) * 1000)
            unjudged += sum(1 for d in ranked if d not in rel)
            n, r = ndcg_at_k(ranked, rel, 10), recall_at_k(ranked, rel, 10)
            s = recall_at_k(ranked, rel, 10, min_grade=2)
            ndcgs.append(n)
            recalls.append(r)
            strict.append(s)
            per_query.setdefault(q.query_id, {"query": q.query})[name] = round(n, 4)
        mean = lambda xs: float(np.mean(xs))  # noqa: E731
        summary[name] = {
            "ndcg@10": round(mean(ndcgs), 4),
            "ndcg@10_ci95": _ci(lambda a: float(np.mean(a.astype(float))), ndcgs),
            "recall@10": round(mean(recalls), 4),
            "recall@10_ci95": _ci(lambda a: float(np.mean(a.astype(float))), recalls),
            "recall@10_grade2": round(mean(strict), 4),
            "unjudged_in_top10": unjudged,
            # the 2022 baseline runs in a separate process and is precomputed: not measured
            "median_latency_ms": None
            if name == "baseline_2022"
            else round(float(np.median(latency)), 1),
        }
    best = max((n for n in summary if n != "baseline_2022"), key=lambda n: summary[n]["ndcg@10"])
    wins = sum(per_query[q][best] > per_query[q]["baseline_2022"] for q in per_query)
    return {
        "n_queries": len(queries),
        "n_judgments": sum(len(v) for v in judgments.values()),
        "relevant_per_query_min": min(
            sum(1 for g in v.values() if g >= 1) for v in judgments.values()
        ),
        "systems": summary,
        "best_new_system": best,
        "queries_where_best_beats_baseline": wins,
        "per_query_ndcg@10": per_query,
    }


def run_all(sections: tuple[str, ...] = ("matching", "extraction", "search")) -> dict:
    set_seeds()
    t0 = time.time()
    world = World.load()
    metrics: dict = {}
    path = REPORTS_DIR / "metrics.json"
    if path.exists():
        metrics = json.loads(path.read_text())
    runners = {
        "matching": evaluate_matching,
        "extraction": evaluate_extraction,
        "search": evaluate_search,
    }
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
