"""Render reports/metrics.json as Markdown tables (reports/results.md and the README)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .config import REPORTS_DIR, ROOT

START, END = "<!-- results:start -->", "<!-- results:end -->"

SYSTEM_NAMES = {
    "baseline_2022": "2022 baseline",
    "bm25": "BM25",
    "char_ngram": "Char n-gram TF-IDF",
    "embedding_e5_small": "Embeddings (e5-small)",
    "embedding_e5_base": "Embeddings (e5-base)",
    "hybrid_bm25_e5_small": "Hybrid: BM25 + e5-small (RRF)",
}


def _p(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _ci(ci: list[float]) -> str:
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def matching_section(m: dict) -> str:
    rows = [
        ("2022 baseline: name TF-IDF cosine", m["baseline_2022"]),
        ("Embedding cosine only (ablation)", m["embedding_cosine"]),
        ("**New: calibrated pairwise model**", m["new"]),
    ]
    out = [
        "### (a) Same-product probability",
        "",
        f"Held-out pairs: {m['n_pairs']['test']} ({m['n_positive']['test']} same-product). "
        f"Training pairs: {m['n_pairs']['train']} ({m['n_positive']['train']} same-product). "
        "Thresholds are chosen on the training pairs. Brackets are 95% bootstrap intervals.",
        "",
        "| System | Precision | Recall | F1 | PR-AUC | Brier | ECE |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in rows:
        note = "" if r["is_probability"] else "¹"
        out.append(
            f"| {name} | {_p(r['precision'])} | {_p(r['recall'])} | "
            f"{_p(r['f1'])} {_ci(r['f1_ci95'])} | {_p(r['pr_auc'])} {_ci(r['pr_auc_ci95'])} | "
            f"{_p(r['brier'])}{note} | {_p(r['ece'])}{note} |"
        )
    b = m["blocking"]
    out += [
        "",
        "¹ A cosine similarity is not a probability; Brier/ECE show what happens if it is read as one.",
        "",
        f"Candidate generation keeps {b['candidate_pairs']:,} of {b['all_pairs']:,} possible pairs "
        f"({_pct(1 - b['reduction_ratio'])}). Its recall is not measured (see Limitations).",
    ]
    return "\n".join(out)


def extraction_section(e: dict) -> str:
    f = e["fields"]
    out = [
        "### (b) Attributes from the name",
        "",
        f"Type, category and brand are scored against the dataset's own labels on "
        f"{e['n_listings']['test']:,} held-out listings. *Answered* = share of listings the system "
        "gives an answer for; *accuracy* = share of those answers that are right; "
        "*overall* = right answers / all listings (an abstention counts as wrong).",
        "",
        "| Field | Classes | Majority class (overall) | 2022: answered / accuracy / overall | New: answered / accuracy / overall |",
        "|---|---|---|---|---|",
    ]
    for name in ("tipo", "categoria", "marca"):
        r = f[name]
        b, n = r["baseline_2022"], r["new"]
        out.append(
            f"| {name} | {r['n_classes_train']} | {_pct(r['baseline_majority']['accuracy_all'])} | "
            f"{_pct(b['coverage'])} / {_pct(b['accuracy_answered'])} / {_pct(b['accuracy_all'])} | "
            f"{_pct(n['coverage'])} / {_pct(n['accuracy_answered'])} / **{_pct(n['accuracy_all'])}** |"
        )
    out += [
        "",
        f"Color and model are scored against {f['cor']['n_test']} held-out names whose color and model "
        "were labelled by an AI assistant (Claude), not yet reviewed by a person. None "
        "(the name does not state it) is a valid answer. *Accuracy* counts None = None as right; "
        "*precision* / *recall* are over the values that are present.",
        "",
        "| Field | Present in name | Always None (accuracy) | 2022 (accuracy) | Dataset field as-is: accuracy / precision / recall | New: accuracy / precision / recall |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("cor", "modelo"):
        r = f[name]
        d, n = r["dataset_field"], r["new"]
        out.append(
            f"| {name} | {r['gold_present']}/{r['n_test']} | {_pct(r['baseline_always_none']['accuracy'])} | "
            f"{_pct(r['baseline_2022']['accuracy'])} | "
            f"{_pct(d['accuracy'])} / {_pct(d['precision'])} / {_pct(d['recall'])} | "
            f"**{_pct(n['accuracy'])}** / {_pct(n['precision'])} / {_pct(n['recall'])} |"
        )
    return "\n".join(out)


def search_section(s: dict) -> str:
    out = [
        "### (c) Search",
        "",
        f"{s['n_queries']} queries, {s['n_judgments']:,} graded judgments (2 = exactly what was asked, "
        "1 = right kind of product, 0 = not relevant). Every listing in every system's top 10 is judged. "
        "Every query has at least "
        f"{s['relevant_per_query_min']} relevant listings, so recall@10 here equals precision@10.",
        "",
        "| System | nDCG@10 | recall@10 (grade ≥ 1) | recall@10 (grade 2) | median latency |",
        "|---|---|---|---|---|",
    ]
    for key, r in s["systems"].items():
        name = SYSTEM_NAMES.get(key, key)
        if key == s["best_new_system"]:
            name = f"**{name}**"
        lat = "n/a" if r["median_latency_ms"] is None else f"{r['median_latency_ms']:.1f} ms"
        out.append(
            f"| {name} | {_p(r['ndcg@10'])} {_ci(r['ndcg@10_ci95'])} | "
            f"{_p(r['recall@10'])} {_ci(r['recall@10_ci95'])} | {_p(r['recall@10_grade2'])} | {lat} |"
        )
    out += [
        "",
        f"The best new system beats the 2022 baseline on {s['queries_where_best_beats_baseline']} "
        f"of {s['n_queries']} queries.",
    ]
    return "\n".join(out)


def render(metrics: dict) -> str:
    parts = []
    if "matching" in metrics:
        parts.append(matching_section(metrics["matching"]))
    if "extraction" in metrics:
        parts.append(extraction_section(metrics["extraction"]))
    if "search" in metrics:
        parts.append(search_section(metrics["search"]))
    return "\n\n".join(parts) + "\n"


REVIEW_START, REVIEW_END = "<!-- review:start -->", "<!-- review:end -->"


def review_section(review: dict | None) -> str:
    """Human review of the AI labels: results if reports/review.json exists, else 'pending'."""
    if review is None:
        return (
            "**Status: pending.** No person has reviewed the labels yet. To review: run "
            "`pm review-sample`, fill the `my_label` column of `labels/review_pairs.csv` with 1 "
            "(same product) or 0 (different) without looking at `labels/pairs.csv`, then run "
            "`pm review-agreement`.\n"
        )

    def row(name: str, b: dict) -> str:
        if not b["n"]:
            return f"| {name} | 0 | n/a | n/a |"
        kappa = "n/a" if b["cohen_kappa"] is None else f"{b['cohen_kappa']:.3f}"
        return f"| {name} | {b['n']} | {_pct(b['agreement'])} ({b['agree']}/{b['n']}) | {kappa} |"

    ids = review["disagreeing_pair_ids"]
    return (
        "\n".join(
            [
                f"A person reviewed {review['n_reviewed']} of the {review['n_in_file']} pairs in "
                "`labels/review_pairs.csv` without seeing the AI labels.",
                "",
                "| Subset | Pairs reviewed | Agreement with AI label | Cohen's kappa |",
                "|---|---|---|---|",
                row("All reviewed", review["overall"]),
                row("Flagged `unsure` by the AI", review["unsure"]),
                row("Test split", review["test"]),
                "",
                f"Disagreeing pair_ids ({len(ids)}): {', '.join(map(str, ids)) if ids else 'none'}.",
            ]
        )
        + "\n"
    )


def _replace_block(text: str, start: str, end: str, body: str) -> str:
    if start not in text or end not in text:
        return text
    return re.sub(
        re.escape(start) + r".*?" + re.escape(end),
        lambda _: f"{start}\n{body}{end}",
        text,
        flags=re.S,
    )


def write_reports(reports_dir: Path = REPORTS_DIR, readme: Path | None = None) -> str:
    """Render results.md from metrics.json (if present) and refresh the README's generated blocks."""
    readme = readme if readme is not None else ROOT / "README.md"
    md = ""
    metrics_path = reports_dir / "metrics.json"
    if metrics_path.exists():
        md = render(json.loads(metrics_path.read_text()))
        (reports_dir / "results.md").write_text(
            "# Results\n\n_Generated by `pm evaluate` from reports/metrics.json. "
            "Do not edit by hand._\n\n" + md
        )
    review_path = reports_dir / "review.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else None
    if readme.exists():
        text = readme.read_text()
        if md:
            text = _replace_block(text, START, END, md)
        text = _replace_block(text, REVIEW_START, REVIEW_END, review_section(review))
        readme.write_text(text)
    return md
