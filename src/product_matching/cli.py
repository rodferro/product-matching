"""Command-line entry point: `pm <command>` (or `python -m product_matching.cli`)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import ARTIFACTS_DIR, DATA_PATH, LABELS_DIR, REPORTS_DIR, ROOT, set_seeds

log = logging.getLogger("product_matching")


def _load_world():
    from .candidates import char_vectorizer, generate_candidates
    from .data import load_products
    from .embeddings import Encoder

    df = load_products(DATA_PATH)
    emb = Encoder().encode_catalog(df["nome"].tolist())
    char_matrix = char_vectorizer().fit_transform(df["nome_norm"])
    cands = generate_candidates(df["nome_norm"], emb, char_matrix=char_matrix)
    return df, emb, cands


def cmd_sample_labels(args: argparse.Namespace) -> None:
    """Write unlabelled samples. Refuses to overwrite existing label files."""
    from .labeling import sample_extraction_check, sample_pairs
    from .splits import extraction_test_mask

    df, _, cands = _load_world()
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    targets = {
        "pairs.csv": lambda: sample_pairs(df, cands),
        "extraction_check.csv": lambda: sample_extraction_check(
            df, extraction_test_mask(df["nome_norm"])
        ),
    }
    for name, make in targets.items():
        path = LABELS_DIR / name
        if path.exists() and not args.force:
            log.warning("%s exists, skipping (use --force to resample and lose labels)", path)
            continue
        make().to_csv(path, index=False)
        log.info("wrote %s", path)


def cmd_pool_queries(args: argparse.Namespace) -> None:
    from .search import write_judgment_pool

    write_judgment_pool(force=args.force)


def cmd_build(args: argparse.Namespace) -> None:
    from .service import build_all

    build_all(ARTIFACTS_DIR)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from .evaluate import run_all

    run_all()


def cmd_review_sample(args: argparse.Namespace) -> None:
    """Blind review file: all `unsure` pairs + 100 random test pairs, without the AI labels."""
    from .data import load_products
    from .labeling import read_labels, review_sample
    from .splits import pair_test_mask

    out = Path(args.labels_dir) / "review_pairs.csv"
    if out.exists() and not args.force:
        log.warning("%s exists, not overwriting (use --force to discard filled-in labels)", out)
        return
    df = load_products(args.data)
    pairs = read_labels("pairs.csv", Path(args.labels_dir))
    is_test = pair_test_mask(
        df["nome_norm"], pairs["id_a"].astype(int).to_numpy(), pairs["id_b"].astype(int).to_numpy()
    )
    sample = review_sample(pairs, is_test)
    sample.to_csv(out, index=False)
    log.info("wrote %s (%d pairs); fill `my_label` with 0/1", out, len(sample))


def cmd_review_agreement(args: argparse.Namespace) -> None:
    """Compare labels/review_pairs.csv (filled in by a person) with the AI labels."""
    from .labeling import read_labels, review_agreement
    from .report import write_reports

    labels_dir = Path(args.labels_dir)
    result = review_agreement(
        read_labels("review_pairs.csv", labels_dir), read_labels("pairs.csv", labels_dir)
    )
    if result["n_reviewed"] == 0:
        log.warning("no rows with my_label = 0 or 1; nothing written")
        return
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "review.json").write_text(json.dumps(result, indent=2))
    write_reports(reports_dir, Path(args.readme))
    o = result["overall"]
    print(
        f"reviewed {result['n_reviewed']}: agreement {o['agreement']:.1%}, kappa {o['cohen_kappa']}"
    )
    print("disagreeing pair_ids:", result["disagreeing_pair_ids"])


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("product_matching.api:app", host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seeds()
    parser = argparse.ArgumentParser(prog="pm", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sample-labels", help="write unlabelled pair / extraction samples")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_sample_labels)

    p = sub.add_parser("pool-queries", help="write the search judgment pool for labels/queries.csv")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_pool_queries)

    p = sub.add_parser("build", help="train all models and indexes into artifacts/")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("evaluate", help="evaluate baseline vs new and write reports/")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("review-sample", help="write labels/review_pairs.csv for blind review")
    p.add_argument("--force", action="store_true")
    p.add_argument("--data", default=str(DATA_PATH))
    p.add_argument("--labels-dir", default=str(LABELS_DIR))
    p.set_defaults(func=cmd_review_sample)

    p = sub.add_parser("review-agreement", help="agreement of a person's labels with the AI labels")
    p.add_argument("--labels-dir", default=str(LABELS_DIR))
    p.add_argument("--reports-dir", default=str(REPORTS_DIR))
    p.add_argument("--readme", default=str(ROOT / "README.md"))
    p.set_defaults(func=cmd_review_agreement)

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
