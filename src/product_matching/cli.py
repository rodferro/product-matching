"""Command-line entry point: `pm <command>` (or `python -m product_matching.cli`)."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import ARTIFACTS_DIR, DATA_PATH, LABELS_DIR, set_seeds

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

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
