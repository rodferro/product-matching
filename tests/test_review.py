"""`pm review-sample` and `pm review-agreement` on a tiny fixture."""

import json

import pandas as pd
import pytest

from product_matching.cli import main
from product_matching.labeling import REVIEW_COLUMNS, review_agreement, review_sample
from product_matching.report import review_section

NAMES = [
    "Geladeira Alfa", "Geladeira Beta", "Fogao Gama", "Fogao Delta", "Notebook Epsilon",
    "Notebook Zeta", "Relogio Eta", "Relogio Theta", "Cadeira Iota", "Cadeira Kappa",
    "Panela Lambda", "Panela Mu",
]  # fmt: skip
# pair_id, id_a, id_b, label, unsure
PAIRS = [
    (1, 0, 1, 1, ""), (2, 2, 3, 0, "1"), (3, 4, 5, 1, "1"), (4, 6, 7, 0, ""),
    (5, 8, 9, 0, ""), (6, 10, 11, 1, ""), (7, 0, 2, 0, "1"), (8, 5, 7, 0, ""),
]  # fmt: skip


@pytest.fixture
def fixture(tmp_path):
    data = tmp_path / "products.csv"
    pd.DataFrame(
        {
            "nome": NAMES,
            "tipo": "None",
            "marca": "Marca",
            "categoria": "None",
            "cor": "None",
            "modelo": "None",
        }  # fmt: skip
    ).to_csv(data, sep="\t", index=False)
    labels = tmp_path / "labels"
    labels.mkdir()
    rows = [
        {
            "pair_id": p,
            "id_a": a,
            "id_b": b,
            "source": "code",
            "nome_a": NAMES[a],
            "nome_b": NAMES[b],
            "marca_a": "Marca",
            "marca_b": "Marca",
            "label": lab,
            "unsure": u,
            "note": "hard" if u else "",
        }  # fmt: skip
        for p, a, b, lab, u in PAIRS
    ]
    pd.DataFrame(rows).to_csv(labels / "pairs.csv", index=False)
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- review:start -->\nold\n<!-- review:end -->\nrest\n")
    return tmp_path


def test_review_sample_is_blind_and_complete(fixture):
    main(["review-sample", "--data", str(fixture / "products.csv"),
          "--labels-dir", str(fixture / "labels")])  # fmt: skip
    out = pd.read_csv(fixture / "labels" / "review_pairs.csv", dtype=str, keep_default_na=False)
    assert list(out.columns) == REVIEW_COLUMNS
    assert "label" not in out.columns and "unsure" not in out.columns
    assert (out["my_label"] == "").all()
    assert {"2", "3", "7"} <= set(out["pair_id"])  # every unsure pair is included
    assert set(out["split"]) <= {"train", "test"}
    extra = out[~out["pair_id"].isin(["2", "3", "7"])]
    assert len(extra) > 0  # the fixture has test pairs, so this check is not vacuous
    assert (extra["split"] == "test").all()  # the random extras come from the test split


def test_review_sample_is_deterministic_and_does_not_overwrite(fixture):
    args = ["review-sample", "--data", str(fixture / "products.csv"),
            "--labels-dir", str(fixture / "labels")]  # fmt: skip
    main(args)
    path = fixture / "labels" / "review_pairs.csv"
    first = path.read_text()
    path.write_text(first.replace(",\n", ",1\n"))  # pretend someone filled it in
    main(args)
    assert path.read_text() != first  # not overwritten without --force
    main([*args, "--force"])
    assert path.read_text() == first  # same seed -> same sample


def test_review_sample_caps_test_pairs():
    pairs = pd.DataFrame({"pair_id": range(1, 7), "source": "code", "nome_a": "a", "nome_b": "b",
                          "marca_a": "", "marca_b": "", "note": "", "label": "0",
                          "unsure": ["1", "", "", "", "", ""]})  # fmt: skip
    is_test = pd.Series([False, True, True, True, False, True]).to_numpy()
    out = review_sample(pairs, is_test, n_test=2)
    assert len(out) == 3 and 1 in set(out["pair_id"])
    assert (out.loc[out["pair_id"] != 1, "split"] == "test").all()


def test_review_agreement_by_hand():
    pairs = pd.DataFrame({"pair_id": ["1", "2", "3", "4", "5"], "label": ["1", "1", "0", "0", "1"],
                          "unsure": ["", "1", "1", "", ""]})  # fmt: skip
    review = pd.DataFrame({"pair_id": ["1", "2", "3", "4", "5"],
                           "split": ["test", "test", "train", "test", "train"],
                           "my_label": ["1", "0", "0", "0", ""]})  # fmt: skip
    r = review_agreement(review, pairs)
    assert r["n_in_file"] == 5 and r["n_reviewed"] == 4  # the empty row is skipped
    # AI 1,1,0,0 vs person 1,0,0,0: agreement 3/4; kappa = (0.75 - 0.5) / (1 - 0.5) = 0.5
    assert r["overall"] == {"n": 4, "agree": 3, "agreement": 0.75, "cohen_kappa": 0.5}
    assert r["unsure"]["n"] == 2 and r["unsure"]["agree"] == 1
    assert r["test"]["n"] == 3 and r["test"]["agree"] == 2
    assert r["disagreeing_pair_ids"] == [2]
    assert r["confusion"] == {"ai0_human0": 2, "ai0_human1": 0, "ai1_human0": 1, "ai1_human1": 1}


def test_review_agreement_command_writes_json_and_readme(fixture):
    main(["review-sample", "--data", str(fixture / "products.csv"),
          "--labels-dir", str(fixture / "labels")])  # fmt: skip
    path = fixture / "labels" / "review_pairs.csv"
    review = pd.read_csv(path, dtype=str, keep_default_na=False)
    ai = {str(p): str(lab) for p, _, _, lab, _ in PAIRS}
    review["my_label"] = [ai[p] if p != "3" else str(1 - int(ai[p])) for p in review["pair_id"]]
    review.to_csv(path, index=False)
    main(["review-agreement", "--labels-dir", str(fixture / "labels"),
          "--reports-dir", str(fixture / "reports"), "--readme", str(fixture / "README.md")])  # fmt: skip
    result = json.loads((fixture / "reports" / "review.json").read_text())
    assert result["disagreeing_pair_ids"] == [3]
    assert result["overall"]["agree"] == result["overall"]["n"] - 1
    readme = (fixture / "README.md").read_text()
    assert "pending" not in readme and "Disagreeing pair_ids (1): 3." in readme
    assert readme.startswith("intro\n") and readme.endswith("rest\n")


def test_readme_says_pending_without_review():
    assert "pending" in review_section(None)
