"""End-to-end smoke test: build every component from a small sample and call the API.

Uses the real embedding model (downloaded on first run), so it takes ~1 minute.
Skip with `pytest -m "not slow"`.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from product_matching.config import DATA_PATH, LABELS_DIR, SEED

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    """~1,000 listings: everything referenced by 300 labelled pairs, plus random extras."""
    tmp = tmp_path_factory.mktemp("e2e")
    raw = pd.read_csv(DATA_PATH, sep="\t", dtype=str, keep_default_na=False)
    pairs = pd.read_csv(LABELS_DIR / "pairs.csv", dtype=str, keep_default_na=False)
    pos, neg = pairs[pairs["label"] == "1"], pairs[pairs["label"] == "0"]
    pairs = pd.concat([pos, neg.sample(n=300 - len(pos), random_state=SEED)])
    ids = set(pairs["id_a"].astype(int)) | set(pairs["id_b"].astype(int))
    rng = np.random.default_rng(SEED)
    extra = rng.choice(len(raw), 1000 - len(ids), replace=False)
    keep = sorted(ids | {int(i) for i in extra})
    remap = {old: new for new, old in enumerate(keep)}
    (tmp / "labels").mkdir()
    raw.iloc[keep].to_csv(tmp / "sample.csv", sep="\t", index=False)
    pairs = pairs.assign(
        id_a=pairs["id_a"].astype(int).map(remap), id_b=pairs["id_b"].astype(int).map(remap)
    )
    pairs.to_csv(tmp / "labels" / "pairs.csv", index=False)
    return tmp


@pytest.fixture(scope="module")
def client(sample):
    from product_matching.api import app, get_service
    from product_matching.service import Service

    service = Service.build(sample / "sample.csv", sample / "labels", cache_dir=sample / "cache")
    service.save(sample / "artifacts")
    service = Service.load(sample / "artifacts")  # also exercises save/load
    app.dependency_overrides[get_service] = lambda: service
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_match_end_to_end(client):
    same = client.post("/match", json={
        "a": {"nome": "Refrigerador Esmaltec ROC30 251 Litros Branco"},
        "b": {"nome": "Refrigerador 251 Litros Cycle Defrost ROC30 Branco 110v - Esmaltec"},
    }).json()  # fmt: skip
    different = client.post("/match", json={
        "a": {"nome": "Refrigerador Esmaltec ROC30 251 Litros Branco"},
        "b": {"nome": "Jogo PS3 Fallout 3 Bethesda"},
    }).json()  # fmt: skip
    assert 0.0 <= different["probability"] < same["probability"] <= 1.0
    assert different["same_product"] is False


def test_extract_end_to_end(client):
    data = client.post("/extract", json={"nome": "Câmera Digital Samsung ES70 12.2MP Preta"}).json()
    assert data["cor"]["value"] == "Preto"
    assert all(data[f]["value"] is None or isinstance(data[f]["value"], str) for f in data)


def test_search_end_to_end(client):
    data = client.get("/search", params={"q": "refrigerador", "k": 5}).json()
    assert len(data["results"]) == 5
    assert any("refrigerador" in r["nome"].lower() for r in data["results"])
