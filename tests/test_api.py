"""API contract tests. The models are replaced by a fake, so these run in milliseconds."""

import pytest
from fastapi.testclient import TestClient

from product_matching.api import app, get_service

FIELDS = ["tipo", "categoria", "marca", "cor", "modelo"]


class FakeService:
    def __init__(self):
        self.calls = []

    def match(self, a, b):
        self.calls.append(("match", a, b))
        same = a["nome"].lower() == b["nome"].lower()
        return {"probability": 0.97 if same else 0.02, "same_product": same, "threshold": 0.4}

    def extract(self, nome):
        self.calls.append(("extract", nome))
        out = {f: {"value": None, "confidence": 0.1} for f in FIELDS}
        out["marca"] = {"value": "Sony", "confidence": 0.99}
        out["cor"] = {"value": None, "confidence": None}
        return out

    def search(self, query, k=10):
        self.calls.append(("search", query, k))
        hits = [{"id": i, "nome": f"Notebook Sony {i}", "tipo": "Notebook", "marca": None,
                 "score": 1.0 / (i + 1)} for i in range(50)]  # fmt: skip
        return hits[:k]


@pytest.fixture
def client():
    fake = FakeService()
    app.dependency_overrides[get_service] = lambda: fake
    with TestClient(app) as c:
        c.fake = fake
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_match_contract(client):
    body = {"a": {"nome": "Polar FT60M"}, "b": {"nome": "polar ft60m", "marca": "Polar"}}
    r = client.post("/match", json=body)
    assert r.status_code == 200
    data = r.json()
    assert set(data) == {"probability", "same_product", "threshold"}
    assert 0.0 <= data["probability"] <= 1.0
    assert data["same_product"] is True
    # optional brand is passed through, missing brand arrives as None
    _, a, b = client.fake.calls[-1]
    assert a == {"nome": "Polar FT60M", "marca": None}
    assert b["marca"] == "Polar"


@pytest.mark.parametrize(
    "body",
    [
        {"a": {"nome": "x"}},  # missing b
        {"a": {"nome": ""}, "b": {"nome": "x"}},  # empty name
        {"a": {"marca": "Sony"}, "b": {"nome": "x"}},  # missing nome
        {"a": {"nome": "x" * 501}, "b": {"nome": "x"}},  # too long
    ],
)
def test_match_rejects_bad_input(client, body):
    assert client.post("/match", json=body).status_code == 422


def test_extract_contract_returns_every_field_with_none_allowed(client):
    r = client.post("/extract", json={"nome": "Notebook Sony Vaio 4GB"})
    assert r.status_code == 200
    data = r.json()
    assert set(data) == set(FIELDS)
    for f in FIELDS:
        assert set(data[f]) == {"value", "confidence"}
    assert data["marca"]["value"] == "Sony"
    assert data["tipo"]["value"] is None  # abstention is serialized as null
    assert data["cor"]["confidence"] is None


def test_extract_requires_name(client):
    assert client.post("/extract", json={}).status_code == 422
    assert client.post("/extract", json={"nome": ""}).status_code == 422


def test_search_contract(client):
    r = client.get("/search", params={"q": "notebook sony 4 GB", "k": 3})
    assert r.status_code == 200
    data = r.json()
    assert data["query"] == "notebook sony 4 GB"
    assert len(data["results"]) == 3
    assert set(data["results"][0]) == {"id", "nome", "tipo", "marca", "score"}
    assert client.fake.calls[-1] == ("search", "notebook sony 4 GB", 3)


def test_search_default_k_is_10(client):
    assert len(client.get("/search", params={"q": "tv"}).json()["results"]) == 10


@pytest.mark.parametrize("params", [{}, {"q": ""}, {"q": "tv", "k": 0}, {"q": "tv", "k": 51}])
def test_search_rejects_bad_input(client, params):
    assert client.get("/search", params=params).status_code == 422
