"""HTTP API.

    POST /match     two listings -> probability that they are the same product
    POST /extract   a product name -> type, category, brand, color, model (None when unsure)
    GET  /search    ?q=notebook sony 4 GB&k=10 -> most relevant listings
    GET  /health

Run with `pm serve` (after `pm build`) or `uvicorn product_matching.api:app`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated, Protocol

from fastapi import Depends, FastAPI, Query
from pydantic import BaseModel, Field

from . import __version__


class Listing(BaseModel):
    nome: str = Field(
        min_length=1, max_length=500, description="Product name as entered by the seller"
    )
    marca: str | None = Field(default=None, max_length=200, description="Brand, if known")

    model_config = {
        "json_schema_extra": {"examples": [{"nome": "Refrigerador Consul CRM33 263 Litros Inox"}]}
    }


class MatchRequest(BaseModel):
    a: Listing
    b: Listing


class MatchResponse(BaseModel):
    probability: float = Field(ge=0.0, le=1.0)
    same_product: bool
    threshold: float


class ExtractRequest(BaseModel):
    nome: str = Field(min_length=1, max_length=500)


class FieldValue(BaseModel):
    value: str | None
    confidence: float | None = Field(
        description="Model confidence for the top value; null for rule-based fields (cor)"
    )


class ExtractResponse(BaseModel):
    tipo: FieldValue
    categoria: FieldValue
    marca: FieldValue
    cor: FieldValue
    modelo: FieldValue


class SearchHit(BaseModel):
    id: int
    nome: str
    tipo: str | None
    marca: str | None
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]


class ServiceLike(Protocol):
    def match(self, a: dict, b: dict) -> dict: ...
    def extract(self, nome: str) -> dict: ...
    def search(self, query: str, k: int = 10) -> list[dict]: ...


@lru_cache(maxsize=1)
def get_service() -> ServiceLike:
    from .service import Service

    return Service.load()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models at startup so the first request is not slow. Tests override get_service.
    if get_service not in app.dependency_overrides:
        service = get_service()
        service.search("aquecimento", k=1)  # warm up the embedding model
    yield


app = FastAPI(
    title="Product matching",
    version=__version__,
    description="Same-product probability, attribute extraction and search for marketplace listings.",
    lifespan=lifespan,
)

ServiceDep = Annotated[ServiceLike, Depends(get_service)]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest, service: ServiceDep) -> dict:
    return service.match(req.a.model_dump(), req.b.model_dump())


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest, service: ServiceDep) -> dict:
    return service.extract(req.nome)


@app.get("/search", response_model=SearchResponse)
def search(
    service: ServiceDep,
    q: Annotated[str, Query(min_length=1, max_length=200, description="Search text")],
    k: Annotated[int, Query(ge=1, le=50)] = 10,
) -> dict:
    return {"query": q, "results": service.search(q, k)}
