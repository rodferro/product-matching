PY ?= .venv/bin/python
PM ?= .venv/bin/pm

.PHONY: install build evaluate test test-fast lint serve notebook docker docker-run all

install:            ## create .venv and install pinned dependencies
	python3.12 -m venv .venv
	.venv/bin/pip install -r requirements-dev.txt
	.venv/bin/pip install -e . --no-deps

build:              ## fit all models on the full data -> artifacts/service.joblib
	$(PM) build

evaluate:           ## score baseline vs new on held-out labels -> reports/
	$(PM) evaluate

test:               ## all tests (includes a ~1 min end-to-end test)
	.venv/bin/pytest

test-fast:          ## skip the end-to-end test
	.venv/bin/pytest -m "not slow"

lint:
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests

serve: build        ## run the API on http://127.0.0.1:8000
	$(PM) serve

notebook:           ## re-execute the analysis notebook in place
	.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb

docker:
	docker build -t product-matching .

docker-run:
	docker run --rm -p 8000:8000 product-matching

all: evaluate build test
