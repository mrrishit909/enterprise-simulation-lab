export DATABASE_URL ?= postgresql://app:app-dev-only@127.0.0.1:55400/app
PY = ./venv/bin/python
BASE = http://127.0.0.1:8400

bootstrap:      ## local virtualenv + database container
	python3 -m venv venv && ./venv/bin/pip install -q -r requirements-dev.txt
	docker compose up -d db
seed:           ## migrate and load the synthetic world (no-op if already seeded)
	$(PY) -m es.seed --if-empty
dev:            ## API with reload on :8400, jobs run by a local worker
	($(PY) -m core.jobs es.api &) && $(PY) -m uvicorn es.api:app --reload --port 8400
test:
	$(PY) -m pytest --cov=es --cov=core --cov-report=term-missing
demo:           ## the full stack in Docker, then the demo scenario against it
	docker compose up --build -d --wait && $(PY) -m core.scenario $(BASE)
load-test:      ## run `make demo` first
	$(PY) -m core.loadtest $(BASE) atlas-viewer-demo 32 10 "GET /v1/scenarios/1234c951-0ca8-5072-8c10-2e29c00a6685" "GET /v1/experiments/2504d70b-9032-5046-8ed9-d3342f143032/distributions" 
evaluate:       ## held-out businesses -> docs/evaluation.md (in the container, like the demo)
	docker compose run --rm -T test python -m es.evaluate > docs/evaluation.md
reset:          ## drop all data (volume included)
	docker compose down -v
.PHONY: bootstrap seed dev test demo load-test evaluate reset
