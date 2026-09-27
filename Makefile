PY ?= .venv/bin/python
PORT ?= 8080

.PHONY: install test run compile score bench

install:
	python3 -m venv .venv
	$(PY) -m pip install -r requirements-dev.txt

test:
	$(PY) -m pytest -q

run:
	$(PY) -m uvicorn app.main:app --host 0.0.0.0 --port $(PORT)

compile:
	$(PY) -m compiler.compile

score:
	$(PY) -m eval.metrics

bench:
	$(PY) -m eval.judge_sim --url http://localhost:$(PORT)
