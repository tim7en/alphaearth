.PHONY: setup run qa publish

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

run:
	python -m aeuz.orchestrator --run all

qa:
	python -m aeuz.orchestrator --run qa

publish:
	python -m aeuz.orchestrator --run publish
