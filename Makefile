.PHONY: all run report docker-run zip clean

PYTHON ?= python3

all: run

run:
	$(PYTHON) run_pipeline.py

report:
	cd report && pdflatex -interaction=nonstopmode technical_report.tex && pdflatex -interaction=nonstopmode technical_report.tex

docker-run:
	docker compose up --build

zip:
	cd .. && zip -r PES1UG24AM392.zip membership-inference-leakage -x "*/__pycache__/*" "*/.DS_Store"
	cd .. && zip -r PES1UG24AM401.zip membership-inference-leakage -x "*/__pycache__/*" "*/.DS_Store"

clean:
	rm -rf src/__pycache__ report/*.aux report/*.log report/*.out
