.PHONY: install test run clean

install:
	python3 -m venv venv
	. venv/bin/activate && python -m pip install --upgrade pip wheel
	. venv/bin/activate && python -m pip install -r requirements.txt

test:
	PYTHONPATH=. pytest tests/ -q

run:
	@if [ -z "$(PDF)" ]; then echo "Usage: make run PDF=inputs/file.pdf"; exit 2; fi
	scripts/run_pdf.sh "$(PDF)"

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
