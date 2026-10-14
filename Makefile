.PHONY: help install lint static-check test demo clean

help:
	@echo "Available make targets:"
	@echo "  install      - Install pinned dependencies from requirements.txt"
	@echo "  lint         - Run ruff linter checks"
	@echo "  static-check - Run static analysis and AST checks across all Python files"
	@echo "  test         - Run pytest test suite across assignments 1-4"
	@echo "  demo         - Run the complete end-to-end MLOps demo script"
	@echo "  clean        - Remove temporary files, caches, and test artifacts"

install:
	pip install -r requirements.txt

lint:
	ruff check .

static-check:
	ruff check .
	python -m compileall -q src tests
	python -m py_compile demo.py streamlit_app.py

test:
	pytest tests/ -v

demo:
	python demo.py

clean:
	rm -rf .pytest_cache __pycache__ src/**/__pycache__ tests/__pycache__ mlruns model_package candidate_package registry.json
