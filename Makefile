PYTHON ?= .venv/bin/python
BOOTSTRAP_PYTHON ?= python3

.PHONY: setup test eval eval-safety run studio
setup:
	$(BOOTSTRAP_PYTHON) -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

test:
	$(PYTHON) -m pytest -q

eval:
	$(PYTHON) -m evals.run $(EVAL_ARGS)

eval-safety:
	$(PYTHON) -m evals.run --suite safety $(EVAL_ARGS)

run:
	$(PYTHON) -m streamlit run app.py

studio:
	LANGGRAPH_DISABLE_FILE_PERSISTENCE=true .venv/bin/langgraph dev --no-reload
