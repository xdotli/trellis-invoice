VENV_DIR=benchmarkthing

venv:
	python3 -m venv $(VENV_DIR)

install: venv
	. $(VENV_DIR)/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt

clean:
	rm -rf $(VENV_DIR)

lint:
	. $(VENV_DIR)/bin/activate && \
	ruff check . --fix --verbose

check: lint

run:
	. $(VENV_DIR)/bin/activate && python main.py
