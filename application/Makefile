.PHONY: run install clean

run:
	@uv run application.py

install:
	@uv sync

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
