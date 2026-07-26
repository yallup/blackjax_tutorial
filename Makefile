.PHONY: install build preview clean

install:
	uv sync --group book

build:
	cd book && uv run jupyter book build --execute --html

preview:
	cd book && uv run jupyter book start --execute

clean:
	uv run jupyter book clean book
