.PHONY: install build preview check clean

install:
	uv sync --group book

build:
	cd book && uv run jupyter book build --html

preview:
	cd book && uv run jupyter book start

check:
	cd book && uv run jupyter book build --execute --html

clean:
	uv run jupyter book clean book
