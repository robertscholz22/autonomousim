# Development tasks for autonomousim. Cargo lives in ~/.cargo/bin, which is not always on PATH.
export PATH := $(HOME)/.cargo/bin:$(PATH)

.PHONY: check fmt test test-rust test-py test-viewer bench dev-py train-deps viewer clean

check:            ## rustfmt + clippy (whole workspace, incl. viewer and bindings)
	cargo fmt --all --check
	cargo clippy --workspace --all-targets -- -D warnings

fmt:
	cargo fmt --all

test: test-rust test-py

test-rust:
	cargo test

test-viewer:      ## the viewer's headless tests (builds Bevy)
	cargo test -p autonomousim-viewer

test-py: dev-py
	uv run pytest -q

bench:
	cargo bench
	uv run python tools/collect_bench.py
	uv run python -m autonomousim.bench --save

dev-py:           ## (re)build the native extension into the uv venv (keeps other groups, e.g. train)
	uv sync --inexact

train-deps:       ## CPU torch + tensorboard for examples/
	uv sync --inexact --group train

viewer:
	cargo run -p autonomousim-viewer --release

clean:
	cargo clean
