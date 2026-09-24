# Development tasks for autonomousim. Cargo lives in ~/.cargo/bin, which is not always on PATH.
export PATH := $(HOME)/.cargo/bin:$(PATH)

.PHONY: check fmt test test-rust test-py test-viewer bench dev-py train-deps viewer clean fixtures-mfeval fixtures-chrono

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

# Oracles outside the repo (see docs/PLAN.md, M2 step 0): Julia + MFeval.jl.
ORACLES ?= $(HOME)/.local/share/autonomousim-oracles
JULIA ?= $(ORACLES)/julia-1.13.0/bin/julia
MFEVAL_JL ?= $(ORACLES)/MFeval_julia
MFEVAL_TYRES := fixtures/tir/MagicFormula52_Parameters.tir fixtures/tir/MagicFormula61_Parameters.tir \
	assets/tires/HMMWV_Pac02Tire.tir assets/tires/Sedan_Pac02Tire.tir

fixtures-mfeval:  ## Magic Formula reference values from MFeval.jl (fixtures/mfeval/)
	mkdir -p fixtures/mfeval
	cargo build -p autonomousim-vehicles --example tir_canonical
	for t in $(MFEVAL_TYRES); do \
		n=$$(basename $$t .tir); \
		target/debug/examples/tir_canonical $$t fixtures/mfeval/$$n.tir && \
		$(JULIA) --project=$(MFEVAL_JL) tools/gen_mfeval_fixtures.jl fixtures/mfeval/$$n.tir $$n fixtures/mfeval/$$n.json || exit 1; \
	done

MICROMAMBA ?= $(HOME)/.local/bin/micromamba
MAMBA_ROOT_PREFIX ?= $(HOME)/.local/share/micromamba
export MAMBA_ROOT_PREFIX

fixtures-chrono:  ## PAC2002, full-vehicle and handling reference runs from Project Chrono (fixtures/chrono/)
	$(MICROMAMBA) run -n chrono python tools/gen_chrono_fixtures.py
	$(MICROMAMBA) run -n chrono python tools/gen_chrono_vehicle_fixtures.py
	$(MICROMAMBA) run -n chrono python tools/gen_chrono_handling_fixtures.py

clean:
	cargo clean
