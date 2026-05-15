.PHONY: build test check clean run dev-test

build:
	cargo build --release

dev:
	cargo build

test:
	cargo test -- --test-threads=1

dev-test:
	cargo test -- --test-threads=1 --nocapture

check:
	cargo check

clippy:
	cargo clippy -- -D warnings

clean:
	cargo clean

run:
	cargo run

check-st:
	cargo run -- check-st $(file)
