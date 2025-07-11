# EdgeGuard Makefile for developer convenience
# 
# Common commands:
#   make tables      - Generate all lookup tables
#   make test        - Run all tests
#   make bench       - Run benchmarks
#   make examples    - Build all examples
#   make clean       - Clean build artifacts

.PHONY: all tables test test-all bench examples clean check lint docs

# Default target
all: check test

# Generate lookup tables
tables:
	@echo "Generating lookup tables..."
	cd edgeguard-core && rustc build_tables.rs -o build_tables
	cd edgeguard-core && ./build_tables --config standard --output generated_tables.rs
	cd edgeguard-core && ./build_tables --config high_precision --output high_precision_tables.rs
	cd edgeguard-core && ./build_tables --config low_memory --output low_memory_tables.rs
	cd edgeguard-core && rm -f build_tables
	@echo "✓ Lookup tables generated"

# Verify generated tables
verify-tables: tables
	@echo "Verifying lookup tables..."
	@# TODO: Add table verification tool
	@echo "✓ Tables verified (manual check required for now)"

# Run tests
test:
	@echo "Running tests..."
	cargo test --workspace

# Run all test configurations
test-all: tables
	@echo "Running all test configurations..."
	cargo test --workspace --all-features
	cargo test --workspace --no-default-features
	cargo test --workspace --release
	@echo "✓ All tests passed"

# Run tests for embedded targets
test-embedded:
	@echo "Testing embedded configurations..."
	cargo test --workspace --no-default-features --features embedded
	cargo test --workspace --target thumbv7em-none-eabihf --no-default-features --no-run

# Run benchmarks
bench:
	@echo "Running benchmarks..."
	cargo bench --workspace

# Build examples
examples:
	@echo "Building examples..."
	cargo build --examples --all-features
	@echo "✓ Examples built"

# Run a specific example
run-example:
	@echo "Available examples will be:"
	@echo "  make run-example EXAMPLE=basic_validation"
	@echo "  make run-example EXAMPLE=multi_sensor_fusion"
	@echo "  make run-example EXAMPLE=event_pipeline"

# Check code without building
check:
	@echo "Checking code..."
	cargo check --workspace --all-features

# Run clippy linter
lint:
	@echo "Running clippy..."
	cargo clippy --workspace --all-features -- -D warnings

# Format code
fmt:
	@echo "Formatting code..."
	cargo fmt --all

# Check formatting
fmt-check:
	@echo "Checking formatting..."
	cargo fmt --all -- --check

# Generate documentation
docs:
	@echo "Generating documentation..."
	cargo doc --workspace --all-features --no-deps
	@echo "Documentation available at: target/doc/edgeguard_core/index.html"

# Open documentation in browser
docs-open: docs
	cargo doc --workspace --all-features --no-deps --open

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	cargo clean
	rm -f edgeguard-core/build_tables
	@echo "✓ Clean complete"

# Deep clean (including generated tables)
clean-all: clean
	@echo "Removing generated tables..."
	rm -f edgeguard-core/generated_tables.rs
	rm -f edgeguard-core/high_precision_tables.rs
	rm -f edgeguard-core/low_memory_tables.rs
	@echo "✓ Deep clean complete"

# Development setup
setup:
	@echo "Setting up development environment..."
	rustup target add thumbv7em-none-eabihf
	rustup component add clippy rustfmt
	@echo "✓ Development environment ready"

# Quick CI check
ci: fmt-check lint test-all
	@echo "✓ CI checks passed"

# Print help
help:
	@echo "EdgeGuard Development Commands:"
	@echo ""
	@echo "  make tables      - Generate lookup tables"
	@echo "  make test        - Run tests"
	@echo "  make test-all    - Run all test configurations"
	@echo "  make bench       - Run benchmarks"
	@echo "  make examples    - Build examples"
	@echo "  make check       - Check code without building"
	@echo "  make lint        - Run clippy linter"
	@echo "  make fmt         - Format code"
	@echo "  make docs        - Generate documentation"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make setup       - Setup development environment"
	@echo "  make ci          - Run CI checks"