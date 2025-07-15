# EdgeGuard Python Bindings

Python bindings for EdgeGuard, a physics-aware data validation and aggregation library for IoT edge devices.

## Installation

### From PyPI (coming soon)

```bash
pip install edgeguard
```

### Development Installation

1. Install Rust and Python dependencies:
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin
```

2. Build and install the development version:
```bash
cd edgeguard-python
maturin develop
```

## Usage

```python
import edgeguard

# Create a temperature validator
validator = edgeguard.TemperatureValidator(min_temp=-20.0, max_temp=60.0)

# Validate sensor readings
result = validator.validate(25.5)
print(f"Valid: {result.is_valid}, Value: {result.value}")
```

## Development

### Building

```bash
# Build for development
maturin develop

# Build release wheels
maturin build --release
```

### Testing

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests
pytest python/tests/
```

## Features

- **High Performance**: Native Rust implementation with Python ergonomics
- **Physics-Aware**: Built-in understanding of sensor physics and constraints
- **Type Safety**: Full Python type hints and runtime validation
- **Cross-Platform**: Works on Linux, macOS, and Windows
- **Memory Efficient**: Minimal overhead for embedded/edge deployments

## License

This project is licensed under the MIT License - see the LICENSE file for details.