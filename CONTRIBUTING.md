# Contributing to EdgeGuard

Thank you for your interest in contributing to EdgeGuard! This guide will help you get started with contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Project Structure](#project-structure)
5. [Contributing Guidelines](#contributing-guidelines)
6. [Pull Request Process](#pull-request-process)
7. [Testing](#testing)
8. [Documentation](#documentation)
9. [Performance Considerations](#performance-considerations)
10. [Release Process](#release-process)

## Code of Conduct

This project adheres to the Rust Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- Rust 1.70.0 or later
- Git
- A GitHub account

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/edgeguard.git
   cd edgeguard
   ```

3. Set up the development environment:
   ```bash
   # Install dependencies
   cargo build
   
   # Run tests to ensure everything works
   cargo test
   
   # Run benchmarks
   ./run_benchmarks.sh
   ```

4. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Project Structure

```
edgeguard/
├── edgeguard-core/          # Core validation library
│   ├── src/
│   │   ├── validators/      # Physics-aware validators
│   │   ├── fusion/         # Sensor fusion algorithms
│   │   ├── pipeline/       # Event processing pipeline
│   │   ├── stream/         # Data streaming
│   │   └── ...
│   ├── examples/           # Usage examples
│   ├── benches/           # Performance benchmarks
│   └── tests/             # Integration tests
├── edgeguard-ml/           # Machine learning features
├── edgeguard-schemas/      # Avro schema support
├── edgeguard-connectors/   # Network connectors
├── examples/               # End-to-end examples
├── docs/                   # Documentation
└── scripts/               # Build and utility scripts
```

## Contributing Guidelines

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues in existing code
- **New features**: Add new functionality
- **Performance improvements**: Optimize existing code
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Examples**: Add practical usage examples

### Coding Standards

#### Rust Code Style

- Follow the official Rust style guide
- Use `cargo fmt` to format code
- Use `cargo clippy` to catch common mistakes
- Write idiomatic Rust code

#### Embedded Considerations

EdgeGuard targets embedded systems, so:

- **No heap allocations** in core algorithms
- **Use `heapless` collections** instead of `std::collections`
- **Fixed-size buffers** with const generics
- **Avoid panics** - use `Result` types
- **No unsafe code** without explicit justification

#### Performance Requirements

- **Validation**: <100μs per reading
- **Pipeline**: 10k+ events/sec on Cortex-M4
- **Memory**: <100KB for embedded configurations
- **No blocking operations** in core paths

### Code Examples

#### Adding a New Validator

```rust
use crate::validators::{Validator, ValidationError};

pub struct MyValidator {
    min_value: f32,
    max_value: f32,
}

impl MyValidator {
    pub fn new() -> Self {
        Self {
            min_value: 0.0,
            max_value: 100.0,
        }
    }
    
    pub fn with_range(mut self, min: f32, max: f32) -> Self {
        self.min_value = min;
        self.max_value = max;
        self
    }
}

impl Validator for MyValidator {
    type Value = f32;
    type Error = ValidationError;
    
    fn validate(&self, value: f32) -> Result<f32, ValidationError> {
        if value < self.min_value || value > self.max_value {
            Err(ValidationError::OutOfRange {
                value,
                min: self.min_value,
                max: self.max_value,
            })
        } else {
            Ok(value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_validator() {
        let validator = MyValidator::new().with_range(0.0, 50.0);
        assert!(validator.validate(25.0).is_ok());
        assert!(validator.validate(100.0).is_err());
    }
}
```

#### Adding a Pipeline Stage

```rust
use crate::pipeline::{PipelineStage, StageOutput, PipelineResult};
use crate::events::Event;

pub struct MyStage {
    threshold: f32,
}

impl MyStage {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl PipelineStage for MyStage {
    fn process(&mut self, event: Event, output: &mut StageOutput) -> PipelineResult<()> {
        // Process event and add to output
        output.push(event)?;
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "my_stage"
    }
    
    fn reset(&mut self) {
        // Reset internal state if needed
    }
}
```

### Documentation Standards

- **All public APIs** must have rustdoc comments
- **Include examples** in documentation
- **Link to related functions** using `[function_name]`
- **Document safety requirements** for any unsafe code
- **Include performance characteristics** where relevant

Example documentation:

```rust
/// Validates temperature readings using physics-based constraints.
/// 
/// This validator checks temperature values against configurable ranges
/// and rate limits based on thermal mass properties.
/// 
/// # Examples
/// 
/// ```rust
/// use edgeguard::validators::TemperatureValidator;
/// 
/// let validator = TemperatureValidator::new()
///     .with_range(-40.0, 85.0)
///     .with_rate_limit(5.0);
/// 
/// assert!(validator.validate(25.0).is_ok());
/// assert!(validator.validate(200.0).is_err());
/// ```
/// 
/// # Performance
/// 
/// Validation typically takes <50μs on Cortex-M4 processors.
pub struct TemperatureValidator {
    // ...
}
```

## Pull Request Process

### Before Submitting

1. **Run all tests**:
   ```bash
   cargo test --all-features
   ```

2. **Check formatting**:
   ```bash
   cargo fmt --all -- --check
   ```

3. **Run clippy**:
   ```bash
   cargo clippy --all-targets --all-features -- -D warnings
   ```

4. **Run benchmarks** (if performance-related):
   ```bash
   ./run_benchmarks.sh
   ```

5. **Update documentation** if needed:
   ```bash
   cargo doc --all-features
   ```

### PR Guidelines

- **Clear title** describing the change
- **Detailed description** explaining the motivation
- **Link to issue** if applicable
- **Breaking changes** clearly documented
- **Performance impact** noted if applicable

### PR Template

```markdown
## Summary

Brief description of the change and its purpose.

## Changes

- Added/modified/removed X
- Updated Y to support Z
- Fixed issue with A

## Testing

- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Performance benchmarks run (if applicable)
- [ ] Documentation updated

## Performance Impact

Describe any performance implications.

## Breaking Changes

List any breaking changes and migration guide.

## Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Performance benchmarks run
```

## Testing

### Test Categories

1. **Unit Tests**: Test individual functions and modules
2. **Integration Tests**: Test component interactions
3. **Property Tests**: Test with random inputs using `proptest`
4. **Benchmark Tests**: Performance and memory tests
5. **Embedded Tests**: Tests for no_std environments

### Writing Tests

#### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_functionality() {
        let validator = MyValidator::new();
        assert!(validator.validate(50.0).is_ok());
    }
    
    #[test]
    fn test_error_conditions() {
        let validator = MyValidator::new().with_range(0.0, 100.0);
        assert!(validator.validate(-10.0).is_err());
    }
}
```

#### Property Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_validator_range(value in -1000.0f32..1000.0f32) {
        let validator = MyValidator::new().with_range(0.0, 100.0);
        let result = validator.validate(value);
        
        if value >= 0.0 && value <= 100.0 {
            assert!(result.is_ok());
        } else {
            assert!(result.is_err());
        }
    }
}
```

#### Benchmark Tests

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_validator(c: &mut Criterion) {
    let validator = MyValidator::new();
    
    c.bench_function("validator_performance", |b| {
        b.iter(|| {
            validator.validate(black_box(25.0))
        })
    });
}

criterion_group!(benches, benchmark_validator);
criterion_main!(benches);
```

### Test Requirements

- **All new code** must have tests
- **Maintain >90% code coverage**
- **Test edge cases** and error conditions
- **Property tests** for validators and algorithms
- **Performance tests** for critical paths

## Documentation

### Types of Documentation

1. **API Documentation** (rustdoc)
2. **User Guide** (markdown)
3. **Examples** (runnable code)
4. **Architecture Documentation**
5. **Performance Guidelines**

### Writing Good Documentation

- **Start with purpose**: Why does this exist?
- **Show examples**: How to use it?
- **Explain constraints**: What are the limits?
- **Document performance**: How fast is it?
- **Link related items**: What else should users know?

### Documentation Checklist

- [ ] Public APIs have rustdoc comments
- [ ] Examples compile and run
- [ ] Performance characteristics documented
- [ ] Error conditions explained
- [ ] Related functions linked
- [ ] Safety requirements noted

## Performance Considerations

### Embedded Constraints

EdgeGuard targets resource-constrained devices:

- **ESP32**: 240MHz CPU, 520KB RAM, 4MB flash
- **Cortex-M4**: 168MHz CPU, 192KB RAM, 1MB flash
- **Raspberry Pi Zero**: 1GHz CPU, 512MB RAM

### Performance Requirements

- **Validation**: <100μs per reading
- **Pipeline**: 10k+ events/sec
- **Memory**: <100KB for basic configuration
- **Flash**: <500KB for embedded builds

### Optimization Guidelines

1. **Use fixed-size data structures**
2. **Avoid heap allocations**
3. **Pre-compute lookup tables**
4. **Use const generics for buffer sizes**
5. **Minimize copying large data**
6. **Use efficient algorithms**

### Performance Testing

```rust
#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn test_validation_performance() {
        let validator = MyValidator::new();
        let start = Instant::now();
        
        for i in 0..10000 {
            validator.validate(i as f32)?;
        }
        
        let duration = start.elapsed();
        assert!(duration < Duration::from_millis(100)); // <10μs per validation
    }
}
```

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. **Update version** in `Cargo.toml`
2. **Update CHANGELOG.md**
3. **Run full test suite**
4. **Run benchmarks** and verify performance
5. **Update documentation**
6. **Create release tag**
7. **Publish to crates.io**

### Breaking Changes

When introducing breaking changes:

1. **Document in CHANGELOG.md**
2. **Provide migration guide**
3. **Consider deprecation warnings** first
4. **Update version number** appropriately

## Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Documentation**: Check the docs first
- **Examples**: Look at practical examples

## Recognition

Contributors are recognized in:

- **CHANGELOG.md**: For significant contributions
- **README.md**: For major features
- **GitHub contributors**: Automatic recognition

## Common Pitfalls

1. **Heap allocations**: Use `heapless` collections
2. **Panics**: Always use `Result` types
3. **Blocking operations**: Use non-blocking APIs
4. **Large stack usage**: Be mindful of stack limits
5. **Floating point**: Consider fixed-point on some targets

## Resources

- [Rust Embedded Book](https://docs.rust-embedded.org/book/)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [EdgeGuard Architecture](docs/architecture.md)

---

Thank you for contributing to EdgeGuard! Your contributions help make IoT data processing more reliable and efficient.