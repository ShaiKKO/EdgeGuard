# Getting Started

EdgeGuard is a physics-aware data validation library for IoT edge devices. This guide covers installation, basic usage, and initial configuration.

## Installation

### Python Projects

Install EdgeGuard Python bindings:

```bash
# Install from PyPI (when published)
pip install edgeguard

# Install from source
pip install maturin
maturin build --release
pip install target/wheels/edgeguard-*.whl
```

### Rust Projects

Add EdgeGuard to your `Cargo.toml`:

```toml
[dependencies]
edgeguard = "0.1.0"
```

For embedded systems without standard library:

```toml
[dependencies]
edgeguard = { version = "0.1.0", default-features = false, features = ["embedded"] }
```

### Platform-Specific Features

```toml
# ESP32 optimized
edgeguard = { version = "0.1.0", features = ["esp32"] }

# Raspberry Pi with full features
edgeguard = { version = "0.1.0", features = ["raspberry-pi"] }

# With ML anomaly detection
edgeguard = { version = "0.1.0", features = ["ml"] }

# With network connectors
edgeguard = { version = "0.1.0", features = ["mqtt", "coap", "http"] }
```

## Basic Usage

### Python Quick Start

```python
import edgeguard as eg

# Create physics-aware temperature validator
validator = eg.TemperatureValidator() \
    .with_range(-20.0, 60.0) \
    .with_rate_limit(5.0)

# Validate sensor reading
try:
    valid_temp = validator.validate(23.5)
    print(f"Valid temperature: {valid_temp}°C")
except eg.ValidationError as e:
    print(f"Validation failed: {e}")
```

### Rust Usage

#### Simple Validation

```rust
use edgeguard::{
    validators::TemperatureValidator,
    traits::Validator,
};

let validator = TemperatureValidator::new()
    .with_range(-20.0, 60.0)
    .with_rate_limit(5.0); // 5°C/s maximum rate

match validator.validate(23.5) {
    Ok(value) => println!("Valid: {}°C", value),
    Err(e) => println!("Invalid: {:?}", e),
}
```

### Pipeline Processing

```rust
use edgeguard::{
    pipeline::{Pipeline, ValidationStage},
    validators::TemperatureValidator,
    events::{EventBuilder, SensorType},
    time::SystemTime,
};

// Create pipeline
let mut pipeline = Pipeline::<64>::builder()
    .add_stage(ValidationStage::new(
        TemperatureValidator::new(),
        SensorType::Temperature
    ))
    .build();

// Process events
let event = EventBuilder::new(SystemTime.now())
    .sensor("temp_001", SensorType::Temperature)
    .reading(23.5, 0.95)
    .unwrap();

pipeline.push_event(event);
pipeline.process_batch(10).unwrap();

// Handle results
while let Some(result) = pipeline.pop_result() {
    match result {
        Event::ValidationResult { status, .. } => {
            println!("Validation: {:?}", status);
        }
        _ => {}
    }
}
```

### Multi-Sensor Fusion

```rust
use edgeguard::{
    fusion::{KalmanFilter, KalmanConfig, StateTransition},
    time::Timestamp,
};

// Configure Kalman filter
let config = KalmanConfig {
    initial_state: [20.0],
    initial_covariance: [[1.0]],
    process_noise: [[0.1]],
    measurement_noise: [[0.5]],
    transition: StateTransition {
        transition_matrix: [[1.0]],
        control_matrix: None,
    },
    measurement_matrix: [[1.0]],
    control_matrix: None,
    convergence_threshold: 0.01,
};

let mut filter = KalmanFilter::<1, 1>::new(config);

// Process measurements
let measurements = [20.1];
let (fused_value, confidence) = filter.update(
    &measurements,
    Timestamp::from_millis(1000),
    None
).unwrap();

println!("Fused: {} (confidence: {})", fused_value, confidence.as_f32());
```

## Configuration

### Memory Constraints

For embedded systems, configure buffer sizes appropriately:

```rust
// Small buffer for ESP32
let pipeline = Pipeline::<64>::builder()
    .add_stage(...)
    .build();

// Larger buffer for Raspberry Pi
let pipeline = Pipeline::<512>::builder()
    .add_stage(...)
    .build();
```

### Validation Thresholds

Configure validators based on your sensor specifications:

```rust
// BME280 sensor specifications
let temp_validator = TemperatureValidator::new()
    .with_range(-40.0, 85.0)    // Operating range
    .with_rate_limit(2.0);      // Conservative rate limit

let humidity_validator = HumidityValidator::new()
    .with_range(0.0, 100.0)
    .with_rate_limit(5.0);

let pressure_validator = PressureValidator::new()
    .with_range(300.0, 1100.0)  // hPa range
    .with_rate_limit(10.0);
```

### Network Integration

#### MQTT Connection

```rust
use edgeguard::connectors::mqtt::{MqttConnector, MqttConfig};

let config = MqttConfig::new("sensor_gateway", "mqtt://localhost:1883")
    .with_credentials("username", "password")
    .with_keep_alive(std::time::Duration::from_secs(60));

let mut client = MqttConnector::new(config)?;
client.subscribe("sensors/+/data", 1)?;
```

#### CoAP for Constrained Devices

```rust
use edgeguard::connectors::coap::{CoapConnector, CoapConfig};

let config = CoapConfig::new("coap://server:5683")
    .with_timeout(std::time::Duration::from_secs(5));

let mut client = CoapConnector::new(config)?;
```

## Performance Considerations

### Target Performance

| Platform | Validation | Pipeline | Memory |
|----------|------------|----------|---------|
| ESP32 | 1k readings/sec | 5k events/sec | <100KB |
| Raspberry Pi | 100k readings/sec | 500k events/sec | <10MB |
| Desktop | 1M+ readings/sec | 1M+ events/sec | <100MB |

### Optimization

For production deployment:

1. **Use appropriate buffer sizes** for your platform
2. **Process events in batches** for better throughput
3. **Monitor pipeline metrics** to detect bottlenecks
4. **Use fixed-size collections** to avoid heap allocation

```rust
// Check performance metrics
let metrics = pipeline.metrics();
if metrics.events_dropped > 0 {
    eprintln!("Pipeline dropping events: {}", metrics.events_dropped);
}
```

## Next Steps

1. **Architecture**: Review [System Architecture](guides/architecture.md) for design decisions
2. **Examples**: Check [Examples](examples/README.md) for complete applications
3. **API Reference**: See [API Documentation](api/README.md) for detailed function reference
4. **Production**: Read [Deployment Guide](guides/deployment.md) for production setup

## Common Issues

### Compilation Errors

For embedded targets, ensure correct feature flags:

```toml
[dependencies]
edgeguard = { version = "0.1.0", default-features = false, features = ["embedded"] }
```

### Memory Issues

Reduce buffer sizes for constrained devices:

```rust
let pipeline = Pipeline::<32>::builder()  // Smaller buffer
    .add_stage(...)
    .build();
```

### Performance Issues

Enable optimizations in `Cargo.toml`:

```toml
[profile.release]
lto = true
codegen-units = 1
```

For more troubleshooting, see [Troubleshooting Guide](guides/troubleshooting.md).