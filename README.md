<div align="center">
  <img src="assets/EdgeGard_Logo.png" alt="EdgeGuard Logo" width="300"/>
  
  
  
  [![Build Status](https://github.com/example/edgeguard/workflows/CI/badge.svg)](https://github.com/example/edgeguard/actions)
  [![Documentation](https://docs.rs/edgeguard/badge.svg)](https://docs.rs/edgeguard)
  [![Crates.io](https://img.shields.io/crates/v/edgeguard.svg)](https://crates.io/crates/edgeguard)
  [![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
</div>

EdgeGuard is a physics-aware data validation and aggregation library for IoT edge devices. It validates sensor data using physics-based constraints and reduces bandwidth usage by 50-90% through intelligent aggregation.

## Features

- **Physics-Aware Validation**: Built-in validators for temperature, humidity, pressure, vibration, and more
- **High Performance**: <100μs validation latency, 10k+ events/sec throughput
- **Embedded Ready**: `no_std` compatible, minimal memory footprint (<100KB)
- **Real-time Processing**: Event-driven pipeline with composable stages
- **Multi-Sensor Fusion**: Kalman filters, weighted averaging, consensus algorithms
- **Network Ready**: MQTT, CoAP, HTTP connectors for cloud integration
- **Built-in Aggregation**: Reduce bandwidth with intelligent data summarization
- **Anomaly Detection**: ML-powered outlier detection (optional)
- **Production Ready**: Comprehensive testing, benchmarking, and documentation

## Quick Start

Add EdgeGuard to your `Cargo.toml`:

```toml
[dependencies]
edgeguard = "0.1.0"

# For embedded systems
edgeguard = { version = "0.1.0", default-features = false, features = ["embedded"] }
```

### Basic Usage

```rust
use edgeguard::{
    validators::TemperatureValidator,
    pipeline::{Pipeline, ValidationStage},
    events::{EventBuilder, SensorType},
    time::SystemTime,
};

// Create validation pipeline
let mut pipeline = Pipeline::<64>::builder()
    .add_stage(ValidationStage::new(
        TemperatureValidator::new()
            .with_range(-20.0, 60.0)
            .with_rate_limit(5.0), // 5°C/s max
        SensorType::Temperature
    ))
    .build();

// Process sensor data
let event = EventBuilder::new(SystemTime.now())
    .sensor("temp_001", SensorType::Temperature)
    .reading(23.5, 0.95)
    .unwrap();

pipeline.push_event(event);
pipeline.process_batch(10)?;

// Handle results
while let Some(result) = pipeline.pop_result() {
    println!("Validation result: {:?}", result);
}
```

## Examples

### Smart Home IoT
Multi-room temperature and humidity monitoring:

```rust
use edgeguard::{
    pipeline::{Pipeline, ValidationStage, AggregationStage},
    validators::{TemperatureValidator, HumidityValidator},
    events::SensorType,
};

let mut pipeline = Pipeline::<512>::builder()
    .add_stage(ValidationStage::new(
        TemperatureValidator::new(),
        SensorType::Temperature
    ))
    .add_stage(ValidationStage::new(
        HumidityValidator::new(),
        SensorType::Humidity
    ))
    .add_stage(AggregationStage::new(
        WindowSpec::TimeWindow(60_000), // 1 minute
        AggregationMethod::Statistics,
        SensorType::Temperature
    ))
    .build();
```

### Industrial Monitoring
Equipment monitoring with predictive maintenance:

```rust
use edgeguard::{
    validators::PressureValidator,
    ml::MLAnomalyStage,
    events::SensorType,
};

let mut pipeline = Pipeline::<1024>::builder()
    .add_stage(ValidationStage::new(
        PressureValidator::new()
            .with_range(0.0, 10.0) // 0-10 bar
            .with_rate_limit(2.0),
        SensorType::Pressure
    ))
    .add_stage(MLAnomalyStage::new(200)) // Anomaly detection
    .build();
```

### ESP32 Deployment
Minimal footprint for microcontrollers:

```rust
#![no_std]
#![no_main]

use edgeguard::{
    validators::TemperatureValidator,
    pipeline::{Pipeline, ValidationStage},
    events::SensorType,
};

// Create minimal pipeline
let mut pipeline = Pipeline::<64>::builder()
    .add_stage(ValidationStage::new(
        TemperatureValidator::new(),
        SensorType::Temperature
    ))
    .build();
```

### MQTT Stream Processor
Real-time cloud integration:

```rust
use edgeguard::{
    connectors::mqtt::{MqttConnector, MqttConfig},
    pipeline::{Pipeline, ValidationStage, FusionStage},
    fusion::KalmanFilter,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mqtt_client = MqttConnector::new(
        MqttConfig::new("sensor-processor", "mqtt://localhost:1883")
    )?;
    
    let pipeline = Pipeline::<1024>::builder()
        .add_stage(ValidationStage::new(
            TemperatureValidator::new(),
            SensorType::Temperature
        ))
        .add_stage(FusionStage::new(
            Box::new(KalmanFilter::default())
        ))
        .build();
    
    // Process MQTT sensor streams...
    Ok(())
}
```

## Performance

Performance characteristics by platform:

| Platform | Validation | Pipeline | Memory Usage |
|----------|------------|----------|--------------|
| ESP32 (240MHz) | 1k readings/sec | 5k events/sec | <100KB |
| Raspberry Pi 4 | 100k readings/sec | 500k events/sec | <10MB |
| Desktop (modern) | 1M+ readings/sec | 1M+ events/sec | <100MB |

### Benchmarks

Run performance benchmarks:

```bash
# Run all benchmarks
./run_benchmarks.sh

# Run specific benchmark
cargo bench --bench performance_benchmarks

# View results
open target/criterion/report/index.html
```

## Architecture

EdgeGuard follows a modular, event-driven architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Sensor Data   │───▶│   Validation     │───▶│   Pipeline      │
│   (Events)      │    │   (Physics)      │    │   (Stages)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Aggregation   │◀───│   Fusion         │◀───│   Processing    │
│   (Bandwidth)   │    │   (Multi-sensor) │    │   (Real-time)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components

- **Validators**: Physics-aware data validation
- **Pipeline**: Event processing with composable stages
- **Fusion**: Multi-sensor data fusion algorithms
- **Streams**: Data ingestion and processing
- **Connectors**: Network integration (MQTT, CoAP, HTTP)
- **ML**: Anomaly detection and pattern recognition

## Configuration

### Feature Flags

```toml
[dependencies]
edgeguard = { version = "0.1.0", features = [
    "std",              # Standard library support
    "embedded",         # Embedded/no_std support
    "ml",               # Machine learning features
    "schemas",          # Avro schema support
    "mqtt",             # MQTT connector
    "coap",             # CoAP connector
    "http",             # HTTP connector
    "high_precision",   # High precision lookup tables
    "low_memory",       # Low memory mode
]}
```

### Platform Presets

```toml
# ESP32 optimized
edgeguard = { version = "0.1.0", features = ["esp32"] }

# Raspberry Pi optimized
edgeguard = { version = "0.1.0", features = ["raspberry-pi"] }
```

## Documentation

- **[Getting Started](docs/getting-started.md)**: Installation and basic usage
- **[API Reference](docs/api/README.md)**: Complete API documentation
- **[Architecture](docs/guides/architecture.md)**: System design and patterns
- **[Examples](examples/)**: Working example applications

## Testing

Run the test suite:

```bash
# Run all tests
cargo test

# Run integration tests
cargo test --test integration

# Run with all features
cargo test --all-features

# Run benchmarks
cargo bench
```

## Deployment

### Supported Platforms

- **ESP32**: Minimal footprint, WiFi connectivity
- **Raspberry Pi**: Full features, GPIO integration
- **NVIDIA Jetson**: AI acceleration, high throughput
- **Custom Hardware**: Flexible configuration

### Network Integration

- **MQTT**: Real-time messaging
- **CoAP**: Constrained device protocol
- **HTTP/HTTPS**: REST API integration
- **Schema Registry**: Avro schema management

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow and guidelines.

## License

EdgeGuard is dual-licensed under MIT and Apache 2.0. See [LICENSE](LICENSE) for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/example/edgeguard/issues)
- **Discussions**: [GitHub Discussions](https://github.com/example/edgeguard/discussions)
- **Documentation**: [docs.rs](https://docs.rs/edgeguard)
