# EdgeGuard Examples

This directory contains comprehensive examples demonstrating EdgeGuard's capabilities across different platforms and use cases.

## Quick Start Examples

### Basic Usage
- **[01_basic_validation.rs](../../edgeguard-core/examples/01_basic_validation.rs)** - Simple temperature validation
- **[02_multi_sensor.rs](../../edgeguard-core/examples/02_multi_sensor.rs)** - Multi-sensor validation
- **[03_lookup_tables.rs](../../edgeguard-core/examples/03_lookup_tables.rs)** - Physics calculations

### Pipeline Examples
- **[04_event_pipeline.rs](../../edgeguard-core/examples/04_event_pipeline.rs)** - Event processing pipeline
- **[05_custom_stages.rs](../../edgeguard-core/examples/05_custom_stages.rs)** - Custom pipeline stages
- **[06_error_handling.rs](../../edgeguard-core/examples/06_error_handling.rs)** - Error handling patterns

### Fusion Examples
- **[07_kalman_fusion.rs](../../edgeguard-core/examples/07_kalman_fusion.rs)** - Kalman filter fusion
- **[08_ekf_models.rs](../../edgeguard-core/examples/08_ekf_models.rs)** - Extended Kalman filter models
- **[09_multi_fusion.rs](../../edgeguard-core/examples/09_multi_fusion.rs)** - Multi-algorithm fusion

### Streaming Examples
- **[10_streaming_data.rs](../../edgeguard-core/examples/10_streaming_data.rs)** - Stream processing
- **[11_ml_anomaly_detection.rs](../../edgeguard-ml/examples/11_ml_anomaly_detection.rs)** - ML anomaly detection

## Platform-Specific Examples

### ESP32 Examples
- **[esp32_basic.rs](../../edgeguard-core/examples/embedded/esp32_basic.rs)** - Basic ESP32 validation
- **[esp32_advanced.rs](../../edgeguard-core/examples/embedded/esp32_advanced.rs)** - Multi-sensor ESP32 fusion
- **[esp32_wifi_stream.rs](../../edgeguard-core/examples/embedded/esp32_wifi_stream.rs)** - WiFi streaming

### Embedded Examples
- **[xiao_quickstart.rs](../../edgeguard-core/examples/embedded/xiao_quickstart.rs)** - Seeed XIAO quick start
- **[xiao_bme688_advanced.rs](../../edgeguard-core/examples/embedded/xiao_bme688_advanced.rs)** - BME688 sensor integration
- **[xiao_edge_ml.rs](../../edgeguard-core/examples/embedded/xiao_edge_ml.rs)** - Edge ML processing
- **[xiao_nrf52840_sense.rs](../../edgeguard-core/examples/embedded/xiao_nrf52840_sense.rs)** - nRF52840 sensor hub

## End-to-End Examples

### IoT Applications
- **[smart_home_iot.rs](../../examples/smart_home_iot.rs)** - Smart home monitoring system
- **[industrial_monitor.rs](../../examples/industrial_monitor.rs)** - Industrial equipment monitoring
- **[mqtt_processor.rs](../../examples/mqtt_processor.rs)** - MQTT data processing
- **[esp32_complete.rs](../../examples/esp32_complete.rs)** - Complete ESP32 solution

## Running Examples

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone EdgeGuard
git clone https://github.com/example/edgeguard.git
cd edgeguard
```

### Desktop Examples

Run basic examples on desktop:

```bash
# Basic validation
cargo run --example 01_basic_validation

# Multi-sensor fusion
cargo run --example 07_kalman_fusion

# Streaming data
cargo run --example 10_streaming_data
```

### ESP32 Examples

Set up ESP32 development environment:

```bash
# Install ESP32 tools
cargo install espup
espup install
source ~/export-esp.sh

# Install flashing tools
cargo install espflash espmonitor
```

Build and flash ESP32 examples:

```bash
# Basic ESP32 example
cargo build --example esp32_basic \
  --target xtensa-esp32-none-elf \
  --no-default-features \
  --features "embedded"

# Flash to ESP32
cargo espflash --example esp32_basic --monitor
```

### Raspberry Pi Examples

Cross-compile for Raspberry Pi:

```bash
# Install cross-compilation target
rustup target add armv7-unknown-linux-gnueabihf

# Build for Raspberry Pi
cargo build --example industrial_monitor \
  --target armv7-unknown-linux-gnueabihf \
  --release

# Copy to Raspberry Pi
scp target/armv7-unknown-linux-gnueabihf/release/examples/industrial_monitor pi@raspberrypi:~/
```

## Example Categories

### 1. Basic Validation Examples

#### Temperature Validation
```rust
use edgeguard::validators::TemperatureValidator;

let validator = TemperatureValidator::new()
    .with_range(-20.0, 60.0)
    .with_rate_limit(5.0);

match validator.validate(25.0) {
    Ok(value) => println!("Valid: {}°C", value),
    Err(e) => println!("Invalid: {:?}", e),
}
```

#### Multi-Sensor Validation
```rust
use edgeguard::{
    validators::{TemperatureValidator, HumidityValidator},
    pipeline::{Pipeline, ValidationStage},
};

let mut pipeline = Pipeline::<64>::builder()
    .add_stage(ValidationStage::new(
        TemperatureValidator::new(),
        SensorType::Temperature
    ))
    .add_stage(ValidationStage::new(
        HumidityValidator::new(),
        SensorType::Humidity
    ))
    .build();
```

### 2. Pipeline Examples

#### Event Processing
```rust
use edgeguard::{
    pipeline::{Pipeline, ValidationStage, CrossValidationStage},
    events::{EventBuilder, SensorType},
};

let mut pipeline = Pipeline::<256>::builder()
    .add_stage(ValidationStage::new(...))
    .add_stage(CrossValidationStage::new())
    .build();

let event = EventBuilder::new(timestamp)
    .sensor("temp_001", SensorType::Temperature)
    .reading(23.5, 0.95)
    .unwrap();

pipeline.push_event(event);
pipeline.process_batch(10)?;
```

#### Custom Pipeline Stages
```rust
use edgeguard::pipeline::{PipelineStage, StageOutput};

struct AlertStage {
    threshold: f32,
}

impl PipelineStage for AlertStage {
    fn process(&mut self, event: Event, output: &mut StageOutput) -> PipelineResult<()> {
        if let Event::SensorReading { value, .. } = event {
            if value > self.threshold {
                // Generate alert
                println!("ALERT: High temperature: {}°C", value);
            }
        }
        output.push(event)
    }
}
```

### 3. Fusion Examples

#### Kalman Filter
```rust
use edgeguard::fusion::{KalmanFilter, KalmanConfig};

let config = KalmanConfig {
    initial_state: [20.0],
    initial_covariance: [[1.0]],
    process_noise: [[0.1]],
    measurement_noise: [[0.5]],
    // ... other config
};

let mut filter = KalmanFilter::new(config);
let (fused_value, confidence) = filter.update(&measurements, timestamp, None)?;
```

#### Weighted Average Fusion
```rust
use edgeguard::fusion::WeightedAverageFusion;

let fusion = WeightedAverageFusion::with_weights([0.4, 0.4, 0.2]);
let (fused_value, confidence) = fusion.update(&measurements, timestamp)?;
```

### 4. Streaming Examples

#### Memory Stream
```rust
use edgeguard::stream::{MemoryStream, Stream};

let events = vec![event1, event2, event3];
let mut stream = MemoryStream::new(&events);

while let Ok(event) = stream.next() {
    println!("Event: {:?}", event);
}
```

#### File Stream
```rust
use edgeguard::stream::{FileStream, CsvFormat};

let mut stream = FileStream::csv("sensor_data.csv", CsvFormat {
    delimiter: b',',
    has_header: true,
    timestamp_column: 0,
    value_column: 1,
    sensor_id_column: 2,
})?;

let mut batch = [Event::default(); 100];
let count = stream.process_batch(&mut batch)?;
```

### 5. Network Examples

#### MQTT Integration
```rust
use edgeguard::connectors::mqtt::{MqttConnector, MqttConfig};

let config = MqttConfig::new("sensor_gateway", "mqtt://localhost:1883")
    .with_credentials("username", "password")
    .with_keep_alive(Duration::from_secs(60));

let mut client = MqttConnector::new(config)?;
client.subscribe("sensors/+/data", 1)?;

let payload = serde_json::to_vec(&sensor_data)?;
client.publish("processed/temperature", payload, 1)?;
```

#### HTTP Integration
```rust
use edgeguard::connectors::http::{HttpConnector, HttpConfig};

let config = HttpConfig::new("https://api.example.com")
    .with_auth_bearer("token123")
    .with_timeout(Duration::from_secs(30));

let mut client = HttpConnector::new(config)?;
let response = client.post("/sensors/data", &sensor_data)?;
```

## Platform-Specific Setup

### ESP32 Setup

#### Hardware Requirements
- ESP32 DevKit (4MB flash, 520KB RAM minimum)
- Sensor modules (BME280, DHT22, etc.)
- Breadboard and jumper wires

#### Software Setup
```bash
# Install ESP32 toolchain
cargo install espup
espup install

# Source environment
source ~/export-esp.sh

# Install flashing tools
cargo install espflash espmonitor
```

#### Memory Configuration
```rust
// ESP32 optimized configuration
let pipeline = Pipeline::<64>::builder()  // Small buffer
    .add_stage(ValidationStage::new(...))
    .build();
```

### Raspberry Pi Setup

#### Hardware Requirements
- Raspberry Pi 4 (2GB RAM minimum)
- MicroSD card (16GB minimum)
- GPIO sensors or I2C/SPI modules

#### Software Setup
```bash
# Install Rust on Raspberry Pi
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Or cross-compile from desktop
rustup target add armv7-unknown-linux-gnueabihf
```

#### Performance Configuration
```rust
// Raspberry Pi configuration
let pipeline = Pipeline::<1024>::builder()  // Larger buffer
    .add_stage(ValidationStage::new(...))
    .add_stage(FusionStage::new(...))
    .build();
```

## Common Patterns

### Error Handling
```rust
match validator.validate(reading) {
    Ok(value) => process_valid_reading(value),
    Err(ValidationError::OutOfRange { value, min, max }) => {
        eprintln!("Value {} outside range [{}, {}]", value, min, max);
    }
    Err(ValidationError::RateOfChangeExceeded { rate, max_rate }) => {
        eprintln!("Rate {} exceeds maximum {}", rate, max_rate);
    }
    Err(e) => eprintln!("Validation error: {:?}", e),
}
```

### Performance Monitoring
```rust
let metrics = pipeline.metrics();
println!("Events processed: {}", metrics.events_processed);
println!("Events dropped: {}", metrics.events_dropped);

if metrics.events_dropped > 0 {
    eprintln!("Pipeline overloaded: {} events dropped", metrics.events_dropped);
}
```

### Configuration Management
```rust
// Environment-based configuration
let temp_range = (
    env::var("TEMP_MIN").unwrap_or("-20.0".to_string()).parse()?,
    env::var("TEMP_MAX").unwrap_or("60.0".to_string()).parse()?,
);

let validator = TemperatureValidator::new()
    .with_range(temp_range.0, temp_range.1);
```

## Next Steps

1. **Start with basic examples** to understand core concepts
2. **Try platform-specific examples** for your target hardware
3. **Build end-to-end solutions** using integration examples
4. **Customize for your use case** by modifying existing examples
5. **Contribute back** by sharing your own examples

## Troubleshooting

### Common Issues

**Build errors on ESP32:**
- Ensure ESP32 toolchain is installed and sourced
- Use `--no-default-features` and appropriate feature flags
- Check target: `--target xtensa-esp32-none-elf`

**Memory issues:**
- Reduce buffer sizes for constrained devices
- Use appropriate feature flags to minimize flash usage
- Monitor heap and stack usage

**Performance issues:**
- Use release builds for performance testing
- Enable optimizations in `Cargo.toml`
- Profile with appropriate tools

For more detailed troubleshooting, see the [Troubleshooting Guide](../guides/troubleshooting.md).

## Contributing

Examples are a great way to contribute to EdgeGuard! When adding new examples:

1. **Follow naming conventions** (numbered for basic examples)
2. **Include comprehensive comments** explaining each step
3. **Add to this README** with description and usage
4. **Test on target platforms** to ensure they work
5. **Document hardware requirements** and setup

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for detailed contribution guidelines.

## License

All examples are part of EdgeGuard and licensed under MIT/Apache-2.0.