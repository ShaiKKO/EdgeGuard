# API Reference

This section provides comprehensive API documentation for EdgeGuard components.

## Core Concepts

### Event System

EdgeGuard processes sensor data through a unified event system:

```rust
use edgeguard::{
    events::{Event, EventBuilder, SensorType},
    time::SystemTime,
};

// Create sensor reading event
let event = EventBuilder::new(SystemTime.now())
    .sensor("temp_001", SensorType::Temperature)
    .reading(23.5, 0.95)
    .unwrap();

// Create validation result event
let validation_event = EventBuilder::new(SystemTime.now())
    .sensor("temp_001", SensorType::Temperature)
    .validation(ValidationStatus::Valid, ConstraintFlags::RANGE_CHECK)
    .unwrap();
```

### Time Management

All events require timestamps for ordering and rate limiting:

```rust
use edgeguard::time::{SystemTime, MonotonicTime, FixedTime};

// System wall clock time
let system_time = SystemTime::new();
let timestamp = system_time.now();

// Monotonic time (preferred for embedded)
let monotonic_time = MonotonicTime::new();
let timestamp = monotonic_time.now();

// Fixed time for testing
let fixed_time = FixedTime::new(1000);
let timestamp = fixed_time.now();
```

### Sensor Types

EdgeGuard supports multiple sensor types with built-in physics validation:

```rust
use edgeguard::events::SensorType;

// Built-in sensor types
let temp_sensor = SensorType::Temperature;
let humidity_sensor = SensorType::Humidity;
let pressure_sensor = SensorType::Pressure;
let vibration_sensor = SensorType::Vibration;
let acoustic_sensor = SensorType::Acoustic;
let emf_sensor = SensorType::Emf;

// Custom sensor types
let custom_sensor = SensorType::Custom(42);
```

## Core APIs

### Validators
Physics-aware validation for sensor data.

- **[TemperatureValidator](validators.md#temperaturevalidator)** - Temperature validation with thermal mass modeling
- **[HumidityValidator](validators.md#humidityvalidator)** - Humidity validation with dew point calculations
- **[PressureValidator](validators.md#pressurevalidator)** - Pressure validation with altitude compensation
- **[Validator Trait](validators.md#validator-trait)** - Custom validator implementation

### Pipeline
Event processing pipeline with composable stages.

- **[Pipeline](pipeline.md#pipeline)** - Main pipeline orchestrator
- **[ValidationStage](pipeline.md#validationstage)** - Validation pipeline stage
- **[FusionStage](pipeline.md#fusionstage)** - Multi-sensor fusion stage
- **[AggregationStage](pipeline.md#aggregationstage)** - Data aggregation stage
- **[CrossValidationStage](pipeline.md#crossvalidationstage)** - Cross-sensor validation
- **[PipelineStage Trait](pipeline.md#pipelinestage-trait)** - Custom stage implementation

### Fusion
Multi-sensor data fusion algorithms.

- **[KalmanFilter](fusion.md#kalmanfilter)** - Kalman filter implementation
- **[ExtendedKalmanFilter](fusion.md#extendedkalmanfilter)** - Extended Kalman filter
- **[WeightedAverageFusion](fusion.md#weightedaveragefusion)** - Weighted average fusion
- **[FusionAlgorithm Trait](fusion.md#fusionalgorithm-trait)** - Custom fusion algorithm

### Connectors
Network connectivity for edge devices.

- **[MqttConnector](connectors.md#mqttconnector)** - MQTT client implementation
- **[CoapConnector](connectors.md#coapconnector)** - CoAP client implementation
- **[HttpConnector](connectors.md#httpconnector)** - HTTP client implementation

### Schemas
Avro schema validation with physics constraints.

- **[SchemaRegistry](schemas.md#schemaregistry)** - Schema management and versioning
- **[SchemaValidator](schemas.md#schemavalidator)** - High-performance schema validation
- **[PhysicsConstraints](schemas.md#physicsconstraints)** - Physics-aware validation rules

### Machine Learning
Anomaly detection and pattern recognition.

- **[IsolationForest](ml.md#isolationforest)** - Unsupervised anomaly detection
- **[MLAnomalyStage](ml.md#mlanomaly-stage)** - Pipeline ML integration
- **[FeatureExtractor](ml.md#featureextractor)** - Feature extraction from sensor data

### Streams
Advanced streaming data processing.

- **[Stream Trait](streams.md#stream-trait)** - Core streaming abstraction
- **[FileStream](streams.md#filestream)** - File-based streaming
- **[Stream Adapters](streams.md#stream-adapters)** - Rate limiting, batching, filtering

### Time Management
Time sources and synchronization.

- **[TimeSource Trait](time.md#timesource-trait)** - Time abstraction
- **[SystemTime](time.md#systemtime)** - Wall clock time
- **[MonotonicTime](time.md#monotonictime)** - Monotonic time source
- **[TimeManager](time.md#timemanager)** - Multi-source time management

## Common Patterns

### Basic Validation
```rust
use edgeguard::validators::TemperatureValidator;

let validator = TemperatureValidator::new()
    .with_range(-20.0, 60.0)
    .with_rate_limit(5.0);

match validator.validate(23.5) {
    Ok(value) => println!("Valid: {}°C", value),
    Err(e) => println!("Invalid: {:?}", e),
}
```

### Pipeline Processing
```rust
use edgeguard::{
    pipeline::{Pipeline, ValidationStage},
    events::{EventBuilder, SensorType},
    time::SystemTime,
};

let mut pipeline = Pipeline::<64>::builder()
    .add_stage(ValidationStage::new(
        TemperatureValidator::new(),
        SensorType::Temperature
    ))
    .build();

let event = EventBuilder::new(SystemTime.now())
    .sensor("temp_001", SensorType::Temperature)
    .reading(23.5, 0.95)
    .unwrap();

pipeline.push_event(event);
pipeline.process_batch(10)?;
```

### Multi-Sensor Fusion
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
let (fused_value, confidence) = filter.update(&[20.1], timestamp, None)?;
```

## Error Handling

### Validation Errors
```rust
use edgeguard::validators::ValidationError;

match validator.validate(reading) {
    Ok(value) => process_valid_reading(value),
    Err(ValidationError::OutOfRange { value, min, max }) => {
        eprintln!("Value {} outside range [{}, {}]", value, min, max);
    }
    Err(ValidationError::RateOfChangeExceeded { rate, max_rate }) => {
        eprintln!("Rate {} exceeds maximum {}", rate, max_rate);
    }
    Err(ValidationError::PhysicsViolation(msg)) => {
        eprintln!("Physics violation: {}", msg);
    }
}
```

### Pipeline Errors
```rust
use edgeguard::pipeline::PipelineError;

match pipeline.process_batch(100) {
    Ok(processed) => println!("Processed {} events", processed),
    Err(PipelineError::QueueFull) => {
        eprintln!("Pipeline queue full");
    }
    Err(PipelineError::ProcessingError(e)) => {
        eprintln!("Processing error: {}", e);
    }
}
```

## Performance Guidelines

### Memory Management
```rust
// Choose appropriate buffer sizes
let pipeline = Pipeline::<64>::builder()   // Small for ESP32
    .add_stage(...)
    .build();

let pipeline = Pipeline::<1024>::builder() // Large for desktop
    .add_stage(...)
    .build();
```

### Batch Processing
```rust
// Process events in batches for better performance
let batch_size = 100;
let processed = pipeline.process_batch(batch_size)?;

// Check metrics
let metrics = pipeline.metrics();
if metrics.events_dropped > 0 {
    eprintln!("Pipeline dropping events: {}", metrics.events_dropped);
}
```

## Type Reference

### Core Types

#### Event
```rust
pub enum Event {
    SensorReading {
        sensor_id: InlineString,
        sensor_type: SensorType,
        value: f32,
        timestamp: Timestamp,
        quality: f32,
    },
    ValidationResult {
        sensor_id: InlineString,
        status: ValidationStatus,
        constraints_applied: ConstraintFlags,
        timestamp: Timestamp,
    },
    // ... other variants
}
```

#### SensorType
```rust
pub enum SensorType {
    Temperature = 0,
    Humidity = 1,
    Pressure = 2,
    Vibration = 3,
    Acoustic = 4,
    Emf = 5,
    Custom(u8),
}
```

#### ValidationStatus
```rust
pub enum ValidationStatus {
    Valid = 0,
    OutOfRange = 1,
    RateExceeded = 2,
    CrossValidationFailed = 3,
    SensorQualityBad = 4,
    InvalidValue = 5,
}
```

### Configuration Types

#### Buffer Sizes
```rust
// Type alias for common buffer sizes
type SmallPipeline = Pipeline<64>;      // ESP32
type MediumPipeline = Pipeline<256>;    // Raspberry Pi
type LargePipeline = Pipeline<1024>;    // Desktop
```

#### Feature Flags
```rust
#[cfg(feature = "embedded")]
use edgeguard::embedded::*;

#[cfg(feature = "ml")]
use edgeguard::ml::*;

#[cfg(feature = "std")]
use edgeguard::std::*;
```

## Best Practices

### Validation Configuration
```rust
// Configure validators based on sensor specifications
let temp_validator = TemperatureValidator::new()
    .with_range(-40.0, 85.0)    // Sensor operating range
    .with_rate_limit(2.0)       // Conservative rate limit
    .with_thermal_mass(10.0);   // Thermal mass in kg

// Use physics-aware validation
let humidity_validator = HumidityValidator::new()
    .with_range(0.0, 100.0)     // Physical limits
    .with_rate_limit(5.0);      // Reasonable rate limit
```

### Pipeline Design
```rust
// Compose stages logically
let pipeline = Pipeline::<512>::builder()
    .add_stage(ValidationStage::new(...))      // Validate first
    .add_stage(CrossValidationStage::new())    // Cross-validate
    .add_stage(FusionStage::new(...))          // Fuse sensors
    .add_stage(AggregationStage::new(...))     // Aggregate data
    .build();
```

### Error Recovery
```rust
// Implement graceful error recovery
match fusion.update(&measurements, timestamp) {
    Ok(result) => use_result(result),
    Err(FusionError::NumericalInstability) => {
        fusion.reset();
        eprintln!("Fusion algorithm reset");
    }
    Err(FusionError::InsufficientData) => {
        // Wait for more data
    }
}
```

## Platform-Specific Considerations

### Embedded Systems
```rust
// Use fixed-size collections
use heapless::Vec;

// Avoid heap allocation
let mut events: Vec<Event, 64> = Vec::new();

// Use const generics for buffer sizes
struct EmbeddedPipeline<const N: usize> {
    buffer: [Event; N],
    // ...
}
```

### Real-time Systems
```rust
// Use deterministic operations
let result = validator.validate(reading);  // O(1) operation

// Avoid blocking operations
match stream.next() {
    Ok(event) => process_event(event),
    Err(nb::Error::WouldBlock) => return,
    Err(nb::Error::Other(e)) => handle_error(e),
}
```

For detailed API documentation, see the individual component pages:
- [Validators API](validators.md)
- [Pipeline API](pipeline.md)
- [Fusion API](fusion.md)
- [Connectors API](connectors.md)
- [Schemas API](schemas.md)
- [Machine Learning API](ml.md)
- [Streams API](streams.md)
- [Time API](time.md)