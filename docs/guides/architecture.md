# Architecture

EdgeGuard implements a modular, event-driven architecture designed for real-time data validation and processing on resource-constrained edge devices.

## System Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Sensors   │───▶│  Validator  │───▶│  Pipeline   │
│  (Events)   │    │  (Physics)  │    │  (Stages)   │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Aggregation │◀───│   Fusion    │◀───│ Processing  │
│(Bandwidth)  │    │(Multi-sensor)│    │(Real-time)  │
└─────────────┘    └─────────────┘    └─────────────┘
```

## Core Components

### Event System

All data flows through a unified event system:

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
    // ... other event types
}
```

**Design Decisions:**
- **Fixed-size strings** (InlineString) avoid heap allocation
- **Enums over traits** for zero-cost abstractions
- **Timestamp required** for ordering and rate limiting

### Validators

Physics-aware validation with configurable constraints:

```rust
pub trait Validator: Send {
    type Value;
    type Error;
    
    fn validate(&self, value: Self::Value) -> Result<Self::Value, Self::Error>;
}
```

**Key Features:**
- **Physics constraints**: Temperature ranges, rate limits, cross-sensor validation
- **No heap allocation**: Fixed-size buffers for rate limiting
- **Configurable thresholds**: Adapt to different sensor specifications

### Pipeline System

Event processing pipeline with composable stages:

```rust
pub trait PipelineStage: Send {
    fn process(&mut self, event: Event, output: &mut StageOutput) -> PipelineResult<()>;
    fn name(&self) -> &'static str;
    fn reset(&mut self);
}
```

**Architecture Benefits:**
- **Composable**: Chain stages for complex processing
- **Backpressure handling**: Configurable queue overflow behavior
- **Metrics collection**: Built-in performance monitoring

### Fusion Algorithms

Multi-sensor data fusion with confidence scoring:

```rust
pub trait FusionAlgorithm: Send {
    fn update(&mut self, measurements: &[f32], timestamp: Timestamp) 
        -> FusionResult<(f32, ConfidenceScore)>;
}
```

**Implemented Algorithms:**
- **Kalman Filter**: Optimal state estimation with process/measurement noise
- **Weighted Average**: Simple fusion with configurable weights
- **Consensus Voting**: Agreement-based fusion for redundant sensors

## Memory Management

### No-Heap Design

EdgeGuard avoids heap allocation through:

```rust
// Fixed-size collections
use heapless::Vec;
let mut events: Vec<Event, 64> = Vec::new();

// Const generic buffers
struct Pipeline<const N: usize> {
    queue: EventQueue<N>,
    // ...
}

// Inline strings
struct InlineString<const N: usize> {
    data: [u8; N],
    len: u8,
}
```

### Memory Layout

| Component | Memory Usage | Notes |
|-----------|-------------|--------|
| Event | 64 bytes | Fixed size, no heap |
| Validator | 32-128 bytes | Depends on constraints |
| Pipeline | N * 64 bytes | N = buffer size |
| Fusion | 1-4 KB | Depends on algorithm |

## Concurrency Model

### Lock-Free Queues

Event queues use atomic operations for thread safety:

```rust
pub struct EventQueue<const N: usize> {
    buffer: [MaybeUninit<Event>; N],
    head: AtomicUsize,
    tail: AtomicUsize,
}
```

### Threading Strategy

- **Single-threaded processing**: No locks in validation pipeline
- **Multi-producer**: Multiple threads can push events
- **Single-consumer**: One thread processes events
- **Backpressure**: Queue overflow handled gracefully

## Error Handling

### No-Panic Design

All operations return `Result` types:

```rust
pub type ValidationResult<T> = Result<T, ValidationError>;
pub type FusionResult<T> = Result<T, FusionError>;
pub type PipelineResult<T> = Result<T, PipelineError>;
```

### Error Recovery

- **Graceful degradation**: Continue processing on validation errors
- **State reset**: Ability to reset algorithms on numerical instability
- **Logging integration**: Structured error reporting

## Platform Adaptations

### ESP32 Configuration

```rust
// Minimal memory footprint
let pipeline = Pipeline::<64>::builder()
    .add_stage(ValidationStage::new(...))
    .build();

// Fixed-point arithmetic
#[cfg(feature = "fixed-point")]
use fixed::types::I16F16;
```

### Raspberry Pi Configuration

```rust
// Larger buffers for higher throughput
let pipeline = Pipeline::<1024>::builder()
    .add_stage(ValidationStage::new(...))
    .add_stage(FusionStage::new(...))
    .build();
```

## Performance Characteristics

### Validation Performance

- **Latency**: <100μs per validation
- **Throughput**: 10k+ validations/sec on Cortex-M4
- **Memory**: O(1) per validation

### Pipeline Performance

- **Batch processing**: Process multiple events per call
- **Zero-copy**: Events moved, not copied
- **Predictable latency**: Fixed-size operations

### Fusion Performance

| Algorithm | Latency | Memory | Accuracy |
|-----------|---------|--------|----------|
| Weighted Average | <10μs | 128B | Good |
| Kalman Filter | <1ms | 2KB | Excellent |
| Consensus | <50μs | 512B | Fair |

## Network Architecture

### Edge-First Processing

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Sensor    │───▶│  EdgeGuard  │───▶│    Cloud    │
│  (Raw Data) │    │ (Validated) │    │(Aggregated) │
└─────────────┘    └─────────────┘    └─────────────┘
    100% volume        50-90% volume      10-50% volume
```

### Connector Architecture

```rust
pub trait Connector: Send {
    type Config;
    type Error;
    
    fn connect(&mut self, config: Self::Config) -> Result<(), Self::Error>;
    fn send(&mut self, data: &[u8]) -> Result<(), Self::Error>;
    fn receive(&mut self) -> Result<Vec<u8>, Self::Error>;
}
```

## Extensibility

### Custom Validators

```rust
struct CustomValidator {
    // Custom constraints
}

impl Validator for CustomValidator {
    type Value = f32;
    type Error = ValidationError;
    
    fn validate(&self, value: f32) -> Result<f32, ValidationError> {
        // Custom validation logic
    }
}
```

### Custom Pipeline Stages

```rust
struct CustomStage {
    // Custom state
}

impl PipelineStage for CustomStage {
    fn process(&mut self, event: Event, output: &mut StageOutput) -> PipelineResult<()> {
        // Custom processing logic
    }
}
```

## Design Rationale

### Why Event-Driven?

1. **Decoupling**: Components don't need to know about each other
2. **Scalability**: Easy to add new processing stages
3. **Testability**: Each component can be tested in isolation
4. **Debugging**: Event flow is traceable

### Why No-Heap?

1. **Predictable performance**: No garbage collection pauses
2. **Memory safety**: No memory leaks or fragmentation
3. **Embedded compatibility**: Works on systems without heap
4. **Real-time guarantees**: Deterministic execution time

### Why Physics-Aware?

1. **Early error detection**: Catch impossible values at source
2. **Domain knowledge**: Leverage physical constraints
3. **Reduced false positives**: Smarter validation than range checks
4. **Cross-sensor validation**: Detect correlated sensor failures

## Future Considerations

### Scalability

- **Distributed processing**: Multiple EdgeGuard instances
- **Load balancing**: Distribute sensors across instances
- **State synchronization**: Consistent fusion across nodes

### Advanced Features

- **Adaptive algorithms**: Self-tuning validation thresholds
- **Temporal validation**: Time-series pattern detection
- **Federated learning**: Distributed model updates

This architecture provides a solid foundation for real-time, physics-aware data validation while maintaining the performance and memory characteristics required for edge deployment.