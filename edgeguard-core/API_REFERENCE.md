# EdgeGuard Core API Reference

This document provides a comprehensive reference for all public APIs in the EdgeGuard Core library. Each module is documented with its types, methods, and usage examples.

## Table of Contents

1. [Events Module](#events-module)
2. [Pipeline Module](#pipeline-module)
3. [Fusion Module](#fusion-module)
4. [Stream Module](#stream-module)
5. [Validators Module](#validators-module)
6. [Time Module](#time-module)
7. [Buffer Module](#buffer-module)
8. [Queue Module](#queue-module)
9. [Lookup Tables Module](#lookup-tables-module)

---

## Events Module

The events module defines the event system that forms the backbone of EdgeGuard's real-time data processing pipeline.

### Types

#### `SensorType` (enum)
```rust
#[repr(u8)]
pub enum SensorType {
    Temperature = 0,
    Humidity = 1,
    Pressure = 2,
    Voc = 3,
    Particulate = 4,
    Acoustic = 5,
    Vibration = 6,
    Emf = 7,
    Custom(u8),
}
```

**Methods:**
- `name(&self) -> &'static str` - Get human-readable name
- `unit(&self) -> &'static str` - Get expected unit of measurement

#### `ValidationStatus` (enum)
```rust
#[repr(u8)]
pub enum ValidationStatus {
    Valid = 0,
    OutOfRange = 1,
    RateExceeded = 2,
    CrossValidationFailed = 3,
    SensorQualityBad = 4,
    InvalidValue = 5,
}
```

#### `ConstraintFlags` (struct)
```rust
pub struct ConstraintFlags(u16);
```

**Constants:**
- `RANGE: ConstraintFlags`
- `RATE: ConstraintFlags`
- `CROSS: ConstraintFlags`
- `QUALITY: ConstraintFlags`
- `PHYSICS: ConstraintFlags`

**Methods:**
- `empty() -> Self` - Create empty flags
- `all() -> Self` - Create with all flags set
- `set(&mut self, other: Self)` - Set flags
- `contains(&self, other: Self) -> bool` - Check if flags are set

#### `CrossValidationType` (enum)
```rust
#[repr(u16)]
pub enum CrossValidationType {
    DewPoint = 0,
    AltitudePressure = 1,
    ThermalConsistency = 2,
    HumidityPressure = 3,
    Custom(u16),
}
```

#### `CrossValidationDetails` (struct)
```rust
pub struct CrossValidationDetails {
    pub expected_value: f32,
    pub actual_value: f32,
    pub deviation_percent: f32,
}
```

#### `SystemEventType` (enum)
```rust
#[repr(u8)]
pub enum SystemEventType {
    PipelineStart = 0,
    PipelineStop = 1,
    QueueOverflow = 2,
    ValidatorError = 3,
    MemoryWarning = 4,
    PerformanceWarning = 5,
}
```

#### `InlineString` (struct)
```rust
pub struct InlineString {
    len: u8,
    data: [u8; MAX_INLINE_ID],
}
```

**Methods:**
- `new(s: &str) -> Option<Self>` - Create from string slice (max 15 chars)
- `as_str(&self) -> &str` - Get as string slice

#### `Event` (enum)
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
    CrossValidationResult {
        primary_sensor: InlineString,
        related_sensor: InlineString,
        validation_type: CrossValidationType,
        status: ValidationStatus,
        details: CrossValidationDetails,
        timestamp: Timestamp,
    },
    BatchReading {
        sensor_id: InlineString,
        sensor_type: SensorType,
        base_timestamp: Timestamp,
        count: u16,
        interval_ms: u16,
        mean_value: f32,
        min_value: f32,
        max_value: f32,
    },
    SystemEvent {
        event_type: SystemEventType,
        timestamp: Timestamp,
        details: u32,
    },
}
```

**Methods:**
- `timestamp(&self) -> Timestamp` - Get event timestamp
- `sensor_id(&self) -> Option<&str>` - Get sensor ID if applicable
- `is_valid(&self) -> bool` - Check if event represents valid data
- `priority(&self) -> u8` - Get event priority for queue management

#### `EventBuilder` (struct)
```rust
pub struct EventBuilder {
    sensor_id: Option<InlineString>,
    sensor_type: Option<SensorType>,
    timestamp: Timestamp,
}
```

**Methods:**
- `new(timestamp: Timestamp) -> Self` - Create new builder
- `sensor(self, id: &str, sensor_type: SensorType) -> Self` - Set sensor info
- `reading(self, value: f32, quality: f32) -> Option<Event>` - Build sensor reading
- `validation(self, status: ValidationStatus, constraints: ConstraintFlags) -> Option<Event>` - Build validation result

### Constants

- `MAX_INLINE_ID: usize = 15` - Maximum length for inline sensor IDs

---

## Pipeline Module

The pipeline module implements a flexible, composable pipeline for processing sensor events through multiple stages.

### Types

#### `PipelineError` (enum)
```rust
pub enum PipelineError {
    StageError { stage: usize, error: ValidationError },
    QueueFull,
    InvalidConfig(&'static str),
    ResourceExhausted,
}
```

#### `BackpressureStrategy` (enum)
```rust
pub enum BackpressureStrategy {
    DropOldest,
    DropNewest,
    Error,
}
```

#### `WindowSpec` (enum)
```rust
pub enum WindowSpec {
    Time { duration_ms: u32 },
    Count { size: u16 },
    Tumbling { duration_ms: u32 },
}
```

#### `AggregationMethod` (enum)
```rust
pub enum AggregationMethod {
    Mean,
    Min,
    Max,
    Sum,
    StdDev,
    Median,
}
```

#### `PipelineMetrics` (struct)
```rust
pub struct PipelineMetrics {
    pub events_processed: [u32; MAX_PIPELINE_STAGES],
    pub events_dropped: [u32; MAX_PIPELINE_STAGES],
    pub processing_time_us: [u32; MAX_PIPELINE_STAGES],
    pub current_depth: u16,
}
```

#### `PipelineStage` (trait)
```rust
pub trait PipelineStage: Send {
    fn process(&mut self, event: Event, output: &mut StageOutput) -> PipelineResult<()>;
    fn name(&self) -> &'static str;
    fn can_handle(&self, event: &Event) -> bool { true }
}
```

#### `StageOutput` (struct)
```rust
pub struct StageOutput {
    events: Vec<Event, 16>,
}
```

**Methods:**
- `emit(&mut self, event: Event) -> PipelineResult<()>` - Emit a new event
- `forward(&mut self, event: Event) -> PipelineResult<()>` - Forward input unchanged

#### `ValidationStage<V>` (struct)
```rust
pub struct ValidationStage<V: Validator + Send> {
    validator: V,
    context: ValidationContext,
    sensor_type: SensorType,
}
```

**Methods:**
- `new(validator: V, sensor_type: SensorType) -> Self` - Create validation stage

#### `FilterStage<F>` (struct)
```rust
pub struct FilterStage<F> {
    predicate: F,
    name: &'static str,
}
```

**Methods:**
- `new(predicate: F, name: &'static str) -> Self` - Create filter stage

#### `RouterStage` (struct)
```rust
pub struct RouterStage {
    routes: [(SensorType, Option<Box<dyn PipelineStage>>); MAX_ROUTES],
    route_count: usize,
}
```

**Methods:**
- `new() -> Self` - Create new router
- `add_route(&mut self, sensor_type: SensorType, stage: Box<dyn PipelineStage>) -> Result<(), ()>` - Add route

#### `CrossValidationStage` (struct)
```rust
pub struct CrossValidationStage {
    sensor_buffer: FnvIndexMap<InlineString, (SensorType, f32, Timestamp), 8>,
    validation_pairs: [(SensorType, SensorType, CrossValidationType); MAX_SENSOR_PAIRS],
    pair_count: usize,
    time_window_ms: u32,
    altitude_m: f32,
}
```

**Methods:**
- `new() -> Self` - Create new stage
- `with_altitude(altitude_m: f32) -> Self` - Create with altitude
- `set_altitude(&mut self, altitude_m: f32)` - Update altitude
- `add_pair(&mut self, primary: SensorType, secondary: SensorType, validation_type: CrossValidationType) -> Result<(), ()>` - Add validation pair

#### `AggregationStage` (struct)
```rust
pub struct AggregationStage {
    window: WindowSpec,
    method: AggregationMethod,
    sensor_type: SensorType,
    value_buffer: Vec<f32, MAX_AGGREGATION_WINDOW>,
    timestamp_buffer: Vec<Timestamp, MAX_AGGREGATION_WINDOW>,
    window_start: Timestamp,
    last_sensor_id: Option<InlineString>,
}
```

**Methods:**
- `new(window: WindowSpec, method: AggregationMethod, sensor_type: SensorType) -> Self` - Create aggregation stage

#### `Pipeline<const N: usize>` (struct)
```rust
pub struct Pipeline<const N: usize> {
    stages: Vec<Box<dyn PipelineStage>, N>,
    input_queue: EventQueue<64>,
    output_queue: EventQueue<64>,
    backpressure: BackpressureStrategy,
    metrics: PipelineMetrics,
}
```

**Methods:**
- `builder() -> PipelineBuilder<N>` - Create pipeline builder
- `process_batch(&mut self, max_events: usize) -> PipelineResult<usize>` - Process events
- `metrics(&self) -> &PipelineMetrics` - Get metrics
- `push_event(&self, event: Event) -> bool` - Push event to input
- `pop_result(&self) -> Option<Event>` - Pop from output

#### `PipelineBuilder<const N: usize>` (struct)
```rust
pub struct PipelineBuilder<const N: usize> {
    stages: Vec<Box<dyn PipelineStage>, N>,
    backpressure: BackpressureStrategy,
}
```

**Methods:**
- `add_stage(self, stage: impl PipelineStage + 'static) -> Self` - Add stage
- `backpressure(self, strategy: BackpressureStrategy) -> Self` - Set strategy
- `build(self) -> Pipeline<N>` - Build pipeline

#### `StreamProcessor<S, const N: usize>` (struct)
```rust
pub struct StreamProcessor<S: Stream, const N: usize> {
    stream: S,
    pipeline: Pipeline<N>,
    stats: ProcessingStats,
}
```

**Methods:**
- `new(stream: S, pipeline: Pipeline<N>) -> Self` - Create processor
- `process_all(self) -> Result<ProcessingStats, PipelineError>` - Process all
- `process_batch(&mut self, batch_size: usize) -> Result<usize, PipelineError>` - Process batch
- `stats(&self) -> &ProcessingStats` - Get statistics
- `pipeline_mut(&mut self) -> &mut Pipeline<N>` - Get pipeline

### Constants

- `MAX_PIPELINE_STAGES: usize = 16` - Maximum stages in pipeline
- `MAX_ROUTES: usize = 8` - Maximum routes in router
- `MAX_SENSOR_PAIRS: usize = 4` - Maximum cross-validation pairs
- `MAX_AGGREGATION_WINDOW: usize = 100` - Maximum aggregation window

---

## Fusion Module

The fusion module implements advanced sensor fusion algorithms to combine data from multiple sensors.

### Types

#### `FusionError` (enum)
```rust
pub enum FusionError {
    InsufficientSensors,
    NumericalInstability,
    SingularMatrix,
    DimensionMismatch,
    UnknownSensor,
    Divergence,
}
```

#### `FusionAlgorithm<const N: usize, const M: usize>` (trait)
```rust
pub trait FusionAlgorithm<const N: usize, const M: usize> {
    type Config;
    
    fn new(config: Self::Config) -> Self;
    fn predict(&mut self, dt_ms: u32) -> FusionResult<()>;
    fn update(&mut self, measurements: &[f32; M], timestamp: Timestamp, mask: Option<u32>) -> FusionResult<(f32, ConfidenceScore)>;
    fn state(&self) -> &[f32; N];
    fn uncertainty(&self) -> [f32; N];
    fn reset(&mut self);
    fn has_converged(&self) -> bool;
}
```

#### `WeightedAverageFusion<const M: usize>` (struct)
```rust
pub struct WeightedAverageFusion<const M: usize> {
    sensors: Vec<Box<dyn SensorModel>, M>,
    weights: [f32; M],
    last_estimate: f32,
    state_array: [f32; 1],
    measurement_count: u32,
}
```

**Methods:**
- `new() -> Self` - Create new fusion
- `add_sensor(self, sensor: Box<dyn SensorModel>) -> Self` - Add sensor
- `fuse(&mut self, measurements: &[f32; M], mask: Option<u32>) -> (f32, ConfidenceScore)` - Fuse measurements

#### `ComplementaryFilter` (struct)
```rust
pub struct ComplementaryFilter {
    fast_weight: f32,
    state: f32,
    state_array: [f32; 1],
    last_timestamp: Timestamp,
}
```

**Methods:**
- `new(config: ComplementaryConfig) -> Self` - Create filter

#### `ConsensusVoting<const M: usize>` (struct)
```rust
pub struct ConsensusVoting<const M: usize> {
    outlier_threshold: f32,
    min_votes: usize,
    last_estimate: f32,
    state_array: [f32; 1],
    measurement_count: u32,
}
```

**Methods:**
- `new(config: VotingConfig) -> Self` - Create voting fusion

### Kalman Filter

#### `KalmanConfig<const N: usize, const M: usize>` (struct)
```rust
pub struct KalmanConfig<const N: usize, const M: usize> {
    pub initial_state: Vector<N>,
    pub initial_covariance: SquareMatrix<N>,
    pub process_noise: SquareMatrix<N>,
    pub measurement_noise: SquareMatrix<M>,
    pub transition: StateTransition<N>,
    pub measurement_matrix: Matrix<M, N>,
    pub control_matrix: Option<Matrix<N, N>>,
    pub convergence_threshold: f32,
}
```

**Methods:**
- `with_process_noise(self, noise: f32) -> Self` - Set process noise
- `with_measurement_noise(self, noise: [f32; M]) -> Self` - Set measurement noise
- `with_transition(self, transition: StateTransition<N>) -> Self` - Set transition

#### `KalmanFilter<const N: usize, const M: usize>` (struct)
```rust
pub struct KalmanFilter<const N: usize, const M: usize> {
    state: Vector<N>,
    covariance: SquareMatrix<N>,
    config: KalmanConfig<N, M>,
    confidence_scorer: ConfidenceScorer<M>,
    last_timestamp: Timestamp,
    update_count: u32,
    workspace: KalmanWorkspace<N, M>,
}
```

**Methods:**
- `new(config: KalmanConfig<N, M>) -> Self` - Create Kalman filter

#### `ExtendedKalmanFilter<const N: usize, const M: usize>` (struct)
```rust
pub struct ExtendedKalmanFilter<const N: usize, const M: usize> {
    kf: KalmanFilter<N, M>,
    state_fn: fn(&Vector<N>, &Vector<N>) -> Vector<N>,
    measurement_fn: fn(&Vector<N>) -> Vector<M>,
    jacobian_epsilon: f32,
}
```

**Methods:**
- `new(config: KalmanConfig<N, M>, state_fn: fn(&Vector<N>, &Vector<N>) -> Vector<N>, measurement_fn: fn(&Vector<N>) -> Vector<M>) -> Self` - Create EKF
- `predict(&mut self, dt_ms: u32) -> FusionResult<()>` - Predict state
- `update(&mut self, measurements: &[f32; M], timestamp: Timestamp, mask: Option<u32>) -> FusionResult<(f32, ConfidenceScore)>` - Update with measurements
- `state(&self) -> &[f32; N]` - Get state
- `uncertainty(&self) -> [f32; N]` - Get uncertainty
- `reset(&mut self)` - Reset filter
- `has_converged(&self) -> bool` - Check convergence

### Confidence

#### `ConfidenceScore` (struct)
```rust
pub struct ConfidenceScore {
    value: u16, // Fixed-point 0.16 format
}
```

**Methods:**
- `from_float(f: f32) -> Self` - Create from float [0,1]
- `as_float(&self) -> f32` - Convert to float
- `value(&self) -> u16` - Get raw value

#### `ConfidenceScorer<const M: usize>` (struct)
```rust
pub struct ConfidenceScorer<const M: usize> {
    base_noise: f32,
    outlier_threshold: f32,
    measurement_count: u32,
}
```

**Methods:**
- `new(base_noise: f32, outlier_threshold: f32) -> Self` - Create scorer
- `increment_count(&mut self)` - Increment count
- `reset(&mut self)` - Reset scorer

### Models

#### `SensorModel` (trait)
```rust
pub trait SensorModel: Send {
    fn sensor_id(&self) -> &str;
    fn sensor_type(&self) -> SensorType;
    fn noise_variance(&self) -> f32;
    fn measurement_model(&self, state: &[f32]) -> f32;
    fn is_healthy(&self) -> bool { true }
}
```

#### `StateTransition<const N: usize>` (struct)
```rust
pub struct StateTransition<const N: usize> {
    pub transition_matrix: SquareMatrix<N>,
    pub process_noise: SquareMatrix<N>,
}
```

### Matrix Operations

#### Type Aliases
```rust
pub type Matrix<const R: usize, const C: usize> = [[f32; C]; R];
pub type SquareMatrix<const N: usize> = Matrix<N, N>;
pub type Vector<const N: usize> = [f32; N];
```

#### Functions
- `multiply<const R: usize, const K: usize, const C: usize>(a: &Matrix<R, K>, b: &Matrix<K, C>, result: &mut Matrix<R, C>)` - Matrix multiplication
- `transpose<const R: usize, const C: usize>(a: &Matrix<R, C>, result: &mut Matrix<C, R>)` - Matrix transpose
- `add<const R: usize, const C: usize>(a: &Matrix<R, C>, b: &Matrix<R, C>, result: &mut Matrix<R, C>)` - Matrix addition
- `make_symmetric<const N: usize>(matrix: &mut SquareMatrix<N>)` - Make matrix symmetric
- `is_well_conditioned<const N: usize>(matrix: &SquareMatrix<N>) -> bool` - Check conditioning
- `matvec<const R: usize, const C: usize>(matrix: &Matrix<R, C>, vector: &Vector<C>, result: &mut Vector<R>)` - Matrix-vector multiplication
- `cholesky<const N: usize>(a: &SquareMatrix<N>, l: &mut SquareMatrix<N>) -> bool` - Cholesky decomposition
- `invert<const N: usize>(a: &SquareMatrix<N>, inv: &mut SquareMatrix<N>) -> bool` - Matrix inversion
- `solve_cholesky<const N: usize>(l: &SquareMatrix<N>, b: &Vector<N>, x: &mut Vector<N>)` - Solve linear system

---

## Stream Module

The stream module provides a unified interface for streaming sensor data from various sources.

### Types

#### `StreamError<E>` (enum)
```rust
pub enum StreamError<E> {
    Transport(E),
    Format(&'static str),
    SchemaViolation,
    EndOfStream,
    Overflow,
    Backpressure,
}
```

#### `BackpressureControl` (struct)
```rust
pub struct BackpressureControl {
    high_watermark: usize,
    low_watermark: usize,
    pending_count: usize,
    backpressure_active: bool,
}
```

**Methods:**
- `new(high_watermark: usize, low_watermark: usize) -> Self` - Create control
- `should_pause(&self) -> bool` - Check if should pause
- `update(&mut self, delta: isize)` - Update count
- `consumed(&mut self, count: usize)` - Mark consumed
- `produced(&mut self, count: usize)` - Mark produced
- `utilization(&self) -> u8` - Get utilization percentage

#### `Stream` (trait)
```rust
pub trait Stream {
    type Item;
    type Error;
    
    fn poll_next(&mut self) -> nb::Result<Self::Item, Self::Error>;
    fn schema(&self) -> Option<&Schema> { None }
    fn size_hint(&self) -> (usize, Option<usize>) { (0, None) }
}
```

#### `BackpressureStream` (trait)
```rust
pub trait BackpressureStream: Stream {
    fn backpressure(&self) -> &BackpressureControl;
    fn backpressure_mut(&mut self) -> &mut BackpressureControl;
    fn can_accept(&self) -> bool { !self.backpressure().should_pause() }
}
```

#### `MemoryStream<'a>` (struct)
```rust
pub struct MemoryStream<'a> {
    events: &'a [Event],
    position: usize,
}
```

**Methods:**
- `new(events: &'a [Event]) -> Self` - Create from slice
- `reset(&mut self)` - Reset to beginning

#### `FileStream` (struct) - requires `std` feature
```rust
pub struct FileStream {
    path: &'static str,
    format: FileFormat,
    buffer: [u8; 4096],
    buffer_pos: usize,
    buffer_len: usize,
    file_offset: usize,
    line_buffer: String<256>,
    eof: bool,
}
```

**Methods:**
- `new(path: &'static str, format: FileFormat) -> Result<Self, StreamError<std::io::Error>>` - Create file stream

#### `FileFormat` (enum) - requires `std` feature
```rust
pub enum FileFormat {
    Csv,
    JsonLines,
    Binary,
}
```

#### `RateLimitedStream<S, T>` (struct)
```rust
pub struct RateLimitedStream<S: Stream, T: TimeSource> {
    inner: S,
    rate: u32,
    tokens: f32,
    last_update: Timestamp,
    time_source: T,
}
```

**Methods:**
- `new(inner: S, events_per_second: u32, time_source: T) -> Self` - Create rate limited stream

#### `BatchingStream<S, T, const N: usize>` (struct)
```rust
pub struct BatchingStream<S: Stream, T: TimeSource, const N: usize> {
    inner: S,
    buffer: Vec<S::Item, N>,
    timeout_ms: u32,
    batch_start: Timestamp,
    time_source: T,
    pending: Option<S::Item>,
}
```

**Methods:**
- `new(inner: S, timeout_ms: u32, time_source: T) -> Self` - Create batching stream

#### `BackpressureWrapper<S>` (struct)
```rust
pub struct BackpressureWrapper<S: Stream> {
    inner: S,
    backpressure: BackpressureControl,
}
```

**Methods:**
- `new(inner: S, backpressure: BackpressureControl) -> Self` - Create wrapper
- `with_defaults(inner: S) -> Self` - Create with defaults

#### `CombinedStream<S1, S2>` (struct)
```rust
pub struct CombinedStream<S1: Stream, S2: Stream> {
    stream1: S1,
    stream2: S2,
    next: bool,
}
```

**Methods:**
- `new(stream1: S1, stream2: S2) -> Self` - Combine streams

#### `RecoveryStrategy` (enum)
```rust
pub enum RecoveryStrategy {
    RetryImmediate,
    RetryBackoff { initial_ms: u32, max_ms: u32 },
    Skip,
    LogAndContinue,
    Fail,
}
```

#### `RecoveryWrapper<S>` (struct)
```rust
pub struct RecoveryWrapper<S: Stream> {
    inner: S,
    strategies: FnvIndexMap<&'static str, RecoveryStrategy, 16>,
    retry_count: u8,
    last_error_time: Option<Timestamp>,
}
```

**Methods:**
- `new(inner: S) -> Self` - Create wrapper
- `with_strategy(self, error_type: &'static str, strategy: RecoveryStrategy) -> Self` - Add strategy

---

## Validators Module

The validators module contains physics-based validators for common IoT sensors.

### Temperature Validator

#### `TemperatureValidator` (struct)
```rust
pub struct TemperatureValidator {
    min_celsius: f32,
    max_celsius: f32,
    max_rate_celsius_per_sec: f32,
}
```

**Methods:**
- `new_with_limits(min: f32, max: f32, max_rate: f32) -> Self` - Create with custom limits
- `indoor() -> Self` - Create for indoor use (-10 to 50°C, 2°C/s)
- `industrial() -> Self` - Create for industrial use (-40 to 200°C, 20°C/s)

**Default limits:**
- Min: -80°C
- Max: 125°C
- Max rate: 10°C/s

### Humidity Validator

#### `HumidityValidator` (struct)
```rust
pub struct HumidityValidator {
    min_percent: f32,
    max_percent: f32,
    max_rate_percent_per_sec: f32,
}
```

**Methods:**
- `new_with_limits(min: f32, max: f32, max_rate: f32) -> Self` - Create with custom limits
- `outdoor() -> Self` - Create for outdoor use (20°C/s rate)

**Default limits:**
- Min: 0%
- Max: 100%
- Max rate: 10%/s

### Pressure Validator

#### `PressureValidator` (struct)
```rust
pub struct PressureValidator {
    min_hpa: f32,
    max_hpa: f32,
    max_rate_hpa_per_sec: f32,
    altitude_m: Option<f32>,
}
```

**Methods:**
- `new_with_limits(min: f32, max: f32, max_rate: f32) -> Self` - Create with custom limits
- `with_altitude(self, altitude_m: f32) -> Self` - Set altitude
- `weather_station() -> Self` - Create for weather station (1 hPa/s)
- `high_altitude(altitude_m: f32) -> Self` - Create for high altitude

**Default limits:**
- Min: 300 hPa
- Max: 1100 hPa
- Max rate: 10 hPa/s

### Common Validator Trait

All validators implement the `Validator` trait:

```rust
pub trait Validator {
    type Value;
    
    fn validate(&self, value: Self::Value, context: &ValidationContext) -> ValidationResult<()>;
    fn constraints(&self) -> ValidatorConstraints;
}
```

#### `ValidatorConstraints` (struct)
```rust
pub struct ValidatorConstraints {
    pub min_value: f32,
    pub max_value: f32,
    pub max_rate_change: f32,
    pub noise_threshold: Option<f32>,
}
```

#### `ValidationContext` (struct)
```rust
pub struct ValidationContext {
    pub timestamp: Timestamp,
    pub sensor_quality: f32,
    pub history: CircularBuffer<MAX_HISTORY_SIZE>,
    pub temperature: Option<f32>,
    pub humidity: Option<f32>,
    pub pressure: Option<f32>,
}
```

**Methods:**
- `add_reading(&mut self, value: f32, timestamp: Timestamp)` - Add to history

---

## Time Module

The time module provides abstractions for handling time on embedded systems.

### Types

#### `Timestamp` (type alias)
```rust
pub type Timestamp = u64; // Milliseconds since epoch or boot
```

#### `TimeSource` (trait)
```rust
pub trait TimeSource {
    fn now(&self) -> Timestamp;
    fn is_wall_clock(&self) -> bool;
    fn precision_ms(&self) -> u32;
}
```

#### `MonotonicTime` (struct)
```rust
pub struct MonotonicTime {
    start_ms: Timestamp,
    start_instant: std::time::Instant, // Only with std feature
}
```

**Methods:**
- `new() -> Self` - Create monotonic timer

#### `SystemTime` (struct) - requires `std` feature
```rust
pub struct SystemTime;
```

#### `FixedTime` (struct)
```rust
pub struct FixedTime {
    timestamp: Timestamp,
}
```

**Methods:**
- `new(timestamp: Timestamp) -> Self` - Create fixed time
- `set(&mut self, timestamp: Timestamp)` - Set time
- `advance(&mut self, ms: u64)` - Advance time

#### `ContextTimeSource<'a>` (struct)
```rust
pub struct ContextTimeSource<'a> {
    context: &'a ValidationContext,
}
```

**Methods:**
- `new(context: &'a ValidationContext) -> Self` - Create from context

#### `TimeManager` (struct)
```rust
pub struct TimeManager {
    primary: Box<dyn TimeSource>,
    fallback: Option<Box<dyn TimeSource>>,
    last_known: Timestamp,
}
```

**Methods:**
- `new(primary: Box<dyn TimeSource>) -> Self` - Create manager
- `with_fallback(self, fallback: Box<dyn TimeSource>) -> Self` - Add fallback
- `now(&mut self) -> Timestamp` - Get current time
- `delta_ms(&self, earlier: Timestamp, later: Timestamp) -> u64` - Calculate delta
- `rate_per_second(&self, value_delta: f32, time_delta_ms: u64) -> f32` - Calculate rate

### Type Aliases

- `MockTimeSource = FixedTime` - Mock time for testing
- `MonotonicClock = MonotonicTime` - Backward compatibility

---

## Buffer Module

The buffer module provides a fixed-size circular buffer for sensor history tracking.

### Types

#### `CircularBuffer<const N: usize>` (struct)
```rust
pub struct CircularBuffer<const N: usize> {
    data: [Option<TimestampedReading>; N],
    write_pos: usize,
    len: usize,
}
```

**Methods:**
- `const fn new() -> Self` - Create empty buffer
- `push(&mut self, reading: TimestampedReading)` - Add reading (overwrites oldest when full)
- `len(&self) -> usize` - Get count
- `is_empty(&self) -> bool` - Check if empty
- `is_full(&self) -> bool` - Check if full
- `last(&self) -> Option<&TimestampedReading>` - Get most recent
- `iter(&self) -> CircularBufferIter<N>` - Iterate oldest to newest
- `clear(&mut self)` - Clear all readings

#### `CircularBufferIter<'a, const N: usize>` (struct)
Iterator over buffer contents from oldest to newest.

#### `TimestampedReading` (struct)
```rust
pub struct TimestampedReading {
    pub value: f32,
    pub timestamp: Timestamp,
}
```

---

## Queue Module

The queue module implements a lock-free Single Producer Multiple Consumer (SPMC) queue.

### Types

#### `EventQueue<const N: usize>` (struct)
```rust
pub struct EventQueue<const N: usize> {
    buffer: UnsafeCell<[MaybeUninit<Event>; N]>,
    head: AtomicUsize,
    tail: AtomicUsize,
    stats: QueueStats,
}
```

**Methods:**
- `const fn new() -> Self` - Create for static context
- `new_runtime() -> Self` - Create at runtime
- `push(&self, event: Event) -> bool` - Push event (single producer)
- `pop(&self) -> Option<Event>` - Pop event (multiple consumers)
- `peek(&self) -> Option<&Event>` - Peek without removing
- `len(&self) -> usize` - Get current length
- `is_empty(&self) -> bool` - Check if empty
- `is_full(&self) -> bool` - Check if full
- `stats(&self) -> &QueueStats` - Get statistics
- `unsafe fn clear(&self)` - Clear queue (not thread-safe)
- `drain(&self) -> QueueDrain<'_, N>` - Drain all events

#### `QueueStats` (struct)
```rust
pub struct QueueStats {
    pub pushed: AtomicU32,
    pub popped: AtomicU32,
    pub dropped: AtomicU32,
    pub max_depth: AtomicU32,
}
```

#### `QueueDrain<'a, const N: usize>` (struct)
Iterator that drains all events from queue.

### Constants

- `QUEUE_CAPACITY: usize = 64` - Default queue capacity

---

## Lookup Tables Module

The lookup module provides pre-computed lookup tables for physics calculations.

### Types

#### `LookupError` (enum)
```rust
pub enum LookupError {
    InputClamped { original: f32, clamped: f32 },
    IndexOutOfBounds,
}
```

#### `DewPointTable<const ROWS: usize, const COLS: usize>` (struct)
```rust
pub struct DewPointTable<const ROWS: usize, const COLS: usize> {
    temp_min: i8,
    temp_max: i8,
    temp_step: u8,
    rh_step: u8,
    values: &'static [[i8; COLS]; ROWS],
}
```

**Methods:**
- `lookup(&self, temp_c: f32, rh_percent: f32) -> LookupResult<f32>` - Look up dew point

#### `AltitudeTable` (struct)
```rust
pub struct AltitudeTable {
    alt_min: i16,
    alt_max: i16,
    alt_step: u16,
    adjustments: &'static [f32],
}
```

**Methods:**
- `get_adjustment(&self, altitude_m: f32) -> LookupResult<f32>` - Get pressure adjustment

### Constants

#### `DEW_POINT_STANDARD`
Standard dew point table:
- Temperature: -40 to 50°C in 5°C steps
- Humidity: 0 to 100% in 10% steps
- Memory: ~200 bytes

#### `AltitudeTable::STANDARD`
Standard altitude table:
- Altitude: -500 to 5000m in 100m steps
- Memory: ~224 bytes

### Functions

- `sin_lookup(angle_rad: f32) -> Option<f32>` - Fast sine approximation
- `cos_lookup(angle_rad: f32) -> Option<f32>` - Fast cosine approximation
- `tan_lookup(angle_rad: f32) -> Option<f32>` - Fast tangent approximation
- `dew_point_lookup(temp_c: f32, rh_percent: f32) -> Option<f32>` - Convenience function

---

## Common Types and Traits

### Error Types

#### `ValidationError` (enum)
```rust
pub enum ValidationError {
    OutOfRange { value: f32, min: f32, max: f32 },
    RateExceeded { rate: f32, max_rate: f32 },
    CrossValidationFailed { reason: &'static str },
    SensorQualityBad { reason: &'static str },
    InvalidValue,
    InsufficientData { required: usize, available: usize },
}
```

#### `ValidationResult<T>` (type alias)
```rust
pub type ValidationResult<T> = Result<T, ValidationError>;
```

#### `FusionResult<T>` (type alias)
```rust
pub type FusionResult<T> = Result<T, FusionError>;
```

#### `PipelineResult<T>` (type alias)
```rust
pub type PipelineResult<T> = Result<T, PipelineError>;
```

#### `LookupResult<T>` (type alias)
```rust
pub type LookupResult<T> = Result<T, LookupError>;
```

### Constants

- `MAX_HISTORY_SIZE: usize = 100` - Maximum history buffer size

---

## Usage Examples

### Basic Sensor Validation
```rust
use edgeguard_core::{
    validators::TemperatureValidator,
    traits::{Validator, ValidationContext},
};

let validator = TemperatureValidator::default();
let mut context = ValidationContext::default();
context.timestamp = 1000;

match validator.validate(25.0, &context) {
    Ok(()) => println!("Temperature valid"),
    Err(e) => println!("Validation failed: {:?}", e),
}
```

### Event Pipeline
```rust
use edgeguard_core::{
    pipeline::{Pipeline, ValidationStage},
    validators::TemperatureValidator,
    events::{EventBuilder, SensorType},
};

let mut pipeline = Pipeline::<8>::builder()
    .add_stage(ValidationStage::new(
        TemperatureValidator::default(),
        SensorType::Temperature,
    ))
    .build();

let event = EventBuilder::new(1000)
    .sensor("temp1", SensorType::Temperature)
    .reading(25.0, 0.95)
    .unwrap();

pipeline.push_event(event);
pipeline.process_batch(10)?;

while let Some(result) = pipeline.pop_result() {
    println!("Processed: {:?}", result);
}
```

### Sensor Fusion
```rust
use edgeguard_core::fusion::{
    KalmanFilter, KalmanConfig,
};

let config = KalmanConfig::<3, 2>::default()
    .with_process_noise(0.01)
    .with_measurement_noise([0.1, 0.15]);

let mut kf = KalmanFilter::new(config);

let measurements = [25.1, 25.3];
let (estimate, confidence) = kf.update(&measurements, 1000, None)?;
println!("Fused estimate: {}°C (confidence: {})", estimate, confidence.as_float());
```

### Stream Processing
```rust
use edgeguard_core::{
    stream::{MemoryStream, Stream},
    events::{Event, EventBuilder, SensorType},
};

let events = [
    EventBuilder::new(1000).sensor("t1", SensorType::Temperature).reading(25.0, 0.95).unwrap(),
    EventBuilder::new(2000).sensor("t1", SensorType::Temperature).reading(25.5, 0.95).unwrap(),
];

let mut stream = MemoryStream::new(&events);

while let Ok(event) = stream.poll_next() {
    match event {
        Event::SensorReading { value, .. } => println!("Reading: {}", value),
        _ => {}
    }
}
```