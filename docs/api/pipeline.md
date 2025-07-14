# Pipeline API

Event-driven processing pipeline with composable stages.

## Pipeline

Main orchestrator for event processing with configurable stages.

### Constructor

```rust
impl<const N: usize> Pipeline<N> {
    pub fn builder() -> PipelineBuilder<N>;
    pub fn push_event(&mut self, event: Event) -> bool;
    pub fn process_batch(&mut self, max_events: usize) -> PipelineResult<usize>;
    pub fn pop_result(&mut self) -> Option<Event>;
    pub fn metrics(&self) -> &PipelineMetrics;
    pub fn reset(&mut self);
}
```

### Example Usage

```rust
let mut pipeline = Pipeline::<256>::builder()
    .add_stage(ValidationStage::new(
        TemperatureValidator::new(),
        SensorType::Temperature
    ))
    .add_stage(CrossValidationStage::new())
    .build();

// Process events
pipeline.push_event(temperature_event);
let processed = pipeline.process_batch(100)?;

// Get results
while let Some(result) = pipeline.pop_result() {
    handle_result(result);
}
```

## PipelineBuilder

Builder pattern for constructing pipelines with stages.

### Methods

```rust
impl<const N: usize> PipelineBuilder<N> {
    pub fn add_stage(self, stage: impl PipelineStage + 'static) -> Self;
    pub fn backpressure(self, strategy: BackpressureStrategy) -> Self;
    pub fn error_handling(self, strategy: ErrorStrategy) -> Self;
    pub fn metrics(self, enabled: bool) -> Self;
    pub fn build(self) -> Pipeline<N>;
}
```

### Configuration

```rust
let pipeline = Pipeline::<512>::builder()
    .add_stage(ValidationStage::new(...))
    .add_stage(FusionStage::new(...))
    .backpressure(BackpressureStrategy::DropOldest)
    .error_handling(ErrorStrategy::Continue)
    .metrics(true)
    .build();
```

## PipelineStage Trait

Core trait for implementing pipeline stages.

```rust
pub trait PipelineStage: Send {
    fn process(&mut self, event: Event, output: &mut StageOutput) -> PipelineResult<()>;
    fn name(&self) -> &'static str { "unnamed" }
    fn reset(&mut self) {}
    fn metrics(&self) -> Option<StageMetrics> { None }
}
```

### StageOutput

Helper for stage output handling:

```rust
pub struct StageOutput {
    pub fn push(&mut self, event: Event) -> bool;
    pub fn push_all(&mut self, events: &[Event]) -> usize;
    pub fn is_full(&self) -> bool;
    pub fn available_space(&self) -> usize;
}
```

## Built-in Stages

### ValidationStage

Validates sensor data using physics-aware validators.

```rust
impl<V: Validator + Send> ValidationStage<V> {
    pub fn new(validator: V, sensor_type: SensorType) -> Self;
    pub fn with_quality_threshold(mut self, threshold: f32) -> Self;
    pub fn with_error_strategy(mut self, strategy: ErrorStrategy) -> Self;
}
```

#### Example

```rust
let stage = ValidationStage::new(
    TemperatureValidator::new().with_range(-20.0, 60.0),
    SensorType::Temperature
).with_quality_threshold(0.8);
```

### FusionStage

Multi-sensor data fusion with confidence scoring.

```rust
impl FusionStage {
    pub fn new(algorithm: Box<dyn FusionAlgorithm>) -> Self;
    pub fn add_sensor_group(&mut self, group_id: &str, sensors: &[&str]) -> Result<(), PipelineError>;
    pub fn with_fusion_window(mut self, window_ms: u32) -> Self;
    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self;
}
```

#### Example

```rust
let mut stage = FusionStage::new(
    Box::new(KalmanFilter::default())
);
stage.add_sensor_group("temperature", &["temp_01", "temp_02", "temp_03"])?;

let stage = stage
    .with_fusion_window(5000)  // 5 second window
    .with_confidence_threshold(0.7);
```

### CrossValidationStage

Cross-sensor validation for detecting correlated failures.

```rust
impl CrossValidationStage {
    pub fn new() -> Self;
    pub fn add_pair(&mut self, primary: SensorType, secondary: SensorType, validation_type: CrossValidationType) -> Result<(), PipelineError>;
    pub fn with_time_window(mut self, window_ms: u32) -> Self;
    pub fn with_correlation_threshold(mut self, threshold: f32) -> Self;
}
```

#### Example

```rust
let mut stage = CrossValidationStage::new()
    .with_time_window(5000)
    .with_correlation_threshold(0.85);

stage.add_pair(
    SensorType::Temperature,
    SensorType::Humidity,
    CrossValidationType::DewPoint
)?;
```

### AggregationStage

Data aggregation for bandwidth reduction.

```rust
impl AggregationStage {
    pub fn new(window: WindowSpec, method: AggregationMethod, sensor_type: SensorType) -> Self;
    pub fn with_compression_ratio(mut self, ratio: f32) -> Self;
    pub fn with_outlier_detection(mut self, enabled: bool) -> Self;
}
```

#### Configuration

```rust
let stage = AggregationStage::new(
    WindowSpec::TimeWindow(60_000),  // 1 minute window
    AggregationMethod::Statistics,
    SensorType::Temperature
).with_compression_ratio(0.1);  // 90% reduction
```

### RouterStage

Routes events to different stages based on conditions.

```rust
impl RouterStage {
    pub fn new() -> Self;
    pub fn add_route(&mut self, condition: RouteCondition, stage: Box<dyn PipelineStage>) -> Result<(), PipelineError>;
    pub fn add_default_route(&mut self, stage: Box<dyn PipelineStage>) -> Result<(), PipelineError>;
}
```

#### Example

```rust
let mut router = RouterStage::new();
router.add_route(
    RouteCondition::SensorType(SensorType::Temperature),
    Box::new(ValidationStage::new(
        TemperatureValidator::new(),
        SensorType::Temperature
    ))
)?;
```

### FilterStage

Filters events based on predicates.

```rust
impl<F> FilterStage<F> 
where F: Fn(&Event) -> bool + Send
{
    pub fn new(predicate: F) -> Self;
    pub fn with_invert(mut self, invert: bool) -> Self;
}
```

#### Example

```rust
let stage = FilterStage::new(|event| {
    match event {
        Event::SensorReading { quality, .. } => *quality > 0.8,
        _ => true,
    }
});
```

## Configuration Types

### BackpressureStrategy

```rust
pub enum BackpressureStrategy {
    DropOldest,      // Drop oldest events when full
    DropNewest,      // Drop newest events when full
    Block,           // Block until space available
    Overflow,        // Allow overflow with warning
}
```

### ErrorStrategy

```rust
pub enum ErrorStrategy {
    Continue,        // Continue processing on errors
    Halt,           // Stop processing on errors
    Retry { count: usize, delay_ms: u32 },  // Retry with backoff
}
```

### WindowSpec

```rust
pub enum WindowSpec {
    TimeWindow(u32),         // Time-based window (milliseconds)
    CountWindow(usize),      // Count-based window
    SlidingWindow { size: usize, step: usize },  // Sliding window
}
```

### AggregationMethod

```rust
pub enum AggregationMethod {
    Statistics,      // Mean, min, max, std dev
    Median,         // Median value
    Percentiles,    // 25th, 50th, 75th percentiles
    Histogram,      // Value distribution
}
```

## Metrics

### PipelineMetrics

```rust
pub struct PipelineMetrics {
    pub events_processed: u64,
    pub events_dropped: u64,
    pub processing_errors: u64,
    pub average_latency_us: f64,
    pub throughput_events_per_sec: f64,
    pub stage_metrics: Vec<StageMetrics>,
}
```

### StageMetrics

```rust
pub struct StageMetrics {
    pub name: &'static str,
    pub events_processed: u64,
    pub processing_time_us: u64,
    pub errors: u64,
    pub utilization: f32,
}
```

### Monitoring

```rust
let metrics = pipeline.metrics();
println!("Processed: {}", metrics.events_processed);
println!("Dropped: {}", metrics.events_dropped);
println!("Throughput: {:.2} events/sec", metrics.throughput_events_per_sec);

for stage in &metrics.stage_metrics {
    println!("Stage {}: {:.2}% utilization", stage.name, stage.utilization * 100.0);
}
```

## StreamProcessor

Integrates streams with pipeline processing.

```rust
impl<S, const N: usize> StreamProcessor<S, N> 
where S: Stream<Item = Event>
{
    pub fn new(stream: S, pipeline: Pipeline<N>) -> Self;
    pub fn process_next(&mut self) -> nb::Result<Option<Event>, StreamError<S::Error>>;
    pub fn process_batch(&mut self, max_events: usize) -> Result<usize, StreamError<S::Error>>;
    pub fn metrics(&self) -> &StreamProcessorMetrics;
}
```

### Example

```rust
let stream = MemoryStream::new(&events);
let pipeline = Pipeline::<256>::builder()
    .add_stage(ValidationStage::new(...))
    .build();

let mut processor = StreamProcessor::new(stream, pipeline);

// Process events
loop {
    match processor.process_next() {
        Ok(Some(result)) => handle_result(result),
        Ok(None) => break,
        Err(nb::Error::WouldBlock) => continue,
        Err(nb::Error::Other(e)) => handle_error(e),
    }
}
```

## Error Handling

### PipelineError

```rust
#[derive(Debug)]
pub enum PipelineError {
    QueueFull,
    InvalidStage(&'static str),
    ProcessingError(Box<dyn std::error::Error + Send>),
    ConfigurationError(&'static str),
    ResourceExhausted,
}
```

### Error Recovery

```rust
match pipeline.process_batch(100) {
    Ok(processed) => println!("Processed {} events", processed),
    Err(PipelineError::QueueFull) => {
        // Handle backpressure
        std::thread::sleep(Duration::from_millis(10));
    }
    Err(PipelineError::ProcessingError(e)) => {
        // Log and continue
        log::error!("Processing error: {}", e);
    }
}
```

## Custom Stages

Implement custom processing stages:

```rust
struct CustomStage {
    counter: u64,
}

impl PipelineStage for CustomStage {
    fn process(&mut self, event: Event, output: &mut StageOutput) -> PipelineResult<()> {
        self.counter += 1;
        
        // Custom processing logic
        let modified_event = match event {
            Event::SensorReading { mut value, .. } => {
                value *= 1.1;  // Apply calibration
                event  // Return modified event
            }
            _ => event,
        };
        
        output.push(modified_event);
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "custom_stage"
    }
    
    fn reset(&mut self) {
        self.counter = 0;
    }
}
```

## Best Practices

### Stage Ordering

```rust
// Logical stage ordering
let pipeline = Pipeline::<512>::builder()
    .add_stage(ValidationStage::new(...))      // Validate first
    .add_stage(CrossValidationStage::new())    // Cross-validate
    .add_stage(FusionStage::new(...))          // Fuse sensors
    .add_stage(AggregationStage::new(...))     // Aggregate last
    .build();
```

### Memory Management

```rust
// Choose appropriate buffer sizes
let pipeline = Pipeline::<64>::builder()    // Small for ESP32
    .add_stage(...)
    .build();

let pipeline = Pipeline::<2048>::builder()  // Large for high throughput
    .add_stage(...)
    .build();
```

### Performance Monitoring

```rust
// Regular metrics collection
let metrics = pipeline.metrics();
if metrics.events_dropped > metrics.events_processed / 10 {
    log::warn!("High drop rate: {:.2}%", 
        metrics.events_dropped as f64 / metrics.events_processed as f64 * 100.0);
}
```

This pipeline API provides flexible, composable event processing with performance monitoring and error handling suitable for edge device deployment.