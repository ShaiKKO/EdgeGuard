# Streams API

Advanced streaming capabilities for real-time sensor data processing.

## Stream Trait

Core abstraction for processing streaming sensor data with non-blocking I/O.

### Trait Definition

```rust
pub trait Stream {
    type Item;
    type Error;
    
    fn next(&mut self) -> nb::Result<Self::Item, Self::Error>;
    fn process_batch(&mut self, items: &mut [Self::Item]) -> nb::Result<usize, Self::Error>;
    fn size_hint(&self) -> (usize, Option<usize>);
    fn is_exhausted(&self) -> bool;
}
```

### Implementation Pattern

```rust
use nb::{Error, Result};

struct CustomStream {
    data: Vec<Event>,
    position: usize,
}

impl Stream for CustomStream {
    type Item = Event;
    type Error = StreamError;
    
    fn next(&mut self) -> Result<Event, StreamError> {
        if self.position >= self.data.len() {
            return Err(Error::Other(StreamError::EndOfStream));
        }
        
        let item = self.data[self.position].clone();
        self.position += 1;
        Ok(item)
    }
    
    fn process_batch(&mut self, items: &mut [Event]) -> Result<usize, StreamError> {
        let available = self.data.len() - self.position;
        let to_copy = items.len().min(available);
        
        items[..to_copy].copy_from_slice(&self.data[self.position..self.position + to_copy]);
        self.position += to_copy;
        
        if to_copy == 0 {
            Err(Error::Other(StreamError::EndOfStream))
        } else {
            Ok(to_copy)
        }
    }
}
```

## MemoryStream

In-memory stream for testing and finite data processing.

### Constructor

```rust
impl<'a> MemoryStream<'a> {
    pub fn new(events: &'a [Event]) -> Self;
    pub fn with_repeat(events: &'a [Event], repeat_count: usize) -> Self;
    pub fn reset(&mut self);
    pub fn position(&self) -> usize;
    pub fn remaining(&self) -> usize;
}
```

### Usage

```rust
let events = vec![
    EventBuilder::new(1000).sensor("temp_01", SensorType::Temperature).reading(23.5, 0.95).unwrap(),
    EventBuilder::new(2000).sensor("temp_01", SensorType::Temperature).reading(24.0, 0.90).unwrap(),
    EventBuilder::new(3000).sensor("temp_01", SensorType::Temperature).reading(24.5, 0.85).unwrap(),
];

let mut stream = MemoryStream::new(&events);

// Process events one by one
while let Ok(event) = stream.next() {
    match event {
        Event::SensorReading { value, timestamp, .. } => {
            println!("Temperature: {}°C at {}", value, timestamp);
        }
        _ => {}
    }
}

// Reset and process in batches
stream.reset();
let mut batch = [Event::default(); 10];
match stream.process_batch(&mut batch) {
    Ok(count) => println!("Processed {} events", count),
    Err(nb::Error::Other(StreamError::EndOfStream)) => println!("Stream exhausted"),
    Err(nb::Error::WouldBlock) => println!("No data available"),
}
```

## FileStream

File-based streaming with CSV and JSON support.

### Constructor

```rust
impl FileStream {
    pub fn csv(path: &str, format: CsvFormat) -> Result<Self, StreamError>;
    pub fn json_lines(path: &str) -> Result<Self, StreamError>;
    pub fn with_buffer_size(mut self, size: usize) -> Self;
    pub fn with_skip_errors(mut self, skip: bool) -> Self;
    pub fn with_schema(mut self, schema: &Schema) -> Self;
}
```

### CSV Processing

```rust
let csv_format = CsvFormat {
    delimiter: b',',
    has_header: true,
    timestamp_column: 0,
    sensor_id_column: 1,
    value_column: 2,
    quality_column: Some(3),
};

let mut stream = FileStream::csv("sensor_data.csv", csv_format)?
    .with_buffer_size(8192)
    .with_skip_errors(true);

// Process CSV data
let mut events_processed = 0;
loop {
    match stream.next() {
        Ok(event) => {
            events_processed += 1;
            process_event(event);
        }
        Err(nb::Error::Other(StreamError::EndOfStream)) => break,
        Err(nb::Error::WouldBlock) => continue,
        Err(nb::Error::Other(e)) => {
            eprintln!("Stream error: {:?}", e);
            break;
        }
    }
}

println!("Processed {} events from CSV", events_processed);
```

### JSON Lines Processing

```rust
let mut stream = FileStream::json_lines("sensor_data.jsonl")?;

// Process JSON lines
while let Ok(event) = stream.next() {
    if let Event::SensorReading { sensor_id, value, .. } = event {
        println!("Sensor {}: {}", sensor_id.as_str(), value);
    }
}
```

## Stream Adapters

Composable stream transformations for advanced processing.

### RateLimitedStream

```rust
impl<S: Stream> RateLimitedStream<S> {
    pub fn new(inner: S, events_per_second: u32) -> Self;
    pub fn with_burst_size(mut self, burst_size: u32) -> Self;
    pub fn with_time_source(mut self, time_source: Box<dyn TimeSource>) -> Self;
    pub fn current_rate(&self) -> f32;
    pub fn tokens_available(&self) -> u32;
}
```

#### Usage

```rust
let base_stream = MemoryStream::new(&events);
let mut rate_limited = RateLimitedStream::new(base_stream, 100)  // 100 events/sec
    .with_burst_size(50);  // Allow bursts up to 50 events

// Rate-limited processing
while let Ok(event) = rate_limited.next() {
    process_event(event);
    // Automatically rate-limited to 100 events/sec
}

// Check current rate
println!("Current rate: {:.2} events/sec", rate_limited.current_rate());
```

### BatchingStream

```rust
impl<S: Stream> BatchingStream<S> {
    pub fn new(inner: S, batch_size: usize, timeout_ms: u32) -> Self;
    pub fn with_min_batch_size(mut self, min_size: usize) -> Self;
    pub fn flush(&mut self) -> Result<Vec<S::Item>, StreamError>;
    pub fn pending_count(&self) -> usize;
}
```

#### Usage

```rust
let base_stream = FileStream::csv("large_dataset.csv", format)?;
let mut batching = BatchingStream::new(base_stream, 100, 1000);  // 100 events or 1 second

// Process in batches
while let Ok(batch) = batching.next() {
    // batch is Vec<Event> with up to 100 events
    process_batch(&batch);
}

// Force flush remaining events
if let Ok(remaining) = batching.flush() {
    if !remaining.is_empty() {
        process_batch(&remaining);
    }
}
```

### BackpressureStream

```rust
impl<S: Stream> BackpressureStream<S> {
    pub fn new(inner: S, config: BackpressureConfig) -> Self;
    pub fn with_strategy(mut self, strategy: BackpressureStrategy) -> Self;
    pub fn is_under_pressure(&self) -> bool;
    pub fn pressure_ratio(&self) -> f32;
}
```

#### Usage

```rust
let config = BackpressureConfig {
    high_watermark: 1000,
    low_watermark: 500,
    buffer_size: 2000,
};

let base_stream = FileStream::csv("sensor_data.csv", format)?;
let mut backpressure = BackpressureStream::new(base_stream, config)
    .with_strategy(BackpressureStrategy::DropOldest);

// Automatic backpressure handling
while let Ok(event) = backpressure.next() {
    if backpressure.is_under_pressure() {
        // Reduce processing complexity under pressure
        process_event_fast(event);
    } else {
        process_event_full(event);
    }
}
```

### FilterStream

```rust
impl<S: Stream, F> FilterStream<S, F> 
where F: Fn(&S::Item) -> bool
{
    pub fn new(inner: S, predicate: F) -> Self;
    pub fn filtered_count(&self) -> usize;
    pub fn passed_count(&self) -> usize;
}
```

#### Usage

```rust
let base_stream = MemoryStream::new(&events);
let mut filtered = FilterStream::new(base_stream, |event| {
    // Only pass high-quality temperature readings
    matches!(event, Event::SensorReading { 
        sensor_type: SensorType::Temperature, 
        quality, 
        .. 
    } if *quality > 0.8)
});

// Process filtered events
while let Ok(event) = filtered.next() {
    // Only high-quality temperature events
    process_high_quality_event(event);
}

println!("Filtered {} events, passed {}", 
    filtered.filtered_count(), 
    filtered.passed_count());
```

### MapStream

```rust
impl<S: Stream, F, T> MapStream<S, F, T>
where F: Fn(S::Item) -> T
{
    pub fn new(inner: S, mapper: F) -> Self;
    pub fn mapped_count(&self) -> usize;
}
```

#### Usage

```rust
let base_stream = MemoryStream::new(&events);
let mut mapped = MapStream::new(base_stream, |event| {
    // Convert temperature events to Celsius if needed
    if let Event::SensorReading { mut value, sensor_type: SensorType::Temperature, .. } = event {
        if value > 100.0 {  // Assume Kelvin if > 100
            value -= 273.15;
        }
        Event::SensorReading { value, ..event }
    } else {
        event
    }
});

while let Ok(event) = mapped.next() {
    // Process normalized temperature events
    process_normalized_event(event);
}
```

## Stream Combinators

Combine multiple streams for complex processing scenarios.

### MergeStream

```rust
impl<S1: Stream, S2: Stream> MergeStream<S1, S2> {
    pub fn new(stream1: S1, stream2: S2) -> Self;
    pub fn with_priority(mut self, priority: MergePriority) -> Self;
    pub fn stream1_count(&self) -> usize;
    pub fn stream2_count(&self) -> usize;
}
```

#### Usage

```rust
let temperature_stream = FileStream::csv("temperature.csv", temp_format)?;
let humidity_stream = FileStream::csv("humidity.csv", humidity_format)?;

let mut merged = MergeStream::new(temperature_stream, humidity_stream)
    .with_priority(MergePriority::Timestamp);  // Merge by timestamp

// Process merged stream
while let Ok(event) = merged.next() {
    match event {
        Event::SensorReading { sensor_type: SensorType::Temperature, .. } => {
            process_temperature(event);
        }
        Event::SensorReading { sensor_type: SensorType::Humidity, .. } => {
            process_humidity(event);
        }
        _ => {}
    }
}
```

### ZipStream

```rust
impl<S1: Stream, S2: Stream> ZipStream<S1, S2> {
    pub fn new(stream1: S1, stream2: S2) -> Self;
    pub fn with_timeout(mut self, timeout_ms: u32) -> Self;
    pub fn sync_count(&self) -> usize;
    pub fn timeout_count(&self) -> usize;
}
```

#### Usage

```rust
let temp_stream = FileStream::csv("temperature.csv", format)?;
let pressure_stream = FileStream::csv("pressure.csv", format)?;

let mut zipped = ZipStream::new(temp_stream, pressure_stream)
    .with_timeout(1000);  // 1 second timeout for synchronization

// Process synchronized pairs
while let Ok((temp_event, pressure_event)) = zipped.next() {
    // Process correlated temperature and pressure readings
    process_correlated_readings(temp_event, pressure_event);
}
```

## Stream Configuration

### BackpressureConfig

```rust
pub struct BackpressureConfig {
    pub high_watermark: usize,
    pub low_watermark: usize,
    pub buffer_size: usize,
    pub strategy: BackpressureStrategy,
}
```

### CsvFormat

```rust
pub struct CsvFormat {
    pub delimiter: u8,
    pub has_header: bool,
    pub timestamp_column: usize,
    pub sensor_id_column: usize,
    pub value_column: usize,
    pub quality_column: Option<usize>,
    pub sensor_type_column: Option<usize>,
}
```

### MergePriority

```rust
pub enum MergePriority {
    RoundRobin,         // Alternate between streams
    Timestamp,          // Order by timestamp
    QualityFirst,       // Higher quality first
    Stream1First,       // Prefer stream 1
    Stream2First,       // Prefer stream 2
}
```

### BackpressureStrategy

```rust
pub enum BackpressureStrategy {
    DropOldest,         // Drop oldest buffered items
    DropNewest,         // Drop newest items
    Block,              // Block until buffer space available
    Panic,              // Panic on buffer overflow
}
```

## Performance Characteristics

### Throughput

- **Memory streams**: 1M+ events/sec
- **File streams**: 100k+ events/sec (depends on I/O)
- **Rate limited**: Configurable up to stream capacity
- **Filtered streams**: 500k+ events/sec (depends on filter complexity)

### Memory Usage

- **Base stream**: <1KB overhead
- **Buffered streams**: Buffer size × item size
- **Batching streams**: Batch size × item size
- **Backpressure streams**: Buffer size × item size

### Latency

- **Memory streams**: <1μs per event
- **File streams**: <100μs per event
- **Network streams**: 1-10ms per event
- **Batch processing**: Batch timeout + processing time

## Error Handling

### StreamError

```rust
pub enum StreamError {
    IoError(std::io::Error),
    ParseError(String),
    SchemaViolation(String),
    EndOfStream,
    Timeout,
    BufferOverflow,
    NetworkError(String),
    InvalidFormat(String),
}
```

### Error Recovery

```rust
// Robust stream processing with error recovery
let mut stream = FileStream::csv("sensor_data.csv", format)?
    .with_skip_errors(true);

let mut error_count = 0;
let mut success_count = 0;

loop {
    match stream.next() {
        Ok(event) => {
            success_count += 1;
            process_event(event);
        }
        Err(nb::Error::Other(StreamError::EndOfStream)) => {
            println!("Stream completed: {} success, {} errors", success_count, error_count);
            break;
        }
        Err(nb::Error::Other(StreamError::ParseError(msg))) => {
            error_count += 1;
            log::warn!("Parse error: {}", msg);
            // Continue processing despite parse errors
        }
        Err(nb::Error::WouldBlock) => {
            // No data available, continue
            continue;
        }
        Err(nb::Error::Other(e)) => {
            error_count += 1;
            log::error!("Stream error: {:?}", e);
            if error_count > 100 {
                log::error!("Too many errors, stopping stream");
                break;
            }
        }
    }
}
```

## Integration Examples

### Pipeline Integration

```rust
use edgeguard::pipeline::StreamProcessor;

let csv_stream = FileStream::csv("sensor_data.csv", format)?
    .with_buffer_size(16384);

let rate_limited = RateLimitedStream::new(csv_stream, 1000);  // 1000 events/sec

let pipeline = Pipeline::<512>::builder()
    .add_stage(ValidationStage::new(
        TemperatureValidator::new(),
        SensorType::Temperature
    ))
    .add_stage(FusionStage::new(
        Box::new(KalmanFilter::default())
    ))
    .build();

let mut processor = StreamProcessor::new(rate_limited, pipeline);

// Process stream through pipeline
while let Ok(Some(result)) = processor.process_next() {
    match result {
        Event::ValidationResult { status: ValidationStatus::Valid, .. } => {
            // Process valid result
        }
        Event::FusionResult { value, confidence, .. } => {
            println!("Fused value: {} (confidence: {})", value, confidence.as_f32());
        }
        _ => {}
    }
}
```

### Real-time Processing

```rust
use std::time::Duration;
use tokio::time::sleep;

// Real-time stream processing with backpressure
let network_stream = NetworkStream::new("tcp://sensor-gateway:8080")?;
let backpressure = BackpressureStream::new(network_stream, BackpressureConfig::default());
let batching = BatchingStream::new(backpressure, 100, 1000);

// Process real-time data
tokio::spawn(async move {
    while let Ok(batch) = batching.next().await {
        // Process batch of events
        for event in batch {
            process_real_time_event(event);
        }
        
        // Small delay to prevent overwhelming the processor
        sleep(Duration::from_millis(10)).await;
    }
});
```

### Multi-Source Processing

```rust
// Process multiple sensor sources simultaneously
let temp_stream = FileStream::csv("temperature.csv", temp_format)?;
let humidity_stream = FileStream::csv("humidity.csv", humidity_format)?;
let pressure_stream = FileStream::csv("pressure.csv", pressure_format)?;

// Merge all streams
let temp_humidity = MergeStream::new(temp_stream, humidity_stream)
    .with_priority(MergePriority::Timestamp);

let all_sensors = MergeStream::new(temp_humidity, pressure_stream)
    .with_priority(MergePriority::Timestamp);

// Apply rate limiting to merged stream
let rate_limited = RateLimitedStream::new(all_sensors, 500);

// Process unified stream
while let Ok(event) = rate_limited.next() {
    match event {
        Event::SensorReading { sensor_type, value, .. } => {
            match sensor_type {
                SensorType::Temperature => process_temperature(value),
                SensorType::Humidity => process_humidity(value),
                SensorType::Pressure => process_pressure(value),
                _ => {}
            }
        }
        _ => {}
    }
}
```

## Best Practices

### Stream Composition

```rust
// Build complex stream pipelines
let base_stream = FileStream::csv("large_dataset.csv", format)?;

let processed_stream = base_stream
    .filter(|event| event.quality() > 0.8)          // High quality only
    .rate_limit(1000)                                // Limit processing rate
    .batch(100, Duration::from_millis(500))          // Batch for efficiency
    .with_backpressure(BackpressureConfig::default()); // Handle pressure

// Process composed stream
while let Ok(batch) = processed_stream.next() {
    process_batch_efficiently(&batch);
}
```

### Memory Management

```rust
// Configure appropriate buffer sizes
let stream = FileStream::csv("data.csv", format)?
    .with_buffer_size(64 * 1024);  // 64KB buffer for file I/O

let batching = BatchingStream::new(stream, 1000, 5000)  // Large batches
    .with_min_batch_size(100);  // Minimum batch size

// Monitor memory usage
let metrics = batching.metrics();
if metrics.memory_usage > 10 * 1024 * 1024 {  // 10MB
    log::warn!("High memory usage: {} bytes", metrics.memory_usage);
}
```

### Error Handling

```rust
// Implement circuit breaker pattern
let mut consecutive_errors = 0;
const MAX_CONSECUTIVE_ERRORS: usize = 10;

while let result = stream.next() {
    match result {
        Ok(event) => {
            consecutive_errors = 0;
            process_event(event);
        }
        Err(nb::Error::Other(e)) => {
            consecutive_errors += 1;
            log::error!("Stream error: {:?}", e);
            
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                log::error!("Too many consecutive errors, stopping stream");
                break;
            }
        }
        Err(nb::Error::WouldBlock) => {
            // Expected for non-blocking streams
            continue;
        }
    }
}
```

This streams API provides comprehensive real-time data processing capabilities with advanced stream composition, backpressure handling, and performance optimization suitable for high-throughput edge applications.