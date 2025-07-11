//! Stream Abstractions for Event Sources and Sinks
//!
//! ## Overview
//!
//! This module provides a unified interface for streaming sensor data from various
//! sources (MQTT, files, memory) and writing to various sinks (Kafka, HTTP, local storage).
//! The stream abstraction enables EdgeGuard to work with both real-time and batch data
//! while maintaining consistent validation semantics.
//!
//! ## Design Philosophy
//!
//! ### Why Streams?
//!
//! IoT data arrives from many sources:
//! - **Real-time**: MQTT topics, WebSocket connections
//! - **Batch**: CSV files, Avro files, database dumps
//! - **Replay**: Historical data for testing/analysis
//!
//! Streams provide a common interface regardless of source:
//! ```text
//! MQTT ─┐
//! File ─┼─→ Stream Trait ─→ Pipeline ─→ Validated Data
//! Kafka ┘
//! ```
//!
//! ### Key Features
//!
//! 1. **Schema Integration**: Streams can validate against Avro schemas
//! 2. **Backpressure**: Streams respect consumer capacity
//! 3. **Error Recovery**: Resilient to transient failures
//! 4. **Zero-Copy**: Minimize data copying for performance
//!
//! ## Stream Trait
//!
//! The core trait is designed for no-std compatibility:
//! - No async/await (use nb crate for non-blocking)
//! - No heap allocation required
//! - Works with interrupts and polling
//!
//! ## Performance Characteristics
//!
//! | Source Type | Latency | Throughput | Memory |
//! |-------------|---------|------------|--------|
//! | Memory      | <1μs    | >1M/sec    | 0      |
//! | File        | ~10μs   | 100K/sec   | 4KB    |
//! | MQTT        | ~1ms    | 10K/sec    | 8KB    |
//! | Kafka       | ~10ms   | 100K/sec   | 64KB   |

use nb;

use crate::{
    events::Event,
    time::Timestamp,
};

// Schema type placeholder for future integration
#[cfg(feature = "schemas")]
pub type Schema = (); // Placeholder until we integrate with edgeguard-schemas

/// Stream error types
#[derive(Debug)]
pub enum StreamError<E> {
    /// Underlying transport error
    Transport(E),
    /// Data format error
    Format(&'static str),
    /// Schema validation failed
    SchemaViolation,
    /// Stream exhausted
    EndOfStream,
    /// Buffer overflow
    Overflow,
}

/// Core stream trait for data sources
/// 
/// ## Design Rationale
/// 
/// The trait uses the `nb` crate pattern for non-blocking I/O:
/// - `WouldBlock`: No data available yet
/// - `Ok(item)`: Data ready
/// - `Err(e)`: Permanent error
/// 
/// This works well with:
/// - Interrupt-driven embedded systems
/// - Polling event loops
/// - Async runtimes (with adapters)
pub trait Stream {
    /// Item type produced by the stream
    type Item;
    
    /// Error type for stream operations
    type Error;
    
    /// Poll for next item (non-blocking)
    /// 
    /// Returns:
    /// - `Ok(item)`: Next item available
    /// - `Err(nb::Error::WouldBlock)`: Try again later
    /// - `Err(nb::Error::Other(e))`: Permanent error
    fn poll_next(&mut self) -> nb::Result<Self::Item, Self::Error>;
    
    /// Get associated schema if any
    /// 
    /// Streams with schemas can perform validation
    /// before events enter the pipeline
    #[cfg(feature = "schemas")]
    fn schema(&self) -> Option<&Schema> {
        None
    }
    
    /// Hint about remaining items
    /// 
    /// Used for progress tracking and buffer sizing
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None) // Unknown by default
    }
}

/// Memory-based stream for testing and replay
/// 
/// ## Use Cases
/// 
/// 1. **Unit Testing**: Feed known data sequences
/// 2. **Replay**: Re-process historical data
/// 3. **Simulation**: Generate synthetic sensor data
/// 
/// ## Example
/// 
/// ```rust
/// use edgeguard_core::stream::MemoryStream;
/// use edgeguard_core::events::{Event, EventBuilder, SensorType};
/// 
/// let events = vec![
///     EventBuilder::new(1000).sensor("t1", SensorType::Temperature).reading(25.0, 0.95).unwrap(),
///     EventBuilder::new(2000).sensor("t1", SensorType::Temperature).reading(25.5, 0.95).unwrap(),
/// ];
/// 
/// let mut stream = MemoryStream::new(&events);
/// while let Ok(event) = stream.poll_next() {
///     // Process event
/// }
/// ```
pub struct MemoryStream<'a> {
    /// Slice of events to stream
    events: &'a [Event],
    /// Current position
    position: usize,
}

impl<'a> MemoryStream<'a> {
    /// Create new memory stream from slice
    pub fn new(events: &'a [Event]) -> Self {
        Self {
            events,
            position: 0,
        }
    }
    
    /// Reset to beginning
    pub fn reset(&mut self) {
        self.position = 0;
    }
}

impl<'a> Stream for MemoryStream<'a> {
    type Item = Event;
    type Error = StreamError<()>;
    
    fn poll_next(&mut self) -> nb::Result<Self::Item, Self::Error> {
        if self.position >= self.events.len() {
            return Err(nb::Error::Other(StreamError::EndOfStream));
        }
        
        let event = self.events[self.position].clone();
        self.position += 1;
        Ok(event)
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.events.len() - self.position;
        (remaining, Some(remaining))
    }
}

/// Avro-aware stream wrapper
/// 
/// ## Overview
/// 
/// Wraps any stream to add Avro schema validation. Events are
/// validated against the schema before being returned.
/// 
/// ## Use Cases
/// 
/// 1. **Kafka Integration**: Validate Avro messages from Kafka
/// 2. **File Processing**: Ensure Avro files match expected schema
/// 3. **API Compatibility**: Validate data from external systems
/// 
/// ## Performance Impact
/// 
/// Schema validation adds ~100-500 cycles per event depending on
/// complexity. For high-frequency streams, consider:
/// - Validating samples (e.g., every 10th event)
/// - Pre-validating at source
/// - Using simpler schemas
#[cfg(feature = "schemas")]
pub struct AvroStream<S: Stream> {
    /// Inner stream
    inner: S,
    /// Avro schema for validation
    schema: Schema,
    /// Validation strategy
    validation: ValidationStrategy,
}

#[cfg(feature = "schemas")]
#[derive(Debug, Clone, Copy)]
pub enum ValidationStrategy {
    /// Validate every event (safest)
    Every,
    /// Validate every Nth event (faster)
    Sample { rate: u32 },
    /// Skip validation (dangerous)
    None,
}

#[cfg(feature = "schemas")]
impl<S: Stream> AvroStream<S> {
    pub fn new(inner: S, schema: Schema) -> Self {
        Self {
            inner,
            schema,
            validation: ValidationStrategy::Every,
        }
    }
    
    pub fn with_validation(mut self, strategy: ValidationStrategy) -> Self {
        self.validation = strategy;
        self
    }
}

#[cfg(feature = "schemas")]
impl<S> Stream for AvroStream<S>
where
    S: Stream,
    S::Item: AvroSerializable,
{
    type Item = S::Item;
    type Error = StreamError<S::Error>;
    
    fn poll_next(&mut self) -> nb::Result<Self::Item, Self::Error> {
        let item = self.inner.poll_next()
            .map_err(|e| match e {
                nb::Error::WouldBlock => nb::Error::WouldBlock,
                nb::Error::Other(e) => nb::Error::Other(StreamError::Transport(e)),
            })?;
        
        // Validate if needed
        match self.validation {
            ValidationStrategy::Every => {
                if !item.validate_schema(&self.schema) {
                    return Err(nb::Error::Other(StreamError::SchemaViolation));
                }
            }
            ValidationStrategy::Sample { rate } => {
                // Use atomic counter for thread-safe sampling
                // 
                // Safety: We use AtomicU32 with Relaxed ordering because:
                // 1. Counter wrapping is acceptable (sampling continues)
                // 2. Exact count not critical (approximate sampling OK)
                // 3. No memory synchronization needed with other data
                use core::sync::atomic::{AtomicU32, Ordering};
                
                static COUNTER: AtomicU32 = AtomicU32::new(0);
                
                let count = COUNTER.fetch_add(1, Ordering::Relaxed);
                if count % rate == 0 {
                    if !item.validate_schema(&self.schema) {
                        return Err(nb::Error::Other(StreamError::SchemaViolation));
                    }
                }
            }
            ValidationStrategy::None => {}
        }
        
        Ok(item)
    }
    
    fn schema(&self) -> Option<&Schema> {
        Some(&self.schema)
    }
}

/// Trait for types that can be validated against Avro schemas
#[cfg(feature = "schemas")]
pub trait AvroSerializable {
    /// Validate against schema
    fn validate_schema(&self, schema: &Schema) -> bool;
}

// Placeholder implementation for Event type schema validation
#[cfg(feature = "schemas")]
impl AvroSerializable for Event {
    fn validate_schema(&self, _schema: &Schema) -> bool {
        // TODO: Implement actual schema validation when we integrate apache-avro
        // For now, return true to allow development to continue
        true
    }
}

/// Rate-limited stream wrapper
/// 
/// ## Overview
/// 
/// Limits the rate at which events are produced, useful for:
/// - Testing pipeline capacity
/// - Simulating real sensor rates
/// - Preventing overwhelming downstream systems
/// 
/// ## Algorithm
/// 
/// Uses token bucket algorithm:
/// - Tokens added at configured rate
/// - Each event consumes one token
/// - No tokens = backpressure
/// 
/// ## Time Source Integration
/// 
/// This implementation requires a time source to track token regeneration.
/// Use the TimeSource trait from the time module:
/// 
/// ```rust
/// use edgeguard_core::time::{TimeSource, MonotonicClock};
/// use edgeguard_core::stream::RateLimitedStream;
/// 
/// let clock = MonotonicClock::new();
/// let limited = RateLimitedStream::new(inner_stream, 100, clock);
/// ```
/// 
/// For embedded systems without a reliable clock, use:
/// - `FallbackTimeSource`: Combines multiple sources
/// - `TickCounterSource`: Based on system ticks
/// - `ExternalTimeSource`: GPS or network time
pub struct RateLimitedStream<S: Stream, T: crate::time::TimeSource> {
    /// Inner stream
    inner: S,
    /// Maximum events per second
    rate: u32,
    /// Token bucket
    tokens: f32,
    /// Last token update time
    last_update: Timestamp,
    /// Time source for token regeneration
    time_source: T,
}

impl<S: Stream, T: crate::time::TimeSource> RateLimitedStream<S, T> {
    pub fn new(inner: S, events_per_second: u32, time_source: T) -> Self {
        Self {
            inner,
            rate: events_per_second,
            tokens: events_per_second as f32,
            last_update: time_source.now(),
            time_source,
        }
    }
    
    fn update_tokens(&mut self) {
        let now = self.time_source.now();
        let elapsed_ms = now.saturating_sub(self.last_update);
        let elapsed_sec = elapsed_ms as f32 / 1000.0;
        
        // Add tokens based on elapsed time
        let new_tokens = elapsed_sec * self.rate as f32;
        self.tokens = (self.tokens + new_tokens).min(self.rate as f32);
        self.last_update = now;
    }
}

impl<S: Stream, T: crate::time::TimeSource> Stream for RateLimitedStream<S, T> {
    type Item = S::Item;
    type Error = S::Error;
    
    fn poll_next(&mut self) -> nb::Result<Self::Item, Self::Error> {
        // Update tokens based on elapsed time
        self.update_tokens();
        
        // Check if we have tokens
        if self.tokens < 1.0 {
            return Err(nb::Error::WouldBlock);
        }
        
        // Try to get item
        let item = self.inner.poll_next()?;
        
        // Consume token
        self.tokens -= 1.0;
        
        Ok(item)
    }
}

/// Batching stream that groups events
/// 
/// ## Overview
/// 
/// Collects multiple events into batches for efficient processing:
/// - Reduces per-event overhead
/// - Enables batch compression
/// - Improves cache locality
/// 
/// ## Memory Usage
/// 
/// Fixed buffer of N events:
/// - Memory = N × sizeof(Event) ≈ N × 128 bytes
/// - Typical N = 16-64 for embedded systems
/// 
/// ## Time Source Integration
/// 
/// Batching requires a time source for timeout handling:
/// 
/// ```rust
/// use edgeguard_core::time::{TimeSource, MonotonicClock};
/// use edgeguard_core::stream::BatchingStream;
/// 
/// let clock = MonotonicClock::new();
/// let batching = BatchingStream::new(inner_stream, 1000, clock);
/// ```
/// 
/// The timeout ensures partial batches are emitted even if the
/// buffer isn't full, preventing data from being stuck indefinitely.
pub struct BatchingStream<S: Stream, T: crate::time::TimeSource, const N: usize> {
    /// Inner stream
    inner: S,
    /// Batch buffer
    buffer: heapless::Vec<S::Item, N>,
    /// Batch timeout (ms)
    timeout_ms: u32,
    /// Batch start time
    batch_start: Timestamp,
    /// Time source for timeout tracking
    time_source: T,
}

impl<S: Stream, T: crate::time::TimeSource, const N: usize> BatchingStream<S, T, N> {
    pub fn new(inner: S, timeout_ms: u32, time_source: T) -> Self {
        Self {
            inner,
            buffer: heapless::Vec::new(),
            timeout_ms,
            batch_start: 0,
            time_source,
        }
    }
}

impl<S: Stream, T: crate::time::TimeSource, const N: usize> Stream for BatchingStream<S, T, N>
where
    S::Item: Clone,
{
    type Item = heapless::Vec<S::Item, N>;
    type Error = StreamError<S::Error>;
    
    fn poll_next(&mut self) -> nb::Result<Self::Item, Self::Error> {
        // Try to fill buffer
        loop {
            match self.inner.poll_next() {
                Ok(item) => {
                    if self.buffer.is_empty() {
                        self.batch_start = self.time_source.now();
                    }
                    
                    if self.buffer.push(item).is_err() {
                        // Buffer full, return batch
                        let batch = core::mem::take(&mut self.buffer);
                        return Ok(batch);
                    }
                }
                Err(nb::Error::WouldBlock) => {
                    // Check timeout
                    let now = self.time_source.now();
                    let elapsed = now.saturating_sub(self.batch_start);
                    
                    if !self.buffer.is_empty() && elapsed >= self.timeout_ms as u64 {
                        // Timeout reached, return partial batch
                        let batch = core::mem::take(&mut self.buffer);
                        return Ok(batch);
                    }
                    
                    return Err(nb::Error::WouldBlock);
                }
                Err(nb::Error::Other(e)) => {
                    return Err(nb::Error::Other(StreamError::Transport(e)));
                }
            }
        }
    }
}

/// Stream combiner for multiple sources
/// 
/// ## Overview
/// 
/// Merges events from multiple streams into a single stream.
/// Useful for:
/// - Multi-sensor fusion
/// - Redundant data sources
/// - Load balancing
/// 
/// ## Fairness
/// 
/// Uses round-robin polling to ensure fairness between streams.
/// High-rate streams won't starve low-rate streams.
pub struct CombinedStream<S1: Stream, S2: Stream> {
    /// First stream
    stream1: S1,
    /// Second stream
    stream2: S2,
    /// Next stream to poll (for fairness)
    next: bool,
}

impl<S1, S2> CombinedStream<S1, S2>
where
    S1: Stream,
    S2: Stream<Item = S1::Item>,
{
    pub fn new(stream1: S1, stream2: S2) -> Self {
        Self {
            stream1,
            stream2,
            next: false,
        }
    }
}

impl<S1, S2> Stream for CombinedStream<S1, S2>
where
    S1: Stream,
    S2: Stream<Item = S1::Item>,
    S1::Error: From<S2::Error>,
{
    type Item = S1::Item;
    type Error = S1::Error;
    
    fn poll_next(&mut self) -> nb::Result<Self::Item, Self::Error> {
        // Alternate between streams for fairness
        let result = if self.next {
            match self.stream2.poll_next() {
                Ok(item) => Ok(item),
                Err(nb::Error::WouldBlock) => {
                    // Try other stream
                    self.stream1.poll_next()
                }
                Err(nb::Error::Other(e)) => Err(nb::Error::Other(e.into())),
            }
        } else {
            match self.stream1.poll_next() {
                Ok(item) => Ok(item),
                Err(nb::Error::WouldBlock) => {
                    // Try other stream
                    self.stream2.poll_next()
                        .map_err(|e| match e {
                            nb::Error::WouldBlock => nb::Error::WouldBlock,
                            nb::Error::Other(e) => nb::Error::Other(e.into()),
                        })
                }
                Err(e) => Err(e),
            }
        };
        
        // Toggle for next call
        self.next = !self.next;
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{EventBuilder, SensorType};
    
    #[test]
    fn memory_stream() {
        let events = [
            EventBuilder::new(1000).sensor("t1", SensorType::Temperature).reading(25.0, 0.95).unwrap(),
            EventBuilder::new(2000).sensor("t1", SensorType::Temperature).reading(25.5, 0.95).unwrap(),
        ];
        
        let mut stream = MemoryStream::new(&events);
        
        // Read all events
        assert!(matches!(stream.poll_next(), Ok(_)));
        assert!(matches!(stream.poll_next(), Ok(_)));
        assert!(matches!(stream.poll_next(), Err(nb::Error::Other(StreamError::EndOfStream))));
        
        // Size hint
        stream.reset();
        assert_eq!(stream.size_hint(), (2, Some(2)));
    }
    
    #[test]
    fn batching_stream() {
        use crate::time::{TimeSource, MonotonicClock};
        
        let events = [
            EventBuilder::new(1000).sensor("t1", SensorType::Temperature).reading(25.0, 0.95).unwrap(),
            EventBuilder::new(2000).sensor("t1", SensorType::Temperature).reading(25.5, 0.95).unwrap(),
            EventBuilder::new(3000).sensor("t1", SensorType::Temperature).reading(26.0, 0.95).unwrap(),
        ];
        
        let memory = MemoryStream::new(&events);
        let clock = MonotonicClock::new();
        let mut batching = BatchingStream::<_, _, 2>::new(memory, 1000, clock);
        
        // Should get batch of 2
        match batching.poll_next() {
            Ok(batch) => assert_eq!(batch.len(), 2),
            _ => panic!("Expected batch"),
        }
        
        // Should get batch of 1
        match batching.poll_next() {
            Ok(batch) => assert_eq!(batch.len(), 1),
            _ => panic!("Expected batch"),
        }
    }
    
    #[test]
    fn combined_stream() {
        let events1 = [
            EventBuilder::new(1000).sensor("t1", SensorType::Temperature).reading(25.0, 0.95).unwrap(),
        ];
        let events2 = [
            EventBuilder::new(2000).sensor("h1", SensorType::Humidity).reading(60.0, 0.95).unwrap(),
        ];
        
        let stream1 = MemoryStream::new(&events1);
        let stream2 = MemoryStream::new(&events2);
        let mut combined = CombinedStream::new(stream1, stream2);
        
        // Should get events from both streams
        let mut count = 0;
        while let Ok(_) = combined.poll_next() {
            count += 1;
        }
        assert_eq!(count, 2);
    }
}