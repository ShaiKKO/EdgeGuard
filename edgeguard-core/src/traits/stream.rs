//! Stream Processing Traits
//!
//! This module defines the core traits for stream-based processing of sensor data.
//! The design follows a pull-based model using the `nb` crate for non-blocking I/O,
//! making it suitable for embedded systems without async runtime overhead.
//!
//! ## Design Philosophy
//!
//! - **Pull-based**: Consumers control when to read data
//! - **Non-blocking**: Uses `nb::Result` for async without runtime
//! - **Backpressure-aware**: Explicit support for flow control
//! - **Memory-efficient**: No hidden allocations
//!
//! ## Common Patterns
//!
//! ```rust
//! use edgeguard_core::traits::Stream;
//! use nb;
//!
//! fn process_stream<S: Stream>(stream: &mut S) -> Result<(), S::Error> {
//!     loop {
//!         match stream.poll_next() {
//!             Ok(item) => process_item(item),
//!             Err(nb::Error::WouldBlock) => {
//!                 // No data available, try again later
//!                 return Ok(());
//!             }
//!             Err(nb::Error::Other(e)) => return Err(e),
//!         }
//!     }
//! }
//! ```

// Note: StreamError enum remains in stream/mod.rs as it's a concrete type, not a trait

/// Core stream trait for event sources
///
/// Streams provide sensor events using a pull-based model compatible
/// with embedded systems. The trait uses `nb::Result` for non-blocking
/// operation without async/await overhead.
///
/// ## Implementation Notes
///
/// - Implementations should be lazy and pull-based
/// - Use `nb::Error::WouldBlock` when no data is available
/// - Memory allocation should be predictable/bounded
/// - Consider implementing `size_hint()` for optimization
///
/// ## Example Implementation
///
/// ```rust
/// use edgeguard_core::traits::Stream;
/// use edgeguard_core::stream::StreamError;
/// use nb;
///
/// struct SensorStream {
///     // ... sensor interface
/// }
///
/// impl Stream for SensorStream {
///     type Item = f32;
///     type Error = StreamError<std::io::Error>;
///     
///     fn poll_next(&mut self) -> nb::Result<Self::Item, Self::Error> {
///         // Try to read from sensor
///         if let Some(value) = self.try_read()? {
///             Ok(value)
///         } else {
///             Err(nb::Error::WouldBlock)
///         }
///     }
///     
///     fn size_hint(&self) -> (usize, Option<usize>) {
///         // Sensor streams are typically unbounded
///         (0, None)
///     }
/// }
/// ```
///
/// ## Error Handling
///
/// Streams use a two-level error model:
/// - `nb::Error::WouldBlock` - Temporary unavailability
/// - `nb::Error::Other(E)` - Actual stream errors
///
/// This allows consumers to distinguish between "try again later"
/// and "something went wrong" scenarios.
pub trait Stream {
    /// Type of items produced by the stream
    type Item;
    
    /// Type of errors that can occur
    type Error;
    
    /// Attempt to pull the next item from the stream
    ///
    /// Returns:
    /// - `Ok(item)` - Next item available
    /// - `Err(nb::Error::WouldBlock)` - No data available yet
    /// - `Err(nb::Error::Other(e))` - Stream error occurred
    ///
    /// ## Contract
    ///
    /// - This method should not block indefinitely
    /// - Multiple `WouldBlock` returns are normal and expected
    /// - After returning an error, the stream may still be usable
    /// - `EndOfStream` errors should be consistent (sticky)
    fn poll_next(&mut self) -> nb::Result<Self::Item, Self::Error>;
    
    /// Returns bounds on remaining items
    ///
    /// Similar to `Iterator::size_hint()`. The first element is the
    /// minimum bound, the second is the optional maximum bound.
    ///
    /// ## Uses
    ///
    /// - Pre-allocating buffers for known-size streams
    /// - Progress reporting for finite streams
    /// - Optimization hints for processing pipelines
    ///
    /// Default implementation returns `(0, None)` indicating unknown size.
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
    
    /// Returns the schema for validation (if applicable)
    ///
    /// This method is only available when the `schemas` feature is enabled.
    /// It allows streams to provide type information for dynamic validation.
    #[cfg(feature = "schemas")]
    fn schema(&self) -> Option<&crate::schemas::Schema> {
        None
    }
}

/// Stream that supports backpressure signaling
///
/// This trait extends `Stream` with explicit backpressure control, allowing
/// consumers to signal when they're overwhelmed and need the producer to
/// slow down or pause.
///
/// ## Use Cases
///
/// - Network streams with flow control
/// - Multi-stage pipelines with different processing speeds
/// - Memory-constrained systems that need to limit buffering
///
/// ## Example
///
/// ```rust
/// use edgeguard_core::traits::{Stream, BackpressureStream};
///
/// fn process_with_backpressure<S: BackpressureStream>(stream: &mut S) {
///     // Check if we should slow down
///     if stream.should_pause() {
///         // Do some work to catch up
///         process_backlog();
///         
///         // Signal we're ready for more
///         stream.resume();
///     }
///     
///     // Continue normal processing
///     match stream.poll_next() {
///         // ...
///     }
/// }
/// ```
pub trait BackpressureStream: Stream {
    /// Check if the stream should pause due to backpressure
    ///
    /// Returns `true` if the consumer is overwhelmed and the producer
    /// should stop sending data temporarily.
    fn should_pause(&self) -> bool;
    
    /// Signal that the consumer is ready for more data
    ///
    /// This method should be called after `should_pause()` returned true
    /// and the consumer has caught up with processing.
    fn resume(&mut self);
    
    /// Get current backpressure metrics
    ///
    /// Returns `(current, capacity)` where:
    /// - `current` is the number of pending items
    /// - `capacity` is the maximum before backpressure triggers
    ///
    /// This can be used for monitoring and adaptive algorithms.
    fn backpressure_stats(&self) -> (usize, usize);
}