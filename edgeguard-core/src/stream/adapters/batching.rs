//! Batching adapter for streams
//!
//! Provides batching capabilities to collect events into groups
//! for more efficient processing.

use crate::stream::{Stream, StreamError};
use crate::time::{TimeSource, Timestamp};

/// Batching strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BatchStrategy {
    /// Batch by count
    Count { size: usize },
    /// Batch by time window
    Time { window_ms: u32 },
    /// Batch by count with timeout
    CountWithTimeout { size: usize, timeout_ms: u32 },
}

/// Batching stream adapter
/// 
/// Collects events into batches based on configured strategy.
/// Useful for reducing processing overhead and network traffic.
/// 
/// ## Example
/// ```rust
/// use edgeguard_core::stream::{BatchingStream, BatchStrategy};
/// use edgeguard_core::time::MonotonicTime;
/// 
/// let inner_stream = /* ... */;
/// 
/// // Batch by count
/// let mut batched = BatchingStream::new(
///     inner_stream, 
///     BatchStrategy::Count { size: 100 },
///     MonotonicTime::new()
/// );
/// 
/// // Batch by time with count limit
/// let mut batched = BatchingStream::new(
///     inner_stream,
///     BatchStrategy::CountWithTimeout { size: 50, timeout_ms: 1000 },
///     MonotonicTime::new()
/// );
/// ```
pub struct BatchingStream<S: Stream, T: TimeSource> {
    /// Inner stream
    inner: S,
    /// Batching strategy
    strategy: BatchStrategy,
    /// Time source
    time_source: T,
    /// Current batch
    batch: heapless::Vec<S::Item, 64>,
    /// Batch start time
    batch_start_time: Timestamp,
    /// Whether we've hit end of stream
    end_of_stream: bool,
}

impl<S: Stream, T: TimeSource> BatchingStream<S, T> {
    /// Create new batching stream
    pub fn new(inner: S, strategy: BatchStrategy, time_source: T) -> Self {
        let now = time_source.now();
        
        Self {
            inner,
            strategy,
            time_source,
            batch: heapless::Vec::new(),
            batch_start_time: now,
            end_of_stream: false,
        }
    }
    
    /// Check if batch is ready
    fn batch_ready(&self) -> bool {
        match self.strategy {
            BatchStrategy::Count { size } => {
                self.batch.len() >= size
            }
            BatchStrategy::Time { window_ms } => {
                let elapsed = self.time_source.now().saturating_sub(self.batch_start_time);
                elapsed >= window_ms as u64
            }
            BatchStrategy::CountWithTimeout { size, timeout_ms } => {
                if self.batch.len() >= size {
                    return true;
                }
                let elapsed = self.time_source.now().saturating_sub(self.batch_start_time);
                elapsed >= timeout_ms as u64 && !self.batch.is_empty()
            }
        }
    }
    
    /// Force emit current batch if not empty
    pub fn flush(&mut self) -> Option<heapless::Vec<S::Item, 64>> {
        if self.batch.is_empty() {
            None
        } else {
            let batch = core::mem::replace(&mut self.batch, heapless::Vec::new());
            self.batch_start_time = self.time_source.now();
            Some(batch)
        }
    }
}

impl<S, T, E> Stream for BatchingStream<S, T>
where
    S: Stream<Error = StreamError<E>>,
    S::Item: Clone,
    T: TimeSource,
{
    type Item = heapless::Vec<S::Item, 64>;
    type Error = StreamError<E>;
    
    fn poll_next(&mut self) -> nb::Result<Self::Item, Self::Error> {
        // If we've already seen end of stream and batch is empty, propagate it
        if self.end_of_stream && self.batch.is_empty() {
            return Err(nb::Error::Other(StreamError::EndOfStream));
        }
        
        // Try to fill batch from inner stream
        loop {
            // Check if batch is ready
            if self.batch_ready() {
                return Ok(self.flush().expect("Batch should not be empty"));
            }
            
            // If end of stream and we have partial batch, emit it
            if self.end_of_stream && !self.batch.is_empty() {
                return Ok(self.flush().expect("Batch should not be empty"));
            }
            
            // Try to get more items
            match self.inner.poll_next() {
                Ok(item) => {
                    // Try to add to batch
                    if self.batch.push(item).is_err() {
                        // Batch is full, emit it
                        let batch = core::mem::replace(&mut self.batch, heapless::Vec::new());
                        self.batch_start_time = self.time_source.now();
                        return Ok(batch);
                    }
                }
                Err(nb::Error::WouldBlock) => {
                    // No more items available right now
                    // Check timeout for partial batches
                    if let BatchStrategy::CountWithTimeout { timeout_ms, .. } = self.strategy {
                        let elapsed = self.time_source.now().saturating_sub(self.batch_start_time);
                        if elapsed >= timeout_ms as u64 && !self.batch.is_empty() {
                            return Ok(self.flush().expect("Batch should not be empty"));
                        }
                    }
                    return Err(nb::Error::WouldBlock);
                }
                Err(nb::Error::Other(e)) => {
                    // Check if it's end of stream
                    if matches!(e, StreamError::EndOfStream) {
                        self.end_of_stream = true;
                        // Emit partial batch if we have one
                        if !self.batch.is_empty() {
                            return Ok(self.flush().expect("Batch should not be empty"));
                        }
                    }
                    return Err(nb::Error::Other(e));
                }
            }
        }
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (inner_min, inner_max) = self.inner.size_hint();
        
        // Calculate how many batches we might produce
        let batch_size = match self.strategy {
            BatchStrategy::Count { size } => size,
            BatchStrategy::Time { .. } => 1, // Unknown batch size for time-based
            BatchStrategy::CountWithTimeout { size, .. } => size,
        };
        
        let current_items = self.batch.len();
        
        (
            (inner_min + current_items) / batch_size,
            inner_max.map(|max| (max + current_items) / batch_size),
        )
    }
}

/// Convenience type for batching events
pub type EventBatchingStream<S, T> = BatchingStream<S, T>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::MemoryStream;
    use crate::time::MockTimeSource;
    use crate::events::{EventBuilder, SensorType};
    
    #[test]
    fn batch_by_count() {
        let events = vec![
            EventBuilder::new(1000).sensor("t1", SensorType::Temperature).reading(25.0, 0.95).unwrap(),
            EventBuilder::new(2000).sensor("t1", SensorType::Temperature).reading(25.5, 0.95).unwrap(),
            EventBuilder::new(3000).sensor("t1", SensorType::Temperature).reading(26.0, 0.95).unwrap(),
            EventBuilder::new(4000).sensor("t1", SensorType::Temperature).reading(26.5, 0.95).unwrap(),
        ];
        
        let inner = MemoryStream::new(&events);
        let time_source = MockTimeSource::new(0);
        let mut batched = BatchingStream::new(
            inner,
            BatchStrategy::Count { size: 2 },
            time_source
        );
        
        // First batch of 2
        let batch1 = batched.poll_next().unwrap();
        assert_eq!(batch1.len(), 2);
        
        // Second batch of 2
        let batch2 = batched.poll_next().unwrap();
        assert_eq!(batch2.len(), 2);
        
        // End of stream
        assert!(matches!(
            batched.poll_next(),
            Err(nb::Error::Other(StreamError::EndOfStream))
        ));
    }
    
    #[test]
    fn batch_by_timeout() {
        let events = vec![
            EventBuilder::new(1000).sensor("t1", SensorType::Temperature).reading(25.0, 0.95).unwrap(),
            EventBuilder::new(2000).sensor("t1", SensorType::Temperature).reading(25.5, 0.95).unwrap(),
            EventBuilder::new(3000).sensor("t1", SensorType::Temperature).reading(26.0, 0.95).unwrap(),
        ];
        
        let inner = MemoryStream::new(&events);
        let time_source = MockTimeSource::new(0);
        
        // Need to wrap in RefCell for interior mutability
        use core::cell::RefCell;
        let time_source_ref = RefCell::new(time_source);
        
        // Wrapper to implement TimeSource
        struct TimeSourceWrapper<'a> {
            inner: &'a RefCell<MockTimeSource>,
        }
        
        impl<'a> TimeSource for TimeSourceWrapper<'a> {
            fn now(&self) -> Timestamp {
                self.inner.borrow().now()
            }
            
            fn is_wall_clock(&self) -> bool {
                false
            }
            
            fn precision_ms(&self) -> u32 {
                1
            }
        }
        
        let time_wrapper = TimeSourceWrapper { inner: &time_source_ref };
        let mut batched = BatchingStream::new(
            inner,
            BatchStrategy::CountWithTimeout { size: 5, timeout_ms: 1000 },
            time_wrapper
        );
        
        // The MemoryStream will exhaust all 3 events immediately and hit EndOfStream
        // This causes the batching stream to emit a partial batch
        let batch = batched.poll_next().unwrap();
        assert_eq!(batch.len(), 3);
        
        // Now we should get EndOfStream
        assert!(matches!(
            batched.poll_next(),
            Err(nb::Error::Other(StreamError::EndOfStream))
        ));
    }
    
    #[test]
    fn partial_batch_on_end_of_stream() {
        let events = vec![
            EventBuilder::new(1000).sensor("t1", SensorType::Temperature).reading(25.0, 0.95).unwrap(),
            EventBuilder::new(2000).sensor("t1", SensorType::Temperature).reading(25.5, 0.95).unwrap(),
            EventBuilder::new(3000).sensor("t1", SensorType::Temperature).reading(26.0, 0.95).unwrap(),
        ];
        
        let inner = MemoryStream::new(&events);
        let time_source = MockTimeSource::new(0);
        let mut batched = BatchingStream::new(
            inner,
            BatchStrategy::Count { size: 5 }, // Larger than available events
            time_source
        );
        
        // Should get partial batch with all 3 events
        let batch = batched.poll_next().unwrap();
        assert_eq!(batch.len(), 3);
        
        // Then end of stream
        assert!(matches!(
            batched.poll_next(),
            Err(nb::Error::Other(StreamError::EndOfStream))
        ));
    }
}