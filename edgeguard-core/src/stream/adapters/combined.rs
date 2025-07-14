//! Stream combiner for merging multiple streams
//!
//! Provides functionality to combine multiple streams into a single stream,
//! useful for processing data from multiple sensors simultaneously.

use crate::stream::{Stream, StreamError};

/// Strategy for combining streams
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CombineStrategy {
    /// Round-robin between streams
    RoundRobin,
    /// Prioritize by timestamp (requires events with timestamps)
    TimeOrdered,
    /// Fair queuing - balance between streams
    FairQueue,
}

/// Combined stream adapter for two streams
/// 
/// Merges two streams into a single stream using the specified strategy.
/// 
/// ## Example
/// ```rust
/// use edgeguard_core::stream::{CombinedStream, CombineStrategy};
/// 
/// let stream1 = /* ... */;
/// let stream2 = /* ... */;
/// 
/// // Round-robin between streams
/// let mut combined = CombinedStream::new(
///     stream1,
///     stream2,
///     CombineStrategy::RoundRobin
/// );
/// ```
pub struct CombinedStream<S1, S2, E>
where
    S1: Stream<Error = StreamError<E>>,
    S2: Stream<Item = S1::Item, Error = StreamError<E>>,
{
    /// First stream
    stream1: S1,
    /// Second stream
    stream2: S2,
    /// Combining strategy
    strategy: CombineStrategy,
    /// Next stream to poll (for round-robin)
    next_stream: u8,
    /// Pending items from each stream
    pending1: Option<S1::Item>,
    pending2: Option<S2::Item>,
    /// Track if streams are exhausted
    stream1_done: bool,
    stream2_done: bool,
}

impl<S1, S2, E> CombinedStream<S1, S2, E>
where
    S1: Stream<Error = StreamError<E>>,
    S2: Stream<Item = S1::Item, Error = StreamError<E>>,
{
    /// Create new combined stream
    pub fn new(stream1: S1, stream2: S2, strategy: CombineStrategy) -> Self {
        Self {
            stream1,
            stream2,
            strategy,
            next_stream: 0,
            pending1: None,
            pending2: None,
            stream1_done: false,
            stream2_done: false,
        }
    }
    
    /// Try to get item from stream 1
    fn poll_stream1(&mut self) -> nb::Result<Option<S1::Item>, StreamError<E>> {
        if self.stream1_done {
            return Ok(None);
        }
        
        match self.stream1.poll_next() {
            Ok(item) => Ok(Some(item)),
            Err(nb::Error::WouldBlock) => Err(nb::Error::WouldBlock),
            Err(nb::Error::Other(e)) => {
                if matches!(e, StreamError::EndOfStream) {
                    self.stream1_done = true;
                    Ok(None)
                } else {
                    Err(nb::Error::Other(e))
                }
            }
        }
    }
    
    /// Try to get item from stream 2
    fn poll_stream2(&mut self) -> nb::Result<Option<S2::Item>, StreamError<E>> {
        if self.stream2_done {
            return Ok(None);
        }
        
        match self.stream2.poll_next() {
            Ok(item) => Ok(Some(item)),
            Err(nb::Error::WouldBlock) => Err(nb::Error::WouldBlock),
            Err(nb::Error::Other(e)) => {
                if matches!(e, StreamError::EndOfStream) {
                    self.stream2_done = true;
                    Ok(None)
                } else {
                    Err(nb::Error::Other(e))
                }
            }
        }
    }
}

impl<S1, S2, E> Stream for CombinedStream<S1, S2, E>
where
    S1: Stream<Error = StreamError<E>>,
    S2: Stream<Item = S1::Item, Error = StreamError<E>>,
{
    type Item = S1::Item;
    type Error = StreamError<E>;
    
    fn poll_next(&mut self) -> nb::Result<Self::Item, Self::Error> {
        // Check if both streams are done
        if self.stream1_done && self.stream2_done {
            return Err(nb::Error::Other(StreamError::EndOfStream));
        }
        
        match self.strategy {
            CombineStrategy::RoundRobin => {
                // Try streams in round-robin order
                for _ in 0..2 {
                    let item = match self.next_stream {
                        0 => {
                            self.next_stream = 1;
                            self.poll_stream1()
                        }
                        _ => {
                            self.next_stream = 0;
                            self.poll_stream2()
                        }
                    };
                    
                    match item {
                        Ok(Some(item)) => return Ok(item),
                        Ok(None) => continue, // Stream exhausted, try next
                        Err(nb::Error::WouldBlock) => continue, // No data, try next
                        Err(e) => return Err(e),
                    }
                }
                
                // Both streams blocked or exhausted
                if self.stream1_done && self.stream2_done {
                    Err(nb::Error::Other(StreamError::EndOfStream))
                } else {
                    Err(nb::Error::WouldBlock)
                }
            }
            
            CombineStrategy::FairQueue => {
                // Try to maintain fairness by checking pending items
                
                // Fill pending slots if empty
                if self.pending1.is_none() && !self.stream1_done {
                    if let Ok(Some(item)) = self.poll_stream1() {
                        self.pending1 = Some(item);
                    }
                }
                
                if self.pending2.is_none() && !self.stream2_done {
                    if let Ok(Some(item)) = self.poll_stream2() {
                        self.pending2 = Some(item);
                    }
                }
                
                // Emit from the stream that has data
                if let Some(item) = self.pending1.take() {
                    return Ok(item);
                }
                
                if let Some(item) = self.pending2.take() {
                    return Ok(item);
                }
                
                // No pending items
                if self.stream1_done && self.stream2_done {
                    Err(nb::Error::Other(StreamError::EndOfStream))
                } else {
                    Err(nb::Error::WouldBlock)
                }
            }
            
            CombineStrategy::TimeOrdered => {
                // For time-ordered, we need to buffer and compare
                // This is a simplified implementation that assumes events have timestamps
                
                // Try to fill pending slots
                if self.pending1.is_none() && !self.stream1_done {
                    if let Ok(Some(item)) = self.poll_stream1() {
                        self.pending1 = Some(item);
                    }
                }
                
                if self.pending2.is_none() && !self.stream2_done {
                    if let Ok(Some(item)) = self.poll_stream2() {
                        self.pending2 = Some(item);
                    }
                }
                
                // For now, just use fair queue behavior
                // A real implementation would compare timestamps
                if let Some(item) = self.pending1.take() {
                    return Ok(item);
                }
                
                if let Some(item) = self.pending2.take() {
                    return Ok(item);
                }
                
                if self.stream1_done && self.stream2_done {
                    Err(nb::Error::Other(StreamError::EndOfStream))
                } else {
                    Err(nb::Error::WouldBlock)
                }
            }
        }
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (min1, max1) = self.stream1.size_hint();
        let (min2, max2) = self.stream2.size_hint();
        
        let pending_count = self.pending1.is_some() as usize + self.pending2.is_some() as usize;
        
        (
            min1.saturating_add(min2).saturating_add(pending_count),
            match (max1, max2) {
                (Some(m1), Some(m2)) => Some(m1.saturating_add(m2).saturating_add(pending_count)),
                _ => None,
            },
        )
    }
}

/// Three-way stream combiner
/// 
/// Combines three streams into one. For more streams, chain CombinedStream instances.
pub struct TripleCombinedStream<S1, S2, S3, E>
where
    S1: Stream<Error = StreamError<E>>,
    S2: Stream<Item = S1::Item, Error = StreamError<E>>,
    S3: Stream<Item = S1::Item, Error = StreamError<E>>,
{
    inner: CombinedStream<CombinedStream<S1, S2, E>, S3, E>,
}

impl<S1, S2, S3, E> TripleCombinedStream<S1, S2, S3, E>
where
    S1: Stream<Error = StreamError<E>>,
    S2: Stream<Item = S1::Item, Error = StreamError<E>>,
    S3: Stream<Item = S1::Item, Error = StreamError<E>>,
{
    /// Create new three-way combined stream
    pub fn new(stream1: S1, stream2: S2, stream3: S3, strategy: CombineStrategy) -> Self {
        let combined12 = CombinedStream::new(stream1, stream2, strategy);
        let inner = CombinedStream::new(combined12, stream3, strategy);
        Self { inner }
    }
}

impl<S1, S2, S3, E> Stream for TripleCombinedStream<S1, S2, S3, E>
where
    S1: Stream<Error = StreamError<E>>,
    S2: Stream<Item = S1::Item, Error = StreamError<E>>,
    S3: Stream<Item = S1::Item, Error = StreamError<E>>,
{
    type Item = S1::Item;
    type Error = StreamError<E>;
    
    fn poll_next(&mut self) -> nb::Result<Self::Item, Self::Error> {
        self.inner.poll_next()
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::MemoryStream;
    use crate::events::{Event, EventBuilder, SensorType};
    
    #[test]
    fn combine_round_robin() {
        let events1 = vec![
            EventBuilder::new(1000).sensor("t1", SensorType::Temperature).reading(25.0, 0.95).unwrap(),
            EventBuilder::new(3000).sensor("t1", SensorType::Temperature).reading(26.0, 0.95).unwrap(),
        ];
        
        let events2 = vec![
            EventBuilder::new(2000).sensor("t2", SensorType::Temperature).reading(24.0, 0.95).unwrap(),
            EventBuilder::new(4000).sensor("t2", SensorType::Temperature).reading(24.5, 0.95).unwrap(),
        ];
        
        let stream1 = MemoryStream::new(&events1);
        let stream2 = MemoryStream::new(&events2);
        
        let mut combined = CombinedStream::new(
            stream1,
            stream2,
            CombineStrategy::RoundRobin
        );
        
        // Should alternate between streams
        let mut results = Vec::new();
        while let Ok(event) = combined.poll_next() {
            if let Event::SensorReading { sensor_id, .. } = &event {
                results.push(sensor_id.as_str().to_string());
            }
        }
        
        // With round-robin, we should see alternating pattern where possible
        assert_eq!(results.len(), 4);
        assert!(results.contains(&"t1".to_string()));
        assert!(results.contains(&"t2".to_string()));
    }
    
    #[test]
    fn combine_with_empty_stream() {
        let events1 = vec![
            EventBuilder::new(1000).sensor("t1", SensorType::Temperature).reading(25.0, 0.95).unwrap(),
            EventBuilder::new(2000).sensor("t1", SensorType::Temperature).reading(25.5, 0.95).unwrap(),
        ];
        
        let events2: Vec<Event> = vec![];
        
        let stream1 = MemoryStream::new(&events1);
        let stream2 = MemoryStream::new(&events2);
        
        let mut combined = CombinedStream::new(
            stream1,
            stream2,
            CombineStrategy::FairQueue
        );
        
        // Should get all events from stream1
        let mut count = 0;
        while let Ok(_) = combined.poll_next() {
            count += 1;
        }
        
        assert_eq!(count, 2);
    }
    
    #[test]
    fn triple_combine() {
        let events1 = vec![
            EventBuilder::new(1000).sensor("t1", SensorType::Temperature).reading(25.0, 0.95).unwrap(),
        ];
        
        let events2 = vec![
            EventBuilder::new(2000).sensor("t2", SensorType::Temperature).reading(24.0, 0.95).unwrap(),
        ];
        
        let events3 = vec![
            EventBuilder::new(3000).sensor("t3", SensorType::Temperature).reading(23.0, 0.95).unwrap(),
        ];
        
        let stream1 = MemoryStream::new(&events1);
        let stream2 = MemoryStream::new(&events2);
        let stream3 = MemoryStream::new(&events3);
        
        let mut combined = TripleCombinedStream::new(
            stream1,
            stream2,
            stream3,
            CombineStrategy::RoundRobin
        );
        
        // Should get all 3 events
        let mut count = 0;
        while let Ok(_) = combined.poll_next() {
            count += 1;
        }
        
        assert_eq!(count, 3);
    }
}