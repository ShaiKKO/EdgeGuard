//! Memory-based streams for testing and replay
//!
//! This module provides in-memory stream implementations that are useful for:
//! - Unit testing
//! - Replaying recorded data
//! - Simulating sensor inputs

use crate::events::Event;
use super::{Stream, StreamError};

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
    
    /// Get current position
    pub fn position(&self) -> usize {
        self.position
    }
    
    /// Check if stream is exhausted
    pub fn is_exhausted(&self) -> bool {
        self.position >= self.events.len()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{EventBuilder, SensorType};
    
    #[test]
    fn memory_stream_basic() {
        let events = vec![
            EventBuilder::new(1000)
                .sensor("t1", SensorType::Temperature)
                .reading(25.0, 0.95)
                .unwrap(),
            EventBuilder::new(2000)
                .sensor("t1", SensorType::Temperature)
                .reading(25.5, 0.95)
                .unwrap(),
        ];
        
        let mut stream = MemoryStream::new(&events);
        
        // Check size hint
        assert_eq!(stream.size_hint(), (2, Some(2)));
        
        // Read first event
        let event1 = stream.poll_next().unwrap();
        if let Event::SensorReading { value, .. } = event1 {
            assert_eq!(value, 25.0);
        } else {
            panic!("Expected sensor reading");
        }
        
        // Check size hint after read
        assert_eq!(stream.size_hint(), (1, Some(1)));
        
        // Read second event
        let event2 = stream.poll_next().unwrap();
        if let Event::SensorReading { value, .. } = event2 {
            assert_eq!(value, 25.5);
        }
        
        // Check exhausted
        assert!(stream.is_exhausted());
        match stream.poll_next() {
            Err(nb::Error::Other(StreamError::EndOfStream)) => {},
            _ => panic!("Expected EndOfStream"),
        }
    }
    
    #[test]
    fn memory_stream_reset() {
        let events = vec![
            EventBuilder::new(1000)
                .sensor("t1", SensorType::Temperature)
                .reading(25.0, 0.95)
                .unwrap(),
        ];
        
        let mut stream = MemoryStream::new(&events);
        
        // Read and exhaust
        stream.poll_next().unwrap();
        assert!(stream.is_exhausted());
        
        // Reset and read again
        stream.reset();
        assert!(!stream.is_exhausted());
        assert_eq!(stream.position(), 0);
        
        let event = stream.poll_next().unwrap();
        if let Event::SensorReading { value, .. } = event {
            assert_eq!(value, 25.0);
        }
    }
}