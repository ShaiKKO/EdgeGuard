//! Rate limiting for streams
//!
//! Provides rate limiting capabilities to control the flow of events.

use crate::stream::Stream;
use crate::time::{TimeSource, Timestamp};

/// Rate-limited stream adapter
/// 
/// Wraps a stream and limits the rate at which events are produced.
/// Uses token bucket algorithm for smooth rate limiting.
/// 
/// ## Example
/// ```rust
/// use edgeguard_core::stream::RateLimitedStream;
/// use edgeguard_core::time::MockTimeSource;
/// 
/// let inner_stream = /* ... */;
/// let time_source = MockTimeSource::new(0);
/// 
/// // Limit to 100 events per second
/// let mut limited = RateLimitedStream::new(inner_stream, 100, time_source);
/// ```
pub struct RateLimitedStream<S: Stream, T: TimeSource> {
    /// Inner stream
    inner: S,
    /// Events per second limit
    events_per_second: u32,
    /// Time source
    time_source: T,
    /// Last event timestamp
    last_event_time: Timestamp,
    /// Token bucket
    tokens: f32,
    /// Maximum tokens (burst capacity)
    max_tokens: f32,
}

impl<S: Stream, T: TimeSource> RateLimitedStream<S, T> {
    /// Create new rate-limited stream
    pub fn new(inner: S, events_per_second: u32, time_source: T) -> Self {
        let max_tokens = events_per_second as f32;
        
        Self {
            inner,
            events_per_second,
            last_event_time: time_source.now(),
            time_source,
            tokens: max_tokens,
            max_tokens,
        }
    }
    
    /// Set burst capacity (max tokens)
    pub fn with_burst(mut self, burst: u32) -> Self {
        self.max_tokens = burst as f32;
        self.tokens = self.tokens.min(self.max_tokens);
        self
    }
    
    /// Update token bucket
    fn update_tokens(&mut self) {
        let now = self.time_source.now();
        let elapsed_ms = now.saturating_sub(self.last_event_time);
        
        if elapsed_ms > 0 {
            let elapsed_sec = elapsed_ms as f32 / 1000.0;
            let new_tokens = elapsed_sec * self.events_per_second as f32;
            
            self.tokens = (self.tokens + new_tokens).min(self.max_tokens);
            self.last_event_time = now;
        }
    }
}

impl<S: Stream, T: TimeSource> Stream for RateLimitedStream<S, T> {
    type Item = S::Item;
    type Error = S::Error;
    
    fn poll_next(&mut self) -> nb::Result<Self::Item, Self::Error> {
        // Update token bucket
        self.update_tokens();
        
        // Check if we have tokens
        if self.tokens < 1.0 {
            return Err(nb::Error::WouldBlock);
        }
        
        // Try to get item from inner stream
        match self.inner.poll_next() {
            Ok(item) => {
                // Consume a token
                self.tokens -= 1.0;
                Ok(item)
            }
            Err(e) => Err(e),
        }
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        // Rate limiting may reduce actual items
        let (_min, max) = self.inner.size_hint();
        (0, max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::MockTimeSource;
    use crate::stream::MemoryStream;
    use crate::events::{EventBuilder, SensorType};
    use core::cell::RefCell;
    
    // Wrapper to make RefCell<MockTimeSource> implement TimeSource
    struct TimeSourceWrapper<'a> {
        inner: &'a RefCell<MockTimeSource>,
    }
    
    impl<'a> TimeSource for TimeSourceWrapper<'a> {
        fn now(&self) -> Timestamp {
            self.inner.borrow().now()
        }
        
        fn is_wall_clock(&self) -> bool {
            false // Mock time source is not wall clock
        }
        
        fn precision_ms(&self) -> u32 {
            1 // 1ms precision
        }
    }
    
    #[test]
    fn rate_limiting() {
        let events = vec![
            EventBuilder::new(1000).sensor("t1", SensorType::Temperature).reading(25.0, 0.95).unwrap(),
            EventBuilder::new(2000).sensor("t1", SensorType::Temperature).reading(25.5, 0.95).unwrap(),
            EventBuilder::new(3000).sensor("t1", SensorType::Temperature).reading(26.0, 0.95).unwrap(),
        ];
        
        let inner = MemoryStream::new(&events);
        let time_source = core::cell::RefCell::new(MockTimeSource::new(0));
        let time_source_wrapper = TimeSourceWrapper { inner: &time_source };
        let mut limited = RateLimitedStream::new(inner, 2, time_source_wrapper); // 2 events/sec
        
        // First event should succeed
        assert!(limited.poll_next().is_ok());
        
        // Second event should succeed (using second token)
        assert!(limited.poll_next().is_ok());
        
        // Third event should block (no tokens)
        assert!(matches!(limited.poll_next(), Err(nb::Error::WouldBlock)));
        
        // Advance time by 500ms (should get 1 token)
        time_source.borrow_mut().advance(500);
        
        // Update tokens and try again
        limited.update_tokens();
        assert!(limited.poll_next().is_ok());
        
        // Now we have no tokens left, should get WouldBlock
        assert!(matches!(limited.poll_next(), Err(nb::Error::WouldBlock)));
        
        // Advance time by another 1000ms to get 2 more tokens
        time_source.borrow_mut().advance(1000);
        limited.update_tokens();
        
        // Now we can exhaust the stream
        match limited.poll_next() {
            Err(nb::Error::Other(_)) => {}, // Expected - EndOfStream from inner
            other => panic!("Expected EndOfStream error, got {:?}", other),
        }
    }
}