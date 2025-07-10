//! Time management for edge devices
//!
//! Provides clock abstraction to handle different time sources:
//! - System clock (when available)
//! - Monotonic counter (for rate calculations)
//! - External RTC (for battery-backed time)
//! - Network time (when connected)

/// Timestamp in milliseconds since epoch (or device boot for monotonic)
pub type Timestamp = u64;

/// Source of time for the system
pub trait TimeSource {
    /// Get current timestamp in milliseconds
    fn now(&self) -> Timestamp;
    
    /// Check if this source provides wall clock time (vs monotonic)
    fn is_wall_clock(&self) -> bool;
    
    /// Get precision in milliseconds
    fn precision_ms(&self) -> u32;
}

/// Monotonic time source using a counter
/// 
/// Starts at 0 on boot, always increases
#[derive(Debug, Clone)]
pub struct MonotonicTime {
    start_ms: Timestamp,
}

impl MonotonicTime {
    pub fn new() -> Self {
        Self { start_ms: 0 }
    }
}

impl TimeSource for MonotonicTime {
    fn now(&self) -> Timestamp {
        // In real implementation, read from hardware timer
        // For now, simulate with a counter
        self.start_ms
    }
    
    fn is_wall_clock(&self) -> bool {
        false
    }
    
    fn precision_ms(&self) -> u32 {
        1 // 1ms precision typical for most timers
    }
}

/// System time source (requires std)
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct SystemTime;

#[cfg(feature = "std")]
impl TimeSource for SystemTime {
    fn now(&self) -> Timestamp {
        use std::time::{SystemTime as StdSystemTime, UNIX_EPOCH};
        
        StdSystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as Timestamp
    }
    
    fn is_wall_clock(&self) -> bool {
        true
    }
    
    fn precision_ms(&self) -> u32 {
        1
    }
}

/// Fixed time source for testing
#[derive(Debug, Clone)]
pub struct FixedTime {
    timestamp: Timestamp,
}

impl FixedTime {
    pub fn new(timestamp: Timestamp) -> Self {
        Self { timestamp }
    }
    
    pub fn set(&mut self, timestamp: Timestamp) {
        self.timestamp = timestamp;
    }
    
    pub fn advance(&mut self, ms: u64) {
        self.timestamp += ms;
    }
}

impl TimeSource for FixedTime {
    fn now(&self) -> Timestamp {
        self.timestamp
    }
    
    fn is_wall_clock(&self) -> bool {
        false
    }
    
    fn precision_ms(&self) -> u32 {
        1
    }
}

/// Manages time with fallback sources
pub struct TimeManager {
    primary: Box<dyn TimeSource>,
    fallback: Option<Box<dyn TimeSource>>,
    last_known: Timestamp,
}

impl TimeManager {
    pub fn new(primary: Box<dyn TimeSource>) -> Self {
        let last_known = primary.now();
        Self {
            primary,
            fallback: None,
            last_known,
        }
    }
    
    pub fn with_fallback(mut self, fallback: Box<dyn TimeSource>) -> Self {
        self.fallback = Some(fallback);
        self
    }
    
    /// Get current time, using fallback if primary fails
    pub fn now(&mut self) -> Timestamp {
        let current = self.primary.now();
        
        // Detect time going backwards (clock adjustment)
        if current < self.last_known {
            if let Some(fallback) = &self.fallback {
                return fallback.now();
            }
        }
        
        self.last_known = current;
        current
    }
    
    /// Calculate time delta between two timestamps
    pub fn delta_ms(&self, earlier: Timestamp, later: Timestamp) -> u64 {
        later.saturating_sub(earlier)
    }
    
    /// Convert delta to rate per second
    pub fn rate_per_second(&self, value_delta: f32, time_delta_ms: u64) -> f32 {
        if time_delta_ms == 0 {
            return 0.0;
        }
        
        value_delta * 1000.0 / time_delta_ms as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn fixed_time_advances() {
        let mut time = FixedTime::new(1000);
        assert_eq!(time.now(), 1000);
        
        time.advance(500);
        assert_eq!(time.now(), 1500);
    }
    
    // TODO: Add test for backwards time detection once we have
    // a way to mock time sources properly
    
    #[test]
    fn rate_calculation() {
        let manager = TimeManager::new(Box::new(FixedTime::new(0)));
        
        // 10 units in 500ms = 20 units/second
        let rate = manager.rate_per_second(10.0, 500);
        assert_eq!(rate, 20.0);
        
        // Zero time delta
        let rate = manager.rate_per_second(10.0, 0);
        assert_eq!(rate, 0.0);
    }
}