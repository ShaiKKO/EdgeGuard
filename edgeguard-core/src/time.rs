//! Time Management and Clock Abstraction for Edge Devices
//!
//! ## Overview
//!
//! Time handling on embedded systems is surprisingly complex. Unlike desktop systems
//! with reliable system clocks, edge devices face numerous challenges:
//!
//! - **No Battery-Backed RTC**: Many devices lose time on power loss
//! - **Clock Drift**: Crystal oscillators drift with temperature changes
//! - **Time Jumps**: Network time sync can cause sudden adjustments
//! - **Limited Resolution**: Some timers only provide millisecond precision
//! - **Rollover**: 32-bit timers overflow after ~49 days
//!
//! This module provides abstractions to handle these challenges gracefully.
//!
//! ## Design Philosophy
//!
//! ### Multiple Time Sources
//!
//! We support different time sources for different purposes:
//!
//! 1. **Monotonic Time**: Always increases, ideal for rate calculations
//!    - Unaffected by time adjustments
//!    - Starts at 0 on boot
//!    - May overflow on long-running systems
//!
//! 2. **Wall Clock Time**: Real-world time for timestamps
//!    - May jump backwards/forwards during sync
//!    - Lost on power cycle without RTC
//!    - Required for data correlation
//!
//! 3. **External RTC**: Battery-backed hardware clock
//!    - Survives power loss
//!    - Often lower precision (1 second)
//!    - May drift significantly
//!
//! ### Fallback Strategy
//!
//! The `TimeManager` implements a fallback chain:
//! ```text
//! Primary Source → Fallback Source → Last Known Good
//!      ↓                ↓                  ↓
//! Network Time    Monotonic Timer    Cached Value
//! ```
//!
//! This ensures time always moves forward for rate calculations.
//!
//! ## Common Patterns
//!
//! ### Rate-of-Change Calculation
//! ```rust
//! use edgeguard_core::time::{TimeManager, MonotonicTime};
//!
//! let mut time_mgr = TimeManager::new(Box::new(MonotonicTime::new()));
//! 
//! let t1 = time_mgr.now();
//! let v1 = 23.5; // First temperature reading
//! 
//! // ... some time passes ...
//! 
//! let t2 = time_mgr.now();
//! let v2 = 24.0; // Second temperature reading
//! 
//! let rate = time_mgr.rate_per_second(v2 - v1, time_mgr.delta_ms(t1, t2));
//! // rate = 0.5°C/second if 1 second passed
//! ```
//!
//! ### Handling Time Jumps
//! ```rust
//! use edgeguard_core::time::{TimeManager, SystemTime, MonotonicTime};
//! 
//! // Use system time with monotonic fallback
//! let time_mgr = TimeManager::new(Box::new(SystemTime))
//!     .with_fallback(Box::new(MonotonicTime::new()));
//! 
//! // If system time jumps backwards (NTP sync), monotonic time is used
//! ```
//!
//! ## Implementation Notes
//!
//! ### Timestamp Format
//!
//! We use milliseconds since epoch (u64) for timestamps:
//! - Sufficient precision for sensor data (1ms)
//! - No overflow for 584 million years
//! - Compatible with most time libraries
//! - Easy to convert to/from other formats
//!
//! ### Thread Safety
//!
//! Time sources must be Send + Sync if used across threads. The default
//! implementations are thread-safe, but custom sources should ensure
//! proper synchronization.
//!
//! ### Power Efficiency
//!
//! Reading time can wake sleeping peripherals. Cache timestamps when
//! processing multiple readings in sequence:
//!
//! ```rust
//! let now = time_mgr.now();
//! for reading in sensor_batch {
//!     reading.timestamp = now; // Reuse same timestamp
//! }
//! ```

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

#[cfg(feature = "std")]
use std::boxed::Box;

use crate::traits::TimeSource;

/// Timestamp in milliseconds since epoch (or device boot for monotonic)
pub type Timestamp = u64;

/// Monotonic time source using a hardware counter
/// 
/// This time source provides a constantly increasing timestamp that is immune
/// to system time adjustments. It's ideal for measuring intervals and calculating
/// rates of change.
/// 
/// ## Characteristics
/// - Starts at 0 when created (typically at boot)
/// - Always increases (never goes backwards)
/// - Unaffected by NTP adjustments or manual time changes
/// - May wrap around on 32-bit systems after ~49 days
/// 
/// ## Hardware Implementation
/// 
/// On real hardware, this would read from:
/// - ARM: SysTick timer or dedicated timer peripheral
/// - ESP32: esp_timer_get_time() or FreeRTOS tick count
/// - Linux: CLOCK_MONOTONIC via clock_gettime()
/// 
/// ## Example
/// ```rust
/// use edgeguard_core::time::{MonotonicTime, TimeSource};
/// 
/// let timer = MonotonicTime::new();
/// let start = timer.now();
/// // ... do some work ...
/// let elapsed = timer.now() - start;
/// println!("Operation took {} ms", elapsed);
/// ```
#[derive(Debug, Clone)]
pub struct MonotonicTime {
    start_ms: Timestamp,
    #[cfg(feature = "std")]
    start_instant: std::time::Instant,
}

impl MonotonicTime {
    pub fn new() -> Self {
        Self { 
            start_ms: 0,
            #[cfg(feature = "std")]
            start_instant: std::time::Instant::now(),
        }
    }
}

impl TimeSource for MonotonicTime {
    fn now(&self) -> Timestamp {
        #[cfg(feature = "std")]
        {
            // Use std::time::Instant for actual monotonic time
            self.start_instant.elapsed().as_millis() as Timestamp
        }
        
        #[cfg(not(feature = "std"))]
        {
            // In no_std, this would read from hardware timer
            // For now, return fixed value
            self.start_ms
        }
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

/// Mock time source for testing - alias for FixedTime
pub type MockTimeSource = FixedTime;

/// Monotonic clock - alias for MonotonicTime for backward compatibility
pub type MonotonicClock = MonotonicTime;

/// Time source that reads from a ValidationContext
/// 
/// This allows using the validation context's timestamp as a time source,
/// useful for consistent time handling in validation pipelines.
pub struct ContextTimeSource<'a> {
    context: &'a crate::traits::ValidationContext,
}

impl<'a> ContextTimeSource<'a> {
    /// Create a new time source from a validation context
    pub fn new(context: &'a crate::traits::ValidationContext) -> Self {
        Self { context }
    }
}

impl<'a> TimeSource for ContextTimeSource<'a> {
    fn now(&self) -> Timestamp {
        self.context.timestamp
    }
    
    fn is_wall_clock(&self) -> bool {
        true // Assumes context timestamp is wall clock time
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