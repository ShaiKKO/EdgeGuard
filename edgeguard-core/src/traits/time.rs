//! Time Source Abstraction for Embedded Systems
//!
//! This module provides the `TimeSource` trait which abstracts time handling
//! across different embedded platforms and test environments.
//!
//! ## Design Goals
//!
//! - **Platform Independence**: Works on bare metal, RTOS, and Linux
//! - **Testability**: Easy to mock for deterministic testing
//! - **Efficiency**: Zero allocation, minimal overhead
//! - **Flexibility**: Supports both wall clock and monotonic time
//!
//! ## Common Implementations
//!
//! - `MonotonicTime`: Hardware timer-based monotonic clock
//! - `SystemTime`: Wall clock time (may jump due to NTP)
//! - `MockTimeSource`: Controllable time for testing
//! - `FallbackTime`: Primary source with automatic fallback

use crate::time::Timestamp;

/// Source of time for the system
/// 
/// This trait abstracts time handling to support different time sources
/// across various embedded platforms. Implementations might use hardware
/// timers, RTC modules, or system calls depending on the platform.
/// 
/// ## Implementation Requirements
/// 
/// - `now()` must be thread-safe if used in multi-threaded context
/// - Timestamp overflow must be handled gracefully
/// - Precision should be documented for each implementation
/// 
/// ## Example Implementation
/// 
/// ```rust
/// use edgeguard_core::traits::TimeSource;
/// use edgeguard_core::time::Timestamp;
/// 
/// struct GpsTimeSource {
///     // ... GPS module interface
/// }
/// 
/// impl TimeSource for GpsTimeSource {
///     fn now(&self) -> Timestamp {
///         // Read time from GPS module
///         // Convert to milliseconds since epoch
///         0 // placeholder
///     }
///     
///     fn is_wall_clock(&self) -> bool {
///         true // GPS provides wall clock time
///     }
///     
///     fn precision_ms(&self) -> u32 {
///         100 // GPS typically updates at 1-10 Hz
///     }
/// }
/// ```
/// 
/// ## Platform-Specific Considerations
/// 
/// ### Bare Metal (no_std)
/// - Use hardware timer peripherals directly
/// - Consider timer overflow and wraparound
/// - Ensure interrupt-safe access if needed
/// 
/// ### RTOS
/// - Use RTOS tick count for monotonic time
/// - May have better precision than bare metal
/// - Check for priority inversion in time access
/// 
/// ### Linux/Unix
/// - Use clock_gettime() with appropriate clock
/// - CLOCK_MONOTONIC for intervals
/// - CLOCK_REALTIME for wall clock
pub trait TimeSource: Send {
    /// Get current timestamp in milliseconds
    /// 
    /// The epoch depends on the implementation:
    /// - Monotonic sources: typically milliseconds since boot
    /// - Wall clock sources: milliseconds since Unix epoch
    /// - Test sources: arbitrary starting point
    fn now(&self) -> Timestamp;
    
    /// Check if this source provides wall clock time (vs monotonic)
    /// 
    /// Wall clock time:
    /// - Can be adjusted (NTP, manual setting)
    /// - May go backwards
    /// - Represents actual time of day
    /// 
    /// Monotonic time:
    /// - Always increases
    /// - Immune to adjustments
    /// - Only useful for measuring intervals
    fn is_wall_clock(&self) -> bool;
    
    /// Get precision in milliseconds
    /// 
    /// This indicates the minimum time difference this source can measure.
    /// For example:
    /// - Hardware timer at 1kHz: 1ms precision
    /// - RTC with 1Hz update: 1000ms precision
    /// - High-resolution timer: <1ms precision (return 0)
    fn precision_ms(&self) -> u32;
}