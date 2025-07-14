//! Common Validation Utilities and Patterns
//!
//! ## Overview
//!
//! This module provides shared functionality used across all validators to ensure
//! consistent behavior and avoid code duplication. These utilities implement the
//! fundamental validation operations that apply regardless of sensor type.
//!
//! ## Design Principles
//!
//! ### 1. Pure Functions
//! All utilities are pure functions with no side effects. This makes them:
//! - Easy to test in isolation
//! - Safe to call from interrupt handlers
//! - Predictable in concurrent environments
//!
//! ### 2. Zero Allocation
//! No dynamic memory allocation - all operations work with provided buffers
//! and return values by copy or reference.
//!
//! ### 3. Defensive Programming
//! Functions handle edge cases gracefully:
//! - Division by zero returns sensible defaults
//! - Overflow/underflow uses saturating arithmetic
//! - Invalid inputs return errors, not panics
//!
//! ## Common Validation Patterns
//!
//! ### Range Validation
//! The most basic check - is the value physically possible?
//! ```rust
//! // Temperature can't be below absolute zero
//! check_range(temp, -273.15, 1000.0)?;
//! ```
//!
//! ### Rate-of-Change Validation
//! Detects impossible changes based on physics:
//! ```rust
//! // Temperature can't change 100°C in 1 second
//! let rate = calculate_rate(new_temp, old_temp, time_ms);
//! if rate > MAX_RATE {
//!     return Err(ValidationError::RateExceeded { rate, max_rate });
//! }
//! ```
//!
//! ### Sensor Health Checks
//! Common failure modes across sensor types:
//! - **Stuck values**: No change over time
//! - **Noise**: Rapid small oscillations
//! - **Drift**: Slow deviation from calibration
//! - **Spikes**: Electrical interference
//!
//! ## Rate Calculation Details
//!
//! Rate calculation seems simple but has subtleties:
//!
//! ### Time Delta Handling
//! ```text
//! If time_delta = 0:
//!   - Could be same timestamp (batch reading)
//!   - Could be clock issue
//!   - Return 0 rate (safe default)
//!
//! If time_delta < sensor_response_time:
//!   - Rate may be artificially high
//!   - Consider sensor lag
//! ```
//!
//! ### Absolute Value
//! We use absolute value for rate because:
//! - Both heating and cooling have physical limits
//! - Simplifies validator configuration
//! - Matches how humans think about rate limits
//!
//! ### Units
//! Always calculate as per-second rates:
//! - Standard SI unit for rates
//! - Easy to compare across sensors
//! - Intuitive for configuration
//!
//! ## Future Enhancements
//!
//! Potential additions for advanced validation:
//! - **Noise estimation**: Calculate signal-to-noise ratio
//! - **Trend detection**: Linear regression over history
//! - **Anomaly scoring**: Statistical deviation from normal
//! - **Sensor fusion**: Combine multiple sensors for validation

use crate::{
    constants::time::MS_PER_SECOND,
    errors::{ValidationError, ValidationResult},
    traits::TimestampedReading,
    buffer::CircularBuffer,
};

/// Check if a value is within the specified range
pub fn check_range(value: f32, min: f32, max: f32) -> ValidationResult<()> {
    if value < min {
        Err(ValidationError::OutOfRange {
            value,
            min,
            max,
        })
    } else if value > max {
        Err(ValidationError::OutOfRange {
            value,
            min,
            max,
        })
    } else {
        Ok(())
    }
}

/// Get the last value from history
pub fn last_value<const N: usize>(history: &CircularBuffer<N>) -> Option<f32> {
    history.last().map(|reading| reading.value)
}

/// Get the last reading from history
pub fn last_reading<const N: usize>(history: &CircularBuffer<N>) -> Option<&TimestampedReading> {
    history.last()
}

/// Calculate rate of change per second
pub fn calculate_rate(current: f32, previous: f32, time_delta_ms: u64) -> f32 {
    if time_delta_ms == 0 {
        return 0.0;
    }
    
    let value_delta = (current - previous).abs();
    value_delta * MS_PER_SECOND as f32 / time_delta_ms as f32
}

/// Calculate rate of change from timestamped readings
pub fn calculate_rate_from_readings(
    current_value: f32,
    current_time: u64,
    last_reading: &TimestampedReading,
) -> f32 {
    let time_delta = current_time.saturating_sub(last_reading.timestamp);
    calculate_rate(current_value, last_reading.value, time_delta)
}

// Import Validatable trait from traits module for f32.is_valid() extension
#[allow(unused_imports)]
use crate::traits::Validatable;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn range_check() {
        assert!(check_range(5.0, 0.0, 10.0).is_ok());
        assert!(check_range(-1.0, 0.0, 10.0).is_err());
        assert!(check_range(11.0, 0.0, 10.0).is_err());
    }
    
    #[test]
    fn rate_calculation() {
        // 10 degree change in 1 second = 10 degrees/second
        assert_eq!(calculate_rate(30.0, 20.0, 1000), 10.0);
        
        // 5 degree change in 500ms = 10 degrees/second
        assert_eq!(calculate_rate(25.0, 20.0, 500), 10.0);
        
        // Zero time = zero rate
        assert_eq!(calculate_rate(30.0, 20.0, 0), 0.0);
    }
    
    #[test]
    fn validatable_floats() {
        assert!(5.0f32.is_valid());
        assert!(!f32::NAN.is_valid());
        assert!(!f32::INFINITY.is_valid());
    }
}