//! Common utilities for validators
//!
//! Shared functions for range checking, rate calculations, etc.

use crate::errors::{ValidationError, ValidationResult};
use crate::traits::TimestampedReading;
use crate::buffer::CircularBuffer;

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
    value_delta * 1000.0 / time_delta_ms as f32
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

/// Check if value represents a valid sensor reading
pub trait Validatable {
    fn is_valid(&self) -> bool;
}

impl Validatable for f32 {
    fn is_valid(&self) -> bool {
        !self.is_nan() && !self.is_infinite()
    }
}

impl Validatable for f64 {
    fn is_valid(&self) -> bool {
        !self.is_nan() && !self.is_infinite()
    }
}

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