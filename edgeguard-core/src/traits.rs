//! Core traits for validators
//!
//! These traits define the interface all validators must implement.
//! Keep them simple - embedded devices don't need complex abstractions.

use crate::errors::ValidationResult;
use crate::time::Timestamp;
use crate::buffer::CircularBuffer;

/// Maximum number of historical samples to keep (adjust for your RAM constraints)
pub const MAX_HISTORY_SIZE: usize = 32;

/// Single reading with timestamp
#[derive(Debug, Clone, Copy)]
pub struct TimestampedReading {
    pub value: f32,
    pub timestamp: Timestamp,
}

/// Context passed to validators containing environmental data and history
#[derive(Clone)]
pub struct ValidationContext {
    /// Recent sensor readings with timestamps for rate-of-change validation
    pub history: CircularBuffer<MAX_HISTORY_SIZE>,
    
    /// Current timestamp in milliseconds (from device RTC or tick counter)
    pub timestamp: Timestamp,
    
    /// Environmental temperature for compensation (optional)
    pub ambient_temp: Option<f32>,
    
    /// Environmental humidity for cross-validation (optional)
    pub ambient_humidity: Option<f32>,
    
    /// Sensor quality indicator (0.0 = bad, 1.0 = perfect)
    pub sensor_quality: f32,
}

impl Default for ValidationContext {
    fn default() -> Self {
        Self {
            history: CircularBuffer::new(),
            timestamp: 0,
            ambient_temp: None,
            ambient_humidity: None,
            sensor_quality: 1.0,
        }
    }
}

impl ValidationContext {
    /// Add a reading to history, maintaining chronological order
    pub fn add_reading(&mut self, value: f32, timestamp: Timestamp) {
        let reading = TimestampedReading { value, timestamp };
        self.history.push(reading);
    }
    
    /// Get the most recent reading if any
    pub fn last_reading(&self) -> Option<&TimestampedReading> {
        self.history.last()
    }
    
    /// Calculate time delta from last reading in milliseconds
    pub fn time_delta_ms(&self) -> Option<u64> {
        self.last_reading()
            .map(|last| self.timestamp.saturating_sub(last.timestamp))
    }
}

/// Core validator trait - implement this for each sensor type
pub trait Validator {
    /// The type of value this validator handles
    type Value;
    
    /// Validate a single reading
    fn validate(&self, value: Self::Value, context: &ValidationContext) -> ValidationResult<()>;
    
    /// Get physical constraints for this validator
    fn constraints(&self) -> ValidatorConstraints;
}

/// Physical constraints for a validator
#[derive(Debug, Clone, Copy)]
pub struct ValidatorConstraints {
    /// Minimum valid value (physics limit)
    pub min_value: f32,
    
    /// Maximum valid value (physics limit)
    pub max_value: f32,
    
    /// Maximum rate of change per second
    pub max_rate_change: f32,
    
    /// Optional: typical noise level for filtering
    pub noise_threshold: Option<f32>,
}

/// Cross-validator for multi-sensor validation
pub trait CrossValidator {
    /// Input type (usually a tuple of readings)
    type Input;
    
    /// Validate readings from multiple sensors
    fn cross_validate(&self, inputs: Self::Input) -> ValidationResult<()>;
}

/// Trait for values that can be validated
pub trait Validatable {
    /// Check if the value is physically valid (not NaN, infinite, etc)
    fn is_valid(&self) -> bool;
}

impl Validatable for f32 {
    fn is_valid(&self) -> bool {
        self.is_finite()
    }
}

impl Validatable for f64 {
    fn is_valid(&self) -> bool {
        self.is_finite()
    }
}

/// Trait for sensors that need environmental compensation
pub trait EnvironmentalCompensation {
    /// Apply compensation based on environmental conditions
    fn compensate(&self, raw_value: f32, context: &ValidationContext) -> f32;
}