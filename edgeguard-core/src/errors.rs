//! Error types for validation failures
//!
//! Keep errors small - they're often returned in hot paths.
//! Each variant should clearly indicate what went wrong and why.

use thiserror_no_std::Error;

/// Result type for validation operations
pub type ValidationResult<T> = Result<T, ValidationError>;

/// Validation errors - kept small for embedded use
#[derive(Error, Debug, Clone, Copy, PartialEq)]
pub enum ValidationError {
    /// Value outside physical limits
    #[error("Value {value} outside range [{min}, {max}]")]
    OutOfRange {
        value: f32,
        min: f32,
        max: f32,
    },
    
    /// Rate of change too high - indicates sensor malfunction or impossible physics
    #[error("Rate {rate}/s exceeds limit {max_rate}/s")]
    RateExceeded {
        rate: f32,
        max_rate: f32,
    },
    
    /// Cross-sensor validation failed (e.g., humidity > 100% at freezing temps)
    #[error("Cross-validation failed: {reason}")]
    CrossValidationFailed {
        reason: &'static str,
    },
    
    /// Sensor reported bad quality or is offline
    #[error("Sensor quality check failed: {reason}")]
    SensorQualityBad {
        reason: &'static str,
    },
    
    /// Value makes no physical sense (NaN, infinity, etc)
    #[error("Invalid value: not a valid number")]
    InvalidValue,
    
    /// Not enough historical data for validation
    #[error("Insufficient data: need {required}, have {available}")]
    InsufficientData {
        required: usize,
        available: usize,
    },
}

#[cfg(feature = "defmt")]
impl defmt::Format for ValidationError {
    fn format(&self, fmt: defmt::Formatter) {
        match self {
            Self::OutOfRange { value, min, max } => 
                defmt::write!(fmt, "Value {} outside [{}, {}]", value, min, max),
            Self::RateExceeded { rate, max_rate } => 
                defmt::write!(fmt, "Rate {}/s exceeds {}/s", rate, max_rate),
            Self::CrossValidationFailed { reason } => 
                defmt::write!(fmt, "Cross-validation: {}", reason),
            Self::SensorQualityBad { reason } => 
                defmt::write!(fmt, "Sensor quality: {}", reason),
            Self::InvalidValue => 
                defmt::write!(fmt, "Invalid value"),
            Self::InsufficientData { required, available } => 
                defmt::write!(fmt, "Need {} samples, have {}", required, available),
        }
    }
}