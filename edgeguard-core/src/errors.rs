//! Error Types for Physics-Based Validation Failures
//!
//! ## Design Philosophy
//!
//! EdgeGuard's error system is designed with embedded systems in mind:
//!
//! 1. **Small Size**: Each error variant is kept minimal (typically 12-16 bytes) since
//!    errors are returned in hot paths and may be stored in queues.
//!
//! 2. **No Heap Allocation**: All error data is inline - no String, only &'static str
//!    for messages. This ensures deterministic memory usage.
//!
//! 3. **Copy Semantics**: Errors implement Copy for efficient return from functions
//!    without move semantics complications.
//!
//! 4. **Actionable Information**: Each error provides enough context to determine
//!    the appropriate response without needing additional queries.
//!
//! ## Error Categories
//!
//! Errors fall into three main categories:
//!
//! ### Physical Violations
//! - `OutOfRange`: Value exceeds physical limits (e.g., -300°C temperature)
//! - `RateExceeded`: Change too rapid for physics (e.g., 100°C/second)
//! - `InvalidValue`: Mathematically invalid (NaN, infinity)
//!
//! ### Cross-Sensor Violations  
//! - `CrossValidationFailed`: Multiple sensors disagree (e.g., impossible dew point)
//!
//! ### System Issues
//! - `SensorQualityBad`: Sensor degraded or offline
//! - `InsufficientData`: Not enough history for validation
//!
//! ## Error Handling Strategy
//!
//! ```rust
//! use edgeguard_core::{ValidationError, Validator, TemperatureValidator, ValidationContext};
//!
//! fn handle_sensor_reading(value: f32, validator: &TemperatureValidator, ctx: &ValidationContext) {
//!     match validator.validate(value, ctx) {
//!         Ok(()) => {
//!             // Reading is valid - proceed with normal processing
//!             // send_to_cloud(value);
//!         }
//!         Err(ValidationError::OutOfRange { .. }) => {
//!             // Sensor likely faulty - reading is impossible
//!             // mark_sensor_faulty();
//!         }
//!         Err(ValidationError::RateExceeded { .. }) => {
//!             // Possible electrical interference or sensor issue
//!             // increment_anomaly_counter();
//!         }
//!         Err(ValidationError::CrossValidationFailed { .. }) => {
//!             // Environmental conditions don't match physics
//!             // trigger_recalibration();
//!         }
//!         Err(ValidationError::SensorQualityBad { .. }) => {
//!             // Sensor needs maintenance
//!             // schedule_maintenance();
//!         }
//!         _ => {
//!             // Other errors - log and investigate
//!         }
//!     }
//! }
//! ```
//!
//! ## Memory Layout
//!
//! The largest error variant determines the enum size:
//! ```text
//! ValidationError size = 16 bytes
//! ├── Discriminant: 1 byte
//! ├── Largest variant (OutOfRange): 12 bytes
//! └── Padding: 3 bytes
//! ```

use thiserror_no_std::Error;

/// Result type for validation operations
pub type ValidationResult<T> = Result<T, ValidationError>;

/// Validation errors - kept small for embedded use
#[derive(Error, Debug, Clone, Copy, PartialEq)]
pub enum ValidationError {
    /// Value outside physical limits
    #[error("Value {value} outside range [{min}, {max}]")]
    OutOfRange {
        /// The actual sensor reading that failed validation
        value: f32,
        /// Minimum acceptable value based on physics
        min: f32,
        /// Maximum acceptable value based on physics
        max: f32,
    },
    
    /// Rate of change too high - indicates sensor malfunction or impossible physics
    #[error("Rate {rate}/s exceeds limit {max_rate}/s")]
    RateExceeded {
        /// Calculated rate of change (units per second)
        rate: f32,
        /// Maximum physically plausible rate
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
        /// Minimum number of samples needed for validation
        required: usize,
        /// Actual number of samples available
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