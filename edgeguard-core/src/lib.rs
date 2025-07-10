//! Core validation engine for EdgeGuard
//!
//! Handles sensor data validation with physics-based constraints.
//! Designed for edge devices with limited resources.
//!
//! Key constraints:
//! - Runs on 32KB RAM (ESP32)
//! - No heap allocation in hot path
//! - Sub-millisecond validation latency
//!
//! ```no_run
//! use edgeguard_core::{Validator, TemperatureValidator, ValidationContext};
//! 
//! let validator = TemperatureValidator::default();
//! let context = ValidationContext::default();
//! 
//! // Validate sensor reading
//! match validator.validate(25.0, &context) {
//!     Ok(_) => {}, // Good to go
//!     Err(e) => {}, // Handle invalid reading
//! }
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_code)]
#![warn(missing_docs)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod buffer;
pub mod errors;
pub mod lookup;
pub mod time;
pub mod traits;
pub mod validators;

// Public API
pub use errors::{ValidationError, ValidationResult};
pub use traits::{Validator, CrossValidator, ValidationContext};
pub use validators::{
    TemperatureValidator,
    HumidityValidator,
    PressureValidator,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_exists() {
        assert!(!VERSION.is_empty());
    }
}