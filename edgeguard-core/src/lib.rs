//! EdgeGuard Core - Physics-Aware Sensor Data Validation Engine
//! 
//! ## Overview
//! 
//! EdgeGuard Core provides a lightweight, embedded-friendly validation engine for IoT sensor data.
//! Unlike traditional threshold-based validation, EdgeGuard understands the physical laws governing
//! sensor behavior, enabling it to detect subtle anomalies that simple range checks would miss.
//! 
//! ## Architecture & Design Philosophy
//! 
//! The validation engine is built around several key principles:
//! 
//! ### 1. Physics-First Validation
//! We don't just check if temperature is between -40°C and 85°C. We understand that:
//! - Temperature cannot change faster than the thermal mass allows
//! - Humidity and temperature are related through dew point physics  
//! - Pressure changes follow atmospheric models based on altitude
//! 
//! ### 2. Zero-Allocation Design
//! Every allocation is planned at compile time:
//! - Fixed-size circular buffers for sensor history
//! - Const generics for compile-time sizing
//! - Stack-based data structures throughout
//! - Pre-computed lookup tables instead of runtime calculations
//! 
//! ### 3. Deterministic Performance
//! Validation always completes in bounded time:
//! - O(1) lookups for expensive calculations (dew point, altitude)
//! - Fixed iteration counts (no dynamic loops)
//! - Predictable memory access patterns
//! 
//! ### 4. Cross-Sensor Validation
//! Sensors don't exist in isolation. We validate relationships:
//! - Dew point must be below air temperature
//! - Pressure must match altitude expectations
//! - Rate of change must be physically plausible
//! 
//! ## Memory Model
//! 
//! The engine is designed for severe memory constraints:
//! 
//! ```text
//! Total RAM budget: 32KB (ESP32 target)
//! ├── Validation engine: <1KB
//! ├── Sensor history: 2KB (configurable)
//! ├── Lookup tables: 1-4KB (depending on precision)
//! └── Application space: ~26KB
//! ```
//! 
//! ## Performance Characteristics
//! 
//! Measured on ESP32-C3 (160MHz RISC-V):
//! - Single validation: <100μs
//! - With cross-validation: <250μs  
//! - Lookup table access: <10μs
//! - History update: <50μs
//! 
//! ## Safety Considerations
//! 
//! EdgeGuard may be used in safety-critical applications. The design ensures:
//! - No panics in production (all errors are handled)
//! - No undefined behavior (unsafe code is forbidden)
//! - Deterministic behavior (no hidden allocations)
//! - Fail-safe defaults (suspicious readings are rejected)
//! 
//! ## Usage Examples
//! 
//! ### Basic Validation
//! ```no_run
//! use edgeguard_core::{Validator, TemperatureValidator, ValidationContext};
//! 
//! let validator = TemperatureValidator::default();
//! let mut context = ValidationContext::default();
//! 
//! // Single reading validation
//! match validator.validate(25.0, &context) {
//!     Ok(_) => println!("Valid reading"),
//!     Err(e) => println!("Invalid: {:?}", e),
//! }
//! ```
//! 
//! ### With History Tracking
//! ```no_run
//! use edgeguard_core::{Validator, TemperatureValidator, ValidationContext};
//! 
//! let validator = TemperatureValidator::default();
//! let mut context = ValidationContext::default();
//! 
//! // Add historical readings for rate-of-change validation
//! context.add_reading(24.5, 1000);
//! context.add_reading(24.7, 2000);
//! context.add_reading(24.9, 3000);
//! context.timestamp = 4000;
//! 
//! // This will check both range AND rate of change
//! validator.validate(25.1, &context)?;
//! # Ok::<(), edgeguard_core::ValidationError>(())
//! ```
//! 
//! ### Cross-Sensor Validation
//! ```no_run
//! use edgeguard_core::{Validator, HumidityValidator, ValidationContext};
//! 
//! let validator = HumidityValidator::default();
//! let mut context = ValidationContext::default();
//! 
//! // Set ambient temperature for dew point validation
//! context.ambient_temp = Some(20.0); // 20°C
//! 
//! // This humidity would create impossible dew point
//! match validator.validate(100.0, &context) {
//!     Err(_) => println!("Physics violation detected!"),
//!     Ok(_) => {},
//! }
//! ```
//! 
//! ## Feature Flags
//! 
//! - `std` (default): Standard library support, includes error formatting
//! - `embedded`: Optimizations for embedded systems, enables defmt logging
//! - `low_memory_tables`: Reduces lookup table size from ~1KB to ~200B
//! - `high_precision_tables`: Increases lookup table precision at cost of ~4KB RAM

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_code)]
#![warn(missing_docs)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod buffer;
pub mod errors;
pub mod events;
pub mod lookup;
pub mod pipeline;
pub mod queue;
pub mod stream;
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

/// EdgeGuard Core library version
/// 
/// Useful for runtime version checks and telemetry
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_exists() {
        assert!(!VERSION.is_empty());
    }
}