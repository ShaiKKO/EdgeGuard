//! Physics-Based Sensor Validators
//!
//! ## Overview
//!
//! This module contains validators for common IoT sensors that go beyond simple
//! range checking to understand the physical laws governing each measurement type.
//! Each validator enforces constraints based on fundamental physics, not just
//! sensor specifications.
//!
//! ## Why Physics-Based Validation?
//!
//! Traditional validation only checks if values are within sensor specs:
//! - Temperature: -40°C to 125°C (sensor limit)
//! - Humidity: 0% to 100% (definition limit)
//! - Pressure: 300 to 1100 hPa (sensor range)
//!
//! But physics tells us much more:
//! - Temperature can't change 50°C in one second (thermal mass)
//! - 100% humidity at -10°C means ice/frost formation
//! - Pressure of 800 hPa at sea level indicates a severe storm
//!
//! ## Validation Layers
//!
//! Each validator implements multiple validation layers:
//!
//! ### 1. Range Validation
//! Basic physical limits - what's theoretically possible:
//! ```rust
//! // Absolute zero is -273.15°C, but no sensor operates there
//! const MIN_TEMP: f32 = -100.0;  // Coldest natural temperature on Earth
//! const MAX_TEMP: f32 = 150.0;   // Above this, most sensors fail
//! ```
//!
//! ### 2. Rate-of-Change Validation  
//! How fast can values physically change:
//! ```rust
//! // Air temperature changes slowly due to thermal mass
//! const MAX_TEMP_RATE: f32 = 5.0;  // °C/second is extreme
//! 
//! // But small sensors in moving air can change faster
//! const MAX_TEMP_RATE_FORCED_AIR: f32 = 10.0;  // With fan/wind
//! ```
//!
//! ### 3. Cross-Sensor Validation
//! Physical relationships between measurements:
//! ```rust
//! // Dew point temperature must be <= air temperature
//! if dew_point > air_temp {
//!     return Err(ValidationError::CrossValidationFailed {
//!         reason: "Dew point exceeds air temperature"
//!     });
//! }
//! ```
//!
//! ### 4. Environmental Context
//! Readings must make sense for the environment:
//! ```rust
//! // At 5000m altitude, pressure should be ~540 hPa
//! // Sea level reading of 1013 hPa would be impossible
//! ```
//!
//! ## Sensor-Specific Considerations
//!
//! ### Temperature
//! - Thermal mass limits rate of change
//! - Self-heating from electronics
//! - Radiation effects in sunlight
//!
//! ### Humidity
//! - Cannot exceed 100% (supersaturation rare)
//! - Temperature-dependent (warm air holds more moisture)
//! - Sensor degradation in condensing conditions
//!
//! ### Pressure  
//! - Altitude-dependent baseline
//! - Weather patterns cause gradual changes
//! - Rapid changes indicate sensor issues
//!
//! ## Usage Example
//!
//! ```rust
//! use edgeguard_core::validators::{TemperatureValidator, HumidityValidator};
//! use edgeguard_core::{ValidationContext, CrossValidator};
//!
//! // Individual sensor validation
//! let temp_validator = TemperatureValidator::default();
//! let humidity_validator = HumidityValidator::default();
//!
//! let mut ctx = ValidationContext::default();
//! ctx.timestamp = 1000;
//!
//! // Validate individual readings
//! temp_validator.validate(25.0, &ctx)?;
//! humidity_validator.validate(60.0, &ctx)?;
//!
//! // Cross-validation ensures physical consistency
//! // If temp=25°C and humidity=60%, dew point ≈ 16.7°C
//! // This is physically valid (dew point < air temp)
//! # Ok::<(), edgeguard_core::ValidationError>(())
//! ```
//!
//! ## Customization
//!
//! Validators can be customized for specific deployments:
//!
//! ```rust
//! use edgeguard_core::validators::TemperatureValidator;
//!
//! // Outdoor sensor in extreme environment
//! let arctic_validator = TemperatureValidator::new_with_limits(
//!     -60.0,  // Extreme cold
//!     40.0,   // Moderate heat  
//!     2.0     // Slower changes due to insulation
//! );
//!
//! // High-temperature industrial sensor
//! let furnace_validator = TemperatureValidator::new_with_limits(
//!     0.0,     // No freezing expected
//!     1000.0,  // Very high temps
//!     50.0     // Rapid changes possible
//! );
//! ```

mod temperature;
mod humidity;
mod pressure;
mod utils;

pub use temperature::TemperatureValidator;
pub use humidity::HumidityValidator;
pub use pressure::PressureValidator;