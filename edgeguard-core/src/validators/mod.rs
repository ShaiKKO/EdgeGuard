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
//! use edgeguard_core::constants::sensors::{TEMP_SENSOR_MIN_C, TEMP_SENSOR_MAX_C};
//! use edgeguard_core::constants::physics::ABSOLUTE_ZERO_CELSIUS;
//! 
//! // Sensor operating range based on commercial datasheets
//! // TEMP_SENSOR_MIN_C = -80.0°C (industrial sensors)
//! // TEMP_SENSOR_MAX_C = 125.0°C (sensor survival limit)
//! ```
//!
//! ### 2. Rate-of-Change Validation  
//! How fast can values physically change:
//! ```rust
//! use edgeguard_core::constants::sensors::TEMP_MAX_RATE_C_PER_S;
//! use edgeguard_core::constants::physics::AIR_TEMP_MAX_RATE_C_PER_S;
//! 
//! // Temperature change rates based on thermal physics
//! // TEMP_MAX_RATE_C_PER_S = 10.0°C/s (sensor tracking limit)
//! // AIR_TEMP_MAX_RATE_C_PER_S = 10.0°C/s (physical air heating limit)
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
//! use edgeguard_core::constants::physics::SEA_LEVEL_PRESSURE_HPA;
//! 
//! // At 5000m altitude, pressure should be ~540 hPa
//! // SEA_LEVEL_PRESSURE_HPA = 1013.25 hPa would be impossible at altitude
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
//! use edgeguard_core::constants::sensors::TEMP_SENSOR_MIN_C;
//! use edgeguard_core::constants::physics::WATER_FREEZING_POINT_C;
//!
//! // Outdoor sensor in extreme environment
//! let arctic_validator = TemperatureValidator::new_with_limits(
//!     -60.0,  // Below standard sensor range for arctic conditions
//!     40.0,   // Moderate heat limit
//!     2.0     // Slower changes due to thermal mass
//! );
//!
//! // High-temperature industrial sensor
//! let furnace_validator = TemperatureValidator::new_with_limits(
//!     WATER_FREEZING_POINT_C,  // No freezing expected (0°C)
//!     1000.0,  // Specialized high-temp sensor
//!     50.0     // Rapid changes in industrial setting
//! );
//! ```

mod temperature;
mod humidity;
mod pressure;
mod utils;

pub use temperature::TemperatureValidator;
pub use humidity::HumidityValidator;
pub use pressure::PressureValidator;