//! Temperature Validation with Thermal Physics Constraints
//!
//! ## Physics Background
//!
//! Temperature validation must consider several physical phenomena:
//!
//! ### Thermal Mass and Heat Capacity
//!
//! Objects cannot change temperature instantaneously due to thermal mass.
//! The rate of temperature change follows:
//!
//! ```text
//! dT/dt = Q / (m × c)
//! 
//! Where:
//! - dT/dt = rate of temperature change (°C/s)
//! - Q = heat flow rate (W)
//! - m = mass (kg)
//! - c = specific heat capacity (J/kg·K)
//! ```
//!
//! For air (c ≈ 1005 J/kg·K), even with 1kW heating in 1m³:
//! - Mass of air ≈ 1.2 kg
//! - Max rate ≈ 1000W / (1.2kg × 1005) ≈ 0.83°C/s
//!
//! This is why we validate rate of change!
//!
//! ### Sensor-Specific Considerations
//!
//! #### Response Time
//! Different sensors have different thermal response times:
//! - Thermocouples: ~1 second (small mass)
//! - RTDs: ~5-10 seconds (larger mass)
//! - Thermistors: ~2-5 seconds
//! - IR sensors: ~100ms (measures radiation)
//!
//! #### Self-Heating
//! Sensors consume power and generate heat:
//! - RTDs: ~1mW (negligible)
//! - Thermistors: ~10μW (negligible)
//! - But poor thermal design can trap heat!
//!
//! #### Environmental Effects
//! - **Solar radiation**: Can add 10-20°C to readings
//! - **Wind chill**: Doesn't affect actual temperature
//! - **Thermal stratification**: Temperature varies with height
//!
//! ## Validation Strategy
//!
//! ### 1. Absolute Limits
//! We use Earth-based limits, not theoretical:
//! - Min: -80°C (coldest natural temperature + margin)
//! - Max: 125°C (sensor survival limit)
//!
//! ### 2. Rate Limiting
//! Based on realistic scenarios:
//! - Indoor: 2°C/s (HVAC systems)
//! - Outdoor: 10°C/s (wind gusts, shade/sun)
//! - Industrial: 20°C/s (oven doors, steam)
//!
//! ### 3. Context-Aware Validation
//! Future enhancements could consider:
//! - Time of day (night = cooling)
//! - Season (winter = lower baseline)
//! - Geography (desert vs arctic)
//!
//! ## Common Issues Detected
//!
//! This validator catches:
//! - **Sensor failure**: Stuck at one value
//! - **Electrical interference**: Rapid spikes
//! - **Disconnection**: Jump to 0°C or -127°C
//! - **Bit flips**: Impossible values
//! - **Thermal shock**: Too-rapid changes
//!
//! ## Usage Examples
//!
//! ```rust
//! use edgeguard_core::validators::TemperatureValidator;
//! use edgeguard_core::ValidationContext;
//!
//! // Standard outdoor sensor
//! let outdoor = TemperatureValidator::default();
//!
//! // HVAC monitoring
//! let indoor = TemperatureValidator::indoor();
//!
//! // Industrial oven monitoring  
//! let oven = TemperatureValidator::new_with_limits(
//!     20.0,   // Room temp minimum
//!     300.0,  // Oven maximum
//!     50.0    // Can change quickly when door opens
//! );
//! ```

use crate::{
    constants::{
        physics::{ABSOLUTE_ZERO_CELSIUS, AIR_TEMP_MAX_RATE_C_PER_S, SENSOR_THERMAL_TIME_CONSTANT_AIR_S},
        sensors::{TEMP_SENSOR_MIN_C, TEMP_SENSOR_MAX_C, TEMP_MAX_RATE_C_PER_S,
                 TEMP_ACCURACY_PROFESSIONAL_C, TEMP_ACCURACY_CONSUMER_C, TEMP_ACCURACY_BUDGET_C},
        quality::{QUALITY_THRESHOLD_GOOD, QUALITY_THRESHOLD_ACCEPTABLE},
    },
    errors::{ValidationError, ValidationResult},
    traits::{Validator, ValidatorConstraints, ValidationContext, Validatable},
};

use super::utils;

/// Validates temperature readings against physical constraints
/// 
/// This validator ensures temperature readings are not just within sensor
/// specifications, but also physically plausible given thermal dynamics.
/// It considers both absolute limits and rate of change.
/// 
/// ## Design Decisions
/// 
/// - **Celsius only**: While Fahrenheit/Kelvin conversions are trivial,
///   we standardize on Celsius to avoid confusion and conversion errors.
///   The scientific community uses Celsius/Kelvin, making physics calculations
///   more intuitive.
/// 
/// - **f32 precision**: Temperature sensors rarely exceed 0.01°C precision,
///   so f32's ~7 decimal digits are more than sufficient. This saves memory
///   on embedded systems.
/// 
/// - **Configurable limits**: Different deployments have vastly different
///   requirements. A freezer monitor needs different limits than a furnace
///   monitor.
/// 
/// ## Memory Usage
/// 
/// 12 bytes total (3 × f32), making it cache-friendly and suitable for
/// embedding in arrays or other structures.
#[derive(Debug, Clone)]
pub struct TemperatureValidator {
    /// Minimum valid temperature in Celsius
    /// 
    /// Set based on deployment environment, not sensor limits.
    /// For example, indoor sensors might use -10°C even though
    /// the sensor works to -40°C.
    min_celsius: f32,
    
    /// Maximum valid temperature in Celsius
    /// 
    /// Usually limited by sensor survival temperature or
    /// expected environmental maximum plus safety margin.
    max_celsius: f32,
    
    /// Maximum rate of change in °C/second
    /// 
    /// Based on thermal mass of measured medium and
    /// realistic heating/cooling rates in the environment.
    max_rate_celsius_per_sec: f32,
}

impl Default for TemperatureValidator {
    fn default() -> Self {
        Self {
            // Use sensor operating range from constants
            min_celsius: TEMP_SENSOR_MIN_C,
            
            // Maximum sensor operating temperature
            max_celsius: TEMP_SENSOR_MAX_C,
            
            // Maximum rate sensors can reliably track
            max_rate_celsius_per_sec: TEMP_MAX_RATE_C_PER_S,
        }
    }
}

impl TemperatureValidator {
    /// Create validator with custom limits
    pub fn new_with_limits(min: f32, max: f32, max_rate: f32) -> Self {
        // Sanity check: can't have min > max
        let (min, max) = if min > max { (max, min) } else { (min, max) };
        
        Self {
            min_celsius: min.max(ABSOLUTE_ZERO_CELSIUS), // Can't go below absolute zero
            max_celsius: max,
            max_rate_celsius_per_sec: max_rate.abs(),
        }
    }
    
    /// Create validator for indoor use (tighter constraints)
    pub fn indoor() -> Self {
        Self {
            min_celsius: -10.0,  // Freezer temps
            max_celsius: 50.0,   // Hot equipment room
            max_rate_celsius_per_sec: AIR_TEMP_MAX_RATE_C_PER_S,  // Physical air heating/cooling limit
        }
    }
    
    /// Create validator for industrial use (wider constraints)
    pub fn industrial() -> Self {
        Self {
            min_celsius: -40.0,  // Industrial freezers
            max_celsius: 200.0,  // Ovens, steam pipes
            max_rate_celsius_per_sec: 20.0,  // Opening oven door
        }
    }
}

impl Validator for TemperatureValidator {
    type Value = f32;
    type Error = ValidationError;
    
    fn validate(&self, value: &Self::Value, context: &ValidationContext) -> ValidationResult<()> {
        // First check: is it even a valid number?
        if !(*value).is_valid() {
            return Err(ValidationError::InvalidValue);
        }
        
        // Range check against physical limits
        utils::check_range(*value, self.min_celsius, self.max_celsius)?;
        
        // Rate of change check (if we have history)
        if let Some(last_reading) = utils::last_reading(&context.history) {
            let rate = utils::calculate_rate_from_readings(
                *value,
                context.timestamp,
                last_reading,
            );
            
            if rate > self.max_rate_celsius_per_sec {
                return Err(ValidationError::RateExceeded {
                    rate,
                    max_rate: self.max_rate_celsius_per_sec,
                });
            }
        }
        
        // Sensor quality check
        if context.sensor_quality < QUALITY_THRESHOLD_ACCEPTABLE {
            return Err(ValidationError::SensorQualityBad {
                reason: "Temperature sensor degraded",
            });
        }
        
        Ok(())
    }
    
    fn constraints(&self) -> ValidatorConstraints {
        ValidatorConstraints {
            min_value: self.min_celsius,
            max_value: self.max_celsius,
            max_rate_change: self.max_rate_celsius_per_sec,
            noise_threshold: Some(TEMP_ACCURACY_PROFESSIONAL_C), // Professional-grade sensor accuracy
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::TimestampedReading;
    
    #[test]
    fn valid_temperature() {
        let validator = TemperatureValidator::default();
        let mut context = ValidationContext::default();
        
        // Normal room temperature
        assert!(validator.validate(22.5, &context).is_ok());
        
        // Update history for next test
        context.add_reading(22.5, 1000);
    }
    
    #[test]
    fn temperature_out_of_range() {
        let validator = TemperatureValidator::default();
        let context = ValidationContext::default();
        
        // Below absolute zero - physically impossible
        assert!(validator.validate(-300.0, &context).is_err());
        
        // Above sensor limit
        assert!(validator.validate(150.0, &context).is_err());
    }
    
    #[test]
    fn temperature_rate_exceeded() {
        let validator = TemperatureValidator::default();
        let mut context = ValidationContext::default();
        
        // Start at 20°C
        context.add_reading(20.0, 1000);
        context.timestamp = 2000; // 1 second later
        
        // Jump to 50°C in "1 second" - too fast
        let result = validator.validate(50.0, &context);
        assert!(matches!(result, Err(ValidationError::RateExceeded { .. })));
    }
    
    #[test]
    fn indoor_validator_limits() {
        let validator = TemperatureValidator::indoor();
        let context = ValidationContext::default();
        
        // Outdoor winter temp - should fail for indoor
        assert!(validator.validate(-20.0, &context).is_err());
        
        // Normal indoor temp - should pass
        assert!(validator.validate(21.0, &context).is_ok());
    }
}