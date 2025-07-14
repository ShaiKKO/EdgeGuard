//! Humidity Validation with Psychrometric Physics
//!
//! ## Physics Background
//!
//! ### What is Relative Humidity?
//!
//! Relative humidity (RH) is the ratio of water vapor present to the maximum
//! possible at a given temperature:
//!
//! ```text
//! RH = (Actual Vapor Pressure / Saturation Vapor Pressure) × 100%
//! ```
//!
//! Key insight: Warm air can hold more moisture than cold air. The relationship
//! is exponential, roughly doubling capacity every 10°C.
//!
//! ### Dew Point Physics
//!
//! The dew point is the temperature at which air becomes saturated (100% RH).
//! It represents the actual moisture content regardless of temperature:
//!
//! ```text
//! If Air Temp = 25°C, RH = 60%
//! Then Dew Point ≈ 16.7°C
//! 
//! Physical constraint: Dew Point ≤ Air Temperature (always!)
//! ```
//!
//! When dew point exceeds air temperature, you've violated physics - this
//! indicates sensor error or miscalibration.
//!
//! ### Sensor Characteristics
//!
//! #### Capacitive Sensors (Most Common)
//! - Polymer absorbs moisture, changing capacitance
//! - Can drift negative when very dry
//! - Degrades in condensing conditions
//! - Response time: 5-30 seconds
//!
//! #### Resistive Sensors
//! - Conductivity changes with moisture
//! - Less accurate but more robust
//! - Used in harsh environments
//!
//! ### Environmental Phenomena
//!
//! #### Supersaturation (>100% RH)
//! Possible in:
//! - Fog/mist (tiny droplets suspended)
//! - Rapid temperature drops
//! - Clean air without condensation nuclei
//!
//! #### Common Scenarios
//! - **Morning dew**: RH approaches 100% as temp drops to dew point
//! - **Air conditioning**: Removes moisture, dropping RH
//! - **Cooking/showers**: Rapid RH increase from water vapor
//! - **Winter heating**: Outdoor air warmed indoors = very low RH
//!
//! ## Validation Strategy
//!
//! ### 1. Physical Limits
//! - Theoretical: 0-100% (definition of percentage)
//! - Practical: -2% to 102% (sensor drift and supersaturation)
//!
//! ### 2. Rate of Change
//! Humidity can change rapidly:
//! - Opening dishwasher: +40% in seconds
//! - AC turning on: -20% in minutes
//! - But sensor response time limits observed rate
//!
//! ### 3. Cross-Sensor Validation
//! The most powerful validation uses temperature:
//! ```rust
//! if calculated_dew_point > air_temperature {
//!     // Physically impossible - sensors are wrong
//! }
//! ```
//!
//! ### 4. Temperature-Dependent Expectations
//! - At -10°C: 80-100% RH is normal (cold air holds little moisture)
//! - At 40°C: >80% RH is rare (would feel oppressive)
//! - Dew point >25°C: Dangerously humid for humans
//!
//! ## Common Issues Detected
//!
//! - **Sensor saturation**: Stuck at 100% after condensation
//! - **Dry-out**: Reads 0% after extreme desiccation  
//! - **Thermal lag**: RH spikes during rapid cooling
//! - **Contamination**: Drift from salt, dust, chemicals
//! - **Cross-talk**: Temperature sensor heating affects RH
//!
//! ## Usage Examples
//!
//! ```rust
//! use edgeguard_core::validators::HumidityValidator;
//! use edgeguard_core::ValidationContext;
//!
//! // Standard indoor/outdoor use
//! let validator = HumidityValidator::default();
//!
//! // Museum/archive - strict control
//! let museum = HumidityValidator::new_with_limits(
//!     40.0,  // Minimum to prevent static
//!     60.0,  // Maximum to prevent mold
//!     5.0    // Slow changes only
//! );
//!
//! // Industrial dryer - extreme conditions
//! let dryer = HumidityValidator::new_with_limits(
//!     0.0,   // Can be bone dry
//!     100.0, // Can be saturated
//!     50.0   // Rapid changes expected
//! );
//! ```

use crate::{
    constants::{
        sensors::{HUMIDITY_SENSOR_MIN_PCT, HUMIDITY_SENSOR_MAX_PCT, HUMIDITY_MAX_RATE_PCT_PER_S,
                 HUMIDITY_ACCURACY_CONSUMER_PCT},
        quality::{QUALITY_THRESHOLD_ACCEPTABLE},
        physics::{DEW_POINT_MARGIN_C, HIGH_TEMP_HUMIDITY_THRESHOLD_C, HIGH_TEMP_MAX_HUMIDITY_PCT},
    },
    errors::{ValidationError, ValidationResult},
    traits::{Validator, ValidatorConstraints, ValidationContext, Validatable},
    lookup::DEW_POINT_STANDARD,
};

use super::utils;

/// Humidity validator for relative humidity percentage
#[derive(Debug, Clone)]
pub struct HumidityValidator {
    /// Minimum valid RH% (sensors can drift negative)
    min_percent: f32,
    
    /// Maximum valid RH% (can exceed 100% in fog/mist)
    max_percent: f32,
    
    /// Maximum rate of change in %/second
    max_rate_percent_per_sec: f32,
}

impl Default for HumidityValidator {
    fn default() -> Self {
        Self {
            // Allow slight negative for sensor drift
            min_percent: HUMIDITY_SENSOR_MIN_PCT - 2.0,  // Account for sensor drift
            
            // Allow slight over 100% for supersaturation
            max_percent: HUMIDITY_SENSOR_MAX_PCT + 2.0,  // Account for fog/mist
            
            // Maximum rate sensors can reliably track
            max_rate_percent_per_sec: HUMIDITY_MAX_RATE_PCT_PER_S,
        }
    }
}

impl HumidityValidator {
    /// Create validator with custom limits
    pub fn new_with_limits(min: f32, max: f32, max_rate: f32) -> Self {
        Self {
            min_percent: min,
            max_percent: max,
            max_rate_percent_per_sec: max_rate.abs(),
        }
    }
    
    /// Strict validator that enforces 0-100% range
    pub fn strict() -> Self {
        Self {
            min_percent: HUMIDITY_SENSOR_MIN_PCT,
            max_percent: HUMIDITY_SENSOR_MAX_PCT,
            max_rate_percent_per_sec: 15.0,  // Slightly stricter than default
        }
    }
    
    /// Calculate dew point - uses lookup table for performance
    /// Falls back to Magnus formula if lookup fails
    fn calculate_dew_point(temp_c: f32, rh_percent: f32) -> Option<f32> {
        // Use lookup table for fast calculation on edge devices
        match DEW_POINT_STANDARD.lookup(temp_c, rh_percent) {
            Ok(dp) => Some(dp),
            Err(_) => {
                // Fallback to Magnus formula if lookup fails
                // This should rarely happen with proper clamping
                const A: f32 = 17.27;
                const B: f32 = 237.7;
                
                if rh_percent <= 0.0 {
                    return None;
                }
                
                // Natural logarithm approximation using Taylor series
                // ln(x) ≈ (x-1) - (x-1)²/2 + (x-1)³/3 - (x-1)⁴/4
                let x = rh_percent / 100.0;
                let ln_x = if (x - 1.0).abs() < 0.5 {
                    // Taylor series for x close to 1
                    let t = x - 1.0;
                    t - t * t * 0.5 + t * t * t / 3.0 - t * t * t * t * 0.25
                } else {
                    // For values farther from 1, use Newton's method
                    let mut ln_val = x - 1.0; // Initial guess
                    for _ in 0..3 {
                        // exp approximation for Newton iteration
                        let exp_ln = 1.0 + ln_val + ln_val * ln_val * 0.5;
                        ln_val = ln_val - (exp_ln - x) / exp_ln;
                    }
                    ln_val
                };
                
                let alpha = (A * temp_c) / (B + temp_c) + ln_x;
                let dew_point = (B * alpha) / (A - alpha);
                
                if dew_point.is_finite() {
                    Some(dew_point)
                } else {
                    None
                }
            }
        }
    }
    
    /// Validate humidity considering temperature (if available)
    fn validate_with_temperature(&self, humidity: f32, temp_c: f32) -> ValidationResult<()> {
        // Physical constraint: dew point can't exceed air temperature
        if let Some(dew_point) = Self::calculate_dew_point(temp_c, humidity) {
            if dew_point > temp_c + DEW_POINT_MARGIN_C {
                return Err(ValidationError::CrossValidationFailed {
                    reason: "Dew point exceeds air temperature",
                });
            }
        }
        
        // At freezing, 100% humidity is normal
        // At high temps, high humidity is unusual outside tropical areas
        if temp_c > HIGH_TEMP_HUMIDITY_THRESHOLD_C && humidity > HIGH_TEMP_MAX_HUMIDITY_PCT {
            return Err(ValidationError::CrossValidationFailed {
                reason: "High humidity at high temperature unlikely",
            });
        }
        
        Ok(())
    }
}

impl Validator for HumidityValidator {
    type Value = f32;
    type Error = ValidationError;
    
    fn validate(&self, value: &Self::Value, context: &ValidationContext) -> ValidationResult<()> {
        // Check for valid number
        if !(*value).is_valid() {
            return Err(ValidationError::InvalidValue);
        }
        
        // Basic range check
        utils::check_range(*value, self.min_percent, self.max_percent)?;
        
        // Rate of change check
        if let Some(last_reading) = utils::last_reading(&context.history) {
            let rate = utils::calculate_rate_from_readings(
                *value,
                context.timestamp,
                last_reading,
            );
            
            if rate > self.max_rate_percent_per_sec {
                return Err(ValidationError::RateExceeded {
                    rate,
                    max_rate: self.max_rate_percent_per_sec,
                });
            }
        }
        
        // Cross-validation with temperature if available
        if let Some(temp) = context.ambient_temp {
            self.validate_with_temperature(*value, temp)?;
        }
        
        // Sensor quality check
        if context.sensor_quality < QUALITY_THRESHOLD_ACCEPTABLE {
            return Err(ValidationError::SensorQualityBad {
                reason: "Humidity sensor degraded",
            });
        }
        
        Ok(())
    }
    
    fn constraints(&self) -> ValidatorConstraints {
        ValidatorConstraints {
            min_value: self.min_percent,
            max_value: self.max_percent,
            max_rate_change: self.max_rate_percent_per_sec,
            noise_threshold: Some(HUMIDITY_ACCURACY_CONSUMER_PCT), // Consumer-grade sensor accuracy
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn valid_humidity() {
        let validator = HumidityValidator::default();
        let context = ValidationContext::default();
        
        // Normal indoor humidity
        assert!(validator.validate(45.0, &context).is_ok());
        
        // High but valid
        assert!(validator.validate(95.0, &context).is_ok());
    }
    
    #[test]
    fn humidity_out_of_range() {
        let validator = HumidityValidator::default();
        let context = ValidationContext::default();
        
        // Way too low
        assert!(validator.validate(-10.0, &context).is_err());
        
        // Way too high
        assert!(validator.validate(150.0, &context).is_err());
    }
    
    #[test]
    fn humidity_cross_validation() {
        let validator = HumidityValidator::default();
        let mut context = ValidationContext::default();
        
        // Set ambient temp to 20°C
        context.ambient_temp = Some(20.0);
        
        // 80% humidity at 20°C is fine
        assert!(validator.validate(80.0, &context).is_ok());
        
        // But at 40°C, 90% humidity fails cross-validation
        context.ambient_temp = Some(40.0);
        let result = validator.validate(90.0, &context);
        assert!(matches!(result, Err(ValidationError::CrossValidationFailed { .. })));
    }
    
    #[test]
    fn dew_point_calculation() {
        // At 20°C and 50% RH, lookup table returns 13°C
        let dew_point = HumidityValidator::calculate_dew_point(20.0, 50.0);
        assert!(dew_point.is_some());
        let dp = dew_point.unwrap();
        assert!((dp - 13.0).abs() < 0.5);
    }
}