//! Humidity validator with dew point cross-validation
//!
//! Validates relative humidity with temperature-dependent constraints.
//! At low temperatures, 100% humidity is common. At high temps, it's rare.

use crate::{
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
            min_percent: -2.0,
            
            // Allow slight over 100% for supersaturation
            max_percent: 102.0,
            
            // Humidity can change quickly when doors open, AC kicks on, etc
            max_rate_percent_per_sec: 20.0,
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
            min_percent: 0.0,
            max_percent: 100.0,
            max_rate_percent_per_sec: 15.0,
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
                
                let alpha = (A * temp_c) / (B + temp_c) + libm::logf(rh_percent / 100.0);
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
            if dew_point > temp_c + 0.5 {  // 0.5°C margin for sensor error
                return Err(ValidationError::CrossValidationFailed {
                    reason: "Dew point exceeds air temperature",
                });
            }
        }
        
        // At freezing, 100% humidity is normal
        // At 40°C, >80% is unusual outside tropical areas
        if temp_c > 35.0 && humidity > 85.0 {
            return Err(ValidationError::CrossValidationFailed {
                reason: "High humidity at high temperature unlikely",
            });
        }
        
        Ok(())
    }
}

impl Validator for HumidityValidator {
    type Value = f32;
    
    fn validate(&self, value: Self::Value, context: &ValidationContext) -> ValidationResult<()> {
        // Check for valid number
        if !value.is_valid() {
            return Err(ValidationError::InvalidValue);
        }
        
        // Basic range check
        utils::check_range(value, self.min_percent, self.max_percent)?;
        
        // Rate of change check
        if let Some(last_reading) = utils::last_reading(&context.history) {
            let rate = utils::calculate_rate_from_readings(
                value,
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
            self.validate_with_temperature(value, temp)?;
        }
        
        // Sensor quality check
        if context.sensor_quality < 0.5 {
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
            noise_threshold: Some(2.0), // ±2% typical sensor noise
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