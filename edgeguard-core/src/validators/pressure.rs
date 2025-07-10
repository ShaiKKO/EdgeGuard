//! Pressure validator with altitude compensation
//!
//! Validates atmospheric pressure readings considering altitude.
//! Pressure decreases predictably with altitude - about 12 hPa per 100m.

use crate::{
    errors::{ValidationError, ValidationResult},
    traits::{Validator, ValidatorConstraints, ValidationContext, Validatable},
    lookup::AltitudeTable,
};

use super::utils;

/// Pressure validator for atmospheric pressure in hPa (hectopascals)
#[derive(Debug, Clone)]
pub struct PressureValidator {
    /// Minimum valid pressure in hPa
    min_hpa: f32,
    
    /// Maximum valid pressure in hPa  
    max_hpa: f32,
    
    /// Maximum rate of change in hPa/second
    max_rate_hpa_per_sec: f32,
    
    /// Altitude in meters (for compensation)
    altitude_m: f32,
}

impl Default for PressureValidator {
    fn default() -> Self {
        Self {
            // Lowest ever recorded: 870 hPa (typhoon)
            min_hpa: 850.0,
            
            // Highest ever recorded: 1084 hPa (Siberian high)
            max_hpa: 1090.0,
            
            // Pressure changes slowly except during severe weather
            max_rate_hpa_per_sec: 0.5,  // ~30 hPa/minute is extreme
            
            // Sea level by default
            altitude_m: 0.0,
        }
    }
}

impl PressureValidator {
    /// Create validator for a specific altitude
    pub fn new_with_altitude(altitude_m: f32) -> Self {
        let mut validator = Self::default();
        validator.altitude_m = altitude_m;
        
        // Use lookup table for accurate altitude adjustment
        if let Ok(adjustment) = AltitudeTable::STANDARD.get_adjustment(altitude_m) {
            // adjustment is negative, so we add it to reduce pressure
            validator.min_hpa += adjustment;
            validator.max_hpa += adjustment;
        } else {
            // Fallback to simple approximation
            let adjustment = (altitude_m / 100.0) * 12.0;
            validator.min_hpa -= adjustment;
            validator.max_hpa -= adjustment;
        }
        
        validator
    }
    
    /// Create validator for aircraft/high altitude use
    pub fn high_altitude() -> Self {
        Self {
            min_hpa: 200.0,  // ~12km altitude
            max_hpa: 1090.0, // Still need sea level max
            max_rate_hpa_per_sec: 2.0, // Rapid altitude changes
            altitude_m: 0.0, // Variable altitude
        }
    }
    
    /// Convert pressure to approximate altitude using barometric formula
    fn pressure_to_altitude(pressure_hpa: f32, sea_level_pressure: f32) -> f32 {
        // Simplified barometric formula
        const TEMP_LAPSE: f32 = 0.0065; // K/m
        const SEA_LEVEL_TEMP: f32 = 288.15; // K
        const G: f32 = 9.80665; // m/s²
        const M: f32 = 0.0289644; // kg/mol
        const R: f32 = 8.31432; // J/(mol·K)
        
        let pressure_ratio = pressure_hpa / sea_level_pressure;
        let exponent = (R * TEMP_LAPSE) / (G * M);
        
        SEA_LEVEL_TEMP / TEMP_LAPSE * (1.0 - libm::powf(pressure_ratio, exponent))
    }
    
    /// Check if pressure makes sense for current altitude
    fn validate_altitude_consistency(&self, pressure: f32) -> ValidationResult<()> {
        // Standard sea level pressure
        const STANDARD_PRESSURE: f32 = 1013.25;
        
        // Calculate expected pressure at our altitude
        let expected_adjustment = (self.altitude_m / 100.0) * 12.0;
        let expected_pressure = STANDARD_PRESSURE - expected_adjustment;
        
        // Allow ±50 hPa for weather variations
        let tolerance = 50.0;
        
        if (pressure - expected_pressure).abs() > tolerance {
            // Check if this could be extreme weather
            if pressure < expected_pressure - tolerance {
                return Err(ValidationError::CrossValidationFailed {
                    reason: "Pressure too low for altitude",
                });
            } else if pressure > expected_pressure + tolerance {
                return Err(ValidationError::CrossValidationFailed {
                    reason: "Pressure too high for altitude",
                });
            }
        }
        
        Ok(())
    }
}

impl Validator for PressureValidator {
    type Value = f32;
    
    fn validate(&self, value: Self::Value, context: &ValidationContext) -> ValidationResult<()> {
        // Check for valid number
        if !value.is_valid() {
            return Err(ValidationError::InvalidValue);
        }
        
        // Basic range check
        utils::check_range(value, self.min_hpa, self.max_hpa)?;
        
        // Rate of change check
        if let Some(last_reading) = utils::last_reading(&context.history) {
            let rate = utils::calculate_rate_from_readings(
                value,
                context.timestamp,
                last_reading,
            );
            
            if rate > self.max_rate_hpa_per_sec {
                return Err(ValidationError::RateExceeded {
                    rate,
                    max_rate: self.max_rate_hpa_per_sec,
                });
            }
        }
        
        // Altitude consistency check
        if self.altitude_m != 0.0 {
            self.validate_altitude_consistency(value)?;
        }
        
        // Sensor quality check
        if context.sensor_quality < 0.5 {
            return Err(ValidationError::SensorQualityBad {
                reason: "Pressure sensor degraded",
            });
        }
        
        Ok(())
    }
    
    fn constraints(&self) -> ValidatorConstraints {
        ValidatorConstraints {
            min_value: self.min_hpa,
            max_value: self.max_hpa,
            max_rate_change: self.max_rate_hpa_per_sec,
            noise_threshold: Some(0.1), // ±0.1 hPa typical sensor noise
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn valid_pressure_sea_level() {
        let validator = PressureValidator::default();
        let context = ValidationContext::default();
        
        // Standard pressure
        assert!(validator.validate(1013.25, &context).is_ok());
        
        // Low pressure system
        assert!(validator.validate(990.0, &context).is_ok());
        
        // High pressure system
        assert!(validator.validate(1030.0, &context).is_ok());
    }
    
    #[test]
    fn pressure_out_of_range() {
        let validator = PressureValidator::default();
        let context = ValidationContext::default();
        
        // Too low - beyond any recorded storm
        assert!(validator.validate(800.0, &context).is_err());
        
        // Too high - impossible at sea level
        assert!(validator.validate(1100.0, &context).is_err());
    }
    
    #[test]
    fn pressure_at_altitude() {
        // Denver is ~1600m elevation
        let validator = PressureValidator::new_with_altitude(1600.0);
        let context = ValidationContext::default();
        
        // Typical Denver pressure ~835 hPa
        assert!(validator.validate(835.0, &context).is_ok());
        
        // Sea level pressure would fail at altitude
        let result = validator.validate(1013.0, &context);
        assert!(result.is_err());
    }
    
    #[test]
    fn pressure_rate_change() {
        let validator = PressureValidator::default();
        let mut context = ValidationContext::default();
        
        // Start at 1013 hPa
        context.add_reading(1013.0, 1000);
        context.timestamp = 2000; // 1 second later
        
        // Drop 10 hPa in "1 second" - way too fast
        let result = validator.validate(1003.0, &context);
        assert!(matches!(result, Err(ValidationError::RateExceeded { .. })));
    }
}