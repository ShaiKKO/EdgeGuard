//! Temperature validator with physics constraints
//!
//! Validates temperature readings based on:
//! - Absolute physical limits (can't go below absolute zero)
//! - Reasonable Earth surface temperatures
//! - Rate of change limits (thermal mass prevents instant changes)

use crate::{
    errors::{ValidationError, ValidationResult},
    traits::{Validator, ValidatorConstraints, ValidationContext, Validatable},
};

use super::utils;

/// Temperature validator for Celsius readings
#[derive(Debug, Clone)]
pub struct TemperatureValidator {
    /// Minimum valid temperature in Celsius
    min_celsius: f32,
    
    /// Maximum valid temperature in Celsius
    max_celsius: f32,
    
    /// Maximum rate of change in °C/second
    max_rate_celsius_per_sec: f32,
}

impl Default for TemperatureValidator {
    fn default() -> Self {
        Self {
            // Absolute zero is -273.15°C, but sensors rarely work that cold
            min_celsius: -80.0,  // Coldest natural Earth temp: -89.2°C Antarctica
            
            // Highest reliable sensor reading before damage
            max_celsius: 125.0,  // Most sensors fail above this
            
            // Realistic rate for air temperature changes
            // Even with direct flame, air temp sensors won't instantly jump
            max_rate_celsius_per_sec: 10.0,
        }
    }
}

impl TemperatureValidator {
    /// Create validator with custom limits
    pub fn new_with_limits(min: f32, max: f32, max_rate: f32) -> Self {
        // Sanity check: can't have min > max
        let (min, max) = if min > max { (max, min) } else { (min, max) };
        
        Self {
            min_celsius: min.max(-273.15), // Can't go below absolute zero
            max_celsius: max,
            max_rate_celsius_per_sec: max_rate.abs(),
        }
    }
    
    /// Create validator for indoor use (tighter constraints)
    pub fn indoor() -> Self {
        Self {
            min_celsius: -10.0,  // Freezer temps
            max_celsius: 50.0,   // Hot equipment room
            max_rate_celsius_per_sec: 2.0,  // HVAC can't change faster
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
    
    fn validate(&self, value: Self::Value, context: &ValidationContext) -> ValidationResult<()> {
        // First check: is it even a valid number?
        if !value.is_valid() {
            return Err(ValidationError::InvalidValue);
        }
        
        // Range check against physical limits
        utils::check_range(value, self.min_celsius, self.max_celsius)?;
        
        // Rate of change check (if we have history)
        if let Some(last_reading) = utils::last_reading(&context.history) {
            let rate = utils::calculate_rate_from_readings(
                value,
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
        if context.sensor_quality < 0.5 {
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
            noise_threshold: Some(0.1), // ±0.1°C typical sensor noise
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