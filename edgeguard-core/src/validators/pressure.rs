//! Atmospheric Pressure Validation with Barometric Physics
//!
//! ## Physics Background
//!
//! ### Atmospheric Pressure Fundamentals
//!
//! Atmospheric pressure is the weight of air above a given point. At sea level,
//! the entire atmosphere (~100km high) weighs down with a force of:
//!
//! ```text
//! 1013.25 hPa = 101,325 Pa = 14.696 psi = 1 atmosphere
//! ```
//!
//! ### Altitude Relationship
//!
//! Pressure decreases exponentially with altitude following the barometric formula:
//!
//! ```text
//! P(h) = P₀ × (1 - 0.0065h/T₀)^5.255
//!
//! Where:
//! - P(h) = pressure at altitude h
//! - P₀ = sea level pressure (1013.25 hPa)
//! - h = altitude in meters
//! - T₀ = sea level temperature (288.15 K)
//! ```
//!
//! Simplified: Pressure drops ~12 hPa per 100m near sea level
//!
//! ### Weather Effects
//!
//! Pressure variations indicate weather patterns:
//! - **High pressure (>1020 hPa)**: Clear, stable weather
//! - **Low pressure (<1010 hPa)**: Storms, precipitation
//! - **Rapid drop (>1 hPa/hour)**: Approaching storm
//!
//! Historical extremes:
//! - Lowest: 870 hPa (Typhoon Tip, 1979)
//! - Highest: 1084 hPa (Siberian High, 1968)
//!
//! ### Sensor Technology
//!
//! #### MEMS Pressure Sensors
//! - Microscopic diaphragm deflects with pressure
//! - Measures capacitance or resistance change
//! - Resolution: 0.01 hPa (10cm altitude)
//! - Temperature sensitive - needs compensation
//!
//! #### Piezoresistive Sensors
//! - Silicon strain gauge changes resistance
//! - Very accurate but power hungry
//! - Used in weather stations
//!
//! ## Validation Strategy
//!
//! ### 1. Altitude-Adjusted Limits
//!
//! Raw pressure limits don't work at altitude:
//! - Sea level: 950-1050 hPa typical
//! - Denver (1600m): 810-870 hPa typical  
//! - Aircraft (10km): ~265 hPa
//!
//! We adjust limits based on deployment altitude.
//!
//! ### 2. Rate of Change Validation
//!
//! Pressure changes indicate:
//! - **0.1 hPa/hour**: Normal variation
//! - **1 hPa/hour**: Weather front passing
//! - **5 hPa/hour**: Severe storm
//! - **>10 hPa/hour**: Sensor error or rapid altitude change
//!
//! ### 3. Cross-Validation Opportunities
//!
//! - **GPS altitude**: Verify pressure matches altitude
//! - **Temperature**: Cold fronts bring pressure drops
//! - **Humidity**: Low pressure often means moisture
//! - **Multiple sensors**: Should agree within 2 hPa
//!
//! ## Common Issues Detected
//!
//! - **Blocked port**: Pressure stuck at one value
//! - **Temperature drift**: Uncompensated thermal effects
//! - **Altitude mismatch**: Sea-level pressure at altitude
//! - **Rapid transit**: Elevator/aircraft pressure changes
//! - **Weather extremes**: Hurricane/tornado conditions
//!
//! ## Practical Applications
//!
//! ### Weather Monitoring
//! ```rust
//! use edgeguard_core::validators::PressureValidator;
//!
//! // Weather station at 500m elevation
//! let weather = PressureValidator::new_with_altitude(500.0);
//! ```
//!
//! ### Indoor Air Quality
//! ```rust
//! // Building HVAC - minimal variation expected
//! let hvac = PressureValidator {
//!     min_hpa: 1000.0,
//!     max_hpa: 1030.0,
//!     max_rate_hpa_per_sec: 0.1,  // Very stable
//!     altitude_m: 100.0,
//! };
//! ```
//!
//! ### Drone/Aircraft
//! ```rust
//! // High altitude, rapid changes
//! let aircraft = PressureValidator::high_altitude();
//! ```
//!
//! ## Altitude Compensation
//!
//! The validator automatically adjusts expectations based on altitude:
//!
//! ```text
//! Location        Altitude   Typical Pressure   Adjusted Range
//! -------------------------------------------------------------
//! Dead Sea        -430m      1065 hPa          1040-1090 hPa
//! Sea Level       0m         1013 hPa          950-1050 hPa
//! Denver          1600m      835 hPa           810-870 hPa
//! Mt. Everest     8848m      315 hPa           300-330 hPa
//! ```

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
    
    /// Convert pressure reading to approximate altitude using the International Standard Atmosphere model
    /// 
    /// This implements the barometric formula for the troposphere (0-11km altitude):
    /// ```text
    /// h = (T₀/L) × [1 - (P/P₀)^(R×L/g×M)]
    /// 
    /// Where:
    /// - h = altitude above sea level (meters)
    /// - T₀ = standard temperature at sea level (288.15 K)
    /// - L = temperature lapse rate (0.0065 K/m)
    /// - P = measured pressure (hPa)
    /// - P₀ = sea level pressure (hPa)
    /// - R = ideal gas constant (8.31432 J/(mol·K))
    /// - g = gravitational acceleration (9.80665 m/s²)
    /// - M = molar mass of dry air (0.0289644 kg/mol)
    /// ```
    /// 
    /// ## Why libm::powf?
    /// 
    /// We use `libm::powf` instead of the standard `f32::powf` because:
    /// - This crate is `no_std` compatible for embedded systems
    /// - Many embedded targets lack hardware floating-point units
    /// - `libm` provides software implementations that work everywhere
    /// - The performance difference is negligible for validation use cases
    /// 
    /// ## Derivation of the Exponent
    /// 
    /// The exponent `(R×L)/(g×M)` comes from solving the hydrostatic equation
    /// with the ideal gas law under the assumption of linear temperature decrease:
    /// 
    /// ```text
    /// Hydrostatic equation: dP/dh = -ρg
    /// Ideal gas law: P = ρRT/M
    /// Temperature model: T = T₀ - L×h
    /// 
    /// Combining and integrating gives:
    /// P/P₀ = (T/T₀)^(g×M/R×L) = (1 - L×h/T₀)^(g×M/R×L)
    /// 
    /// Solving for h yields our formula with exponent ≈ 0.1902
    /// ```
    /// 
    /// ## Accuracy and Limitations
    /// 
    /// - Accurate to ±30m for altitudes up to 6000m
    /// - Assumes standard atmosphere (15°C at sea level)
    /// - Ignores humidity effects (dry air assumption)
    /// - Not valid above 11km (troposphere limit)
    fn pressure_to_altitude(pressure_hpa: f32, sea_level_pressure: f32) -> f32 {
        // Temperature lapse rate: rate at which temperature decreases with altitude
        // Standard value: 6.5°C per 1000m in the troposphere
        const TEMP_LAPSE: f32 = 0.0065; // K/m
        
        // Standard temperature at sea level (15°C = 288.15 K)
        // This is the global average used in aviation
        const SEA_LEVEL_TEMP: f32 = 288.15; // K
        
        // Standard gravitational acceleration at Earth's surface
        // Varies slightly with latitude but standardized for calculations
        const G: f32 = 9.80665; // m/s²
        
        // Molar mass of dry air (weighted average of N₂, O₂, Ar, etc.)
        // 78% N₂ (28.014) + 21% O₂ (31.998) + 1% Ar (39.948) ≈ 28.9644 g/mol
        const M: f32 = 0.0289644; // kg/mol
        
        // Universal gas constant (energy per temperature per mole)
        const R: f32 = 8.31432; // J/(mol·K)
        
        // Calculate the pressure ratio (dimensionless)
        let pressure_ratio = pressure_hpa / sea_level_pressure;
        
        // Calculate the exponent for the barometric formula
        // This equals approximately 0.1902 for Earth's atmosphere
        let exponent = (R * TEMP_LAPSE) / (G * M);
        
        // Apply the barometric formula
        // Note: libm::powf is used for no_std compatibility
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