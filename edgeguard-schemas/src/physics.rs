//! Physics Constraints for Sensor Validation
//!
//! This module defines physics-based constraints that can be embedded in Avro schemas
//! and used for runtime validation. These constraints encode fundamental physical laws
//! and sensor limitations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete set of physics constraints for a sensor type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorConstraints {
    /// Absolute minimum based on physics laws
    pub absolute_min: Option<f64>,
    
    /// Absolute maximum based on physics laws
    pub absolute_max: Option<f64>,
    
    /// Typical minimum for this sensor type
    pub typical_min: Option<f64>,
    
    /// Typical maximum for this sensor type
    pub typical_max: Option<f64>,
    
    /// Maximum rate of change per second
    pub max_rate_change: Option<f64>,
    
    /// SI unit for this measurement
    pub si_unit: String,
    
    /// Common display unit
    pub display_unit: String,
    
    /// Cross-sensor validation rules
    pub cross_validation: Vec<CrossValidationRule>,
}

/// Cross-validation rule between sensors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationRule {
    /// Other sensor type required
    pub requires_sensor: String,
    
    /// Type of validation
    pub validation_type: CrossValidationType,
    
    /// Parameters for validation
    pub parameters: HashMap<String, f64>,
}

/// Types of cross-sensor validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrossValidationType {
    /// Dew point must not exceed temperature
    DewPointConstraint,
    
    /// Pressure must match altitude
    AltitudePressureConsistency,
    
    /// Heat index calculation
    HeatIndexValidation,
    
    /// Custom validation rule
    Custom(String),
}

impl SensorConstraints {
    /// Temperature sensor constraints (Celsius)
    pub fn temperature_celsius() -> Self {
        Self {
            absolute_min: Some(-273.15), // Absolute zero
            absolute_max: Some(5778.0),  // Sun's surface (theoretical)
            typical_min: Some(-80.0),    // Practical sensor limit
            typical_max: Some(125.0),    // Practical sensor limit
            max_rate_change: Some(10.0), // 10°C/s for small sensors
            si_unit: "kelvin".to_string(),
            display_unit: "celsius".to_string(),
            cross_validation: vec![
                CrossValidationRule {
                    requires_sensor: "humidity".to_string(),
                    validation_type: CrossValidationType::DewPointConstraint,
                    parameters: HashMap::new(),
                },
            ],
        }
    }
    
    /// Pressure sensor constraints (hectopascals/mbar)
    pub fn pressure_hpa() -> Self {
        Self {
            absolute_min: Some(0.0),      // Vacuum
            absolute_max: Some(1100.0),   // Deep ocean equivalent
            typical_min: Some(870.0),     // Severe storm
            typical_max: Some(1085.0),    // Siberian high
            max_rate_change: Some(10.0),  // 10 hPa/s for rapid changes
            si_unit: "pascal".to_string(),
            display_unit: "hectopascal".to_string(),
            cross_validation: vec![
                CrossValidationRule {
                    requires_sensor: "altitude".to_string(),
                    validation_type: CrossValidationType::AltitudePressureConsistency,
                    parameters: HashMap::from([
                        ("sea_level_pressure".to_string(), 1013.25),
                    ]),
                },
            ],
        }
    }
    
    /// Humidity sensor constraints (relative humidity %)
    pub fn humidity_percent() -> Self {
        Self {
            absolute_min: Some(0.0),
            absolute_max: Some(100.0),
            typical_min: Some(0.0),
            typical_max: Some(100.0),
            max_rate_change: Some(20.0), // 20%/s for small volumes
            si_unit: "ratio".to_string(),
            display_unit: "percent".to_string(),
            cross_validation: vec![
                CrossValidationRule {
                    requires_sensor: "temperature".to_string(),
                    validation_type: CrossValidationType::DewPointConstraint,
                    parameters: HashMap::new(),
                },
            ],
        }
    }
    
    /// VOC sensor constraints (parts per billion)
    pub fn voc_ppb() -> Self {
        Self {
            absolute_min: Some(0.0),
            absolute_max: Some(10000.0), // 10 ppm max
            typical_min: Some(0.0),
            typical_max: Some(1000.0),   // 1 ppm typical indoor max
            max_rate_change: Some(100.0), // 100 ppb/s
            si_unit: "ppb".to_string(),
            display_unit: "ppb".to_string(),
            cross_validation: vec![],
        }
    }
    
    /// Particulate matter constraints (µg/m³)
    pub fn pm25_ugm3() -> Self {
        Self {
            absolute_min: Some(0.0),
            absolute_max: Some(1000.0),   // Hazardous levels
            typical_min: Some(0.0),
            typical_max: Some(500.0),     // Very unhealthy
            max_rate_change: Some(50.0),  // 50 µg/m³/s
            si_unit: "kg/m3".to_string(),
            display_unit: "ug/m3".to_string(),
            cross_validation: vec![],
        }
    }
    
    /// Acoustic sensor constraints (decibels SPL)
    pub fn acoustic_db_spl() -> Self {
        Self {
            absolute_min: Some(0.0),      // Threshold of hearing
            absolute_max: Some(194.0),    // Maximum in air at 1 atm
            typical_min: Some(20.0),      // Quiet room
            typical_max: Some(140.0),     // Pain threshold
            max_rate_change: Some(50.0),  // 50 dB/s for impulse sounds
            si_unit: "pascal".to_string(),
            display_unit: "dB_SPL".to_string(),
            cross_validation: vec![],
        }
    }
    
    /// Acoustic frequency constraints (Hz)
    pub fn acoustic_frequency_hz() -> Self {
        Self {
            absolute_min: Some(0.0),      // DC
            absolute_max: Some(100000.0), // Ultrasonic
            typical_min: Some(20.0),      // Human hearing lower limit
            typical_max: Some(20000.0),   // Human hearing upper limit
            max_rate_change: Some(10000.0), // 10 kHz/s for sweeps
            si_unit: "hertz".to_string(),
            display_unit: "Hz".to_string(),
            cross_validation: vec![],
        }
    }
    
    /// Vibration sensor constraints (g-force acceleration)
    pub fn vibration_g() -> Self {
        Self {
            absolute_min: Some(-100.0),   // Extreme shock
            absolute_max: Some(100.0),    // Extreme shock
            typical_min: Some(-10.0),     // Industrial vibration
            typical_max: Some(10.0),      // Industrial vibration
            max_rate_change: Some(1000.0), // 1000 g/s for impact
            si_unit: "m/s2".to_string(),
            display_unit: "g".to_string(),
            cross_validation: vec![],
        }
    }
    
    /// EMF sensor constraints (V/m electric field)
    pub fn emf_electric_vm() -> Self {
        Self {
            absolute_min: Some(0.0),
            absolute_max: Some(30000.0),  // Near high voltage lines
            typical_min: Some(0.1),       // Background levels
            typical_max: Some(300.0),     // Residential exposure limit
            max_rate_change: Some(100.0), // 100 V/m/s
            si_unit: "V/m".to_string(),
            display_unit: "V/m".to_string(),
            cross_validation: vec![],
        }
    }
}

/// Property key for physics constraints in Avro schemas
pub const CONSTRAINTS_PROPERTY_KEY: &str = "edgeguard.constraints";

/// Extract constraints from schema custom properties
/// 
/// Looks for "edgeguard.constraints" property in Avro schema
pub fn extract_from_schema(schema: &apache_avro::Schema) -> Option<SensorConstraints> {
    match schema {
        apache_avro::Schema::Record { attributes, .. } => {
            // Look for our custom property
            if let Some(value) = attributes.get(CONSTRAINTS_PROPERTY_KEY) {
                // Deserialize from JSON value
                serde_json::from_value(value.clone()).ok()
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Embed constraints into schema as custom property
pub fn embed_in_schema(
    schema: &mut apache_avro::Schema,
    constraints: &SensorConstraints,
) -> Result<(), crate::SchemaError> {
    match schema {
        apache_avro::Schema::Record { attributes, .. } => {
            // Serialize constraints to JSON
            let constraints_json = serde_json::to_value(constraints)
                .map_err(|e| crate::SchemaError::SerializationError(e.to_string()))?;
            
            // Add as custom property
            attributes.insert(CONSTRAINTS_PROPERTY_KEY.to_string(), constraints_json);
            Ok(())
        }
        _ => Err(crate::SchemaError::InvalidSchema(
            "Can only embed constraints in Record schemas".to_string()
        )),
    }
}

/// Create a validator from constraints
pub fn create_validator_from_constraints(
    constraints: &SensorConstraints,
    sensor_type: &str,
) -> Result<Box<dyn edgeguard_core::traits::Validator<Value = f32, Error = edgeguard_core::errors::ValidationError> + Send>, String> {
    use edgeguard_core::validators::{TemperatureValidator, HumidityValidator, PressureValidator};
    
    match sensor_type {
        "temperature" => {
            let min = constraints.typical_min.unwrap_or(-80.0) as f32;
            let max = constraints.typical_max.unwrap_or(125.0) as f32;
            let rate = constraints.max_rate_change.unwrap_or(10.0) as f32;
            Ok(Box::new(TemperatureValidator::new_with_limits(min, max, rate)))
        }
        "humidity" => {
            let min = constraints.typical_min.unwrap_or(0.0) as f32;
            let max = constraints.typical_max.unwrap_or(100.0) as f32;
            let rate = constraints.max_rate_change.unwrap_or(20.0) as f32;
            Ok(Box::new(HumidityValidator::new_with_limits(min, max, rate)))
        }
        "pressure" => {
            // For pressure, we use altitude 0 by default
            Ok(Box::new(PressureValidator::new_with_altitude(0.0)))
        }
        _ => Err(format!("No validator available for sensor type: {}", sensor_type)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn temperature_constraints_valid() {
        let constraints = SensorConstraints::temperature_celsius();
        assert_eq!(constraints.absolute_min, Some(-273.15));
        assert_eq!(constraints.typical_max, Some(125.0));
        assert_eq!(constraints.display_unit, "celsius");
        assert!(!constraints.cross_validation.is_empty());
    }
    
    #[test]
    fn pressure_has_altitude_validation() {
        let constraints = SensorConstraints::pressure_hpa();
        let has_altitude = constraints.cross_validation.iter().any(|rule| {
            matches!(rule.validation_type, CrossValidationType::AltitudePressureConsistency)
        });
        assert!(has_altitude);
    }
    
    #[test]
    fn humidity_range_is_percentage() {
        let constraints = SensorConstraints::humidity_percent();
        assert_eq!(constraints.absolute_min, Some(0.0));
        assert_eq!(constraints.absolute_max, Some(100.0));
    }
    
    #[test]
    fn acoustic_constraints_valid() {
        let constraints = SensorConstraints::acoustic_db_spl();
        assert_eq!(constraints.absolute_min, Some(0.0));
        assert_eq!(constraints.absolute_max, Some(194.0)); // Physical limit in air
        assert_eq!(constraints.typical_max, Some(140.0));  // Pain threshold
        assert_eq!(constraints.display_unit, "dB_SPL");
    }
    
    #[test]
    fn vibration_constraints_symmetric() {
        let constraints = SensorConstraints::vibration_g();
        assert_eq!(constraints.absolute_min, Some(-100.0));
        assert_eq!(constraints.absolute_max, Some(100.0));
        assert_eq!(constraints.display_unit, "g");
        // High rate of change for impacts
        assert_eq!(constraints.max_rate_change, Some(1000.0));
    }
    
    #[test]
    fn emf_constraints_residential_safety() {
        let constraints = SensorConstraints::emf_electric_vm();
        assert_eq!(constraints.typical_max, Some(300.0)); // Residential limit
        assert_eq!(constraints.si_unit, "V/m");
        assert!(constraints.absolute_max.unwrap() > constraints.typical_max.unwrap());
    }
}