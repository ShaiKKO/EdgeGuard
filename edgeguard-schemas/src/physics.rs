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
}

/// Extract constraints from schema custom properties
/// 
/// Looks for "edgeguard.constraints" property in Avro schema
pub fn extract_from_schema(_schema: &apache_avro::Schema) -> Option<SensorConstraints> {
    // This would parse custom properties from the schema
    // For now, returning None as it requires schema traversal
    None
}

/// Embed constraints into schema as custom property
pub fn embed_in_schema(
    _schema: &mut apache_avro::Schema,
    _constraints: &SensorConstraints,
) -> Result<(), crate::SchemaError> {
    // This would add constraints as custom properties
    // Implementation depends on apache-avro API for custom properties
    Ok(())
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
}