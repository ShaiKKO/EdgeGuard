//! Helper Functions for EdgeGuard Python API
//!
//! This module provides high-level helper functions that make common EdgeGuard
//! operations more convenient for Python users. These functions combine multiple
//! low-level operations into single, easy-to-use interfaces.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use crate::errors::{ErrorConverter, PyValidationResult};
use crate::events::{PySensorType, PySensorReading};
use crate::validators::{ValidationResult, PyTemperatureValidator, PyHumidityValidator, PyPressureValidator};
use crate::time::PyTimestamp;
use crate::conversion::ConversionUtils;

/// High-level helper functions for common EdgeGuard operations
#[pyclass]
pub struct EdgeGuardHelpers;

#[pymethods]
impl EdgeGuardHelpers {
    /// Quick validate a single sensor reading with automatic validator creation
    ///
    /// Args:
    ///     sensor_type: Type of sensor (string or PySensorType)
    ///     value: Sensor reading value
    ///     quality: Quality indicator 0.0-1.0 (optional, defaults to 1.0)
    ///
    /// Returns:
    ///     ValidationResult with validation outcome
    ///
    /// # Examples
    ///
    /// ```python
    /// import edgeguard
    /// 
    /// # Quick validation with defaults
    /// result = edgeguard.quick_validate("temperature", 25.0)
    /// print(f"Valid: {result.is_valid}")
    /// 
    /// # With quality indicator
    /// result = edgeguard.quick_validate("humidity", 65.0, quality=0.95)
    /// ```
    #[staticmethod]
    #[pyo3(signature = (sensor_type, value, quality = None))]
    pub fn quick_validate(
        sensor_type: &PyAny,
        value: f64,
        quality: Option<f64>,
    ) -> PyValidationResult<ValidationResult> {
        let quality = quality.unwrap_or(1.0);
        
        // Parse sensor type (handle both string and enum)
        let sensor_type = if let Ok(type_str) = sensor_type.extract::<String>() {
            ConversionUtils::parse_sensor_type(&type_str)?
        } else if let Ok(sensor_enum) = sensor_type.extract::<PySensorType>() {
            sensor_enum
        } else {
            return Err(ErrorConverter::configuration_error(
                "sensor_type",
                "string or PySensorType enum",
                "unknown type",
            ));
        };
        
        // Use the helper function for validation
        Self::validate_with_sensor_type(sensor_type, value, Some(quality))
    }
    
    /// Batch validate multiple readings with automatic validator selection
    ///
    /// Args:
    ///     readings: List of sensor readings (dicts or PySensorReading objects)
    ///
    /// Returns:
    ///     List of ValidationResult objects
    ///
    /// # Examples
    ///
    /// ```python
    /// import edgeguard
    /// 
    /// readings = [
    ///     {"sensor_type": "temperature", "value": 25.0, "quality": 0.95},
    ///     {"sensor_type": "humidity", "value": 65.0, "quality": 0.90},
    /// ]
    /// 
    /// results = edgeguard.batch_validate(readings)
    /// for result in results:
    ///     print(f"Valid: {result.is_valid}, Value: {result.value}")
    /// ```
    #[staticmethod]
    pub fn batch_validate(py: Python, readings: &PyList) -> PyValidationResult<Vec<ValidationResult>> {
        let mut results = Vec::new();
        
        for (i, item) in readings.iter().enumerate() {
            let result = if let Ok(reading) = item.extract::<PySensorReading>() {
                // Already a PySensorReading object - validate directly
                Self::validate_with_sensor_type(reading.sensor_type, reading.value, Some(reading.quality))?
            } else if let Ok(dict) = item.downcast::<PyDict>() {
                // Dictionary format
                let sensor_type_str: String = dict.get_item("sensor_type")?
                    .ok_or_else(|| ErrorConverter::configuration_error(
                        &format!("readings[{}]", i),
                        "sensor_type field",
                        "missing",
                    ))?
                    .extract()?;
                    
                let value: f64 = dict.get_item("value")?
                    .ok_or_else(|| ErrorConverter::configuration_error(
                        &format!("readings[{}]", i),
                        "value field",
                        "missing",
                    ))?
                    .extract()?;
                    
                let quality: f64 = if let Some(q) = dict.get_item("quality")? {
                    q.extract()?
                } else {
                    1.0
                };
                
                let sensor_type = ConversionUtils::parse_sensor_type(&sensor_type_str)?;
                Self::validate_with_sensor_type(sensor_type, value, Some(quality))?
            } else {
                return Err(ErrorConverter::configuration_error(
                    &format!("readings[{}]", i),
                    "PySensorReading or dict",
                    "unknown type",
                ));
            };
            
            results.push(result);
        }
        
        Ok(results)
    }
    
    /// Create a sensor reading with automatic timestamp
    ///
    /// Args:
    ///     sensor_id: Unique sensor identifier
    ///     sensor_type: Type of sensor (string or PySensorType)
    ///     value: Measured value
    ///     quality: Quality indicator 0.0-1.0 (optional, defaults to 1.0)
    ///
    /// Returns:
    ///     PySensorReading with current timestamp
    #[staticmethod]
    #[pyo3(signature = (sensor_id, sensor_type, value, quality = None))]
    pub fn create_reading(
        sensor_id: String,
        sensor_type: &PyAny,
        value: f64,
        quality: Option<f64>,
    ) -> PyValidationResult<PySensorReading> {
        let quality = quality.unwrap_or(1.0);
        
        // Parse sensor type
        let sensor_type = if let Ok(type_str) = sensor_type.extract::<String>() {
            ConversionUtils::parse_sensor_type(&type_str)?
        } else if let Ok(sensor_enum) = sensor_type.extract::<PySensorType>() {
            sensor_enum
        } else {
            return Err(ErrorConverter::configuration_error(
                "sensor_type",
                "string or PySensorType enum",
                "unknown type",
            ));
        };
        
        let timestamp = PyTimestamp::now();
        
        PySensorReading::new(sensor_id, sensor_type, value, timestamp, quality)
    }
    
    /// Validate reading and return summary dict
    ///
    /// Args:
    ///     sensor_type: Type of sensor
    ///     value: Sensor reading value
    ///     quality: Quality indicator (optional)
    ///
    /// Returns:
    ///     Dictionary with validation summary
    #[staticmethod]
    #[pyo3(signature = (sensor_type, value, quality = None))]
    fn validate_and_summarize(
        py: Python,
        sensor_type: &PyAny,
        value: f64,
        quality: Option<f64>,
    ) -> PyValidationResult<PyObject> {
        let result = Self::quick_validate(sensor_type, value, quality)?;
        
        let dict = PyDict::new(py);
        dict.set_item("value", result.value)?;
        dict.set_item("is_valid", result.is_valid)?;
        dict.set_item("status", result.status.name())?;
        dict.set_item("message", &result.message)?;
        dict.set_item("confidence", result.confidence)?;
        dict.set_item("corrections_count", result.corrections_applied.len())?;
        dict.set_item("corrections", &result.corrections_applied)?;
        
        Ok(dict.into())
    }
    
    /// Get typical ranges for all sensor types
    ///
    /// Returns:
    ///     Dictionary mapping sensor type names to their typical ranges
    #[staticmethod]
    pub fn get_all_sensor_ranges(py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        
        let sensor_types = [
            PySensorType::Temperature,
            PySensorType::Humidity,
            PySensorType::Pressure,
            PySensorType::Voc,
            PySensorType::Particulate,
            PySensorType::Acoustic,
            PySensorType::Vibration,
            PySensorType::Emf,
        ];
        
        for sensor_type in &sensor_types {
            let (min, max) = sensor_type.typical_range();
            let range_dict = PyDict::new(py);
            range_dict.set_item("min", min)?;
            range_dict.set_item("max", max)?;
            range_dict.set_item("unit", sensor_type.unit())?;
            
            dict.set_item(sensor_type.name(), range_dict)?;
        }
        
        Ok(dict.into())
    }
    
    /// Check if a value is within typical range for sensor type
    ///
    /// Args:
    ///     sensor_type: Type of sensor
    ///     value: Value to check
    ///
    /// Returns:
    ///     True if value is within typical range
    #[staticmethod]
    fn is_value_in_typical_range(sensor_type: &PyAny, value: f64) -> PyValidationResult<bool> {
        // Parse sensor type
        let sensor_type = if let Ok(type_str) = sensor_type.extract::<String>() {
            ConversionUtils::parse_sensor_type(&type_str)?
        } else if let Ok(sensor_enum) = sensor_type.extract::<PySensorType>() {
            sensor_enum
        } else {
            return Err(ErrorConverter::configuration_error(
                "sensor_type",
                "string or PySensorType enum",
                "unknown type",
            ));
        };
        
        let (min, max) = sensor_type.typical_range();
        Ok(value >= min && value <= max)
    }
}

impl EdgeGuardHelpers {
    /// Internal helper: validate with sensor type enum directly
    fn validate_with_sensor_type(
        sensor_type: PySensorType,
        value: f64,
        quality: Option<f64>,
    ) -> PyValidationResult<ValidationResult> {
        let quality = quality.unwrap_or(1.0);
        
        // Create appropriate validator and validate
        match sensor_type {
            PySensorType::Temperature => {
                let validator = PyTemperatureValidator::new(
                    crate::validators::constants::DEFAULT_TEMP_MIN_C,
                    crate::validators::constants::DEFAULT_TEMP_MAX_C,
                    crate::validators::constants::DEFAULT_TEMP_RATE_LIMIT_C_PER_S,
                    crate::validators::constants::DEFAULT_THERMAL_MASS_KG,
                    None,
                )?;
                validator.validate(value, None, Some(quality))
            }
            PySensorType::Humidity => {
                let validator = PyHumidityValidator::new(
                    crate::validators::constants::DEFAULT_HUMIDITY_MIN_PCT,
                    crate::validators::constants::DEFAULT_HUMIDITY_MAX_PCT,
                    crate::validators::constants::DEFAULT_HUMIDITY_RATE_LIMIT_PCT_PER_S,
                    None,
                )?;
                validator.validate(value, None, Some(quality))
            }
            PySensorType::Pressure => {
                let validator = PyPressureValidator::new(
                    crate::validators::constants::DEFAULT_PRESSURE_MIN_HPA,
                    crate::validators::constants::DEFAULT_PRESSURE_MAX_HPA,
                    crate::validators::constants::DEFAULT_PRESSURE_RATE_LIMIT_HPA_PER_S,
                    crate::validators::constants::DEFAULT_ALTITUDE_M,
                )?;
                validator.validate(value, None, Some(quality))
            }
            _ => Err(ErrorConverter::configuration_error(
                "sensor_type",
                "Temperature, Humidity, or Pressure",
                sensor_type.name(),
            )),
        }
    }
}