//! Type Conversion Utilities for Python Bindings
//!
//! This module provides utilities for seamless conversion between Python and Rust types,
//! with special focus on maintaining precision and handling edge cases appropriately.
//!
//! ## Design Principles
//!
//! 1. **Preserve Precision**: All conversions maintain maximum possible precision
//! 2. **Fail Fast**: Invalid conversions fail immediately with clear error messages
//! 3. **Type Safety**: Strong typing prevents silent data corruption
//! 4. **Performance**: Optimized for high-frequency conversion operations

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use crate::errors::{ErrorConverter, PyValidationResult};
use crate::events::{PySensorType, PySensorReading, PyValidationStatus};
use crate::time::PyTimestamp;

/// Utility functions for converting between Python and Rust types
#[pyclass]
pub struct ConversionUtils;

#[pymethods]
impl ConversionUtils {
    /// Convert Python list to Vec<f64> with validation
    ///
    /// Args:
    ///     py_list: Python list of numbers
    ///     max_length: Maximum allowed length (optional)
    ///
    /// Returns:
    ///     Vec<f64> with validated numeric values
    ///
    /// Raises:
    ///     ConfigurationError: If list is too long or contains invalid values
    #[staticmethod]
    fn list_to_vec_f64(py_list: &PyList, max_length: Option<usize>) -> PyValidationResult<Vec<f64>> {
        let length = py_list.len();
        
        if let Some(max_len) = max_length {
            if length > max_len {
                return Err(ErrorConverter::configuration_error(
                    "list_length",
                    &format!("max {} items", max_len),
                    &format!("{} items", length),
                ));
            }
        }
        
        let mut result = Vec::with_capacity(length);
        
        for (i, item) in py_list.iter().enumerate() {
            let value: f64 = item.extract()
                .map_err(|_| ErrorConverter::configuration_error(
                    &format!("list[{}]", i),
                    "numeric value",
                    "non-numeric type",
                ))?;
                
            if !value.is_finite() {
                return Err(ErrorConverter::configuration_error(
                    &format!("list[{}]", i),
                    "finite number",
                    if value.is_nan() { "NaN" } else { "infinite" },
                ));
            }
            
            result.push(value);
        }
        
        Ok(result)
    }
    
    /// Convert Vec<f64> to Python list
    ///
    /// Args:
    ///     py: Python interpreter
    ///     vec: Vector of f64 values
    ///
    /// Returns:
    ///     Python list containing the values
    #[staticmethod]
    fn vec_f64_to_list(py: Python, vec: Vec<f64>) -> PyResult<PyObject> {
        let py_list = PyList::new(py, vec);
        Ok(py_list.into())
    }
    
    /// Convert Python dict to sensor reading with validation
    ///
    /// Args:
    ///     py: Python interpreter
    ///     data: Dictionary with sensor reading data
    ///
    /// Returns:
    ///     PySensorReading instance
    ///
    /// Raises:
    ///     ConfigurationError: If required fields are missing or invalid
    #[staticmethod]
    fn dict_to_sensor_reading(py: Python, data: &PyDict) -> PyValidationResult<PySensorReading> {
        PySensorReading::from_dict(py, data)
    }
    
    /// Convert multiple sensor readings to list of dicts
    ///
    /// Args:
    ///     py: Python interpreter
    ///     readings: List of sensor readings
    ///
    /// Returns:
    ///     Python list of dictionaries
    #[staticmethod]
    fn sensor_readings_to_dicts(py: Python, readings: Vec<PySensorReading>) -> PyResult<PyObject> {
        let mut dict_list = Vec::new();
        
        for reading in readings {
            dict_list.push(reading.to_dict(py)?);
        }
        
        let py_list = PyList::new(py, dict_list);
        Ok(py_list.into())
    }
    
    /// Parse sensor type from string with fuzzy matching
    ///
    /// Args:
    ///     type_str: String representation of sensor type
    ///
    /// Returns:
    ///     PySensorType enum value
    ///
    /// Raises:
    ///     ConfigurationError: If sensor type cannot be determined
    #[staticmethod]
    pub fn parse_sensor_type(type_str: &str) -> PyValidationResult<PySensorType> {
        let normalized = type_str.to_lowercase().trim().to_string();
        
        let sensor_type = match normalized.as_str() {
            "temperature" | "temp" | "t" => PySensorType::Temperature,
            "humidity" | "hum" | "h" | "rh" => PySensorType::Humidity,
            "pressure" | "press" | "p" | "hpa" | "bar" => PySensorType::Pressure,
            "voc" | "volatile" | "gas" => PySensorType::Voc,
            "particulate" | "pm" | "pm2.5" | "pm10" | "dust" => PySensorType::Particulate,
            "acoustic" | "sound" | "noise" | "db" => PySensorType::Acoustic,
            "vibration" | "vib" | "accelerometer" | "accel" => PySensorType::Vibration,
            "emf" | "electromagnetic" | "magnetic" => PySensorType::Emf,
            "custom" => PySensorType::Custom,
            _ => return Err(ErrorConverter::configuration_error(
                "sensor_type",
                "valid sensor type (temperature, humidity, pressure, etc.)",
                &type_str,
            )),
        };
        
        Ok(sensor_type)
    }
    
    /// Convert timestamp to Python datetime with timezone
    ///
    /// Args:
    ///     py: Python interpreter
    ///     timestamp: PyTimestamp instance
    ///     timezone: Timezone name (optional, defaults to UTC)
    ///
    /// Returns:
    ///     Python datetime object
    #[staticmethod]
    fn timestamp_to_datetime_tz(
        py: Python, 
        timestamp: &PyTimestamp,
        timezone: Option<&str>
    ) -> PyResult<PyObject> {
        if timezone.is_some() && timezone != Some("UTC") {
            // For non-UTC timezones, we'd need pytz or zoneinfo
            // For now, just convert to UTC and let Python handle timezone conversion
            return Err(ErrorConverter::configuration_error(
                "timezone",
                "UTC (other timezones not yet supported)",
                timezone.unwrap_or("None"),
            ).into());
        }
        
        timestamp.to_datetime(py)
    }
    
    /// Validate and convert quality value
    ///
    /// Args:
    ///     quality: Quality value (0.0-1.0 or percentage 0-100)
    ///
    /// Returns:
    ///     Normalized quality value (0.0-1.0)
    ///
    /// Raises:
    ///     ConfigurationError: If quality value is invalid
    #[staticmethod]
    fn normalize_quality(quality: f64) -> PyValidationResult<f64> {
        if !quality.is_finite() {
            return Err(ErrorConverter::configuration_error(
                "quality",
                "finite number",
                if quality.is_nan() { "NaN" } else { "infinite" },
            ));
        }
        
        // Handle percentage format (0-100) by normalizing to 0.0-1.0
        let normalized = if quality > 1.0 && quality <= 100.0 {
            quality / 100.0
        } else if quality >= 0.0 && quality <= 1.0 {
            quality
        } else {
            return Err(ErrorConverter::configuration_error(
                "quality",
                "value between 0.0-1.0 or 0-100",
                &format!("{}", quality),
            ));
        };
        
        Ok(normalized)
    }
    
    /// Convert validation status to descriptive dictionary
    ///
    /// Args:
    ///     py: Python interpreter
    ///     status: Validation status enum
    ///
    /// Returns:
    ///     Dictionary with status details
    #[staticmethod]
    fn validation_status_to_dict(py: Python, status: &PyValidationStatus) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("name", status.name())?;
        dict.set_item("description", status.description())?;
        dict.set_item("is_valid", status.is_valid())?;
        dict.set_item("is_critical", status.is_critical())?;
        dict.set_item("code", status.clone() as u8)?;
        Ok(dict.into())
    }
    
    /// Batch convert multiple values with error tracking
    ///
    /// Args:
    ///     py: Python interpreter
    ///     values: List of values to convert
    ///     converter_name: Name of conversion function for error reporting
    ///
    /// Returns:
    ///     Tuple of (successful_conversions, failed_indices, error_messages)
    #[staticmethod]
    fn batch_convert_f64(
        py: Python,
        values: &PyList,
        converter_name: &str
    ) -> PyResult<(Vec<f64>, Vec<usize>, Vec<String>)> {
        let mut successful = Vec::new();
        let mut failed_indices = Vec::new();
        let mut error_messages = Vec::new();
        
        for (i, item) in values.iter().enumerate() {
            match item.extract::<f64>() {
                Ok(value) if value.is_finite() => {
                    successful.push(value);
                }
                Ok(value) => {
                    failed_indices.push(i);
                    error_messages.push(format!(
                        "{}: index {}: non-finite value ({})",
                        converter_name, i,
                        if value.is_nan() { "NaN" } else { "infinite" }
                    ));
                }
                Err(e) => {
                    failed_indices.push(i);
                    error_messages.push(format!(
                        "{}: index {}: conversion error: {}",
                        converter_name, i, e
                    ));
                }
            }
        }
        
        Ok((successful, failed_indices, error_messages))
    }
}

/// Helper functions for common conversion patterns
impl ConversionUtils {
    /// Internal helper: extract f64 with validation
    pub(crate) fn extract_finite_f64(obj: &PyAny, field_name: &str) -> PyValidationResult<f64> {
        let value: f64 = obj.extract()
            .map_err(|_| ErrorConverter::configuration_error(
                field_name,
                "numeric value",
                "non-numeric type",
            ))?;
            
        if !value.is_finite() {
            return Err(ErrorConverter::configuration_error(
                field_name,
                "finite number",
                if value.is_nan() { "NaN" } else { "infinite" },
            ));
        }
        
        Ok(value)
    }
    
    /// Internal helper: extract optional f64 with validation
    pub(crate) fn extract_optional_finite_f64(
        dict: &PyDict, 
        key: &str, 
        default: f64
    ) -> PyValidationResult<f64> {
        match dict.get_item(key)? {
            Some(obj) => Self::extract_finite_f64(obj, key),
            None => Ok(default),
        }
    }
    
    /// Internal helper: extract string with length validation
    pub(crate) fn extract_string_with_length(
        obj: &PyAny, 
        field_name: &str, 
        max_length: usize
    ) -> PyValidationResult<String> {
        let value: String = obj.extract()
            .map_err(|_| ErrorConverter::configuration_error(
                field_name,
                "string",
                "non-string type",
            ))?;
            
        if value.len() > max_length {
            return Err(ErrorConverter::configuration_error(
                field_name,
                &format!("string with max {} characters", max_length),
                &format!("{} characters", value.len()),
            ));
        }
        
        if value.is_empty() {
            return Err(ErrorConverter::configuration_error(
                field_name,
                "non-empty string",
                "empty string",
            ));
        }
        
        Ok(value)
    }
}