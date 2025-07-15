//! EdgeGuard Python Bindings
//! 
//! This module provides Python bindings for EdgeGuard's physics-aware IoT sensor validation library.
//! 
//! ## Features
//! 
//! - Physics-aware validators (Temperature, Humidity, Pressure)
//! - Event-driven processing pipeline
//! - Multi-sensor fusion algorithms
//! - Real-time streaming and aggregation
//! - High-performance native implementation
//! 
//! ## Usage
//! 
//! ```python
//! import edgeguard
//! 
//! # Create a temperature validator
//! validator = edgeguard.TemperatureValidator(min_temp=-20.0, max_temp=60.0)
//! 
//! # Validate a reading
//! result = validator.validate(25.5)
//! print(f"Valid: {result.is_valid}, Value: {result.value}")
//! ```

use pyo3::prelude::*;

mod errors;
mod time;
mod events;
mod validators;
mod conversion;
mod helpers;

use errors::*;
use time::*;
use events::*;
use validators::{ValidationResult, *};
use conversion::*;
use helpers::*;

/// EdgeGuard Python module
/// 
/// Provides Python bindings for physics-aware IoT sensor validation
#[pymodule]
fn edgeguard(_py: Python, m: &PyModule) -> PyResult<()> {
    // Version information
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__doc__", "Physics-aware data validation for IoT edge devices")?;

    // Exception classes
    m.add("EdgeGuardError", _py.get_type::<EdgeGuardError>())?;
    m.add("ValidationError", _py.get_type::<PyValidationError>())?;
    m.add("FusionError", _py.get_type::<PyFusionError>())?;
    m.add("PipelineError", _py.get_type::<PyPipelineError>())?;
    m.add("ConfigurationError", _py.get_type::<PyConfigurationError>())?;

    // Time handling
    m.add_class::<PyTimestamp>()?;
    m.add_class::<TimeUtils>()?;

    // Events and types
    m.add_class::<PySensorType>()?;
    m.add_class::<PyValidationStatus>()?;
    m.add_class::<PySensorReading>()?;
    m.add_class::<PyEventBuilder>()?;

    // Validators (note: PyValidationResult is in validators module)
    m.add_class::<ValidationResult>()?;
    m.add_class::<PyTemperatureValidator>()?;
    m.add_class::<PyHumidityValidator>()?;
    m.add_class::<PyPressureValidator>()?;
    
    // Conversion utilities
    m.add_class::<ConversionUtils>()?;
    
    // Helper functions
    m.add_class::<EdgeGuardHelpers>()?;

    // Create sensor type constants for easier access
    let sensor_type_class = _py.get_type::<PySensorType>();
    m.add("Temperature", PySensorType::Temperature)?;
    m.add("Humidity", PySensorType::Humidity)?;
    m.add("Pressure", PySensorType::Pressure)?;
    m.add("VOC", PySensorType::Voc)?;
    m.add("Particulate", PySensorType::Particulate)?;
    m.add("Acoustic", PySensorType::Acoustic)?;
    m.add("Vibration", PySensorType::Vibration)?;
    m.add("EMF", PySensorType::Emf)?;
    m.add("Custom", PySensorType::Custom)?;

    // Validator constants
    let constants = PyModule::new(_py, "constants")?;
    constants.add("DEFAULT_TEMP_MIN_C", validators::constants::DEFAULT_TEMP_MIN_C)?;
    constants.add("DEFAULT_TEMP_MAX_C", validators::constants::DEFAULT_TEMP_MAX_C)?;
    constants.add("DEFAULT_TEMP_RATE_LIMIT_C_PER_S", validators::constants::DEFAULT_TEMP_RATE_LIMIT_C_PER_S)?;
    constants.add("DEFAULT_HUMIDITY_MIN_PCT", validators::constants::DEFAULT_HUMIDITY_MIN_PCT)?;
    constants.add("DEFAULT_HUMIDITY_MAX_PCT", validators::constants::DEFAULT_HUMIDITY_MAX_PCT)?;
    constants.add("DEFAULT_HUMIDITY_RATE_LIMIT_PCT_PER_S", validators::constants::DEFAULT_HUMIDITY_RATE_LIMIT_PCT_PER_S)?;
    constants.add("DEFAULT_PRESSURE_MIN_HPA", validators::constants::DEFAULT_PRESSURE_MIN_HPA)?;
    constants.add("DEFAULT_PRESSURE_MAX_HPA", validators::constants::DEFAULT_PRESSURE_MAX_HPA)?;
    constants.add("DEFAULT_PRESSURE_RATE_LIMIT_HPA_PER_S", validators::constants::DEFAULT_PRESSURE_RATE_LIMIT_HPA_PER_S)?;
    constants.add("DEFAULT_MIN_QUALITY", validators::constants::DEFAULT_MIN_QUALITY)?;
    constants.add("DEFAULT_GOOD_QUALITY", validators::constants::DEFAULT_GOOD_QUALITY)?;
    m.add_submodule(constants)?;

    // Convenience functions at module level
    m.add_function(wrap_pyfunction!(quick_validate, m)?)?;
    m.add_function(wrap_pyfunction!(create_reading, m)?)?;
    m.add_function(wrap_pyfunction!(get_sensor_ranges, m)?)?;

    Ok(())
}

// Convenience functions exposed at module level for easier access

/// Quick validate a sensor reading with default settings
#[pyfunction]
#[pyo3(signature = (sensor_type, value, quality = None))]
fn quick_validate(
    sensor_type: &PyAny,
    value: f64,
    quality: Option<f64>,
) -> PyValidationResult<ValidationResult> {
    EdgeGuardHelpers::quick_validate(sensor_type, value, quality)
}

/// Create a sensor reading with current timestamp
#[pyfunction]
#[pyo3(signature = (sensor_id, sensor_type, value, quality = None))]
fn create_reading(
    sensor_id: String,
    sensor_type: &PyAny,
    value: f64,
    quality: Option<f64>,
) -> PyValidationResult<PySensorReading> {
    EdgeGuardHelpers::create_reading(sensor_id, sensor_type, value, quality)
}

/// Get typical sensor ranges for all sensor types
#[pyfunction]
fn get_sensor_ranges(py: Python) -> PyResult<PyObject> {
    EdgeGuardHelpers::get_all_sensor_ranges(py)
}