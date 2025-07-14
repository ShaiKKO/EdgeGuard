//! Python Error Handling for EdgeGuard
//!
//! This module provides comprehensive error handling that maps Rust errors to Python exceptions
//! while preserving all error context and maintaining the same precision as the core library.
//!
//! ## Design Philosophy
//!
//! The error handling follows these principles:
//! 1. **Preserve Error Context**: All Rust error information is maintained
//! 2. **Python Native**: Errors map to appropriate Python exception types
//! 3. **Detailed Messages**: Clear, actionable error messages with context
//! 4. **Type Safety**: Strong type checking at the Python-Rust boundary
//!
//! ## Error Mapping Strategy
//!
//! ```text
//! Rust ValidationError → Python ValidationError (custom exception)
//! Rust FusionError     → Python FusionError (custom exception)
//! Rust PipelineError   → Python PipelineError (custom exception)
//! Panic/Internal       → Python RuntimeError
//! ```

use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError, PyTypeError};
use edgeguard_core::{ValidationError as RustValidationError, ValidationResult};

/// Base exception class for all EdgeGuard errors
///
/// This provides a common base for all EdgeGuard-specific exceptions,
/// allowing Python code to catch all EdgeGuard errors with a single except clause.
#[pyclass]
pub struct EdgeGuardError {
    /// Error code for programmatic handling
    #[pyo3(get)]
    pub code: String,
    
    /// Detailed error message
    #[pyo3(get)]
    pub message: String,
    
    /// Additional context information
    #[pyo3(get)]
    pub context: Option<String>,
}

#[pymethods]
impl EdgeGuardError {
    #[new]
    fn new(code: String, message: String, context: Option<String>) -> Self {
        Self { code, message, context }
    }
    
    fn __str__(&self) -> String {
        match &self.context {
            Some(ctx) => format!("[{}] {}: {}", self.code, self.message, ctx),
            None => format!("[{}] {}", self.code, self.message),
        }
    }
    
    fn __repr__(&self) -> String {
        format!(
            "EdgeGuardError(code='{}', message='{}', context={:?})",
            self.code, self.message, self.context
        )
    }
}

/// Validation error for sensor data validation failures
///
/// Raised when sensor data fails physics-aware validation checks.
/// Contains detailed information about what validation rule was violated.
#[pyclass]
pub struct PyValidationError {
    /// Error code for programmatic handling
    #[pyo3(get)]
    pub code: String,
    
    /// Detailed error message
    #[pyo3(get)]
    pub message: String,
    
    /// Additional context information
    #[pyo3(get)]
    pub context: Option<String>,
    
    /// The invalid value that caused the error
    #[pyo3(get)]
    pub value: Option<f64>,
    
    /// Expected range for the value (if applicable)
    #[pyo3(get)]
    pub expected_range: Option<(f64, f64)>,
    
    /// Validation rule that was violated
    #[pyo3(get)]
    pub rule_violated: String,
}

#[pymethods]
impl PyValidationError {
    #[new]
    #[pyo3(signature = (code, message, rule_violated, context = None, value = None, expected_range = None))]
    fn new(
        code: String,
        message: String,
        rule_violated: String,
        context: Option<String>,
        value: Option<f64>,
        expected_range: Option<(f64, f64)>,
    ) -> Self {
        Self {
            code,
            message,
            context,
            value,
            expected_range,
            rule_violated,
        }
    }
    
    fn __str__(&self) -> String {
        let base = format!("ValidationError: {}", self.rule_violated);
        match (self.value, &self.expected_range) {
            (Some(val), Some((min, max))) => {
                format!("{} (value: {}, expected: {}-{})", base, val, min, max)
            }
            (Some(val), None) => format!("{} (value: {})", base, val),
            _ => base,
        }
    }
}

/// Fusion algorithm error for multi-sensor data fusion failures
///
/// Raised when sensor fusion algorithms encounter issues such as
/// numerical instability or insufficient data.
#[pyclass]
pub struct PyFusionError {
    /// Error code for programmatic handling
    #[pyo3(get)]
    pub code: String,
    
    /// Detailed error message
    #[pyo3(get)]
    pub message: String,
    
    /// Additional context information
    #[pyo3(get)]
    pub context: Option<String>,
    
    /// Algorithm that encountered the error
    #[pyo3(get)]
    pub algorithm: String,
    
    /// Number of sensors involved
    #[pyo3(get)]
    pub sensor_count: Option<usize>,
}

#[pymethods]
impl PyFusionError {
    #[new]
    #[pyo3(signature = (code, message, algorithm, context = None, sensor_count = None))]
    fn new(
        code: String,
        message: String,
        algorithm: String,
        context: Option<String>,
        sensor_count: Option<usize>,
    ) -> Self {
        Self {
            code,
            message,
            context,
            algorithm,
            sensor_count,
        }
    }
}

/// Pipeline processing error for event pipeline failures
///
/// Raised when the event processing pipeline encounters errors
/// such as queue overflow or stage processing failures.
#[pyclass]
pub struct PyPipelineError {
    /// Error code for programmatic handling
    #[pyo3(get)]
    pub code: String,
    
    /// Detailed error message
    #[pyo3(get)]
    pub message: String,
    
    /// Additional context information
    #[pyo3(get)]
    pub context: Option<String>,
    
    /// Stage where the error occurred
    #[pyo3(get)]
    pub stage: Option<String>,
    
    /// Event count when error occurred
    #[pyo3(get)]
    pub event_count: Option<usize>,
}

#[pymethods]
impl PyPipelineError {
    #[new]
    #[pyo3(signature = (code, message, context = None, stage = None, event_count = None))]
    fn new(
        code: String,
        message: String,
        context: Option<String>,
        stage: Option<String>,
        event_count: Option<usize>,
    ) -> Self {
        Self {
            code,
            message,
            context,
            stage,
            event_count,
        }
    }
}

/// Configuration error for invalid EdgeGuard configuration
///
/// Raised when EdgeGuard components are configured with invalid parameters
/// or incompatible settings.
#[pyclass]
pub struct PyConfigurationError {
    /// Error code for programmatic handling
    #[pyo3(get)]
    pub code: String,
    
    /// Detailed error message
    #[pyo3(get)]
    pub message: String,
    
    /// Additional context information
    #[pyo3(get)]
    pub context: Option<String>,
    
    /// Configuration parameter that is invalid
    #[pyo3(get)]
    pub parameter: String,
    
    /// Expected value type or range
    #[pyo3(get)]
    pub expected: String,
}

#[pymethods]
impl PyConfigurationError {
    #[new]
    #[pyo3(signature = (code, message, parameter, expected, context = None))]
    fn new(
        code: String,
        message: String,
        parameter: String,
        expected: String,
        context: Option<String>,
    ) -> Self {
        Self {
            code,
            message,
            context,
            parameter,
            expected,
        }
    }
}

/// Error conversion utilities
pub struct ErrorConverter;

impl ErrorConverter {
    /// Convert Rust ValidationError to Python ValidationError
    pub fn validation_error_to_py(
        py: Python,
        error: RustValidationError,
    ) -> PyErr {
        let (code, message, value, expected_range, rule) = match error {
            RustValidationError::InvalidValue => (
                "INVALID_VALUE".to_string(),
                "Invalid sensor value (NaN or infinite)".to_string(),
                None,
                None::<(f64, f64)>,
                "value_validity".to_string(),
            ),
            RustValidationError::OutOfRange { value, min, max } => (
                "OUT_OF_RANGE".to_string(),
                format!("Value {} outside valid range [{}, {}]", value, min, max),
                Some(value as f64),
                Some((min as f64, max as f64)),
                "range_check".to_string(),
            ),
            RustValidationError::RateExceeded { rate, max_rate } => (
                "RATE_EXCEEDED".to_string(),
                format!("Rate of change {} exceeds maximum {}", rate, max_rate),
                Some(rate as f64),
                Some((0.0, max_rate as f64)),
                "rate_of_change".to_string(),
            ),
            RustValidationError::SensorQualityBad { reason } => (
                "SENSOR_QUALITY_BAD".to_string(),
                format!("Sensor quality check failed: {}", reason),
                None,
                None::<(f64, f64)>,
                "quality_check".to_string(),
            ),
            RustValidationError::CrossValidationFailed { reason } => (
                "CROSS_VALIDATION_FAILED".to_string(),
                format!("Cross-validation failed: {}", reason),
                None,
                None::<(f64, f64)>,
                "cross_validation".to_string(),
            ),
            RustValidationError::InsufficientData { required, available } => (
                "INSUFFICIENT_DATA".to_string(),
                format!("Insufficient data: need {} samples, have {}", required, available),
                None,
                None::<(f64, f64)>,
                "data_sufficiency".to_string(),
            ),
        };

        PyErr::new::<PyValidationError, _>((
            code.clone(),
            message.clone(),
            rule,
            None::<String>,
            value,
            expected_range,
        ))
    }

    /// Convert any Result to Python exception
    pub fn result_to_py<T>(
        py: Python,
        result: ValidationResult<T>,
    ) -> PyResult<T> {
        result.map_err(|e| Self::validation_error_to_py(py, e))
    }

    /// Create a configuration error
    pub fn configuration_error(
        parameter: &str,
        expected: &str,
        actual: &str,
    ) -> PyErr {
        PyErr::new::<PyConfigurationError, _>((
            "INVALID_CONFIG".to_string(),
            format!("Invalid configuration parameter '{}': expected {}, got {}", 
                   parameter, expected, actual),
            parameter.to_string(),
            expected.to_string(),
            None::<String>,
        ))
    }

    /// Create a type error for invalid Python types
    pub fn type_error(expected: &str, actual: &str) -> PyErr {
        PyErr::new::<PyTypeError, _>(format!(
            "Expected {}, got {}",
            expected, actual
        ))
    }

    /// Create a runtime error for internal errors
    pub fn runtime_error(message: &str) -> PyErr {
        PyErr::new::<PyRuntimeError, _>(message.to_string())
    }
}

/// Result type alias for Python binding functions
pub type PyValidationResult<T> = PyResult<T>;

/// Helper macro for converting Rust results to Python results
#[macro_export]
macro_rules! rust_to_py {
    ($result:expr, $py:expr) => {
        $crate::errors::ErrorConverter::result_to_py($py, $result)
    };
}

/// Helper macro for creating configuration errors
#[macro_export]
macro_rules! config_error {
    ($param:expr, $expected:expr, $actual:expr) => {
        $crate::errors::ErrorConverter::configuration_error($param, $expected, $actual)
    };
}