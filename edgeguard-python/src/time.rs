//! Time Handling for Python Bindings
//!
//! This module provides Python-friendly time abstractions that maintain compatibility
//! with EdgeGuard's precise timestamp requirements while offering Pythonic interfaces.
//!
//! ## Design Principles
//!
//! 1. **Precision Preservation**: Maintains millisecond precision from Rust
//! 2. **Python Integration**: Compatible with datetime, time, and numpy
//! 3. **Timezone Awareness**: Proper UTC handling and timezone conversion
//! 4. **Performance**: Minimal overhead for high-frequency operations
//!
//! ## Time Representation
//!
//! EdgeGuard internally uses milliseconds since Unix epoch (u64) for:
//! - Consistent precision across platforms
//! - Efficient comparison and arithmetic
//! - Lock-free atomic operations
//! - Embedded system compatibility
//!
//! Python bindings convert between:
//! - Rust Timestamp (u64 milliseconds) ↔ Python datetime objects
//! - Rust Timestamp (u64 milliseconds) ↔ Python float (seconds)
//! - Rust Timestamp (u64 milliseconds) ↔ NumPy datetime64[ms]

use pyo3::prelude::*;
use pyo3::types::PyFloat;
use edgeguard_core::time::Timestamp;
use crate::errors::{ErrorConverter, PyValidationResult};

/// Timestamp constants for validation and conversion
pub mod constants {
    /// Minimum valid timestamp (January 1, 2000 UTC)
    /// Used to detect obviously invalid timestamps
    pub const MIN_VALID_TIMESTAMP_MS: u64 = 946_684_800_000;
    
    /// Maximum valid timestamp (January 1, 2100 UTC)
    /// Used to detect timestamps that are likely errors
    pub const MAX_VALID_TIMESTAMP_MS: u64 = 4_102_444_800_000;
    
    /// Milliseconds per second
    pub const MS_PER_SECOND: u64 = 1_000;
    
    /// Milliseconds per minute
    pub const MS_PER_MINUTE: u64 = 60_000;
    
    /// Milliseconds per hour
    pub const MS_PER_HOUR: u64 = 3_600_000;
    
    /// Milliseconds per day
    pub const MS_PER_DAY: u64 = 86_400_000;
    
    /// Default timeout for time operations (5 seconds)
    pub const DEFAULT_TIMEOUT_MS: u64 = 5_000;
}

/// Python-compatible timestamp wrapper
///
/// Provides a bridge between EdgeGuard's internal timestamp representation
/// and Python's datetime ecosystem. Maintains millisecond precision while
/// offering convenient conversion methods.
///
/// # Examples
///
/// ```python
/// import edgeguard
/// from datetime import datetime, timezone
///
/// # Create from current time
/// ts = edgeguard.Timestamp.now()
/// 
/// # Create from datetime
/// dt = datetime.now(timezone.utc)
/// ts = edgeguard.Timestamp.from_datetime(dt)
/// 
/// # Convert back to datetime
/// dt_back = ts.to_datetime()
/// 
/// # Get as float seconds
/// seconds = ts.to_seconds()
/// ```
#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct PyTimestamp {
    /// Internal timestamp in milliseconds since Unix epoch
    pub(crate) inner: Timestamp,
}

#[pymethods]
impl PyTimestamp {
    /// Create a new timestamp from milliseconds since Unix epoch
    ///
    /// Args:
    ///     milliseconds: Milliseconds since January 1, 1970 UTC
    ///
    /// Returns:
    ///     New PyTimestamp instance
    ///
    /// Raises:
    ///     ValueError: If timestamp is outside valid range (2000-2100)
    #[new]
    fn new(milliseconds: u64) -> PyValidationResult<Self> {
        if milliseconds < constants::MIN_VALID_TIMESTAMP_MS {
            return Err(ErrorConverter::configuration_error(
                "milliseconds",
                "timestamp after 2000-01-01",
                &format!("{} (before year 2000)", milliseconds),
            ));
        }
        
        if milliseconds > constants::MAX_VALID_TIMESTAMP_MS {
            return Err(ErrorConverter::configuration_error(
                "milliseconds",
                "timestamp before 2100-01-01", 
                &format!("{} (after year 2100)", milliseconds),
            ));
        }
        
        Ok(Self {
            inner: milliseconds,
        })
    }

    /// Create timestamp representing current time
    ///
    /// Returns:
    ///     PyTimestamp with current UTC time
    #[staticmethod]
    pub fn now() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        
        Self {
            inner: duration.as_millis() as u64,
        }
    }

    /// Create timestamp from Python datetime object
    ///
    /// Automatically handles timezone conversion to UTC. If the datetime
    /// is naive (no timezone), it's assumed to be UTC.
    ///
    /// Args:
    ///     dt: Python datetime object
    ///
    /// Returns:
    ///     PyTimestamp representing the datetime in UTC
    ///
    /// Raises:
    ///     ValueError: If datetime cannot be converted or is out of range
    #[staticmethod]
    pub fn from_datetime(py: Python, dt: &PyAny) -> PyValidationResult<Self> {
        // Convert Python datetime to timestamp
        let timestamp_seconds: f64 = dt.call_method0("timestamp")?.extract()?;
        let milliseconds = (timestamp_seconds * 1000.0) as u64;
        
        Self::new(milliseconds)
    }

    /// Create timestamp from seconds since Unix epoch
    ///
    /// Args:
    ///     seconds: Seconds since January 1, 1970 UTC (can be fractional)
    ///
    /// Returns:
    ///     PyTimestamp with millisecond precision
    ///
    /// Raises:
    ///     ValueError: If timestamp is out of valid range
    #[staticmethod]
    fn from_seconds(seconds: f64) -> PyValidationResult<Self> {
        if !seconds.is_finite() {
            return Err(ErrorConverter::configuration_error(
                "seconds",
                "finite number",
                "NaN or infinite",
            ));
        }
        
        let milliseconds = (seconds * 1000.0) as u64;
        Self::new(milliseconds)
    }

    /// Get milliseconds since Unix epoch
    ///
    /// Returns:
    ///     Milliseconds as integer
    #[getter]
    fn milliseconds(&self) -> u64 {
        self.inner
    }

    /// Convert to seconds since Unix epoch
    ///
    /// Returns:
    ///     Seconds as float with millisecond precision
    fn to_seconds(&self) -> f64 {
        self.inner as f64 / 1000.0
    }

    /// Convert to Python datetime object (UTC)
    ///
    /// Returns:
    ///     datetime.datetime object in UTC timezone
    pub fn to_datetime(&self, py: Python) -> PyResult<PyObject> {
        let datetime_module = py.import("datetime")?;
        let timezone_class = datetime_module.getattr("timezone")?;
        let utc = timezone_class.getattr("utc")?;
        
        let timestamp_seconds = self.to_seconds();
        let datetime_class = datetime_module.getattr("datetime")?;
        
        datetime_class.call_method1(
            "fromtimestamp",
            (timestamp_seconds, utc)
        ).map(|dt| dt.into())
    }

    /// Add milliseconds to this timestamp
    ///
    /// Args:
    ///     milliseconds: Milliseconds to add (can be negative)
    ///
    /// Returns:
    ///     New PyTimestamp with added time
    ///
    /// Raises:
    ///     ValueError: If result would be out of valid range
    fn add_milliseconds(&self, milliseconds: i64) -> PyValidationResult<Self> {
        let new_ms = if milliseconds >= 0 {
            self.inner.saturating_add(milliseconds as u64)
        } else {
            self.inner.saturating_sub((-milliseconds) as u64)
        };
        
        Self::new(new_ms)
    }

    /// Add seconds to this timestamp
    ///
    /// Args:
    ///     seconds: Seconds to add (can be fractional and negative)
    ///
    /// Returns:
    ///     New PyTimestamp with added time
    fn add_seconds(&self, seconds: f64) -> PyValidationResult<Self> {
        let milliseconds = (seconds * 1000.0) as i64;
        self.add_milliseconds(milliseconds)
    }

    /// Calculate duration since another timestamp
    ///
    /// Args:
    ///     other: Earlier timestamp to compare against
    ///
    /// Returns:
    ///     Duration in milliseconds (positive if self is later)
    fn duration_since(&self, other: &PyTimestamp) -> i64 {
        self.inner as i64 - other.inner as i64
    }

    /// Calculate duration until another timestamp
    ///
    /// Args:
    ///     other: Later timestamp to compare against
    ///
    /// Returns:
    ///     Duration in milliseconds (positive if other is later)
    fn duration_until(&self, other: &PyTimestamp) -> i64 {
        other.inner as i64 - self.inner as i64
    }

    /// Check if this timestamp is before another
    ///
    /// Args:
    ///     other: Timestamp to compare against
    ///
    /// Returns:
    ///     True if this timestamp is earlier
    fn is_before(&self, other: &PyTimestamp) -> bool {
        self.inner < other.inner
    }

    /// Check if this timestamp is after another
    ///
    /// Args:
    ///     other: Timestamp to compare against
    ///
    /// Returns:
    ///     True if this timestamp is later
    fn is_after(&self, other: &PyTimestamp) -> bool {
        self.inner > other.inner
    }

    /// Format timestamp as ISO 8601 string
    ///
    /// Returns:
    ///     ISO 8601 formatted string (e.g., "2023-12-25T10:30:45.123Z")
    fn to_iso_string(&self, py: Python) -> PyResult<String> {
        let dt = self.to_datetime(py)?;
        let iso_string: String = dt.call_method0(py, "isoformat")?.extract(py)?;
        Ok(iso_string)
    }

    // Python magic methods for natural comparison and arithmetic
    
    fn __eq__(&self, other: &PyTimestamp) -> bool {
        self.inner == other.inner
    }
    
    fn __ne__(&self, other: &PyTimestamp) -> bool {
        self.inner != other.inner
    }
    
    fn __lt__(&self, other: &PyTimestamp) -> bool {
        self.inner < other.inner
    }
    
    fn __le__(&self, other: &PyTimestamp) -> bool {
        self.inner <= other.inner
    }
    
    fn __gt__(&self, other: &PyTimestamp) -> bool {
        self.inner > other.inner
    }
    
    fn __ge__(&self, other: &PyTimestamp) -> bool {
        self.inner >= other.inner
    }
    
    fn __sub__(&self, other: &PyTimestamp) -> i64 {
        self.duration_since(other)
    }
    
    fn __str__(&self) -> PyResult<String> {
        Python::with_gil(|py| self.to_iso_string(py))
    }
    
    fn __repr__(&self) -> String {
        format!("Timestamp({}ms)", self.inner)
    }
    
    fn __hash__(&self) -> u64 {
        self.inner
    }
}

impl PyTimestamp {
    /// Create from internal Rust timestamp
    pub(crate) fn from_rust(timestamp: Timestamp) -> Self {
        Self { inner: timestamp }
    }
    
    /// Convert to internal Rust timestamp
    pub(crate) fn to_rust(&self) -> Timestamp {
        self.inner
    }
}

/// Time utilities for working with timestamps and durations
#[pyclass]
pub struct TimeUtils;

#[pymethods]
impl TimeUtils {
    /// Parse ISO 8601 timestamp string
    ///
    /// Args:
    ///     iso_string: ISO 8601 formatted timestamp
    ///
    /// Returns:
    ///     PyTimestamp parsed from string
    ///
    /// Raises:
    ///     ValueError: If string cannot be parsed
    #[staticmethod]
    fn parse_iso(py: Python, iso_string: &str) -> PyResult<PyTimestamp> {
        let datetime_module = py.import("datetime")?;
        let datetime_class = datetime_module.getattr("datetime")?;
        
        let dt = datetime_class.call_method1("fromisoformat", (iso_string,))?;
        PyTimestamp::from_datetime(py, dt)
    }

    /// Get current Unix timestamp in milliseconds
    ///
    /// Returns:
    ///     Current time as milliseconds since Unix epoch
    #[staticmethod]
    fn now_millis() -> u64 {
        PyTimestamp::now().inner
    }

    /// Convert duration to human-readable string
    ///
    /// Args:
    ///     duration_ms: Duration in milliseconds
    ///
    /// Returns:
    ///     Human-readable duration string (e.g., "2h 30m 15s")
    #[staticmethod]
    fn format_duration(duration_ms: u64) -> String {
        if duration_ms < constants::MS_PER_SECOND {
            return format!("{}ms", duration_ms);
        }
        
        let seconds = duration_ms / constants::MS_PER_SECOND;
        let minutes = seconds / 60;
        let hours = minutes / 60;
        let days = hours / 24;
        
        if days > 0 {
            format!("{}d {}h {}m", days, hours % 24, minutes % 60)
        } else if hours > 0 {
            format!("{}h {}m {}s", hours, minutes % 60, seconds % 60)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, seconds % 60)
        } else {
            format!("{}s", seconds)
        }
    }

    /// Validate timestamp is within reasonable bounds
    ///
    /// Args:
    ///     timestamp: Timestamp to validate
    ///
    /// Returns:
    ///     True if timestamp appears valid
    #[staticmethod]
    fn is_valid_timestamp(timestamp: &PyTimestamp) -> bool {
        timestamp.inner >= constants::MIN_VALID_TIMESTAMP_MS &&
        timestamp.inner <= constants::MAX_VALID_TIMESTAMP_MS
    }
}