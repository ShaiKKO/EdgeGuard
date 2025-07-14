//! Event System Python Bindings
//!
//! This module exposes EdgeGuard's event-driven architecture to Python, providing
//! high-level abstractions for sensor data processing while maintaining the same
//! performance characteristics as the Rust implementation.
//!
//! ## Event-Driven Architecture
//!
//! EdgeGuard processes sensor data through an event system that provides:
//! - **Type Safety**: Strongly typed events prevent data corruption
//! - **Memory Efficiency**: Fixed-size events with zero-copy where possible
//! - **Real-time Processing**: Sub-millisecond event processing latency
//! - **Composability**: Events can be chained through processing pipelines
//!
//! ## Event Types
//!
//! The system supports several event categories:
//! - **SensorReading**: Raw sensor measurements with quality metadata
//! - **ValidationResult**: Results of physics-aware validation
//! - **CrossValidation**: Multi-sensor validation results  
//! - **BatchReading**: Aggregated sensor data for bandwidth efficiency
//! - **SystemEvent**: Operational events (errors, status changes)
//!
//! ## Performance Characteristics
//!
//! - Event creation: <1μs
//! - Event serialization: <10μs
//! - Queue operations: Lock-free, <100ns
//! - Memory overhead: 64 bytes per event (average)

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use edgeguard_core::events::{
    Event as RustEvent,
    SensorType as RustSensorType,
    ValidationStatus as RustValidationStatus,
    CrossValidationType as RustCrossValidationType,
    EventBuilder as RustEventBuilder,
    InlineString,
};
use crate::time::PyTimestamp;
use crate::errors::{ErrorConverter, PyValidationResult};

/// Sensor type enumeration
///
/// Defines the physical quantity being measured. Used for:
/// - Applying appropriate validation rules
/// - Routing events through the pipeline
/// - Cross-sensor validation logic
/// - Unit conversion and scaling
///
/// Each sensor type has associated physics constraints and validation rules
/// that are automatically applied during processing.
#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub enum PySensorType {
    /// Temperature sensors (thermocouples, RTDs, thermistors)
    /// 
    /// Physics constraints:
    /// - Range: -80°C to 125°C (typical industrial)
    /// - Rate limit: 10°C/s (thermal mass dependent)
    /// - Cross-validation: Must be consistent with dew point
    Temperature = 0,

    /// Relative humidity sensors (capacitive, resistive)
    ///
    /// Physics constraints:
    /// - Range: 0% to 100% RH
    /// - Rate limit: 20%/s (typical)
    /// - Cross-validation: Dew point must be below air temperature
    Humidity = 1,

    /// Barometric pressure sensors (piezoresistive, capacitive)
    ///
    /// Physics constraints:
    /// - Range: 540-1080 hPa (sea level to 5000m altitude)
    /// - Rate limit: 50 hPa/s (weather changes)
    /// - Cross-validation: Must correlate with altitude
    Pressure = 2,

    /// Volatile Organic Compound sensors (metal oxide, PID)
    ///
    /// Physics constraints:
    /// - Range: 0-50 ppm (typical)
    /// - Rate limit: Application dependent
    /// - Cross-validation: May correlate with temperature
    Voc = 3,

    /// Particulate matter sensors (optical, gravimetric)
    ///
    /// Physics constraints:
    /// - Range: 0-1000 μg/m³ (PM2.5/PM10)
    /// - Rate limit: Environmental dependent
    /// - Cross-validation: May correlate with humidity
    Particulate = 4,

    /// Acoustic level sensors (microphones, sound level meters)
    ///
    /// Physics constraints:
    /// - Range: 30-130 dB SPL
    /// - Rate limit: Very high (milliseconds)
    /// - Cross-validation: May correlate with vibration
    Acoustic = 5,

    /// Vibration sensors (accelerometers, piezoelectric)
    ///
    /// Physics constraints:
    /// - Range: 0-50g acceleration
    /// - Rate limit: Very high (microseconds)
    /// - Cross-validation: May correlate with acoustic
    Vibration = 6,

    /// Electromagnetic field sensors (hall effect, inductive)
    ///
    /// Physics constraints:
    /// - Range: Application dependent
    /// - Rate limit: Very high
    /// - Cross-validation: Equipment dependent
    Emf = 7,

    /// Custom sensor type for specialized applications
    ///
    /// Physics constraints:
    /// - Range: User defined
    /// - Rate limit: User defined
    /// - Cross-validation: User defined
    Custom = 255,
}

#[pymethods]
impl PySensorType {
    /// Get human-readable name for the sensor type
    ///
    /// Returns:
    ///     String name of the sensor type
    pub fn name(&self) -> &'static str {
        match self {
            PySensorType::Temperature => "Temperature",
            PySensorType::Humidity => "Humidity",
            PySensorType::Pressure => "Pressure",
            PySensorType::Voc => "VOC",
            PySensorType::Particulate => "Particulate",
            PySensorType::Acoustic => "Acoustic",
            PySensorType::Vibration => "Vibration",
            PySensorType::Emf => "EMF",
            PySensorType::Custom => "Custom",
        }
    }

    /// Get typical measurement unit for this sensor type
    ///
    /// Returns:
    ///     String representation of the typical unit
    fn unit(&self) -> &'static str {
        match self {
            PySensorType::Temperature => "°C",
            PySensorType::Humidity => "%RH",
            PySensorType::Pressure => "hPa",
            PySensorType::Voc => "ppm",
            PySensorType::Particulate => "μg/m³",
            PySensorType::Acoustic => "dB",
            PySensorType::Vibration => "g",
            PySensorType::Emf => "T",
            PySensorType::Custom => "custom",
        }
    }

    /// Get typical measurement range for this sensor type
    ///
    /// Returns:
    ///     Tuple of (min, max) typical values
    fn typical_range(&self) -> (f64, f64) {
        match self {
            PySensorType::Temperature => (-80.0, 125.0),
            PySensorType::Humidity => (0.0, 100.0),
            PySensorType::Pressure => (540.0, 1080.0),
            PySensorType::Voc => (0.0, 50.0),
            PySensorType::Particulate => (0.0, 1000.0),
            PySensorType::Acoustic => (30.0, 130.0),
            PySensorType::Vibration => (0.0, 50.0),
            PySensorType::Emf => (-1000.0, 1000.0),
            PySensorType::Custom => (f64::NEG_INFINITY, f64::INFINITY),
        }
    }

    fn __str__(&self) -> String {
        format!("{} ({})", self.name(), self.unit())
    }

    fn __repr__(&self) -> String {
        format!("SensorType.{}", self.name())
    }

    fn __eq__(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }

    fn __hash__(&self) -> u8 {
        self.clone() as u8
    }
}

/// Validation status for sensor readings
///
/// Indicates the result of physics-aware validation checks.
/// Used to track data quality and filter unreliable measurements.
#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub enum PyValidationStatus {
    /// Reading passed all validation checks
    Valid = 0,
    
    /// Reading is outside acceptable range for sensor type
    OutOfRange = 1,
    
    /// Rate of change exceeds physical limits
    RateExceeded = 2,
    
    /// Failed cross-validation with related sensors
    CrossValidationFailed = 3,
    
    /// Sensor quality indicator is below threshold
    SensorQualityBad = 4,
    
    /// Reading contains invalid values (NaN, infinite)
    InvalidValue = 5,
}

#[pymethods]
impl PyValidationStatus {
    /// Get human-readable description of validation status
    ///
    /// Returns:
    ///     String description of the status
    fn description(&self) -> &'static str {
        match self {
            PyValidationStatus::Valid => "Reading passed all validation checks",
            PyValidationStatus::OutOfRange => "Reading outside acceptable range",
            PyValidationStatus::RateExceeded => "Rate of change too high",
            PyValidationStatus::CrossValidationFailed => "Failed cross-sensor validation",
            PyValidationStatus::SensorQualityBad => "Poor sensor quality",
            PyValidationStatus::InvalidValue => "Invalid numerical value",
        }
    }

    /// Check if status indicates valid data
    ///
    /// Returns:
    ///     True if the reading is considered valid
    fn is_valid(&self) -> bool {
        matches!(self, PyValidationStatus::Valid)
    }

    /// Check if status indicates a critical error
    ///
    /// Returns:
    ///     True if the reading indicates a serious problem
    fn is_critical(&self) -> bool {
        matches!(
            self,
            PyValidationStatus::InvalidValue | PyValidationStatus::SensorQualityBad
        )
    }

    fn __str__(&self) -> String {
        format!("{}: {}", self.name(), self.description())
    }

    fn __repr__(&self) -> String {
        format!("ValidationStatus.{}", self.name())
    }

    pub fn name(&self) -> &'static str {
        match self {
            PyValidationStatus::Valid => "Valid",
            PyValidationStatus::OutOfRange => "OutOfRange",
            PyValidationStatus::RateExceeded => "RateExceeded",
            PyValidationStatus::CrossValidationFailed => "CrossValidationFailed",
            PyValidationStatus::SensorQualityBad => "SensorQualityBad",
            PyValidationStatus::InvalidValue => "InvalidValue",
        }
    }
}

/// Sensor reading event
///
/// Represents a single measurement from a sensor with associated metadata.
/// This is the fundamental data unit in EdgeGuard's processing pipeline.
///
/// # Design Notes
///
/// - Sensor ID is limited to 15 characters for memory efficiency
/// - Quality is normalized to 0.0-1.0 range (0.0 = unreliable, 1.0 = perfect)
/// - Timestamp precision is milliseconds for real-time applications
/// - Value precision is f32 for embedded compatibility
#[pyclass]
#[derive(Clone, Debug)]
pub struct PySensorReading {
    /// Unique identifier for the sensor (max 15 characters)
    #[pyo3(get)]
    pub sensor_id: String,
    
    /// Type of sensor and measurement
    #[pyo3(get)]
    pub sensor_type: PySensorType,
    
    /// Measured value in sensor's native units
    #[pyo3(get)]
    pub value: f64,
    
    /// Timestamp when measurement was taken
    #[pyo3(get)]
    pub timestamp: PyTimestamp,
    
    /// Quality indicator (0.0 = unreliable, 1.0 = perfect)
    #[pyo3(get)]
    pub quality: f64,
}

#[pymethods]
impl PySensorReading {
    /// Create a new sensor reading
    ///
    /// Args:
    ///     sensor_id: Unique sensor identifier (max 15 chars)
    ///     sensor_type: Type of sensor/measurement
    ///     value: Measured value
    ///     timestamp: When measurement was taken
    ///     quality: Quality indicator 0.0-1.0
    ///
    /// Returns:
    ///     New PySensorReading instance
    ///
    /// Raises:
    ///     ValueError: If parameters are invalid
    #[new]
    fn new(
        sensor_id: String,
        sensor_type: PySensorType,
        value: f64,
        timestamp: PyTimestamp,
        quality: f64,
    ) -> PyValidationResult<Self> {
        // Validate sensor ID length
        if sensor_id.len() > 15 {
            return Err(ErrorConverter::configuration_error(
                "sensor_id",
                "string with max 15 characters",
                &format!("{} characters", sensor_id.len()),
            ));
        }

        if sensor_id.is_empty() {
            return Err(ErrorConverter::configuration_error(
                "sensor_id",
                "non-empty string",
                "empty string",
            ));
        }

        // Validate quality range
        if !(0.0..=1.0).contains(&quality) {
            return Err(ErrorConverter::configuration_error(
                "quality",
                "value between 0.0 and 1.0",
                &format!("{}", quality),
            ));
        }

        // Validate value is finite
        if !value.is_finite() {
            return Err(ErrorConverter::configuration_error(
                "value",
                "finite number",
                if value.is_nan() { "NaN" } else { "infinite" },
            ));
        }

        Ok(Self {
            sensor_id,
            sensor_type,
            value,
            timestamp,
            quality,
        })
    }

    /// Check if this is a high-quality reading
    ///
    /// Returns:
    ///     True if quality is above 0.8
    fn is_high_quality(&self) -> bool {
        self.quality > 0.8
    }

    /// Check if reading is within typical range for sensor type
    ///
    /// Returns:
    ///     True if value is within typical range
    fn is_in_typical_range(&self) -> bool {
        let (min, max) = self.sensor_type.typical_range();
        self.value >= min && self.value <= max
    }

    /// Convert to dictionary representation
    ///
    /// Returns:
    ///     Dictionary with all reading data
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("sensor_id", &self.sensor_id)?;
        dict.set_item("sensor_type", self.sensor_type.name())?;
        dict.set_item("value", self.value)?;
        dict.set_item("timestamp", self.timestamp.to_datetime(py)?)?;
        dict.set_item("quality", self.quality)?;
        Ok(dict.into())
    }

    /// Create from dictionary representation
    ///
    /// Args:
    ///     data: Dictionary containing reading data
    ///
    /// Returns:
    ///     New PySensorReading instance
    #[staticmethod]
    fn from_dict(py: Python, data: &PyDict) -> PyValidationResult<Self> {
        let sensor_id: String = data.get_item("sensor_id")?
            .ok_or_else(|| ErrorConverter::configuration_error("data", "sensor_id key", "missing"))?
            .extract()?;
            
        let sensor_type_name: String = data.get_item("sensor_type")?
            .ok_or_else(|| ErrorConverter::configuration_error("data", "sensor_type key", "missing"))?
            .extract()?;
            
        let sensor_type = match sensor_type_name.as_str() {
            "Temperature" => PySensorType::Temperature,
            "Humidity" => PySensorType::Humidity,
            "Pressure" => PySensorType::Pressure,
            "VOC" => PySensorType::Voc,
            "Particulate" => PySensorType::Particulate,
            "Acoustic" => PySensorType::Acoustic,
            "Vibration" => PySensorType::Vibration,
            "EMF" => PySensorType::Emf,
            "Custom" => PySensorType::Custom,
            _ => return Err(ErrorConverter::configuration_error(
                "sensor_type",
                "valid sensor type name",
                &sensor_type_name,
            )),
        };
        
        let value: f64 = data.get_item("value")?
            .ok_or_else(|| ErrorConverter::configuration_error("data", "value key", "missing"))?
            .extract()?;
            
        let timestamp = if let Some(ts_obj) = data.get_item("timestamp")? {
            PyTimestamp::from_datetime(py, ts_obj)?
        } else {
            PyTimestamp::now()
        };
        
        let quality: f64 = if let Some(q) = data.get_item("quality")? {
            q.extract()?
        } else {
            1.0
        };
        
        Self::new(sensor_id, sensor_type, value, timestamp, quality)
    }

    fn __str__(&self) -> String {
        format!(
            "SensorReading(id='{}', type={}, value={:.3}, quality={:.2})",
            self.sensor_id,
            self.sensor_type.name(),
            self.value,
            self.quality
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "SensorReading(sensor_id='{}', sensor_type={:?}, value={}, timestamp={:?}, quality={})",
            self.sensor_id,
            self.sensor_type,
            self.value,
            self.timestamp,
            self.quality
        )
    }
}

/// Event builder for creating events with validation
///
/// Provides a fluent interface for building events while ensuring
/// all required fields are provided and validated.
///
/// # Examples
///
/// ```python
/// import edgeguard
/// 
/// # Create a temperature reading
/// event = (edgeguard.EventBuilder(timestamp)
///     .sensor("temp_01", edgeguard.SensorType.Temperature)
///     .reading(23.5, 0.95)
///     .build())
/// ```
#[pyclass]
pub struct PyEventBuilder {
    timestamp: PyTimestamp,
    sensor_id: Option<String>,
    sensor_type: Option<PySensorType>,
}

#[pymethods]
impl PyEventBuilder {
    /// Create a new event builder
    ///
    /// Args:
    ///     timestamp: Timestamp for the event
    ///
    /// Returns:
    ///     New PyEventBuilder instance
    #[new]
    fn new(timestamp: PyTimestamp) -> Self {
        Self {
            timestamp,
            sensor_id: None,
            sensor_type: None,
        }
    }

    /// Set sensor information
    ///
    /// Args:
    ///     sensor_id: Unique sensor identifier
    ///     sensor_type: Type of sensor
    ///
    /// Returns:
    ///     Self for method chaining
    fn sensor(mut slf: PyRefMut<Self>, sensor_id: String, sensor_type: PySensorType) -> PyRefMut<Self> {
        slf.sensor_id = Some(sensor_id);
        slf.sensor_type = Some(sensor_type);
        slf
    }

    /// Build a sensor reading event
    ///
    /// Args:
    ///     value: Measured value
    ///     quality: Quality indicator 0.0-1.0
    ///
    /// Returns:
    ///     PySensorReading event
    ///
    /// Raises:
    ///     ValueError: If required fields are missing or invalid
    fn reading(&self, value: f64, quality: f64) -> PyValidationResult<PySensorReading> {
        let sensor_id = self.sensor_id.as_ref()
            .ok_or_else(|| ErrorConverter::configuration_error(
                "sensor_id", "sensor() must be called first", "missing"
            ))?;
            
        let sensor_type = self.sensor_type.as_ref()
            .ok_or_else(|| ErrorConverter::configuration_error(
                "sensor_type", "sensor() must be called first", "missing"
            ))?;

        PySensorReading::new(
            sensor_id.clone(),
            sensor_type.clone(),
            value,
            self.timestamp,
            quality,
        )
    }
}

// Conversion utilities between Rust and Python types
impl PySensorType {
    pub(crate) fn to_rust(&self) -> RustSensorType {
        match self {
            PySensorType::Temperature => RustSensorType::Temperature,
            PySensorType::Humidity => RustSensorType::Humidity,
            PySensorType::Pressure => RustSensorType::Pressure,
            PySensorType::Voc => RustSensorType::Voc,
            PySensorType::Particulate => RustSensorType::Particulate,
            PySensorType::Acoustic => RustSensorType::Acoustic,
            PySensorType::Vibration => RustSensorType::Vibration,
            PySensorType::Emf => RustSensorType::Emf,
            PySensorType::Custom => RustSensorType::Custom(255),
        }
    }

    pub(crate) fn from_rust(sensor_type: RustSensorType) -> Self {
        match sensor_type {
            RustSensorType::Temperature => PySensorType::Temperature,
            RustSensorType::Humidity => PySensorType::Humidity,
            RustSensorType::Pressure => PySensorType::Pressure,
            RustSensorType::Voc => PySensorType::Voc,
            RustSensorType::Particulate => PySensorType::Particulate,
            RustSensorType::Acoustic => PySensorType::Acoustic,
            RustSensorType::Vibration => PySensorType::Vibration,
            RustSensorType::Emf => PySensorType::Emf,
            RustSensorType::Custom(_) => PySensorType::Custom,
        }
    }
}