//! Event Types for Sensor Data Processing Pipeline
//!
//! ## Overview
//!
//! This module defines the event system that forms the backbone of EdgeGuard's
//! real-time data processing pipeline. Events flow through the system carrying
//! sensor data, validation results, and system status information.
//!
//! ## Design Philosophy
//!
//! ### Why Events?
//!
//! Traditional IoT systems often use synchronous, blocking validation:
//! ```text
//! Sensor → Read → Validate → Store → Transmit
//!          ↓        ↓         ↓        ↓
//!        Block    Block     Block    Block
//! ```
//!
//! This approach has several problems:
//! - **Blocking**: Each stage waits for the previous one
//! - **Rigid**: Hard to add new processing stages
//! - **Memory**: Buffers data at each stage
//! - **Latency**: Slow sensors block fast ones
//!
//! Event-driven architecture solves these issues:
//! ```text
//! Sensor → Event Queue → Pipeline → Output
//!    ↓         ↓            ↓         ↓
//!  Async   Buffered    Parallel   Batched
//! ```
//!
//! ### Memory Model
//!
//! Events are designed for embedded constraints:
//! - **Size**: <128 bytes to fit in cache lines
//! - **Alignment**: 8-byte aligned for atomic operations
//! - **Lifetime**: Stack-allocated, no heap required
//! - **Zero-Copy**: Use references where possible
//!
//! ```text
//! Event size breakdown (worst case):
//! ├── Discriminant: 4 bytes
//! ├── Largest variant (BatchReading): 120 bytes
//! │   ├── sensor_id: 16 bytes (inline string)
//! │   ├── base_timestamp: 8 bytes
//! │   ├── values ptr: 8 bytes
//! │   ├── values len: 8 bytes
//! │   └── interval_ms: 2 bytes
//! └── Padding: 4 bytes
//! Total: 128 bytes
//! ```
//!
//! ### Type Safety
//!
//! Events use Rust's type system to prevent errors:
//! - **Tagged Unions**: Can't misinterpret event type
//! - **Lifetime Tracking**: Prevents use-after-free
//! - **Exhaustive Matching**: Compiler ensures all cases handled
//!
//! ## Event Flow
//!
//! Events flow through the pipeline in stages:
//!
//! 1. **Generation**: Sensors produce `SensorReading` events
//! 2. **Validation**: Validators consume readings, produce `ValidationResult`
//! 3. **Correlation**: Cross-validators produce `CrossValidationResult`
//! 4. **Aggregation**: High-frequency data batched into `BatchReading`
//! 5. **Monitoring**: System produces `SystemEvent` for health tracking
//!
//! ## Cross-Sensor Validation
//!
//! Physical phenomena often involve multiple sensors:
//! - **Dew Point**: Requires temperature AND humidity
//! - **Altitude**: Requires pressure AND temperature
//! - **Heat Index**: Requires temperature AND humidity
//!
//! The `CrossValidationResult` event captures these relationships,
//! enabling detection of subtle inconsistencies that single-sensor
//! validation would miss.

use crate::time::Timestamp;
use core::fmt;

/// Maximum length for inline sensor IDs
/// 
/// IDs longer than this use external storage (not implemented in no_std)
pub const MAX_INLINE_ID: usize = 15;

/// Sensor type enumeration
/// 
/// Maps to specific validators and physical constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SensorType {
    Temperature = 0,
    Humidity = 1,
    Pressure = 2,
    Voc = 3,
    Particulate = 4,
    Acoustic = 5,
    Vibration = 6,
    Emf = 7,
    Custom(u8),
}

impl SensorType {
    /// Get human-readable name
    pub const fn name(&self) -> &'static str {
        match self {
            SensorType::Temperature => "temperature",
            SensorType::Humidity => "humidity",
            SensorType::Pressure => "pressure",
            SensorType::Voc => "voc",
            SensorType::Particulate => "particulate",
            SensorType::Acoustic => "acoustic",
            SensorType::Vibration => "vibration",
            SensorType::Emf => "emf",
            SensorType::Custom(_) => "custom",
        }
    }
    
    /// Get expected unit of measurement
    pub const fn unit(&self) -> &'static str {
        match self {
            SensorType::Temperature => "°C",
            SensorType::Humidity => "%",
            SensorType::Pressure => "hPa",
            SensorType::Voc => "ppb",
            SensorType::Particulate => "μg/m³",
            SensorType::Acoustic => "dB",
            SensorType::Vibration => "m/s²",
            SensorType::Emf => "V/m",
            SensorType::Custom(_) => "",
        }
    }
}

/// Validation status for events
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ValidationStatus {
    /// Reading passed all validation checks
    Valid = 0,
    /// Reading is outside acceptable range
    OutOfRange = 1,
    /// Rate of change exceeds physical limits
    RateExceeded = 2,
    /// Cross-sensor validation failed
    CrossValidationFailed = 3,
    /// Sensor quality is too low
    SensorQualityBad = 4,
    /// Reading failed basic sanity checks (NaN, Inf)
    InvalidValue = 5,
}

/// Bit flags for applied constraints
/// 
/// Tracks which validation rules were checked
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstraintFlags(u16);

impl ConstraintFlags {
    pub const RANGE: Self = Self(1 << 0);
    pub const RATE: Self = Self(1 << 1);
    pub const CROSS: Self = Self(1 << 2);
    pub const QUALITY: Self = Self(1 << 3);
    pub const PHYSICS: Self = Self(1 << 4);
    
    pub const fn empty() -> Self {
        Self(0)
    }
    
    pub const fn all() -> Self {
        Self(0b11111)
    }
    
    pub fn set(&mut self, other: Self) {
        self.0 |= other.0;
    }
    
    pub const fn contains(&self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

/// Cross-validation types for multi-sensor checks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum CrossValidationType {
    /// Dew point must be <= air temperature
    DewPoint = 0,
    /// Pressure must match altitude expectations
    AltitudePressure = 1,
    /// Multiple temperature sensors should agree
    ThermalConsistency = 2,
    /// Humidity affects pressure slightly
    HumidityPressure = 3,
    /// Custom validation type
    Custom(u16),
}

/// Details about cross-validation failures
#[derive(Debug, Clone, Copy)]
pub struct CrossValidationDetails {
    /// Expected value based on physics
    pub expected_value: f32,
    /// Actual value from sensor
    pub actual_value: f32,
    /// Percentage deviation from expected
    pub deviation_percent: f32,
}

/// System event types for monitoring
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SystemEventType {
    /// Pipeline started
    PipelineStart = 0,
    /// Pipeline stopped
    PipelineStop = 1,
    /// Event queue overflow
    QueueOverflow = 2,
    /// Validator error
    ValidatorError = 3,
    /// Memory pressure warning
    MemoryWarning = 4,
    /// Performance degradation
    PerformanceWarning = 5,
}

/// Inline string for sensor IDs
/// 
/// Avoids heap allocation for common ID lengths
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct InlineString {
    len: u8,
    data: [u8; MAX_INLINE_ID],
}

impl InlineString {
    /// Create from string slice
    pub fn new(s: &str) -> Option<Self> {
        let bytes = s.as_bytes();
        if bytes.len() > MAX_INLINE_ID {
            return None;
        }
        
        let mut data = [0u8; MAX_INLINE_ID];
        data[..bytes.len()].copy_from_slice(bytes);
        
        Some(Self {
            len: bytes.len() as u8,
            data,
        })
    }
    
    /// Get as string slice
    pub fn as_str(&self) -> &str {
        // We only store valid UTF-8 from new(), so this should never panic
        core::str::from_utf8(&self.data[..self.len as usize])
            .expect("InlineString contains invalid UTF-8")
    }
}

impl fmt::Debug for InlineString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.as_str())
    }
}

/// Main event type for the processing pipeline
/// 
/// This enum represents all possible events that flow through the system.
/// Each variant is carefully sized to fit within 128 bytes total.
#[derive(Debug, Clone)]
pub enum Event {
    /// Raw sensor reading from a single sensor
    /// 
    /// Size: ~48 bytes
    SensorReading {
        /// Sensor identifier (e.g., "temp_01")
        sensor_id: InlineString,
        /// Type of sensor for routing
        sensor_type: SensorType,
        /// Measured value in sensor units
        value: f32,
        /// Timestamp in milliseconds
        timestamp: Timestamp,
        /// Quality indicator (0.0-1.0)
        quality: f32,
    },
    
    /// Validation result for a single sensor
    /// 
    /// Size: ~32 bytes
    ValidationResult {
        /// Sensor that was validated
        sensor_id: InlineString,
        /// Validation outcome
        status: ValidationStatus,
        /// Which constraints were checked
        constraints_applied: ConstraintFlags,
        /// When validation occurred
        timestamp: Timestamp,
    },
    
    /// Cross-sensor validation result
    /// 
    /// Size: ~64 bytes
    CrossValidationResult {
        /// Primary sensor in validation
        primary_sensor: InlineString,
        /// Related sensor used for validation
        related_sensor: InlineString,
        /// Type of cross-validation performed
        validation_type: CrossValidationType,
        /// Validation outcome
        status: ValidationStatus,
        /// Detailed failure information
        details: CrossValidationDetails,
        /// When validation occurred
        timestamp: Timestamp,
    },
    
    /// Batched readings for high-frequency sensors
    /// 
    /// Size: ~40 bytes (+ external data)
    BatchReading {
        /// Sensor identifier
        sensor_id: InlineString,
        /// Type of sensor
        sensor_type: SensorType,
        /// Timestamp of first reading
        base_timestamp: Timestamp,
        /// Number of readings
        count: u16,
        /// Milliseconds between readings
        interval_ms: u16,
        /// Average value for quick checks
        mean_value: f32,
        /// Min value in batch
        min_value: f32,
        /// Max value in batch
        max_value: f32,
    },
    
    /// System events for monitoring
    /// 
    /// Size: ~24 bytes
    SystemEvent {
        /// Type of system event
        event_type: SystemEventType,
        /// When event occurred
        timestamp: Timestamp,
        /// Event-specific details (bit-packed)
        details: u32,
    },
}

impl Event {
    /// Get event timestamp
    pub fn timestamp(&self) -> Timestamp {
        match self {
            Event::SensorReading { timestamp, .. } => *timestamp,
            Event::ValidationResult { timestamp, .. } => *timestamp,
            Event::CrossValidationResult { timestamp, .. } => *timestamp,
            Event::BatchReading { base_timestamp, .. } => *base_timestamp,
            Event::SystemEvent { timestamp, .. } => *timestamp,
        }
    }
    
    /// Get sensor ID if applicable
    pub fn sensor_id(&self) -> Option<&str> {
        match self {
            Event::SensorReading { sensor_id, .. } => Some(sensor_id.as_str()),
            Event::ValidationResult { sensor_id, .. } => Some(sensor_id.as_str()),
            Event::CrossValidationResult { primary_sensor, .. } => Some(primary_sensor.as_str()),
            Event::BatchReading { sensor_id, .. } => Some(sensor_id.as_str()),
            Event::SystemEvent { .. } => None,
        }
    }
    
    /// Check if event represents valid data
    pub fn is_valid(&self) -> bool {
        match self {
            Event::ValidationResult { status, .. } => *status == ValidationStatus::Valid,
            Event::CrossValidationResult { status, .. } => *status == ValidationStatus::Valid,
            _ => true, // Other events don't have validation status
        }
    }
    
    /// Get event priority for queue management
    /// 
    /// Lower numbers = higher priority
    pub fn priority(&self) -> u8 {
        match self {
            Event::SystemEvent { .. } => 0, // Highest priority
            Event::CrossValidationResult { .. } => 1,
            Event::ValidationResult { .. } => 2,
            Event::SensorReading { .. } => 3,
            Event::BatchReading { .. } => 4, // Lowest priority
        }
    }
}

/// Event builder for convenient construction
pub struct EventBuilder {
    sensor_id: Option<InlineString>,
    sensor_type: Option<SensorType>,
    timestamp: Timestamp,
}

impl EventBuilder {
    /// Create new builder with current timestamp
    pub fn new(timestamp: Timestamp) -> Self {
        Self {
            sensor_id: None,
            sensor_type: None,
            timestamp,
        }
    }
    
    /// Set sensor ID
    pub fn sensor(mut self, id: &str, sensor_type: SensorType) -> Self {
        self.sensor_id = InlineString::new(id);
        self.sensor_type = Some(sensor_type);
        self
    }
    
    /// Build sensor reading event
    pub fn reading(self, value: f32, quality: f32) -> Option<Event> {
        Some(Event::SensorReading {
            sensor_id: self.sensor_id?,
            sensor_type: self.sensor_type?,
            value,
            timestamp: self.timestamp,
            quality,
        })
    }
    
    /// Build validation result event
    pub fn validation(self, status: ValidationStatus, constraints: ConstraintFlags) -> Option<Event> {
        Some(Event::ValidationResult {
            sensor_id: self.sensor_id?,
            status,
            constraints_applied: constraints,
            timestamp: self.timestamp,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn event_size() {
        // Ensure events fit in cache line
        assert!(core::mem::size_of::<Event>() <= 128);
    }
    
    #[test]
    fn inline_string() {
        let s = InlineString::new("temp_01").unwrap();
        assert_eq!(s.as_str(), "temp_01");
        
        // Too long
        assert!(InlineString::new("this_is_a_very_long_sensor_id").is_none());
    }
    
    #[test]
    fn event_builder() {
        let event = EventBuilder::new(1000)
            .sensor("temp_01", SensorType::Temperature)
            .reading(25.0, 0.95)
            .unwrap();
        
        assert_eq!(event.sensor_id(), Some("temp_01"));
        assert_eq!(event.timestamp(), 1000);
    }
    
    #[test]
    fn constraint_flags() {
        let mut flags = ConstraintFlags::empty();
        flags.set(ConstraintFlags::RANGE);
        flags.set(ConstraintFlags::RATE);
        
        assert!(flags.contains(ConstraintFlags::RANGE));
        assert!(flags.contains(ConstraintFlags::RATE));
        assert!(!flags.contains(ConstraintFlags::CROSS));
    }
}