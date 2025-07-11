//! Core Traits and Abstractions for Physics-Based Validation
//!
//! ## Design Philosophy
//!
//! EdgeGuard uses a trait-based architecture to provide flexibility while maintaining
//! zero-cost abstractions. The trait system allows:
//!
//! - **Pluggable Validators**: Add new sensor types without modifying core code
//! - **Static Dispatch**: No runtime overhead from dynamic dispatch
//! - **Compile-Time Optimization**: Monomorphization eliminates unused code
//! - **Type Safety**: Catch configuration errors at compile time
//!
//! ## Trait Hierarchy
//!
//! ```text
//! Validator (core validation logic)
//!     ├── TemperatureValidator
//!     ├── HumidityValidator  
//!     └── PressureValidator
//!
//! CrossValidator (multi-sensor correlation)
//!     ├── DewPointValidator (temp + humidity)
//!     └── AltitudeValidator (pressure + temp)
//!
//! EnvironmentalCompensation (sensor correction)
//!     └── PressureValidator (altitude compensation)
//! ```
//!
//! ## Memory Model
//!
//! The validation context is carefully sized for embedded use:
//!
//! ```text
//! ValidationContext size = ~544 bytes
//! ├── history: CircularBuffer<32> = 512 bytes
//! ├── timestamp: u64 = 8 bytes
//! ├── ambient_temp: Option<f32> = 8 bytes
//! ├── ambient_humidity: Option<f32> = 8 bytes
//! ├── sensor_quality: f32 = 4 bytes
//! └── padding = 4 bytes
//! ```
//!
//! Adjust `MAX_HISTORY_SIZE` based on your RAM budget and validation needs.
//!
//! ## Usage Pattern
//!
//! The typical validation flow:
//!
//! ```rust
//! use edgeguard_core::{Validator, TemperatureValidator, ValidationContext};
//!
//! // 1. Create validator with physics constraints
//! let validator = TemperatureValidator::new_with_limits(-40.0, 125.0, 5.0);
//!
//! // 2. Build context with history and environment
//! let mut ctx = ValidationContext::default();
//! ctx.add_reading(20.0, 1000);
//! ctx.add_reading(20.5, 2000);
//! ctx.timestamp = 3000;
//! ctx.sensor_quality = 0.95;
//!
//! // 3. Validate new reading
//! match validator.validate(21.0, &ctx) {
//!     Ok(()) => {
//!         // Reading passed all physics checks
//!         ctx.add_reading(21.0, ctx.timestamp);
//!     }
//!     Err(e) => {
//!         // Handle validation failure
//!         println!("Rejected: {:?}", e);
//!     }
//! }
//! ```
//!
//! ## Extension Points
//!
//! To add a new sensor type:
//!
//! 1. Define a struct for your validator
//! 2. Implement the `Validator` trait
//! 3. Optionally implement `CrossValidator` for multi-sensor checks
//! 4. Optionally implement `EnvironmentalCompensation` for corrections
//!
//! The trait system ensures your validator integrates seamlessly with the
//! existing infrastructure.

use crate::errors::ValidationResult;
use crate::time::Timestamp;
use crate::buffer::CircularBuffer;

/// Maximum number of historical samples to keep (adjust for your RAM constraints)
pub const MAX_HISTORY_SIZE: usize = 32;

/// A sensor reading paired with its timestamp
/// 
/// This structure captures both the sensor value and when it was measured,
/// enabling rate-of-change calculations and time-series analysis.
/// 
/// ## Size
/// 12 bytes total (4 byte f32 + 8 byte u64)
/// 
/// ## Usage
/// ```rust
/// use edgeguard_core::traits::TimestampedReading;
/// 
/// let reading = TimestampedReading {
///     value: 23.5,      // Temperature in Celsius
///     timestamp: 1000,  // Milliseconds since epoch/boot
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TimestampedReading {
    /// Sensor measurement value
    pub value: f32,
    /// When the measurement was taken (milliseconds)
    pub timestamp: Timestamp,
}

/// Validation context containing sensor history and environmental conditions
/// 
/// This structure provides validators with the information needed to perform
/// physics-aware validation beyond simple range checking. It includes:
/// 
/// - Historical data for rate-of-change validation
/// - Environmental conditions for cross-sensor validation
/// - Sensor health metrics for reliability assessment
/// 
/// ## Memory Layout
/// 
/// The structure is designed to fit within typical embedded RAM constraints:
/// - Total size: ~544 bytes with default history size
/// - Can be stack-allocated for deterministic memory usage
/// - History size is configurable via `MAX_HISTORY_SIZE`
/// 
/// ## Thread Safety
/// 
/// ValidationContext is not thread-safe. In multi-threaded environments,
/// each thread should have its own context or use synchronization.
#[derive(Clone)]
pub struct ValidationContext {
    /// Recent sensor readings for trend analysis
    /// 
    /// Stores up to MAX_HISTORY_SIZE readings in chronological order.
    /// Used for:
    /// - Rate-of-change validation
    /// - Trend detection
    /// - Noise analysis
    pub history: CircularBuffer<MAX_HISTORY_SIZE>,
    
    /// Current timestamp for the reading being validated
    /// 
    /// Should use the same time source as historical readings
    /// to ensure accurate rate calculations.
    pub timestamp: Timestamp,
    
    /// Ambient temperature from a reference sensor
    /// 
    /// Used for:
    /// - Temperature compensation of pressure sensors
    /// - Cross-validation of humidity readings
    /// - Thermal drift correction
    pub ambient_temp: Option<f32>,
    
    /// Ambient humidity from a reference sensor
    /// 
    /// Used for:
    /// - Dew point validation
    /// - Condensation risk assessment
    pub ambient_humidity: Option<f32>,
    
    /// Sensor health metric (0.0 = failing, 1.0 = perfect)
    /// 
    /// Derived from:
    /// - Self-diagnostic data
    /// - Historical accuracy
    /// - Calibration drift
    pub sensor_quality: f32,
}

impl Default for ValidationContext {
    fn default() -> Self {
        Self {
            history: CircularBuffer::new(),
            timestamp: 0,
            ambient_temp: None,
            ambient_humidity: None,
            sensor_quality: 1.0,
        }
    }
}

impl ValidationContext {
    /// Add a reading to history, maintaining chronological order
    pub fn add_reading(&mut self, value: f32, timestamp: Timestamp) {
        let reading = TimestampedReading { value, timestamp };
        self.history.push(reading);
    }
    
    /// Get the most recent reading if any
    pub fn last_reading(&self) -> Option<&TimestampedReading> {
        self.history.last()
    }
    
    /// Calculate time delta from last reading in milliseconds
    pub fn time_delta_ms(&self) -> Option<u64> {
        self.last_reading()
            .map(|last| self.timestamp.saturating_sub(last.timestamp))
    }
}

/// Core trait for implementing physics-based sensor validation
/// 
/// This is the primary extension point for adding new sensor types to EdgeGuard.
/// Implementors define validation logic that goes beyond simple range checking
/// to understand the physical constraints of the measured phenomenon.
/// 
/// ## Implementation Guidelines
/// 
/// 1. **Range Validation**: Check against physical limits
/// 2. **Rate Validation**: Ensure changes are physically plausible  
/// 3. **Cross-Validation**: Correlate with related sensors
/// 4. **Quality Checks**: Consider sensor degradation
/// 
/// ## Example Implementation
/// 
/// ```rust
/// use edgeguard_core::{Validator, ValidationContext, ValidationResult, ValidatorConstraints};
/// 
/// struct AirQualityValidator {
///     max_pm25: f32,
///     max_rate: f32,
/// }
/// 
/// impl Validator for AirQualityValidator {
///     type Value = f32;
///     
///     fn validate(&self, value: f32, context: &ValidationContext) -> ValidationResult<()> {
///         // Check physical limits
///         if value < 0.0 || value > self.max_pm25 {
///             return Err(ValidationError::OutOfRange {
///                 value,
///                 min: 0.0,
///                 max: self.max_pm25,
///             });
///         }
///         
///         // Check rate of change
///         // ... additional validation logic
///         
///         Ok(())
///     }
///     
///     fn constraints(&self) -> ValidatorConstraints {
///         ValidatorConstraints {
///             min_value: 0.0,
///             max_value: self.max_pm25,
///             max_rate_change: self.max_rate,
///             noise_threshold: Some(5.0), // ±5 μg/m³ sensor noise
///         }
///     }
/// }
/// ```
pub trait Validator {
    /// The type of value this validator handles
    type Value;
    
    /// Validate a single reading against physics constraints
    /// 
    /// Returns Ok(()) if the reading is plausible, or an error describing
    /// why the reading was rejected.
    fn validate(&self, value: Self::Value, context: &ValidationContext) -> ValidationResult<()>;
    
    /// Get the physical constraints for this validator
    /// 
    /// Used by the system to understand validation bounds for optimization
    /// and reporting purposes.
    fn constraints(&self) -> ValidatorConstraints;
}

/// Physical constraints for a validator
#[derive(Debug, Clone, Copy)]
pub struct ValidatorConstraints {
    /// Minimum valid value (physics limit)
    pub min_value: f32,
    
    /// Maximum valid value (physics limit)
    pub max_value: f32,
    
    /// Maximum rate of change per second
    pub max_rate_change: f32,
    
    /// Optional: typical noise level for filtering
    pub noise_threshold: Option<f32>,
}

/// Cross-validator for multi-sensor validation
pub trait CrossValidator {
    /// Input type (usually a tuple of readings)
    type Input;
    
    /// Validate readings from multiple sensors
    fn cross_validate(&self, inputs: Self::Input) -> ValidationResult<()>;
}

/// Trait for values that can be validated
pub trait Validatable {
    /// Check if the value is physically valid (not NaN, infinite, etc)
    fn is_valid(&self) -> bool;
}

impl Validatable for f32 {
    fn is_valid(&self) -> bool {
        self.is_finite()
    }
}

impl Validatable for f64 {
    fn is_valid(&self) -> bool {
        self.is_finite()
    }
}

/// Trait for sensors that need environmental compensation
pub trait EnvironmentalCompensation {
    /// Apply compensation based on environmental conditions
    fn compensate(&self, raw_value: f32, context: &ValidationContext) -> f32;
}