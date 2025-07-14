//! Constants for EdgeGuard Core
//!
//! This module provides centralized, well-documented constants used throughout
//! the EdgeGuard system. All numeric values are defined here with clear
//! explanations of their purpose, source, and rationale.
//!
//! ## Organization
//!
//! Constants are grouped by domain:
//! - **Physics**: Fundamental physical constants and limits
//! - **Sensors**: Sensor specifications and characteristics  
//! - **Time**: Time-related constants and intervals
//! - **Quality**: Quality scores and thresholds
//! - **Buffers**: Memory and buffer size limits
//!
//! ## Usage Guidelines
//!
//! 1. Always use these constants instead of magic numbers
//! 2. When adding new constants, include comprehensive documentation
//! 3. Reference industry standards or datasheets where applicable
//! 4. Group related constants together
//! 5. Use descriptive names that include units

/// Physical constants and limits based on laws of physics and thermodynamics.
pub mod physics;

/// Sensor specifications including accuracy grades and operational limits.
pub mod sensors;

/// Time-related constants for intervals, timeouts, and sampling rates.
pub mod time;

/// Quality thresholds and sensor accuracy classifications.
pub mod quality;

/// Buffer sizes and memory constraints for embedded systems.
pub mod buffers;

/// Fusion algorithm parameters and thresholds.
pub mod fusion;

/// Pipeline processing parameters and limits.
pub mod pipeline;

// Re-export commonly used constants for convenience
pub use physics::{
    ABSOLUTE_ZERO_CELSIUS, SEA_LEVEL_PRESSURE_HPA,
    INDOOR_TEMP_MIN_C, INDOOR_TEMP_MAX_C, INDOOR_TEMP_NOMINAL_C,
    INDOOR_HUMIDITY_MIN_PCT, INDOOR_HUMIDITY_MAX_PCT,
};

pub use sensors::{
    TEMP_SENSOR_MIN_C, TEMP_SENSOR_MAX_C, TEMP_MAX_RATE_C_PER_S,
    HUMIDITY_SENSOR_MIN_PCT, HUMIDITY_SENSOR_MAX_PCT, HUMIDITY_MAX_RATE_PCT_PER_S,
    PRESSURE_SENSOR_MIN_HPA, PRESSURE_SENSOR_MAX_HPA, PRESSURE_MAX_RATE_HPA_PER_S,
};

pub use time::{
    MS_PER_SECOND, SECONDS_PER_MINUTE, MINUTES_PER_HOUR,
    DEFAULT_SAMPLE_INTERVAL_MS, HIGH_FREQ_SAMPLE_INTERVAL_MS,
};

pub use quality::{
    QUALITY_PROFESSIONAL, QUALITY_CONSUMER, QUALITY_BUDGET,
    QUALITY_THRESHOLD_GOOD, QUALITY_THRESHOLD_ACCEPTABLE,
};

pub use buffers::{
    DEFAULT_EVENT_BUFFER_SIZE, SMALL_EVENT_BUFFER_SIZE,
    MAX_PIPELINE_STAGES, MAX_SENSOR_GROUPS,
};

pub use fusion::{
    MIN_SENSORS_FOR_FUSION, CONVERGENCE_MEASUREMENT_COUNT,
    FIXED_POINT_SCALE, NEWTON_METHOD_ITERATIONS,
};