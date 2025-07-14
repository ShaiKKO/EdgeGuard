//! Time-Related Constants
//!
//! This module defines time intervals, durations, and conversion factors
//! used throughout the EdgeGuard system for scheduling and timing operations.

// ===== TIME UNIT CONVERSIONS =====

/// Milliseconds per second.
pub const MS_PER_SECOND: u64 = 1000;

/// Microseconds per millisecond.
pub const US_PER_MS: u64 = 1000;

/// Microseconds per second.
pub const US_PER_SECOND: u64 = 1_000_000;

/// Seconds per minute.
pub const SECONDS_PER_MINUTE: u32 = 60;

/// Minutes per hour.
pub const MINUTES_PER_HOUR: u32 = 60;

/// Hours per day.
pub const HOURS_PER_DAY: u32 = 24;

/// Seconds per hour.
pub const SECONDS_PER_HOUR: u32 = SECONDS_PER_MINUTE * MINUTES_PER_HOUR;

/// Milliseconds per minute.
pub const MS_PER_MINUTE: u64 = MS_PER_SECOND * SECONDS_PER_MINUTE as u64;

/// Milliseconds per hour.
pub const MS_PER_HOUR: u64 = MS_PER_MINUTE * MINUTES_PER_HOUR as u64;

// ===== SAMPLING INTERVALS =====

/// Default sensor sampling interval (milliseconds).
/// 
/// 1 Hz sampling rate is standard for environmental monitoring.
/// Balances data quality with power consumption.
/// 
/// Source: Industry best practices for IoT
pub const DEFAULT_SAMPLE_INTERVAL_MS: u64 = 1000;

/// High-frequency sampling interval (milliseconds).
/// 
/// 10 Hz for applications requiring rapid response.
/// Used for vibration monitoring, control loops.
/// 
/// Source: Industrial control requirements
pub const HIGH_FREQ_SAMPLE_INTERVAL_MS: u64 = 100;

/// Low-frequency sampling interval (milliseconds).
/// 
/// 0.1 Hz for battery-powered, long-term monitoring.
/// Common in LoRaWAN and agricultural applications.
/// 
/// Source: LoRaWAN specification
pub const LOW_FREQ_SAMPLE_INTERVAL_MS: u64 = 10000;

/// Environmental monitoring interval (seconds).
/// 
/// Standard interval for HVAC and weather stations.
/// Provides sufficient resolution for slow-changing parameters.
/// 
/// Source: ASHRAE monitoring guidelines
pub const ENV_MONITOR_INTERVAL_S: u32 = 60;

// ===== TIMEOUT VALUES =====

/// Default operation timeout (milliseconds).
/// 
/// Maximum time to wait for sensor response or operation completion.
/// Prevents indefinite blocking in error conditions.
/// 
/// Source: Embedded systems best practices
pub const DEFAULT_TIMEOUT_MS: u32 = 5000;

/// Network operation timeout (milliseconds).
/// 
/// Longer timeout for network operations (HTTP, MQTT).
/// Accounts for network latency and retries.
/// 
/// Source: IoT protocol recommendations
pub const NETWORK_TIMEOUT_MS: u32 = 30000;

/// Sensor warmup timeout (milliseconds).
/// 
/// Maximum time to wait for sensor stabilization.
/// Some sensors (MOx, electrochemical) need warmup.
/// 
/// Source: Sensor manufacturer specifications
pub const SENSOR_WARMUP_TIMEOUT_MS: u32 = 60000;

/// Critical operation timeout (milliseconds).
/// 
/// Short timeout for time-critical operations.
/// Used in control loops and safety systems.
/// 
/// Source: Real-time system requirements
pub const CRITICAL_TIMEOUT_MS: u32 = 100;

// ===== AGGREGATION WINDOWS =====

/// Default aggregation window (milliseconds).
/// 
/// Time window for computing statistics (mean, min, max).
/// 1 minute provides good balance of resolution and smoothing.
/// 
/// Source: Time-series database practices
pub const DEFAULT_AGG_WINDOW_MS: u32 = 60000;

/// Short aggregation window (milliseconds).
/// 
/// For near real-time statistics and rapid changes.
/// 10 seconds captures transient events.
/// 
/// Source: SCADA system practices
pub const SHORT_AGG_WINDOW_MS: u32 = 10000;

/// Long aggregation window (milliseconds).
/// 
/// For trend analysis and bandwidth reduction.
/// 15 minutes is standard for historical data.
/// 
/// Source: Building automation standards
pub const LONG_AGG_WINDOW_MS: u32 = 900000;

// ===== RETRY AND BACKOFF =====

/// Initial retry delay (milliseconds).
/// 
/// First retry attempt after failure.
/// Short delay for transient errors.
/// 
/// Source: Exponential backoff best practices
pub const INITIAL_RETRY_DELAY_MS: u32 = 1000;

/// Maximum retry delay (milliseconds).
/// 
/// Upper bound for exponential backoff.
/// Prevents excessive wait times.
/// 
/// Source: Network protocol standards
pub const MAX_RETRY_DELAY_MS: u32 = 60000;

/// Default number of retry attempts.
/// 
/// Balance between reliability and resource usage.
/// Three attempts handle most transient failures.
/// 
/// Source: Reliability engineering practices
pub const DEFAULT_RETRY_COUNT: u32 = 3;

// ===== MAINTENANCE INTERVALS =====

/// Sensor calibration interval (days).
/// 
/// Recommended recalibration frequency.
/// Varies by sensor type and application.
/// 
/// Source: Metrology standards
pub const CALIBRATION_INTERVAL_DAYS: u32 = 365;

/// System health check interval (seconds).
/// 
/// Frequency of self-diagnostic checks.
/// Detects degradation before failure.
/// 
/// Source: Preventive maintenance practices
pub const HEALTH_CHECK_INTERVAL_S: u32 = 3600;

/// Data retention period (days).
/// 
/// How long to keep historical data.
/// Balance between storage and analysis needs.
/// 
/// Source: Data governance policies
pub const DATA_RETENTION_DAYS: u32 = 90;

// ===== REAL-TIME CONSTRAINTS =====

/// Maximum acceptable latency (microseconds).
/// 
/// For real-time control applications.
/// Includes sensor reading and processing time.
/// 
/// Source: Industrial control requirements
pub const MAX_LATENCY_US: u64 = 1000;

/// Control loop period (milliseconds).
/// 
/// Update rate for PID and other control algorithms.
/// 100 Hz is typical for motor control.
/// 
/// Source: Control theory practices
pub const CONTROL_LOOP_PERIOD_MS: u32 = 10;

/// Watchdog timeout (milliseconds).
/// 
/// Maximum time between watchdog resets.
/// Detects system hangs and crashes.
/// 
/// Source: Embedded system safety standards
pub const WATCHDOG_TIMEOUT_MS: u32 = 5000;