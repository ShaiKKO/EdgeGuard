//! Quality Thresholds and Sensor Grades
//!
//! This module defines quality scores, confidence thresholds, and sensor
//! accuracy classifications used for data validation and fusion.

// ===== SENSOR QUALITY GRADES =====

/// Professional-grade sensor quality score (0.0-1.0).
/// 
/// Represents calibrated, high-accuracy sensors:
/// - Laboratory equipment
/// - Calibrated RTDs (Class A)
/// - Research-grade instruments
/// 
/// Typical accuracy: ±0.5% of reading or better
pub const QUALITY_PROFESSIONAL: f32 = 0.95;

/// Consumer-grade sensor quality score (0.0-1.0).
/// 
/// Represents typical commercial sensors:
/// - DS18B20, DHT22, BME280
/// - Factory-calibrated sensors
/// - Smart home devices
/// 
/// Typical accuracy: ±2% of reading
pub const QUALITY_CONSUMER: f32 = 0.90;

/// Budget/hobby-grade sensor quality score (0.0-1.0).
/// 
/// Represents low-cost sensors:
/// - DHT11, basic thermistors
/// - Uncalibrated sensors
/// - DIY projects
/// 
/// Typical accuracy: ±5% of reading
pub const QUALITY_BUDGET: f32 = 0.80;

/// Degraded sensor quality score (0.0-1.0).
/// 
/// Sensor showing signs of drift or noise:
/// - Aging sensors needing calibration
/// - Environmental interference
/// - Power supply issues
/// 
/// Data still usable but less reliable
pub const QUALITY_DEGRADED: f32 = 0.70;

/// Poor sensor quality score (0.0-1.0).
/// 
/// Sensor near failure or highly unreliable:
/// - Significant drift or bias
/// - Intermittent connections
/// - Environmental damage
/// 
/// Data should be used with caution
pub const QUALITY_POOR: f32 = 0.50;

// ===== QUALITY THRESHOLDS =====

/// Minimum quality for critical applications.
/// 
/// Below this, readings should not be used for:
/// - Safety systems
/// - Control loops
/// - Regulatory compliance
/// 
/// Source: Industrial safety standards
pub const QUALITY_THRESHOLD_CRITICAL: f32 = 0.90;

/// Minimum quality for normal operations.
/// 
/// Standard threshold for accepting sensor data.
/// Below this triggers quality warnings.
/// 
/// Source: IoT best practices
pub const QUALITY_THRESHOLD_GOOD: f32 = 0.80;

/// Minimum quality for any use.
/// 
/// Absolute minimum for data acceptance.
/// Below this indicates sensor failure.
/// 
/// Source: Statistical reliability theory
pub const QUALITY_THRESHOLD_ACCEPTABLE: f32 = 0.50;

/// Quality degradation rate per year.
/// 
/// Expected annual decline in sensor quality.
/// Used for predictive maintenance.
/// 
/// Source: Sensor aging studies
pub const QUALITY_ANNUAL_DEGRADATION: f32 = 0.02;

// ===== CONFIDENCE SCORES =====

/// High confidence threshold.
/// 
/// Indicates strong agreement between:
/// - Multiple sensors
/// - Model predictions
/// - Historical patterns
/// 
/// Source: Statistical confidence intervals (95%)
pub const CONFIDENCE_HIGH: f32 = 0.95;

/// Medium confidence threshold.
/// 
/// Acceptable for most applications.
/// May warrant additional validation.
/// 
/// Source: Statistical confidence intervals (80%)
pub const CONFIDENCE_MEDIUM: f32 = 0.80;

/// Low confidence threshold.
/// 
/// Minimum for data acceptance.
/// Requires corroboration or filtering.
/// 
/// Source: Statistical significance (p < 0.5)
pub const CONFIDENCE_LOW: f32 = 0.50;

/// Initial confidence for new sensors.
/// 
/// Starting confidence before calibration data.
/// Improves with successful validations.
/// 
/// Source: Bayesian prior assumption
pub const CONFIDENCE_INITIAL: f32 = 0.70;

// ===== FUSION WEIGHTS =====

/// Weight multiplier for professional sensors.
/// 
/// Higher weight in fusion algorithms due to accuracy.
/// Normalized with other sensor weights.
/// 
/// Source: Sensor accuracy ratios
pub const FUSION_WEIGHT_PROFESSIONAL: f32 = 1.0;

/// Weight multiplier for consumer sensors.
/// 
/// Standard weight for typical sensors.
/// Baseline for fusion calculations.
/// 
/// Source: Relative accuracy assessment
pub const FUSION_WEIGHT_CONSUMER: f32 = 0.5;

/// Weight multiplier for budget sensors.
/// 
/// Lower weight due to higher uncertainty.
/// Still contributes to redundancy.
/// 
/// Source: Error propagation analysis
pub const FUSION_WEIGHT_BUDGET: f32 = 0.2;

/// Minimum weight for sensor inclusion.
/// 
/// Below this, sensor excluded from fusion.
/// Prevents bad data from affecting results.
/// 
/// Source: Outlier rejection theory
pub const FUSION_WEIGHT_MINIMUM: f32 = 0.1;

// ===== ANOMALY DETECTION =====

/// Anomaly score threshold for warnings.
/// 
/// Triggers alert but continues operation.
/// Indicates unusual but not impossible values.
/// 
/// Source: Statistical process control (2σ)
pub const ANOMALY_SCORE_WARNING: f32 = 0.6;

/// Anomaly score threshold for alarms.
/// 
/// Indicates likely sensor or system fault.
/// May trigger protective actions.
/// 
/// Source: Statistical process control (3σ)
pub const ANOMALY_SCORE_ALARM: f32 = 0.8;

/// Anomaly score threshold for rejection.
/// 
/// Data rejected as physically impossible.
/// Sensor marked as failed.
/// 
/// Source: Physical constraint violations
pub const ANOMALY_SCORE_REJECT: f32 = 0.95;

// ===== DATA VALIDATION SCORES =====

/// Score multiplier for in-range values.
/// 
/// Reward for values within expected bounds.
/// Contributes to overall quality score.
pub const SCORE_IN_RANGE: f32 = 1.0;

/// Score multiplier for edge values.
/// 
/// Values near limits but still valid.
/// Slightly reduced confidence.
pub const SCORE_EDGE_CASE: f32 = 0.8;

/// Score multiplier for rate compliance.
/// 
/// Values with acceptable rate of change.
/// Indicates sensor responding properly.
pub const SCORE_RATE_OK: f32 = 1.0;

/// Score multiplier for high rate of change.
/// 
/// Rapid but possible changes.
/// May indicate external disturbance.
pub const SCORE_RATE_HIGH: f32 = 0.7;

/// Score multiplier for cross-validation pass.
/// 
/// Agreement with related sensors.
/// Increases confidence significantly.
pub const SCORE_CROSS_VALID: f32 = 1.2;

/// Score multiplier for cross-validation fail.
/// 
/// Disagreement with related sensors.
/// Reduces confidence, may indicate issue.
pub const SCORE_CROSS_INVALID: f32 = 0.5;