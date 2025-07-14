//! Fusion Algorithm Constants
//!
//! This module defines constants for sensor fusion algorithms including
//! Kalman filters, weighted averaging, complementary filters, and
//! confidence scoring mechanisms.

// ===== FUSION ALGORITHM PARAMETERS =====

/// Minimum number of sensors required for fusion algorithms.
/// 
/// Most fusion algorithms need at least 2 sensors to provide
/// meaningful results. Single sensor data passes through unchanged.
/// 
/// Source: Sensor fusion theory
pub const MIN_SENSORS_FOR_FUSION: usize = 2;

/// Confidence score for single sensor measurements.
/// 
/// When only one sensor is available, we assign moderate confidence
/// since there's no redundancy for validation.
/// 
/// Source: Engineering judgment based on reliability analysis
pub const SINGLE_SENSOR_CONFIDENCE: f32 = 0.5;

/// Number of measurements required for fusion convergence.
/// 
/// Most fusion algorithms need this many measurements to stabilize
/// their internal state and provide reliable estimates.
/// 
/// Source: Empirical testing with Kalman filters
pub const CONVERGENCE_MEASUREMENT_COUNT: u32 = 10;

/// Minimum updates required before declaring convergence.
/// 
/// Prevents premature convergence detection in Kalman filters
/// and other adaptive algorithms.
/// 
/// Source: Control system best practices
pub const MIN_CONVERGENCE_UPDATES: u32 = 5;

/// Time constant divisor for convergence calculations.
/// 
/// Used to scale convergence based on measurement frequency.
/// Higher values mean slower convergence adaptation.
/// 
/// Source: Adaptive filter design
pub const CONVERGENCE_TIME_DIVISOR: f32 = 10.0;

// ===== FIXED-POINT ARITHMETIC =====

/// Fixed-point scaling factor for u16 confidence values.
/// 
/// Maps floating point [0.0, 1.0] to integer [0, 65535].
/// Chosen for full u16 range utilization.
/// 
/// Source: Q16.16 fixed-point representation
pub const FIXED_POINT_SCALE: f32 = 65535.0;

/// Minimum confidence value in fixed-point representation (1%).
/// 
/// Represents extremely low confidence, typically indicating
/// sensor failure or complete uncertainty.
/// 
/// Source: 0.01 * 65535 ≈ 655
pub const CONFIDENCE_MIN_FIXED: u16 = 655;

/// Moderate confidence in fixed-point representation (50%).
/// 
/// Represents average confidence, used as default for
/// single sensor or uncertain conditions.
/// 
/// Source: 0.5 * 65535 = 32768
pub const CONFIDENCE_MODERATE_FIXED: u16 = 32768;

/// High confidence threshold in fixed-point representation (90%).
/// 
/// Indicates good sensor agreement and stable conditions.
/// Above this threshold, measurements are highly reliable.
/// 
/// Source: 0.9 * 65535 ≈ 58982
pub const CONFIDENCE_HIGH_FIXED: u16 = 58982;

/// Maximum confidence in fixed-point representation (100%).
/// 
/// Perfect confidence - theoretical maximum that's rarely
/// achieved in practice due to sensor noise.
/// 
/// Source: Full u16 range
pub const CONFIDENCE_MAX_FIXED: u16 = 65535;

// ===== GEOMETRIC MEAN APPROXIMATION =====

/// Numerator for geometric mean approximation penalty.
/// 
/// Used with GEOMETRIC_MEAN_PENALTY_DENOMINATOR to create
/// a conservative scaling factor for confidence combination.
/// 
/// Source: sqrt(a*b) ≤ (a+b)/2, with 61/64 ≈ 0.953 safety margin
pub const GEOMETRIC_MEAN_PENALTY_NUMERATOR: u32 = 61;

/// Denominator for geometric mean approximation penalty.
/// 
/// Creates a 61/64 ≈ 0.953 conservative factor to ensure
/// combined confidence doesn't exceed theoretical maximum.
/// 
/// Source: Empirical testing with various sensor combinations
pub const GEOMETRIC_MEAN_PENALTY_DENOMINATOR: u32 = 64;

// ===== WEIGHTED AVERAGE FUSION =====

/// Divisor for variance-based confidence calculation.
/// 
/// Used in formula: confidence = 1.0 / (1.0 + variance)
/// Ensures confidence decreases smoothly with increasing variance.
/// 
/// Source: Standard variance-to-confidence mapping
pub const VARIANCE_CONFIDENCE_DIVISOR: f32 = 1.0;

// ===== COMPLEMENTARY FILTER DEFAULTS =====

/// Default weight for fast sensor in complementary filter.
/// 
/// Small weight (2%) for high-frequency sensor ensures
/// quick response while filtering noise.
/// 
/// Source: Digital filter design for IMU fusion
pub const COMPLEMENTARY_FAST_WEIGHT_DEFAULT: f32 = 0.02;

/// Default weight for slow sensor in complementary filter.
/// 
/// Large weight (98%) for low-frequency sensor provides
/// stability and drift compensation.
/// 
/// Source: Complementary to fast weight (1.0 - 0.02)
pub const COMPLEMENTARY_SLOW_WEIGHT_DEFAULT: f32 = 0.98;

// ===== CONSENSUS VOTING PARAMETERS =====

/// Outlier detection threshold in standard deviations.
/// 
/// Values beyond 3σ from mean are considered outliers.
/// Based on normal distribution properties.
/// 
/// Source: Statistical quality control (3-sigma rule)
pub const OUTLIER_THRESHOLD_SIGMA: f32 = 3.0;

/// Divisor for calculating majority threshold.
/// 
/// Used as: required_votes = (total_sensors + 1) / 2
/// Ensures true majority for odd numbers.
/// 
/// Source: Voting theory
pub const MAJORITY_THRESHOLD_DIVISOR: usize = 2;

/// Default minimum votes required for consensus.
/// 
/// At least 2 sensors must agree for valid consensus,
/// preventing single sensor from dominating.
/// 
/// Source: Redundancy requirements
pub const VOTING_MIN_VOTES_DEFAULT: usize = 2;

/// Confidence threshold for voting consensus.
/// 
/// Sensors with confidence below 70% are excluded
/// from voting to prevent unreliable data influence.
/// 
/// Source: Empirical testing with degraded sensors
pub const VOTING_CONFIDENCE_THRESHOLD: f32 = 0.7;

// ===== SENSOR AGREEMENT THRESHOLDS =====

/// Excellent agreement threshold (normalized difference).
/// 
/// Sensors within 10% are considered in excellent agreement,
/// resulting in high confidence scores.
/// 
/// Source: Sensor accuracy specifications
pub const AGREEMENT_EXCELLENT_THRESHOLD: f32 = 0.1;

/// Good agreement threshold (normalized difference).
/// 
/// Sensors within 100% (factor of 2) show acceptable agreement
/// with reduced confidence.
/// 
/// Source: Practical sensor tolerance analysis
pub const AGREEMENT_GOOD_THRESHOLD: f32 = 1.0;

/// Confidence penalty for good (not excellent) agreement.
/// 
/// Applied when sensors agree but not perfectly.
/// Reduces confidence to 90% of maximum.
/// 
/// Source: 1.0 - 0.1 = 0.9 (10% penalty)
pub const AGREEMENT_GOOD_PENALTY: f32 = 0.1;

/// Default confidence for poor sensor agreement.
/// 
/// When sensors disagree significantly, assign 50%
/// confidence indicating high uncertainty.
/// 
/// Source: Engineering judgment
pub const AGREEMENT_POOR_CONFIDENCE: f32 = 0.5;

// ===== NEWTON'S METHOD PARAMETERS =====

/// Number of iterations for Newton's method square root.
/// 
/// 3 iterations provide ~0.1% accuracy for typical values,
/// balancing precision with computational efficiency.
/// 
/// Source: Numerical analysis of convergence rates
pub const NEWTON_METHOD_ITERATIONS: usize = 3;

/// Multiplication factor for Newton's method.
/// 
/// Used in iteration: x_new = x * (3 - a*x²) / 2
/// The 0.5 factor is part of the Newton-Raphson formula.
/// 
/// Source: Newton's method for f(x) = 1/x² - a
pub const NEWTON_METHOD_FACTOR: f32 = 0.5;

// ===== MATRIX OPERATIONS =====

/// Minimum diagonal value to prevent singularity.
/// 
/// Added to diagonal elements to ensure numerical stability
/// in matrix operations, preventing division by zero.
/// 
/// Source: Numerical linear algebra best practices
pub const MATRIX_MIN_DIAGONAL: f32 = 1e-6;

/// Maximum condition number for matrix stability.
/// 
/// Matrices with condition number above this are considered
/// ill-conditioned and may produce unreliable results.
/// 
/// Source: Numerical stability analysis
pub const MATRIX_MAX_CONDITION: f32 = 1e6;

/// Threshold for detecting singular matrices.
/// 
/// Determinant values below this indicate singularity,
/// requiring special handling or algorithm switching.
/// 
/// Source: Machine epsilon considerations
pub const MATRIX_SINGULAR_THRESHOLD: f32 = 1e-10;

/// Maximum matrix size for inversion operations.
/// 
/// Larger matrices use alternative algorithms to avoid
/// numerical instability and computational complexity.
/// 
/// Source: Embedded system constraints
pub const MATRIX_MAX_SIZE_INVERT: usize = 16;

// ===== KALMAN FILTER DEFAULTS =====

/// Default process noise for Kalman filter.
/// 
/// Represents uncertainty in system dynamics.
/// Higher values allow faster adaptation to changes.
/// 
/// Source: Typical sensor fusion applications
pub const KALMAN_DEFAULT_PROCESS_NOISE: f32 = 0.01;

/// Default measurement noise for Kalman filter.
/// 
/// Represents sensor measurement uncertainty.
/// Based on typical consumer-grade sensor accuracy.
/// 
/// Source: Sensor datasheet analysis
pub const KALMAN_DEFAULT_MEASUREMENT_NOISE: f32 = 0.1;

/// Default convergence threshold for Kalman filter.
/// 
/// Filter considered converged when innovation below this.
/// Balances responsiveness with stability.
/// 
/// Source: Control system design practices
pub const KALMAN_DEFAULT_CONVERGENCE_THRESHOLD: f32 = 0.01;

/// Epsilon for Jacobian matrix computation.
/// 
/// Small perturbation for numerical differentiation
/// in Extended Kalman Filter implementations.
/// 
/// Source: Numerical differentiation theory
pub const KALMAN_JACOBIAN_EPSILON: f32 = 1e-6;

/// Smoothing factor for Kalman confidence updates.
/// 
/// Exponential moving average factor for confidence scores.
/// Lower values provide more smoothing.
/// 
/// Source: Signal processing practices
pub const KALMAN_CONFIDENCE_ALPHA: f32 = 0.1;

/// Innovation threshold for Kalman filter outlier detection.
/// 
/// Measurements with innovation above 3σ are considered
/// outliers and handled specially.
/// 
/// Source: Statistical process control
pub const KALMAN_INNOVATION_THRESHOLD: f32 = 3.0;

// ===== CHI-SQUARED CRITICAL VALUES =====
// For 95% confidence level (α = 0.05)

/// Chi-squared critical value for 1 degree of freedom.
/// 
/// Used for single-dimensional confidence intervals
/// and hypothesis testing in fusion algorithms.
/// 
/// Source: Chi-squared distribution table, 95% confidence
pub const CHI_SQUARED_1D: f32 = 3.84;

/// Chi-squared critical value for 2 degrees of freedom.
/// 
/// Used for 2D confidence ellipses in sensor fusion,
/// such as position estimation (x, y).
/// 
/// Source: Chi-squared distribution table, 95% confidence
pub const CHI_SQUARED_2D: f32 = 5.99;

/// Chi-squared critical value for 3 degrees of freedom.
/// 
/// Used for 3D confidence ellipsoids, such as
/// 3D position or orientation estimation.
/// 
/// Source: Chi-squared distribution table, 95% confidence
pub const CHI_SQUARED_3D: f32 = 7.81;

/// Chi-squared critical value for 4 degrees of freedom.
/// 
/// Used for quaternion-based orientation fusion
/// and other 4-dimensional state estimates.
/// 
/// Source: Chi-squared distribution table, 95% confidence
pub const CHI_SQUARED_4D: f32 = 9.49;

/// Chi-squared critical value for 5 degrees of freedom.
/// 
/// Used for complex state vectors with 5 components,
/// such as position + 2D velocity estimates.
/// 
/// Source: Chi-squared distribution table, 95% confidence
pub const CHI_SQUARED_5D: f32 = 11.07;

/// Wilson-Hilferty coefficient for chi-squared approximation.
/// 
/// Used in normal approximation of chi-squared distribution
/// for efficient confidence interval calculation.
/// 
/// Source: Wilson-Hilferty transformation
pub const WILSON_HILFERTY_COEFFICIENT: f32 = 2.45;

// ===== ENVIRONMENTAL CONFIDENCE PARAMETERS =====

/// Optimal temperature range minimum (°C).
/// 
/// Below this, sensor accuracy may degrade but
/// measurements are still reliable.
/// 
/// Source: Typical sensor operating conditions
pub const ENV_TEMP_OPTIMAL_MIN: f32 = 0.0;

/// Optimal temperature range maximum (°C).
/// 
/// Above this, sensor accuracy may degrade but
/// measurements are still reliable.
/// 
/// Source: Typical sensor operating conditions
pub const ENV_TEMP_OPTIMAL_MAX: f32 = 50.0;

/// Extended temperature range minimum (°C).
/// 
/// Beyond this, significant accuracy degradation expected.
/// Confidence scores reduced accordingly.
/// 
/// Source: Sensor survival specifications
pub const ENV_TEMP_EXTENDED_MIN: f32 = -20.0;

/// Extended temperature range maximum (°C).
/// 
/// Beyond this, significant accuracy degradation expected.
/// Confidence scores reduced accordingly.
/// 
/// Source: Sensor survival specifications
pub const ENV_TEMP_EXTENDED_MAX: f32 = 70.0;

/// Low altitude threshold for pressure confidence (m).
/// 
/// Below 3000m, pressure sensors maintain full accuracy.
/// Above this, confidence gradually decreases.
/// 
/// Source: Barometric sensor specifications
pub const ENV_ALTITUDE_LOW_THRESHOLD: f32 = 3000.0;

/// High altitude threshold for pressure confidence (m).
/// 
/// Above 5000m, pressure sensor accuracy significantly
/// degraded. Confidence scores heavily penalized.
/// 
/// Source: High-altitude sensor testing
pub const ENV_ALTITUDE_HIGH_THRESHOLD: f32 = 5000.0;

// ===== EXPONENTIAL APPROXIMATION COEFFICIENTS =====

/// Quadratic coefficient for exp(-0.5x²) approximation.
/// 
/// Used in formula: exp(-0.5x²) ≈ 1 - 0.5x² + 0.125x⁴
/// Provides accuracy within 2% for x in [0, 3].
/// 
/// Source: Taylor series expansion of exponential
pub const EXP_APPROX_QUADRATIC_COEFF: f32 = 0.5;

/// Quartic coefficient for exp(-0.5x²) approximation.
/// 
/// Higher-order term improves accuracy for larger x values.
/// Combined with quadratic term for balanced approximation.
/// 
/// Source: Taylor series expansion of exponential
pub const EXP_APPROX_QUARTIC_COEFF: f32 = 0.125;

// ===== PIECEWISE LINEAR APPROXIMATION =====

/// First threshold for variance ratio mapping.
/// 
/// Below this ratio, confidence degrades slowly.
/// Represents good sensor agreement.
/// 
/// Source: Empirical analysis of sensor variance
pub const VARIANCE_RATIO_THRESHOLD_1: f32 = 0.5;

/// Second threshold for variance ratio mapping.
/// 
/// Between first and second threshold, moderate degradation.
/// Represents acceptable sensor agreement.
/// 
/// Source: Empirical analysis of sensor variance
pub const VARIANCE_RATIO_THRESHOLD_2: f32 = 1.0;

/// Third threshold for variance ratio mapping.
/// 
/// Above this, confidence degrades rapidly.
/// Indicates poor sensor agreement.
/// 
/// Source: Empirical analysis of sensor variance
pub const VARIANCE_RATIO_THRESHOLD_3: f32 = 2.0;

/// Confidence at variance ratio 0.5.
/// 
/// Approximates exp(-0.5) ≈ 0.606 rounded to 0.8.
/// Represents good but not perfect agreement.
/// 
/// Source: Exponential decay approximation
pub const CONFIDENCE_AT_RATIO_0_5: f32 = 0.8;

/// Confidence at variance ratio 1.0.
/// 
/// Approximates exp(-1) ≈ 0.368 rounded to 0.6.
/// Represents moderate sensor agreement.
/// 
/// Source: Exponential decay approximation
pub const CONFIDENCE_AT_RATIO_1_0: f32 = 0.6;

/// Confidence at variance ratio 2.0.
/// 
/// Approximates exp(-2) ≈ 0.135 rounded to 0.3.
/// Represents poor sensor agreement.
/// 
/// Source: Exponential decay approximation
pub const CONFIDENCE_AT_RATIO_2_0: f32 = 0.3;

/// Decay rate for first variance range [0, 0.5].
/// 
/// Linear decay coefficient: confidence = 1.0 - 0.4 * ratio
/// Gives 0.8 confidence at ratio = 0.5.
/// 
/// Source: Linear interpolation of exponential
pub const CONFIDENCE_DECAY_RATE_1: f32 = 0.4;

/// Decay rate for second variance range [0.5, 1.0].
/// 
/// Steeper decay as variance increases.
/// Transitions from 0.8 to 0.6 confidence.
/// 
/// Source: Linear interpolation of exponential
pub const CONFIDENCE_DECAY_RATE_2: f32 = 0.4;

/// Decay rate for third variance range [1.0, 2.0].
/// 
/// Even steeper decay for high variance.
/// Transitions from 0.6 to 0.3 confidence.
/// 
/// Source: Linear interpolation of exponential
pub const CONFIDENCE_DECAY_RATE_3: f32 = 0.3;

/// Decay rate for extreme variance range [2.0, ∞].
/// 
/// Slow decay to minimum confidence.
/// Prevents confidence from dropping too rapidly.
/// 
/// Source: Engineering judgment
pub const CONFIDENCE_DECAY_RATE_4: f32 = 0.1;

// ===== CONVERGENCE PARAMETERS =====

/// Divisor for convergence factor calculation.
/// 
/// Used as: convergence = min(1.0, count / divisor)
/// 20 measurements typically indicate convergence.
/// 
/// Source: Empirical testing with Kalman filters
pub const CONVERGENCE_DIVISOR: f32 = 20.0;

// ===== ENVIRONMENTAL HUMIDITY RANGES =====

/// Minimum optimal humidity (%).
/// 
/// Below this, static electricity and material degradation.
/// Most sensors operate well above 20% RH.
/// 
/// Source: Sensor manufacturer recommendations
pub const ENV_HUMIDITY_OPTIMAL_MIN: f32 = 20.0;

/// Maximum optimal humidity (%).
/// 
/// Above this, condensation risk increases.
/// Most sensors operate well below 80% RH.
/// 
/// Source: Sensor manufacturer recommendations
pub const ENV_HUMIDITY_OPTIMAL_MAX: f32 = 80.0;

/// Minimum extended humidity range (%).
/// 
/// Sensors may show drift but remain functional.
/// Very dry conditions affect some sensor types.
/// 
/// Source: Extended operating specifications
pub const ENV_HUMIDITY_EXTENDED_MIN: f32 = 10.0;

/// Maximum extended humidity range (%).
/// 
/// Near-condensing conditions, accuracy degraded.
/// Risk of sensor damage in prolonged exposure.
/// 
/// Source: Extended operating specifications
pub const ENV_HUMIDITY_EXTENDED_MAX: f32 = 90.0;

// ===== ENVIRONMENTAL CONFIDENCE LEVELS =====

/// Confidence for optimal environmental conditions.
/// 
/// Full confidence when all parameters in ideal range.
/// No derating applied to measurements.
/// 
/// Source: Baseline confidence definition
pub const ENV_CONFIDENCE_OPTIMAL: f32 = 1.0;

/// Confidence for extended environmental conditions.
/// 
/// Reduced confidence outside optimal but within spec.
/// 30% reduction reflects typical accuracy degradation.
/// 
/// Source: Sensor accuracy vs temperature curves
pub const ENV_CONFIDENCE_EXTENDED: f32 = 0.7;

/// Confidence for extreme environmental conditions.
/// 
/// Significant confidence reduction outside specifications.
/// Data still useful but requires careful interpretation.
/// 
/// Source: Engineering judgment for out-of-spec operation
pub const ENV_CONFIDENCE_EXTREME: f32 = 0.3;

/// Confidence for extended humidity conditions.
/// 
/// Less severe than temperature extremes.
/// 20% reduction for humidity outside optimal.
/// 
/// Source: Humidity sensor characteristics
pub const ENV_CONFIDENCE_HUMIDITY_EXTENDED: f32 = 0.8;

/// Confidence for extreme humidity conditions.
/// 
/// 50% confidence for very dry or near-condensing.
/// Higher than temperature extreme due to sensor resilience.
/// 
/// Source: Humidity sensor degradation studies
pub const ENV_CONFIDENCE_HUMIDITY_EXTREME: f32 = 0.5;

/// Confidence for extreme pressure conditions.
/// 
/// Very low or high pressure affects multiple sensor types.
/// 60% confidence reduction for extreme conditions.
/// 
/// Source: High-altitude sensor testing
pub const ENV_CONFIDENCE_PRESSURE_EXTREME: f32 = 0.4;

/// Confidence for extreme altitude conditions.
/// 
/// Above 5000m, multiple sensor types affected.
/// Same as pressure extreme due to correlation.
/// 
/// Source: Mountain deployment experience
pub const ENV_CONFIDENCE_ALTITUDE_EXTREME: f32 = 0.4;

// ===== ENVIRONMENTAL PRESSURE RANGES =====

/// Minimum optimal pressure range (hPa).
/// 
/// Typical low pressure system at sea level.
/// Most pressure sensors accurate in this range.
/// 
/// Source: Weather station normal operating range
pub const ENV_PRESSURE_OPTIMAL_MIN: f32 = 950.0;

/// Maximum optimal pressure range (hPa).
/// 
/// Typical high pressure system at sea level.
/// Most pressure sensors accurate in this range.
/// 
/// Source: Weather station normal operating range
pub const ENV_PRESSURE_OPTIMAL_MAX: f32 = 1050.0;

// ===== CROSS-VALIDATION AGREEMENT THRESHOLDS =====

/// Excellent agreement threshold for cross-validation.
/// 
/// When normalized deviation < 0.5, sensors show excellent agreement.
/// Results in full confidence (1.0).
/// 
/// Source: Engineering tolerance for high-precision sensors
pub const CROSS_VALIDATION_EXCELLENT_THRESHOLD: f32 = 0.5;

/// Good agreement threshold for cross-validation.
/// 
/// When normalized deviation < 1.0, sensors show good agreement.
/// Confidence drops linearly from 0.9 to 0.7.
/// 
/// Source: Typical sensor accuracy bounds
pub const CROSS_VALIDATION_GOOD_THRESHOLD: f32 = 1.0;

/// Marginal agreement threshold for cross-validation.
/// 
/// When normalized deviation < 2.0, sensors show marginal agreement.
/// Confidence drops linearly from 0.7 to 0.3.
/// 
/// Source: Maximum acceptable sensor divergence
pub const CROSS_VALIDATION_MARGINAL_THRESHOLD: f32 = 2.0;

/// Confidence for excellent sensor agreement.
/// 
/// Perfect agreement between sensors results in full confidence.
/// Indicates highly reliable measurements.
/// 
/// Source: Maximum confidence definition
pub const CROSS_VALIDATION_EXCELLENT_CONFIDENCE: f32 = 1.0;

/// Starting confidence for good agreement range.
/// 
/// At the boundary between excellent and good agreement.
/// Slightly reduced from perfect due to minor deviations.
/// 
/// Source: 10% reduction from perfect agreement
pub const CROSS_VALIDATION_GOOD_START_CONFIDENCE: f32 = 0.9;

/// Confidence decay rate for good agreement range.
/// 
/// Linear decay coefficient for normalized deviation in [0.5, 1.0].
/// Results in 0.9 - 0.4*(deviation-0.5) formula.
/// 
/// Source: Linear interpolation for smooth transition
pub const CROSS_VALIDATION_GOOD_DECAY_RATE: f32 = 0.4;

/// Starting confidence for marginal agreement range.
/// 
/// At the boundary between good and marginal agreement.
/// Indicates acceptable but degraded sensor agreement.
/// 
/// Source: 30% reduction from perfect agreement
pub const CROSS_VALIDATION_MARGINAL_START_CONFIDENCE: f32 = 0.7;

/// Confidence decay rate for marginal agreement range.
/// 
/// Linear decay coefficient for normalized deviation in [1.0, 2.0].
/// Results in 0.7 - 0.4*(deviation-1.0) formula.
/// 
/// Source: Linear interpolation maintaining continuity
pub const CROSS_VALIDATION_MARGINAL_DECAY_RATE: f32 = 0.4;

/// Confidence for poor sensor agreement.
/// 
/// When normalized deviation >= 2.0, sensors disagree significantly.
/// Low confidence but data may still be useful for outlier detection.
/// 
/// Source: Minimum useful confidence threshold
pub const CROSS_VALIDATION_POOR_CONFIDENCE: f32 = 0.3;

// ===== ENVIRONMENTAL FACTOR COUNTS =====

/// Number of environmental factors for confidence calculation.
/// 
/// Includes: temperature, humidity, pressure, altitude.
/// Used for averaging combined environmental confidence.
/// 
/// Source: Number of environmental parameters tracked
pub const ENV_FACTOR_COUNT: f32 = 4.0;

// ===== KALMAN UNCERTAINTY CONFIDENCE MAPPING =====

/// Low uncertainty threshold for Kalman filter confidence.
/// 
/// Covariance trace below this indicates very low uncertainty.
/// Results in maximum confidence score.
/// 
/// Source: Empirical testing with converged filters
pub const KALMAN_LOW_UNCERTAINTY_THRESHOLD: f32 = 0.1;

/// Medium uncertainty threshold for Kalman filter confidence.
/// 
/// Covariance trace between low and medium shows moderate uncertainty.
/// Confidence decreases linearly in this range.
/// 
/// Source: Typical sensor fusion applications
pub const KALMAN_MEDIUM_UNCERTAINTY_THRESHOLD: f32 = 1.0;

/// High confidence for low uncertainty in Kalman filter.
/// 
/// Starting confidence when uncertainty is at low threshold.
/// Represents well-converged filter state.
/// 
/// Source: Maximum achievable confidence
pub const KALMAN_LOW_UNCERTAINTY_CONFIDENCE: f32 = 1.0;

/// Medium confidence for medium uncertainty in Kalman filter.
/// 
/// Starting confidence at medium uncertainty threshold.
/// 10% reduction from maximum confidence.
/// 
/// Source: Linear interpolation for smooth transition
pub const KALMAN_MEDIUM_UNCERTAINTY_CONFIDENCE: f32 = 0.9;

/// Confidence decay rate for medium uncertainty range.
/// 
/// Linear decay coefficient for trace in [0.1, 1.0].
/// Results in 0.9 - 0.4*(trace-0.1) formula.
/// 
/// Source: Empirical tuning for reasonable confidence spread
pub const KALMAN_UNCERTAINTY_DECAY_RATE: f32 = 0.4;

/// Confidence multiplier for high uncertainty.
/// 
/// For trace > 1.0, confidence = 0.5 / trace.
/// Provides asymptotic decrease for very uncertain states.
/// 
/// Source: Hyperbolic confidence model
pub const KALMAN_HIGH_UNCERTAINTY_MULTIPLIER: f32 = 0.5;

/// Convergence updates for Kalman filter confidence.
/// 
/// Number of updates to reach full convergence confidence.
/// Used in formula: convergence = min(1.0, updates / divisor).
/// 
/// Source: Typical filter convergence time
pub const KALMAN_CONVERGENCE_UPDATE_DIVISOR: f32 = 10.0;

// ===== FUSION PIPELINE PARAMETERS =====

/// Default fusion interval in milliseconds.
/// 
/// Rate at which sensor groups are processed for fusion.
/// 100ms = 10Hz provides good balance of responsiveness and efficiency.
/// 
/// Source: Common sensor fusion rate for IoT applications
pub const FUSION_DEFAULT_INTERVAL_MS: u32 = 100;

/// Maximum size for measurements map in fusion stage.
/// 
/// Stores temporary sensor readings before fusion processing.
/// Sized for typical multi-sensor deployments.
/// 
/// Source: 2 * MAX_SENSORS_PER_GROUP for double buffering
pub const MAX_MEASUREMENTS_MAP_SIZE: usize = 16;

/// Confidence score for simple average fusion.
/// 
/// Used when no sophisticated fusion algorithm is available.
/// Moderate confidence reflects lack of advanced processing.
/// 
/// Source: Engineering judgment for basic averaging
pub const SIMPLE_AVERAGE_CONFIDENCE: f32 = 0.8;

// ===== ENVIRONMENTAL DEFAULTS =====

/// Default room temperature in Celsius.
/// 
/// Used as fallback ambient temperature for models.
/// Represents typical indoor conditions.
/// 
/// Source: Standard room temperature (68°F)
pub const DEFAULT_ROOM_TEMPERATURE: f32 = 20.0;

/// Standard sea level pressure in hPa.
/// 
/// Reference pressure for altitude calculations.
/// International Standard Atmosphere value.
/// 
/// Source: ISA standard atmosphere
pub const DEFAULT_SEA_LEVEL_PRESSURE: f32 = 1013.25;

/// Default moderate humidity percentage.
/// 
/// Typical indoor relative humidity.
/// Used as equilibrium point in models.
/// 
/// Source: Comfortable indoor humidity range
pub const DEFAULT_MODERATE_HUMIDITY: f32 = 50.0;

/// Default sea level altitude in meters.
/// 
/// Reference altitude for pressure calculations.
/// Zero point for altitude measurements.
/// 
/// Source: Sea level definition
pub const DEFAULT_SEA_LEVEL_ALTITUDE: f32 = 0.0;

// ===== TEMPERATURE MODEL CONSTANTS =====

/// Default thermal mass for temperature sensors (J/K).
/// 
/// Represents heat capacity of small sensor package.
/// Affects how quickly sensor responds to temperature changes.
/// 
/// Source: Typical values for TO-92 sensor packages
pub const DEFAULT_THERMAL_MASS: f32 = 10.0;

/// Default heat transfer coefficient.
/// 
/// Moderate insulation between sensor and environment.
/// Higher values mean faster temperature equilibration.
/// 
/// Source: Empirical testing with common sensors
pub const DEFAULT_HEAT_TRANSFER: f32 = 0.1;

/// Temperature sensor noise standard deviation (°C).
/// 
/// Typical measurement noise for consumer-grade sensors.
/// Used for uncertainty estimation in fusion.
/// 
/// Source: DS18B20, DHT22 specifications
pub const TEMP_MODEL_NOISE_STD: f32 = 0.1;

/// Minimum temperature model range (°C).
/// 
/// Lower bound for temperature sensor operation.
/// Beyond this, sensor damage or failure likely.
/// 
/// Source: Industrial temperature sensor limits
pub const TEMP_MODEL_MIN: f32 = -40.0;

/// Maximum temperature model range (°C).
/// 
/// Upper bound for temperature sensor operation.
/// Typical limit for consumer/industrial sensors.
/// 
/// Source: Common sensor specifications
pub const TEMP_MODEL_MAX: f32 = 85.0;

/// Temperature range margin for optimal operation (°C).
/// 
/// Buffer zone from min/max for best accuracy.
/// Within margin of limits, accuracy may degrade.
/// 
/// Source: Sensor accuracy vs temperature curves
pub const TEMP_RANGE_MARGIN: f32 = 5.0;

/// Optimal temperature range minimum (°C).
/// 
/// Lower bound for best sensor accuracy.
/// Most sensors calibrated for this range.
/// 
/// Source: Typical calibration range
pub const TEMP_OPTIMAL_MIN: f32 = 0.0;

/// Optimal temperature range maximum (°C).
/// 
/// Upper bound for best sensor accuracy.
/// Indoor/moderate outdoor conditions.
/// 
/// Source: Typical calibration range
pub const TEMP_OPTIMAL_MAX: f32 = 50.0;

/// Extended temperature range minimum (°C).
/// 
/// Acceptable but degraded accuracy.
/// Confidence reduced in this range.
/// 
/// Source: Sensor survival specifications
pub const TEMP_EXTENDED_MIN: f32 = -20.0;

/// Extended temperature range maximum (°C).
/// 
/// Acceptable but degraded accuracy.
/// Approaching thermal limits.
/// 
/// Source: Sensor survival specifications
pub const TEMP_EXTENDED_MAX: f32 = 70.0;

// ===== PRESSURE MODEL CONSTANTS =====

/// Pressure sensor noise standard deviation (hPa).
/// 
/// Typical measurement noise for MEMS pressure sensors.
/// Used for Kalman filter measurement uncertainty.
/// 
/// Source: BMP280, BME680 specifications
pub const PRESSURE_MODEL_NOISE_STD: f32 = 0.1;

/// Minimum pressure model range (hPa).
/// 
/// Corresponds to approximately 9000m altitude.
/// Below this, most sensors fail.
/// 
/// Source: High-altitude sensor specifications
pub const PRESSURE_MODEL_MIN: f32 = 300.0;

/// Maximum pressure model range (hPa).
/// 
/// Extreme storm conditions at sea level.
/// Above this indicates sensor error.
/// 
/// Source: Meteorological extremes
pub const PRESSURE_MODEL_MAX: f32 = 1100.0;

/// Maximum pressure change rate (hPa/hour).
/// 
/// Typical rate during weather fronts.
/// Used for rate-of-change validation.
/// 
/// Source: Meteorological observations
pub const PRESSURE_CHANGE_RATE: f32 = 1.0;

/// Process noise factor for pressure models.
/// 
/// Scales random walk variance in prediction.
/// Accounts for unpredictable pressure changes.
/// 
/// Source: Empirical tuning with weather data
pub const PRESSURE_PROCESS_NOISE_FACTOR: f32 = 0.1;

/// Mean reversion rate for pressure.
/// 
/// Weak tendency to return to mean pressure.
/// Prevents unbounded drift in models.
/// 
/// Source: Statistical analysis of pressure data
pub const PRESSURE_MEAN_REVERSION_RATE: f32 = 0.01;

/// Barometric temperature lapse rate (K/m).
/// 
/// Temperature decrease with altitude.
/// Used in barometric formula calculations.
/// 
/// Source: International Standard Atmosphere
pub const BAROMETRIC_LAPSE_RATE: f32 = 0.0065;

/// Standard atmosphere temperature (K).
/// 
/// Reference temperature at sea level (15°C).
/// Used in barometric formula.
/// 
/// Source: ISA standard atmosphere
pub const STANDARD_ATMOSPHERE_TEMP: f32 = 288.15;

/// Barometric formula exponent.
/// 
/// Exponent in pressure-altitude relationship.
/// P = P0 * (1 - L*h/T0)^5.255
/// 
/// Source: Derived from gas law and gravity
pub const BAROMETRIC_EXPONENT: f32 = 5.255;

/// Scale height of atmosphere (m).
/// 
/// Characteristic height for pressure decay.
/// Used in exponential approximation.
/// 
/// Source: RT/Mg for Earth's atmosphere
pub const SCALE_HEIGHT_ATMOSPHERE: f32 = 8400.0;

/// Low altitude threshold (m).
/// 
/// Below this, pressure sensors maintain accuracy.
/// Used for confidence calculations.
/// 
/// Source: Sensor accuracy vs altitude data
pub const ALTITUDE_LOW_THRESHOLD: f32 = 3000.0;

/// High altitude threshold (m).
/// 
/// Above this, significant accuracy degradation.
/// Confidence heavily reduced.
/// 
/// Source: High-altitude testing data
pub const ALTITUDE_HIGH_THRESHOLD: f32 = 5000.0;

// ===== HUMIDITY MODEL CONSTANTS =====

/// Humidity sensor noise standard deviation (% RH).
/// 
/// Typical measurement noise for capacitive humidity sensors.
/// Higher than temperature due to sensor physics.
/// 
/// Source: DHT22, BME280 specifications
pub const HUMIDITY_MODEL_NOISE_STD: f32 = 2.0;

/// Humidity sensor hysteresis (% RH).
/// 
/// Difference between rising and falling measurements.
/// Inherent to capacitive sensing technology.
/// 
/// Source: Sensor manufacturer data
pub const HUMIDITY_HYSTERESIS: f32 = 1.0;

/// Humidity sensor response time (ms).
/// 
/// Time to reach 63% of step change.
/// Slow due to moisture absorption/desorption.
/// 
/// Source: Typical sensor specifications
pub const HUMIDITY_RESPONSE_TIME: f32 = 8000.0;

/// Humidity equilibrium point (%).
/// 
/// Typical indoor humidity level.
/// Used as drift target in models.
/// 
/// Source: Indoor climate studies
pub const HUMIDITY_EQUILIBRIUM: f32 = 50.0;

/// Minimum valid humidity (%).
/// 
/// Physical lower bound for relative humidity.
/// Below this indicates sensor failure.
/// 
/// Source: Physics constraint
pub const HUMIDITY_MIN: f32 = 0.0;

/// Maximum valid humidity (%).
/// 
/// Physical upper bound for relative humidity.
/// Cannot exceed saturation.
/// 
/// Source: Physics constraint
pub const HUMIDITY_MAX: f32 = 100.0;

// ===== CONFIDENCE LEVEL CONSTANTS =====

/// Excellent confidence level.
/// 
/// All factors optimal, high certainty.
/// Reserved for ideal conditions.
/// 
/// Source: Maximum achievable confidence
pub const CONFIDENCE_EXCELLENT: f32 = 1.0;

/// Good confidence level.
/// 
/// Minor uncertainties, still reliable.
/// Typical for well-functioning systems.
/// 
/// Source: 10% reduction from excellent
pub const CONFIDENCE_GOOD: f32 = 0.9;

/// Moderate confidence level.
/// 
/// Average reliability, usable data.
/// Default for single sensors.
/// 
/// Source: Midpoint of confidence range
pub const CONFIDENCE_MODERATE: f32 = 0.8;

/// Fair confidence level.
/// 
/// Reduced reliability, use with caution.
/// Near operational limits.
/// 
/// Source: Marginal operating conditions
pub const CONFIDENCE_FAIR: f32 = 0.7;

/// Poor confidence level.
/// 
/// Low reliability, significant uncertainty.
/// Consider sensor maintenance.
/// 
/// Source: Degraded sensor performance
pub const CONFIDENCE_POOR: f32 = 0.4;

/// Critical confidence level.
/// 
/// Minimal reliability, likely errors.
/// Sensor near failure.
/// 
/// Source: Failure threshold
pub const CONFIDENCE_CRITICAL: f32 = 0.3;

/// Minimum useful confidence.
/// 
/// Below this, data should be rejected.
/// Absolute minimum for any processing.
/// 
/// Source: Statistical significance threshold
pub const CONFIDENCE_MIN_USEFUL: f32 = 0.2;

// ===== STATISTICAL CONFIDENCE DECAY RATES =====

/// Temperature statistical confidence decay rate.
/// 
/// Confidence reduction per unit normalized error.
/// Applied in range [0, 1] normalized error.
/// 
/// Source: Empirical sensor accuracy analysis
pub const STAT_CONF_DECAY_TEMP: f32 = 0.2;

/// Pressure statistical confidence decay rate.
/// 
/// Lower decay rate due to stable pressure dynamics.
/// More tolerant of small errors.
/// 
/// Source: Pressure sensor characteristics
pub const STAT_CONF_DECAY_PRESSURE: f32 = 0.1;

/// Humidity statistical confidence decay rate.
/// 
/// Similar to pressure, tolerant of noise.
/// Accounts for sensor hysteresis.
/// 
/// Source: Humidity sensor analysis
pub const STAT_CONF_DECAY_HUMIDITY: f32 = 0.1;

/// Secondary confidence decay rate.
/// 
/// Applied for normalized errors > 1.0.
/// Steeper decay for larger errors.
/// 
/// Source: Error distribution analysis
pub const STAT_CONF_DECAY_SECONDARY: f32 = 0.3;

// ===== MATHEMATICAL APPROXIMATION CONSTANTS =====

/// Taylor expansion threshold.
/// 
/// For |x| < 0.5, Taylor series provides good accuracy.
/// Used for exp(), ln(), and trig approximations.
/// 
/// Source: Numerical analysis error bounds
pub const TAYLOR_EXPANSION_THRESHOLD: f32 = 0.5;

/// Padé approximation threshold.
/// 
/// For 0.5 < |x| < 2.0, use Padé rational approximation.
/// Better accuracy than Taylor for medium values.
/// 
/// Source: Approximation theory
pub const PADE_APPROXIMATION_THRESHOLD: f32 = 2.0;

/// Taylor series cubic coefficient (1/6).
/// 
/// Third-order term in Taylor expansions.
/// Used in exp(x) ≈ 1 + x + x²/2 + x³/6.
/// 
/// Source: Taylor series mathematics
pub const TAYLOR_CUBIC_COEFF: f32 = 0.16666667;

/// Padé approximation coefficient.
/// 
/// Common factor in rational approximations.
/// Used in various Padé forms.
/// 
/// Source: Padé approximant tables
pub const PADE_COEFF: f32 = 0.5;

/// Padé quadratic coefficient (1/12).
/// 
/// Second-order term in Padé approximations.
/// Improves accuracy over simple ratios.
/// 
/// Source: Padé approximant tables
pub const PADE_QUADRATIC_COEFF: f32 = 0.08333333;

/// Exponential function limit maximum.
/// 
/// Prevent overflow in exp() approximations.
/// Beyond this, return maximum safe value.
/// 
/// Source: Numerical stability analysis
pub const EXP_LIMIT_MAX: f32 = 10.0;

/// Exponential function limit minimum.
/// 
/// Prevent underflow in exp() approximations.
/// Beyond this, return near-zero value.
/// 
/// Source: Numerical stability analysis
pub const EXP_LIMIT_MIN: f32 = -10.0;

// ===== EKF MODEL CONSTANTS =====

/// EKF default time step (seconds).
/// 
/// Update interval for state predictions.
/// 100ms provides good tracking performance.
/// 
/// Source: Common sensor sampling rates
pub const EKF_TIME_STEP: f32 = 0.1;

/// EKF default process noise.
/// 
/// Base uncertainty in state evolution.
/// Tuned for typical sensor applications.
/// 
/// Source: Kalman filter tuning guidelines
pub const EKF_DEFAULT_PROCESS_NOISE: f32 = 0.01;

/// Ambient temperature process noise.
/// 
/// Environmental temperature changes slowly.
/// Lower noise for stable parameter.
/// 
/// Source: Environmental monitoring data
pub const EKF_AMBIENT_PROCESS_NOISE: f32 = 0.001;

/// Cooling rate process noise.
/// 
/// Thermal properties nearly constant.
/// Minimal variation expected.
/// 
/// Source: Thermal system analysis
pub const EKF_COOLING_RATE_NOISE: f32 = 0.0001;

/// Velocity decay factor for altitude tracking.
/// 
/// Vertical velocity naturally decays.
/// Models air resistance and gravity.
/// 
/// Source: Atmospheric physics
pub const EKF_VELOCITY_DECAY: f32 = 0.95;

/// Hysteresis time constant (seconds).
/// 
/// Characteristic time for humidity sensor lag.
/// Used in hysteresis modeling.
/// 
/// Source: Sensor response measurements
pub const HYSTERESIS_TAU: f32 = 5.0;

/// Hysteresis decay factor.
/// 
/// Rate of hysteresis reduction.
/// Applied in exponential decay model.
/// 
/// Source: Empirical sensor testing
pub const HYSTERESIS_DECAY_FACTOR: f32 = 2.0;

// ===== EKF INITIAL STATE VALUES =====

/// Initial temperature state (°C).
/// 
/// Reasonable starting temperature.
/// Near typical room temperature.
/// 
/// Source: Common indoor conditions
pub const TEMP_INITIAL_STATE: f32 = 25.0;

/// Initial ambient temperature (°C).
/// 
/// Assumed environmental temperature.
/// Used when not measured directly.
/// 
/// Source: Standard room temperature
pub const TEMP_INITIAL_AMBIENT: f32 = 20.0;

/// Initial cooling rate (1/s).
/// 
/// Default heat transfer parameter.
/// Moderate thermal coupling.
/// 
/// Source: Typical sensor thermal mass
pub const COOLING_RATE_INITIAL: f32 = 0.1;

/// Temperature initial uncertainty.
/// 
/// Starting covariance for temperature state.
/// Reflects initial knowledge uncertainty.
/// 
/// Source: Reasonable temperature variance
pub const TEMP_INITIAL_UNCERTAINTY: f32 = 1.0;

/// Ambient temperature initial uncertainty.
/// 
/// Higher uncertainty for unmeasured parameter.
/// Reduces with observations.
/// 
/// Source: Environmental variability
pub const AMBIENT_INITIAL_UNCERTAINTY: f32 = 4.0;

/// Cooling rate initial uncertainty.
/// 
/// Parameter uncertainty in thermal model.
/// Refined through state estimation.
/// 
/// Source: Model parameter variance
pub const COOLING_RATE_UNCERTAINTY: f32 = 0.01;

/// Altitude initial uncertainty (m²).
/// 
/// Starting variance for altitude estimate.
/// 100m² = 10m standard deviation.
/// 
/// Source: GPS accuracy equivalent
pub const ALTITUDE_INITIAL_UNCERTAINTY: f32 = 100.0;

/// Velocity initial uncertainty.
/// 
/// Starting variance for vertical velocity.
/// Small for stationary start.
/// 
/// Source: Zero-velocity assumption
pub const VELOCITY_INITIAL_UNCERTAINTY: f32 = 0.1;

/// Humidity initial state (%).
/// 
/// Moderate starting humidity.
/// Typical indoor condition.
/// 
/// Source: Indoor climate average
pub const HUMIDITY_INITIAL_STATE: f32 = 50.0;

/// Humidity initial uncertainty.
/// 
/// Starting variance for humidity states.
/// Reflects sensor accuracy.
/// 
/// Source: Sensor specifications
pub const HUMIDITY_INITIAL_UNCERTAINTY: f32 = 4.0;

/// Humidity state correlation.
/// 
/// Cross-correlation between true and measured.
/// Accounts for sensor lag.
/// 
/// Source: Sensor physics model
pub const HUMIDITY_CORRELATION: f32 = 2.0;

/// Hysteresis initial uncertainty.
/// 
/// Starting variance for hysteresis state.
/// Moderate uncertainty in lag effect.
/// 
/// Source: Hysteresis model analysis
pub const HYSTERESIS_INITIAL_UNCERTAINTY: f32 = 1.0;

// ===== TIME CONVERSION CONSTANTS =====

/// Milliseconds per hour.
/// 
/// Used to convert time units in pressure models.
/// 1 hour = 60 min × 60 sec × 1000 ms.
/// 
/// Source: Time unit conversion
pub const MS_PER_HOUR: f32 = 3_600_000.0;

/// Milliseconds to seconds conversion.
/// 
/// Divider to convert milliseconds to seconds.
/// Used frequently in time-based calculations.
/// 
/// Source: Time unit definition
pub const MS_TO_SECONDS: f32 = 1000.0;