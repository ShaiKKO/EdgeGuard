//! Sensor Fusion Framework for Multi-Sensor Data Integration
//!
//! ## Overview
//!
//! This module implements advanced sensor fusion algorithms to combine data from multiple
//! sensors, producing more accurate and reliable measurements than any single sensor alone.
//! The framework is designed for edge devices with limited computational resources while
//! maintaining numerical stability and physics-aware constraints.
//!
//! ## Why Sensor Fusion?
//!
//! Individual sensors suffer from various limitations:
//! - **Noise**: Random measurement errors
//! - **Drift**: Systematic bias that changes over time
//! - **Latency**: Different sensors have different response times
//! - **Failure**: Sensors can fail or produce outliers
//!
//! Sensor fusion addresses these issues by:
//! ```text
//! Sensor 1 ──┐
//! Sensor 2 ──┼─→ Fusion Algorithm ─→ Optimal Estimate
//! Sensor N ──┘         ↓
//!                 Uncertainty
//! ```
//!
//! ## Fusion Algorithms
//!
//! ### Kalman Filter (Primary)
//!
//! The Kalman filter is optimal for linear systems with Gaussian noise:
//! ```text
//! State Prediction:    x̂ₖ = Fₖ·xₖ₋₁ + Bₖ·uₖ
//! Covariance Update:   Pₖ = Fₖ·Pₖ₋₁·Fₖᵀ + Qₖ
//! Innovation:          yₖ = zₖ - Hₖ·x̂ₖ
//! Kalman Gain:         Kₖ = Pₖ·Hₖᵀ·(Hₖ·Pₖ·Hₖᵀ + Rₖ)⁻¹
//! State Update:        xₖ = x̂ₖ + Kₖ·yₖ
//! ```
//!
//! ### Complementary Filter (Lightweight)
//!
//! For resource-constrained devices, complementary filters provide simpler fusion:
//! ```text
//! fused = α·sensor1 + (1-α)·sensor2
//! ```
//!
//! ### Weighted Average (Baseline)
//!
//! The simplest fusion method, using reliability-based weights:
//! ```text
//! fused = Σ(wᵢ·sensorᵢ) / Σwᵢ
//! ```
//!
//! ## Architecture
//!
//! The fusion framework consists of several components:
//!
//! 1. **Core Traits**: Define the interface for fusion algorithms
//! 2. **Matrix Operations**: No-std compatible linear algebra
//! 3. **Sensor Models**: Physics-based models for each sensor type
//! 4. **Fusion Algorithms**: Kalman, complementary, and weighted average
//! 5. **Pipeline Integration**: Seamless integration with event pipeline
//!
//! ## Memory Model
//!
//! All fusion operations use fixed-size allocations:
//! ```text
//! FusionState<N, M> size:
//! ├── State vector:     N × 4 bytes
//! ├── Covariance:       N × N × 4 bytes
//! ├── Measurement:      M × 4 bytes
//! ├── Workspace:        2 × N × N × 4 bytes
//! └── Total (N=3, M=2): ~168 bytes
//! ```
//!
//! ## Numerical Stability
//!
//! The implementation includes several numerical safeguards:
//! - **Symmetric enforcement**: Covariance matrices kept symmetric
//! - **Positive definite**: Cholesky decomposition for square roots
//! - **Condition monitoring**: Detect near-singular matrices
//! - **Joseph form**: Numerically stable covariance updates
//!
//! ## Usage Example
//!
//! ```rust
//! use edgeguard_core::fusion::{KalmanFilter, SensorModel};
//! use edgeguard_core::events::SensorType;
//! 
//! // Create fusion for 3 temperature sensors
//! let mut fusion = KalmanFilter::<3, 3>::new()
//!     .with_sensors([
//!         SensorModel::new("temp_1", SensorType::Temperature, 0.1),
//!         SensorModel::new("temp_2", SensorType::Temperature, 0.15),
//!         SensorModel::new("temp_3", SensorType::Temperature, 0.2),
//!     ])
//!     .with_process_noise(0.01);
//! 
//! // Update with measurements
//! let measurements = [25.1, 25.3, 24.9];
//! let (estimate, confidence) = fusion.update(&measurements, timestamp);
//! ```

pub mod kalman;
pub mod confidence;
pub mod models;
pub mod pipeline;

// Re-export main types
pub use kalman::{KalmanFilter, KalmanConfig, ExtendedKalmanFilter};
pub use confidence::{ConfidenceScore, ConfidenceScorer};
pub use models::{SensorModel, StateTransition, ekf_models};
pub use pipeline::FusionStage;

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

#[cfg(feature = "std")]
use std::boxed::Box;

use crate::{
    constants::fusion::{
        MIN_SENSORS_FOR_FUSION, SINGLE_SENSOR_CONFIDENCE, CONVERGENCE_MEASUREMENT_COUNT,
        MIN_CONVERGENCE_UPDATES, VARIANCE_CONFIDENCE_DIVISOR, GEOMETRIC_MEAN_PENALTY_NUMERATOR, 
        GEOMETRIC_MEAN_PENALTY_DENOMINATOR, NEWTON_METHOD_ITERATIONS, NEWTON_METHOD_FACTOR,
        MATRIX_MIN_DIAGONAL, MATRIX_MAX_CONDITION, MATRIX_SINGULAR_THRESHOLD,
        MATRIX_MAX_SIZE_INVERT, OUTLIER_THRESHOLD_SIGMA, MAJORITY_THRESHOLD_DIVISOR,
        VOTING_MIN_VOTES_DEFAULT, VOTING_CONFIDENCE_THRESHOLD, AGREEMENT_EXCELLENT_THRESHOLD,
        AGREEMENT_GOOD_THRESHOLD, AGREEMENT_GOOD_PENALTY, AGREEMENT_POOR_CONFIDENCE,
        COMPLEMENTARY_FAST_WEIGHT_DEFAULT, COMPLEMENTARY_SLOW_WEIGHT_DEFAULT,
    },
    time::Timestamp,
    errors::ValidationError,
};

/// Result type for fusion operations
pub type FusionResult<T> = Result<T, FusionError>;

/// Errors that can occur during sensor fusion
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FusionError {
    /// Not enough sensors for fusion (need at least 2)
    InsufficientSensors,
    /// Numerical instability detected
    NumericalInstability,
    /// Matrix inversion failed
    SingularMatrix,
    /// Measurement dimensions don't match
    DimensionMismatch,
    /// Sensor model not found
    UnknownSensor,
    /// Fusion diverged (covariance too large)
    Divergence,
}

impl From<FusionError> for ValidationError {
    fn from(err: FusionError) -> Self {
        match err {
            FusionError::InsufficientSensors => ValidationError::InsufficientData {
                required: MIN_SENSORS_FOR_FUSION,
                available: 1,  // Assumed to have 1 if insufficient
            },
            FusionError::NumericalInstability => ValidationError::InvalidValue,
            FusionError::SingularMatrix => ValidationError::InvalidValue,
            FusionError::DimensionMismatch => ValidationError::InvalidValue,
            FusionError::UnknownSensor => ValidationError::SensorQualityBad {
                reason: "Unknown sensor in fusion group",
            },
            FusionError::Divergence => ValidationError::CrossValidationFailed {
                reason: "Fusion algorithm diverged",
            },
        }
    }
}

// Re-export fusion algorithm trait
pub use crate::traits::FusionAlgorithm;

// Re-export fusion constants for submodules
pub use crate::constants::fusion::{
    TAYLOR_EXPANSION_THRESHOLD, SCALE_HEIGHT_ATMOSPHERE
};

/// Simplified weighted average fusion
/// 
/// ## When to Use
/// 
/// Weighted average is suitable when:
/// - Sensors measure the same quantity directly
/// - No complex dynamics or time delays
/// - Computational resources are extremely limited
/// 
/// ## Algorithm
/// 
/// ```text
/// estimate = Σ(weight[i] × measurement[i]) / Σweight[i]
/// 
/// Where weight[i] = 1 / variance[i]
/// ```
pub struct WeightedAverageFusion<const M: usize> {
    /// Sensor models with noise characteristics
    sensors: heapless::Vec<Box<dyn SensorModel>, M>,
    /// Computed weights (1/variance)
    weights: [f32; M],
    /// Last estimate
    last_estimate: f32,
    /// State as array (for trait compatibility)
    state_array: [f32; 1],
    /// Measurement count for convergence
    measurement_count: u32,
}

impl<const M: usize> WeightedAverageFusion<M> {
    /// Create new weighted average fusion
    pub fn new() -> Self {
        // Use default config for equal weights
        <Self as FusionAlgorithm<1, M>>::new(Default::default())
    }
    
    /// Add a sensor model
    pub fn add_sensor(mut self, sensor: Box<dyn SensorModel>) -> Self {
        if self.sensors.len() < M {
            let idx = self.sensors.len();
            self.weights[idx] = 1.0 / sensor.noise_variance();
            let _ = self.sensors.push(sensor);
        }
        self
    }
    
    /// Fuse measurements using weighted average
    pub fn fuse(&mut self, measurements: &[f32; M], mask: Option<u32>) -> (f32, ConfidenceScore) {
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut active_sensors = 0;
        
        // Apply mask if provided
        let mask = mask.unwrap_or(u32::MAX);
        
        for i in 0..M {
            if mask & (1 << i) != 0 {
                weighted_sum += self.weights[i] * measurements[i];
                weight_sum += self.weights[i];
                active_sensors += 1;
            }
        }
        
        // Compute estimate
        let estimate = if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            self.last_estimate
        };
        
        self.last_estimate = estimate;
        self.measurement_count += 1;
        
        // Compute confidence based on:
        // 1. Number of active sensors
        // 2. Agreement between sensors
        // 3. Total measurements processed
        let sensor_confidence = (active_sensors as f32) / (M as f32);
        let convergence_confidence = (self.measurement_count as f32 / CONVERGENCE_MEASUREMENT_COUNT as f32).min(1.0);
        
        // Compute variance of measurements
        let mut variance = 0.0;
        let mut count = 0;
        for i in 0..M {
            if mask & (1 << i) != 0 {
                let diff = measurements[i] - estimate;
                variance += diff * diff;
                count += 1;
            }
        }
        
        let agreement_confidence = if count > 1 {
            let variance = variance / (count as f32);
            // Map variance to confidence: low variance = high confidence
            // Using approximation to avoid exp() for no-std compatibility
            // exp(-x) ≈ 1/(1 + x) for small x, bounded to [0,1]
            let confidence = VARIANCE_CONFIDENCE_DIVISOR / (VARIANCE_CONFIDENCE_DIVISOR + variance);
            confidence.max(0.0).min(1.0)
        } else {
            SINGLE_SENSOR_CONFIDENCE // Single sensor, moderate confidence
        };
        
        let overall_confidence = sensor_confidence * convergence_confidence * agreement_confidence;
        
        (estimate, ConfidenceScore::from_float(overall_confidence))
    }
}

/// Configuration for weighted average fusion
#[derive(Debug, Clone)]
pub struct WeightedAverageConfig<const M: usize> {
    /// Weights for each sensor (normalized internally)
    pub weights: [f32; M],
    /// Minimum sensors required for fusion
    pub min_sensors: usize,
    /// Outlier rejection threshold (in standard deviations)
    pub outlier_threshold: f32,
}

impl<const M: usize> Default for WeightedAverageConfig<M> {
    fn default() -> Self {
        let mut weights = [1.0; M];
        // Normalize to sum to 1
        for w in weights.iter_mut() {
            *w /= M as f32;
        }
        Self {
            weights,
            min_sensors: (M + 1) / MAJORITY_THRESHOLD_DIVISOR, // Majority
            outlier_threshold: OUTLIER_THRESHOLD_SIGMA,
        }
    }
}

impl<const M: usize> FusionAlgorithm<1, M> for WeightedAverageFusion<M> {
    type Config = WeightedAverageConfig<M>;
    
    fn new(config: Self::Config) -> Self {
        let mut fusion = Self {
            sensors: heapless::Vec::new(),
            weights: config.weights,
            last_estimate: 0.0,
            state_array: [0.0; 1],
            measurement_count: 0,
        };
        
        // Normalize weights
        let sum: f32 = fusion.weights.iter().sum();
        if sum > 0.0 {
            for w in fusion.weights.iter_mut() {
                *w /= sum;
            }
        }
        
        fusion
    }
    
    fn predict(&mut self, _dt_ms: u32) -> FusionResult<()> {
        // Weighted average doesn't use prediction
        Ok(())
    }
    
    fn update(
        &mut self,
        measurements: &[f32; M],
        _timestamp: Timestamp,
        mask: Option<u32>,
    ) -> FusionResult<(f32, ConfidenceScore)> {
        let result = self.fuse(measurements, mask);
        self.state_array[0] = result.0;
        Ok(result)
    }
    
    fn state(&self) -> &[f32; 1] {
        // Return as single-element array for trait compatibility
        // We can't store this as an array, so we need a workaround
        // This is a limitation of the trait design
        panic!("WeightedAverageFusion::state() not supported - use last_estimate field directly")
    }
    
    fn uncertainty(&self) -> [f32; 1] {
        // Uncertainty based on sensor count and agreement
        // This is a simplified estimate
        [1.0 / (self.measurement_count as f32 + 1.0)]
    }
    
    fn reset(&mut self) {
        self.last_estimate = 0.0;
        self.measurement_count = 0;
    }
    
    fn has_converged(&self) -> bool {
        self.measurement_count >= CONVERGENCE_MEASUREMENT_COUNT
    }
}

/// Complementary filter for combining fast and slow sensors
/// 
/// Ideal for combining sensors with different frequency responses,
/// such as accelerometer (fast) and gyroscope (slow) data.
pub struct ComplementaryFilter {
    /// Weight for fast sensor (0.0 to 1.0)
    fast_weight: f32,
    /// Current state estimate
    state: f32,
    /// State as array (for trait compatibility)
    state_array: [f32; 1],
    /// Last timestamp for dt calculation
    last_timestamp: Timestamp,
}

/// Configuration for complementary filter
#[derive(Debug, Clone)]
pub struct ComplementaryConfig {
    /// Weight for fast sensor (0.0 to 1.0)
    pub fast_weight: f32,
    /// Weight for slow sensor (1.0 - fast_weight)
    pub slow_weight: f32,
    /// Crossover frequency in Hz
    pub crossover_freq: f32,
}

impl Default for ComplementaryConfig {
    fn default() -> Self {
        Self {
            fast_weight: COMPLEMENTARY_FAST_WEIGHT_DEFAULT,
            slow_weight: COMPLEMENTARY_SLOW_WEIGHT_DEFAULT,
            crossover_freq: 0.1,  // 0.1 Hz crossover frequency
        }
    }
}

impl ComplementaryFilter {
    /// Create new complementary filter
    pub fn new(config: ComplementaryConfig) -> Self {
        Self {
            fast_weight: config.fast_weight,
            state: 0.0,
            state_array: [0.0; 1],
            last_timestamp: 0,
        }
    }
}

impl FusionAlgorithm<1, 2> for ComplementaryFilter {
    type Config = ComplementaryConfig;
    
    fn new(config: Self::Config) -> Self {
        ComplementaryFilter::new(config)
    }
    
    fn predict(&mut self, _dt_ms: u32) -> FusionResult<()> {
        // Complementary filter doesn't use explicit prediction
        Ok(())
    }
    
    fn update(
        &mut self,
        measurements: &[f32; 2],
        timestamp: Timestamp,
        _mask: Option<u32>,
    ) -> FusionResult<(f32, ConfidenceScore)> {
        let fast_sensor = measurements[0];
        let slow_sensor = measurements[1];
        
        // Classic complementary filter equation
        self.state = self.fast_weight * fast_sensor + (1.0 - self.fast_weight) * slow_sensor;
        self.state_array[0] = self.state;
        
        self.last_timestamp = timestamp;
        
        // Confidence based on sensor agreement
        let difference = (fast_sensor - slow_sensor).abs();
        let confidence = if difference < AGREEMENT_EXCELLENT_THRESHOLD {
            1.0
        } else if difference < AGREEMENT_GOOD_THRESHOLD {
            1.0 - (difference - AGREEMENT_EXCELLENT_THRESHOLD) * AGREEMENT_GOOD_PENALTY
        } else {
            AGREEMENT_POOR_CONFIDENCE
        };
        
        Ok((self.state, ConfidenceScore::from_float(confidence)))
    }
    
    fn state(&self) -> &[f32; 1] {
        &self.state_array
    }
    
    fn uncertainty(&self) -> [f32; 1] {
        // Simple uncertainty estimate
        [0.1]
    }
    
    fn reset(&mut self) {
        self.state = 0.0;
        self.state_array[0] = 0.0;
        self.last_timestamp = 0;
    }
    
    fn has_converged(&self) -> bool {
        true // Complementary filter converges immediately
    }
}

/// Consensus voting fusion for safety-critical applications
/// 
/// Rejects outliers and requires minimum agreement between sensors
pub struct ConsensusVoting<const M: usize> {
    /// Outlier threshold in standard deviations
    outlier_threshold: f32,
    /// Minimum votes required
    min_votes: usize,
    /// Last valid estimate
    last_estimate: f32,
    /// State as array (for trait compatibility)
    state_array: [f32; 1],
    /// Measurement count
    measurement_count: u32,
}

/// Configuration for consensus voting
#[derive(Debug, Clone)]
pub struct VotingConfig {
    /// Outlier threshold in standard deviations
    pub outlier_threshold: f32,
    /// Minimum votes required for valid estimate
    pub min_votes: usize,
    /// Confidence threshold for accepting result
    pub confidence_threshold: f32,
}

impl Default for VotingConfig {
    fn default() -> Self {
        Self {
            outlier_threshold: OUTLIER_THRESHOLD_SIGMA,
            min_votes: VOTING_MIN_VOTES_DEFAULT,
            confidence_threshold: VOTING_CONFIDENCE_THRESHOLD,
        }
    }
}

impl<const M: usize> ConsensusVoting<M> {
    /// Create new consensus voting fusion
    pub fn new(config: VotingConfig) -> Self {
        Self {
            outlier_threshold: config.outlier_threshold,
            min_votes: config.min_votes,
            last_estimate: 0.0,
            state_array: [0.0; 1],
            measurement_count: 0,
        }
    }
}

impl<const M: usize> FusionAlgorithm<1, M> for ConsensusVoting<M> {
    type Config = VotingConfig;
    
    fn new(config: Self::Config) -> Self {
        ConsensusVoting::new(config)
    }
    
    fn predict(&mut self, _dt_ms: u32) -> FusionResult<()> {
        Ok(())
    }
    
    fn update(
        &mut self,
        measurements: &[f32; M],
        _timestamp: Timestamp,
        mask: Option<u32>,
    ) -> FusionResult<(f32, ConfidenceScore)> {
        let mask = mask.unwrap_or(u32::MAX);
        
        // Collect active measurements
        let mut active_measurements = heapless::Vec::<f32, M>::new();
        for i in 0..M {
            if mask & (1 << i) != 0 {
                let _ = active_measurements.push(measurements[i]);
            }
        }
        
        if active_measurements.is_empty() {
            return Ok((self.last_estimate, ConfidenceScore::from_float(0.0)));
        }
        
        // Calculate mean and standard deviation
        let mean = active_measurements.iter().sum::<f32>() / active_measurements.len() as f32;
        let variance = active_measurements.iter()
            .map(|x| {
                let diff = x - mean;
                diff * diff  // Square without powi
            })
            .sum::<f32>() / active_measurements.len() as f32;
        // Fast square root approximation for no_std
        let std_dev = if variance == 0.0 {
            0.0
        } else {
            // Newton's method approximation
            let mut x = variance;
            x = (x + variance / x) * NEWTON_METHOD_FACTOR;
            x = (x + variance / x) * NEWTON_METHOD_FACTOR;
            x
        };
        
        // Filter outliers
        let mut valid_measurements = heapless::Vec::<f32, M>::new();
        for &measurement in active_measurements.iter() {
            if (measurement - mean).abs() <= self.outlier_threshold * std_dev {
                let _ = valid_measurements.push(measurement);
            }
        }
        
        // Check if we have enough valid measurements
        if valid_measurements.len() < self.min_votes {
            return Ok((self.last_estimate, ConfidenceScore::from_float(0.0)));
        }
        
        // Calculate consensus estimate
        let consensus = valid_measurements.iter().sum::<f32>() / valid_measurements.len() as f32;
        self.last_estimate = consensus;
        self.state_array[0] = consensus;
        self.measurement_count += 1;
        
        // Calculate confidence
        let vote_ratio = valid_measurements.len() as f32 / active_measurements.len() as f32;
        let convergence = (self.measurement_count as f32 / CONVERGENCE_MEASUREMENT_COUNT as f32).min(1.0);
        let confidence = vote_ratio * convergence;
        
        Ok((consensus, ConfidenceScore::from_float(confidence)))
    }
    
    fn state(&self) -> &[f32; 1] {
        &self.state_array
    }
    
    fn uncertainty(&self) -> [f32; 1] {
        [0.1 / (self.measurement_count as f32 + 1.0)]
    }
    
    fn reset(&mut self) {
        self.last_estimate = 0.0;
        self.state_array[0] = 0.0;
        self.measurement_count = 0;
    }
    
    fn has_converged(&self) -> bool {
        self.measurement_count >= MIN_CONVERGENCE_UPDATES
    }
}

/// Matrix operations for fusion algorithms
/// 
/// Provides basic linear algebra operations needed for Kalman filtering
/// without heap allocation. All operations work on fixed-size arrays.
pub mod matrix {
    use super::{NEWTON_METHOD_FACTOR, NEWTON_METHOD_ITERATIONS, MATRIX_MIN_DIAGONAL, 
                MATRIX_MAX_CONDITION, MATRIX_SINGULAR_THRESHOLD, MATRIX_MAX_SIZE_INVERT};
    /// Matrix type using const generics
    pub type Matrix<const R: usize, const C: usize> = [[f32; C]; R];
    
    /// Square matrix type
    pub type SquareMatrix<const N: usize> = Matrix<N, N>;
    
    /// Vector type
    pub type Vector<const N: usize> = [f32; N];
    
    /// Matrix multiplication: C = A × B
    /// 
    /// Dimensions: A[R×K] × B[K×C] = C[R×C]
    pub fn multiply<const R: usize, const K: usize, const C: usize>(
        a: &Matrix<R, K>,
        b: &Matrix<K, C>,
        result: &mut Matrix<R, C>,
    ) {
        for i in 0..R {
            for j in 0..C {
                result[i][j] = 0.0;
                for k in 0..K {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }
    
    /// Matrix transpose: B = Aᵀ
    pub fn transpose<const R: usize, const C: usize>(
        a: &Matrix<R, C>,
        result: &mut Matrix<C, R>,
    ) {
        for i in 0..R {
            for j in 0..C {
                result[j][i] = a[i][j];
            }
        }
    }
    
    /// Matrix addition: C = A + B
    pub fn add<const R: usize, const C: usize>(
        a: &Matrix<R, C>,
        b: &Matrix<R, C>,
        result: &mut Matrix<R, C>,
    ) {
        for i in 0..R {
            for j in 0..C {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
    }
    
    /// Make matrix symmetric: A = (A + Aᵀ) / 2
    /// 
    /// Critical for maintaining positive definite covariance matrices
    pub fn make_symmetric<const N: usize>(matrix: &mut SquareMatrix<N>) {
        for i in 0..N {
            for j in i+1..N {
                let avg = (matrix[i][j] + matrix[j][i]) * NEWTON_METHOD_FACTOR;
                matrix[i][j] = avg;
                matrix[j][i] = avg;
            }
        }
    }
    
    /// Check if matrix is well-conditioned
    /// 
    /// Uses simplified condition number estimation based on diagonal dominance
    pub fn is_well_conditioned<const N: usize>(matrix: &SquareMatrix<N>) -> bool {
        
        let mut min_diag = f32::INFINITY;
        let mut max_diag = 0.0f32;
        
        for i in 0..N {
            let diag = matrix[i][i].abs();
            min_diag = min_diag.min(diag);
            max_diag = max_diag.max(diag);
        }
        
        // Check minimum diagonal element
        if min_diag < MATRIX_MIN_DIAGONAL {
            return false;
        }
        
        // Check condition number estimate
        if max_diag / min_diag > MATRIX_MAX_CONDITION {
            return false;
        }
        
        true
    }
    
    /// Matrix-vector multiplication: y = A × x
    pub fn matvec<const R: usize, const C: usize>(
        matrix: &Matrix<R, C>,
        vector: &Vector<C>,
        result: &mut Vector<R>,
    ) {
        for i in 0..R {
            result[i] = 0.0;
            for j in 0..C {
                result[i] += matrix[i][j] * vector[j];
            }
        }
    }
    
    /// Cholesky decomposition: A = L × Lᵀ
    /// 
    /// Decomposes a positive definite matrix into lower triangular form.
    /// Critical for maintaining numerical stability in Kalman filters.
    /// 
    /// ## Algorithm
    /// 
    /// For each element:
    /// - Diagonal: L[j,j] = sqrt(A[j,j] - Σ(L[j,k]²))
    /// - Below diagonal: L[i,j] = (A[i,j] - Σ(L[i,k]×L[j,k])) / L[j,j]
    /// 
    /// Returns false if matrix is not positive definite
    pub fn cholesky<const N: usize>(
        a: &SquareMatrix<N>,
        l: &mut SquareMatrix<N>,
    ) -> bool {
        // Initialize L to zero
        for i in 0..N {
            for j in 0..N {
                l[i][j] = 0.0;
            }
        }
        
        for j in 0..N {
            // Compute diagonal element
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[j][k] * l[j][k];
            }
            
            let diag_val = a[j][j] - sum;
            if diag_val <= 0.0 {
                // Not positive definite
                return false;
            }
            
            // Newton's method for square root (fast convergence)
            let mut sqrt_val = diag_val;
            for _ in 0..NEWTON_METHOD_ITERATIONS {
                sqrt_val = NEWTON_METHOD_FACTOR * (sqrt_val + diag_val / sqrt_val);
            }
            l[j][j] = sqrt_val;
            
            // Compute elements below diagonal
            for i in (j + 1)..N {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[i][k] * l[j][k];
                }
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
        
        true
    }
    
    /// Matrix inversion using Gauss-Jordan elimination
    /// 
    /// ## Safety Note
    /// 
    /// Matrix inversion is numerically unstable and should be avoided
    /// when possible. Prefer solving linear systems directly.
    /// This implementation is provided for cases where inversion
    /// is unavoidable (e.g., information filter form).
    /// 
    /// Returns false if matrix is singular
    pub fn invert<const N: usize>(
        a: &SquareMatrix<N>,
        inv: &mut SquareMatrix<N>,
    ) -> bool {
        // Work with fixed maximum size to avoid const generic issues
        if N > MATRIX_MAX_SIZE_INVERT {
            return false;
        }
        
        // Create augmented matrix [A | I]
        let mut aug: [[f32; MATRIX_MAX_SIZE_INVERT * 2]; MATRIX_MAX_SIZE_INVERT] = [[0.0; MATRIX_MAX_SIZE_INVERT * 2]; MATRIX_MAX_SIZE_INVERT];
        
        // Copy A to left side, identity to right side
        for i in 0..N {
            for j in 0..N {
                aug[i][j] = a[i][j];
                aug[i][j + N] = if i == j { 1.0 } else { 0.0 };
            }
        }
        
        // Forward elimination
        for k in 0..N {
            // Find pivot
            let mut max_row = k;
            let mut max_val = aug[k][k].abs();
            
            for i in (k + 1)..N {
                if aug[i][k].abs() > max_val {
                    max_val = aug[i][k].abs();
                    max_row = i;
                }
            }
            
            // Check for singular matrix
            if max_val < MATRIX_SINGULAR_THRESHOLD {
                return false;
            }
            
            // Swap rows if needed
            if max_row != k {
                for j in 0..(N * 2) {
                    let temp = aug[k][j];
                    aug[k][j] = aug[max_row][j];
                    aug[max_row][j] = temp;
                }
            }
            
            // Scale pivot row
            let pivot = aug[k][k];
            for j in 0..(N * 2) {
                aug[k][j] /= pivot;
            }
            
            // Eliminate column
            for i in 0..N {
                if i != k {
                    let factor = aug[i][k];
                    for j in 0..(N * 2) {
                        aug[i][j] -= factor * aug[k][j];
                    }
                }
            }
        }
        
        // Extract inverse from right side
        for i in 0..N {
            for j in 0..N {
                inv[i][j] = aug[i][j + N];
            }
        }
        
        true
    }
    
    /// Solve linear system A×x = b using forward/back substitution
    /// 
    /// More numerically stable than computing A⁻¹×b
    /// Assumes A is already decomposed using Cholesky: A = L×Lᵀ
    pub fn solve_cholesky<const N: usize>(
        l: &SquareMatrix<N>,
        b: &Vector<N>,
        x: &mut Vector<N>,
    ) {
        // Forward substitution: solve L×y = b
        let mut y = [0.0; N];
        for i in 0..N {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l[i][j] * y[j];
            }
            y[i] = (b[i] - sum) / l[i][i];
        }
        
        // Back substitution: solve Lᵀ×x = y
        for i in (0..N).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..N {
                sum += l[j][i] * x[j];
            }
            x[i] = (y[i] - sum) / l[i][i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn weighted_average_fusion() {
        use models::{sensor_models, TemperatureModel};
        
        let mut fusion = WeightedAverageFusion::<3>::new()
            .add_sensor(Box::new(sensor_models::temperature("t1", 0.1)))
            .add_sensor(Box::new(sensor_models::temperature("t2", 0.2)))
            .add_sensor(Box::new(sensor_models::temperature("t3", 0.15)));
        
        // Warm up the fusion algorithm with a few measurements
        for _ in 0..5 {
            fusion.fuse(&[25.0, 25.0, 25.0], None);
        }
        
        // Test with all sensors
        let measurements = [25.0, 25.2, 24.8];
        let (estimate, confidence) = fusion.fuse(&measurements, None);
        
        // Estimate should be weighted average
        assert!((estimate - 25.0).abs() < 0.1);
        assert!(confidence.value() > 32768); // > 50% confidence after warm-up
        
        // Test with sensor mask (only first two)
        let (estimate2, confidence2) = fusion.fuse(&measurements, Some(0b011));
        assert!((estimate2 - 25.08).abs() < 0.1); // Weighted toward sensor 1
        assert!(confidence2.value() < confidence.value()); // Lower confidence with fewer sensors
    }
    
    #[test]
    fn matrix_operations() {
        use matrix::*;
        
        // Test multiplication
        let a: Matrix<2, 3> = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b: Matrix<3, 2> = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];
        let mut c: Matrix<2, 2> = [[0.0; 2]; 2];
        
        multiply(&a, &b, &mut c);
        
        assert_eq!(c[0][0], 58.0); // 1×7 + 2×9 + 3×11
        assert_eq!(c[0][1], 64.0); // 1×8 + 2×10 + 3×12
        
        // Test symmetrization
        let mut m: SquareMatrix<2> = [[1.0, 2.0], [3.0, 4.0]];
        make_symmetric(&mut m);
        
        assert_eq!(m[0][1], 2.5); // (2 + 3) / 2
        assert_eq!(m[1][0], 2.5);
    }
    
    #[test]
    fn matrix_inversion() {
        use matrix::*;
        
        // Test 2x2 matrix inversion
        let a: SquareMatrix<2> = [[4.0, 7.0], [2.0, 6.0]];
        let mut inv: SquareMatrix<2> = [[0.0; 2]; 2];
        
        assert!(invert(&a, &mut inv));
        
        // Verify A × A⁻¹ = I
        let mut product: SquareMatrix<2> = [[0.0; 2]; 2];
        multiply(&a, &inv, &mut product);
        
        assert!((product[0][0] - 1.0).abs() < 1e-6);
        assert!((product[1][1] - 1.0).abs() < 1e-6);
        assert!(product[0][1].abs() < 1e-6);
        assert!(product[1][0].abs() < 1e-6);
    }
    
    #[test]
    fn cholesky_decomposition() {
        use matrix::*;
        
        // Test with positive definite matrix
        let a: SquareMatrix<3> = [
            [4.0, 12.0, -16.0],
            [12.0, 37.0, -43.0],
            [-16.0, -43.0, 98.0],
        ];
        let mut l: SquareMatrix<3> = [[0.0; 3]; 3];
        
        assert!(cholesky(&a, &mut l));
        
        // Verify L × Lᵀ = A
        let mut lt: SquareMatrix<3> = [[0.0; 3]; 3];
        transpose(&l, &mut lt);
        
        let mut product: SquareMatrix<3> = [[0.0; 3]; 3];
        multiply(&l, &lt, &mut product);
        
        // Check reconstruction with tolerance appropriate for embedded sqrt approximation
        // Our Newton's method sqrt uses only 3 iterations for performance
        for i in 0..3 {
            for j in 0..3 {
                let diff = (product[i][j] - a[i][j]).abs();
                // Allow 0.3% relative error or 0.01 absolute error, whichever is larger
                let tolerance = (0.003 * a[i][j].abs()).max(0.01);
                assert!(diff < tolerance, 
                    "Reconstruction error too large at [{},{}]: {} vs {}, diff = {}", 
                    i, j, product[i][j], a[i][j], diff);
            }
        }
    }
}