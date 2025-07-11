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

use crate::{
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
                required: 2,
                available: 1,
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

/// Core trait for sensor fusion algorithms
/// 
/// ## Design Rationale
/// 
/// This trait provides a generic interface for different fusion algorithms
/// while maintaining zero-allocation guarantees through const generics:
/// - `N`: State vector dimension (e.g., [position, velocity, acceleration])
/// - `M`: Measurement vector dimension (number of sensors)
/// 
/// ## Safety Requirements
/// 
/// Implementations must:
/// 1. Never panic on invalid inputs
/// 2. Detect and handle numerical instabilities
/// 3. Maintain bounded execution time
/// 4. Use only stack allocation
pub trait FusionAlgorithm<const N: usize, const M: usize> {
    /// Configuration type for the algorithm
    type Config;
    
    /// Create new fusion instance with configuration
    fn new(config: Self::Config) -> Self;
    
    /// Predict next state based on time delta
    /// 
    /// This step propagates the state forward in time using the
    /// system dynamics model. For sensor fusion, this often includes:
    /// - Drift modeling
    /// - Environmental effects
    /// - Known control inputs
    fn predict(&mut self, dt_ms: u32) -> FusionResult<()>;
    
    /// Update state with new measurements
    /// 
    /// Incorporates new sensor readings to refine the state estimate.
    /// Returns the fused estimate and confidence score.
    /// 
    /// ## Parameters
    /// - `measurements`: Array of sensor readings
    /// - `timestamp`: Measurement timestamp for synchronization
    /// - `mask`: Optional bit mask for available sensors
    fn update(
        &mut self,
        measurements: &[f32; M],
        timestamp: Timestamp,
        mask: Option<u32>,
    ) -> FusionResult<(f32, ConfidenceScore)>;
    
    /// Get current state estimate
    fn state(&self) -> &[f32; N];
    
    /// Get estimation uncertainty (covariance diagonal)
    fn uncertainty(&self) -> [f32; N];
    
    /// Reset fusion to initial state
    fn reset(&mut self);
    
    /// Check if fusion has converged
    /// 
    /// Convergence criteria:
    /// - Uncertainty below threshold
    /// - Innovation within bounds
    /// - Sufficient measurements processed
    fn has_converged(&self) -> bool;
}

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
    /// Measurement count for convergence
    measurement_count: u32,
}

impl<const M: usize> WeightedAverageFusion<M> {
    /// Create new weighted average fusion
    pub fn new() -> Self {
        Self {
            sensors: heapless::Vec::new(),
            weights: [0.0; M],
            last_estimate: 0.0,
            measurement_count: 0,
        }
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
        let convergence_confidence = (self.measurement_count as f32 / 10.0).min(1.0);
        
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
            let confidence = 1.0 / (1.0 + variance);
            confidence.max(0.0).min(1.0)
        } else {
            0.5 // Single sensor, moderate confidence
        };
        
        let overall_confidence = sensor_confidence * convergence_confidence * agreement_confidence;
        
        (estimate, ConfidenceScore::from_float(overall_confidence))
    }
}

/// Matrix operations for fusion algorithms
/// 
/// Provides basic linear algebra operations needed for Kalman filtering
/// without heap allocation. All operations work on fixed-size arrays.
pub mod matrix {
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
                let avg = (matrix[i][j] + matrix[j][i]) * 0.5;
                matrix[i][j] = avg;
                matrix[j][i] = avg;
            }
        }
    }
    
    /// Check if matrix is well-conditioned
    /// 
    /// Uses simplified condition number estimation based on diagonal dominance
    pub fn is_well_conditioned<const N: usize>(matrix: &SquareMatrix<N>) -> bool {
        const MIN_DIAGONAL: f32 = 1e-6;
        const MAX_CONDITION: f32 = 1e6;
        
        let mut min_diag = f32::INFINITY;
        let mut max_diag = 0.0f32;
        
        for i in 0..N {
            let diag = matrix[i][i].abs();
            min_diag = min_diag.min(diag);
            max_diag = max_diag.max(diag);
        }
        
        // Check minimum diagonal element
        if min_diag < MIN_DIAGONAL {
            return false;
        }
        
        // Check condition number estimate
        if max_diag / min_diag > MAX_CONDITION {
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
            for _ in 0..3 { // 3 iterations sufficient for f32 precision
                sqrt_val = 0.5 * (sqrt_val + diag_val / sqrt_val);
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
        const MAX_N: usize = 16;
        if N > MAX_N {
            return false;
        }
        
        // Create augmented matrix [A | I]
        let mut aug: [[f32; MAX_N * 2]; MAX_N] = [[0.0; MAX_N * 2]; MAX_N];
        
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
            if max_val < 1e-10 {
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
        
        // Test with all sensors
        let measurements = [25.0, 25.2, 24.8];
        let (estimate, confidence) = fusion.fuse(&measurements, None);
        
        // Estimate should be weighted average
        assert!((estimate - 25.0).abs() < 0.1);
        assert!(confidence.value() > 0.5);
        
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
        
        for i in 0..3 {
            for j in 0..3 {
                assert!((product[i][j] - a[i][j]).abs() < 1e-4);
            }
        }
    }
}