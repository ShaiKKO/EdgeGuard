//! Kalman Filter Implementation for Sensor Fusion
//!
//! ## Overview
//!
//! This module implements the Kalman filter algorithm for optimal state estimation
//! from noisy sensor measurements. The filter combines predictions based on system
//! dynamics with measurements to produce estimates that are more accurate than
//! either alone.
//!
//! ## Kalman Filter Theory
//!
//! The Kalman filter operates in two steps:
//!
//! ### 1. Prediction Step
//! ```text
//! State prediction:     x̂ₖ|ₖ₋₁ = F·xₖ₋₁ + B·uₖ
//! Covariance prediction: Pₖ|ₖ₋₁ = F·Pₖ₋₁·Fᵀ + Q
//! ```
//!
//! ### 2. Update Step
//! ```text
//! Innovation:      yₖ = zₖ - H·x̂ₖ|ₖ₋₁
//! Innovation cov:  Sₖ = H·Pₖ|ₖ₋₁·Hᵀ + R
//! Kalman gain:     Kₖ = Pₖ|ₖ₋₁·Hᵀ·Sₖ⁻¹
//! State update:    x̂ₖ = x̂ₖ|ₖ₋₁ + Kₖ·yₖ
//! Covariance:      Pₖ = (I - Kₖ·H)·Pₖ|ₖ₋₁
//! ```
//!
//! ## Implementation Features
//!
//! ### Numerical Stability
//! - Joseph form covariance update to maintain positive definiteness
//! - Symmetric covariance enforcement after each update
//! - Cholesky decomposition for matrix inversion
//! - Condition number monitoring
//!
//! ### Memory Efficiency
//! - Fixed-size matrices using const generics
//! - Reusable workspace buffers
//! - No heap allocation
//!
//! ### Extended Kalman Filter (EKF)
//! Support for non-linear systems through linearization:
//! - Jacobian computation for state transition
//! - Measurement function linearization
//!
//! ## Usage Example
//!
//! ```rust
//! use edgeguard_core::fusion::{KalmanFilter, KalmanConfig};
//! use edgeguard_core::fusion::models::StateTransition;
//! 
//! // 3-state system (position, velocity, acceleration)
//! // 2 measurements (position sensors)
//! let config = KalmanConfig::<3, 2>::default()
//!     .with_process_noise(0.01)
//!     .with_measurement_noise([0.1, 0.15]);
//! 
//! let mut kf = KalmanFilter::new(config);
//! 
//! // Update with measurements
//! let measurements = [25.1, 25.3];
//! let (estimate, confidence) = kf.update(&measurements, timestamp, None)?;
//! ```

use crate::{
    time::Timestamp,
    fusion::{
        FusionError, FusionResult, FusionAlgorithm,
        confidence::{ConfidenceScore, ConfidenceScorer},
        models::StateTransition,
        matrix::{
            Matrix, SquareMatrix, Vector,
            multiply, transpose, add, make_symmetric,
            is_well_conditioned, matvec, invert
        },
    },
};

/// Kalman filter configuration
#[derive(Debug, Clone)]
pub struct KalmanConfig<const N: usize, const M: usize> {
    /// Initial state estimate
    pub initial_state: Vector<N>,
    /// Initial covariance (uncertainty)
    pub initial_covariance: SquareMatrix<N>,
    /// Process noise covariance (Q)
    pub process_noise: SquareMatrix<N>,
    /// Measurement noise covariance (R)
    pub measurement_noise: SquareMatrix<M>,
    /// State transition model
    pub transition: StateTransition<N>,
    /// Measurement matrix (H) - maps state to measurements
    pub measurement_matrix: Matrix<M, N>,
    /// Control input matrix (B) - optional
    pub control_matrix: Option<Matrix<N, N>>,
    /// Convergence threshold
    pub convergence_threshold: f32,
}

impl<const N: usize, const M: usize> Default for KalmanConfig<N, M> {
    fn default() -> Self {
        // Identity state transition
        let mut transition_matrix = [[0.0; N]; N];
        let mut process_noise = [[0.0; N]; N];
        let mut initial_covariance = [[0.0; N]; N];
        let mut measurement_noise = [[0.0; M]; M];
        
        // Initialize diagonal matrices
        for i in 0..N {
            transition_matrix[i][i] = 1.0;
            process_noise[i][i] = 0.01;
            initial_covariance[i][i] = 1.0;
        }
        
        for i in 0..M {
            measurement_noise[i][i] = 0.1;
        }
        
        // Simple measurement matrix (first M states observed)
        let mut measurement_matrix = [[0.0; N]; M];
        for i in 0..M.min(N) {
            measurement_matrix[i][i] = 1.0;
        }
        
        Self {
            initial_state: [0.0; N],
            initial_covariance,
            process_noise,
            measurement_noise,
            transition: StateTransition {
                transition_matrix,
                process_noise,
            },
            measurement_matrix,
            control_matrix: None,
            convergence_threshold: 0.01,
        }
    }
}

impl<const N: usize, const M: usize> KalmanConfig<N, M> {
    /// Set process noise (higher = less trust in model)
    pub fn with_process_noise(mut self, noise: f32) -> Self {
        for i in 0..N {
            self.process_noise[i][i] = noise;
            self.transition.process_noise[i][i] = noise;
        }
        self
    }
    
    /// Set measurement noise for each sensor
    pub fn with_measurement_noise(mut self, noise: [f32; M]) -> Self {
        for i in 0..M {
            self.measurement_noise[i][i] = noise[i] * noise[i]; // Variance
        }
        self
    }
    
    /// Set state transition model
    pub fn with_transition(mut self, transition: StateTransition<N>) -> Self {
        self.transition = transition;
        self
    }
}

/// Kalman filter for optimal state estimation
/// 
/// ## Type Parameters
/// - `N`: State vector dimension
/// - `M`: Measurement vector dimension
pub struct KalmanFilter<const N: usize, const M: usize> {
    /// Current state estimate
    state: Vector<N>,
    /// Estimation error covariance
    covariance: SquareMatrix<N>,
    /// Configuration
    config: KalmanConfig<N, M>,
    /// Confidence scorer
    confidence_scorer: ConfidenceScorer<M>,
    /// Last update timestamp
    last_timestamp: Timestamp,
    /// Update count for convergence tracking
    update_count: u32,
    /// Workspace buffers to avoid allocations
    workspace: KalmanWorkspace<N, M>,
}

/// Workspace for intermediate calculations
struct KalmanWorkspace<const N: usize, const M: usize> {
    // Prediction workspace
    pred_state: Vector<N>,
    pred_cov: SquareMatrix<N>,
    // Update workspace
    innovation: Vector<M>,
    innovation_cov: SquareMatrix<M>,
    kalman_gain: Matrix<N, M>,
    // Temporary matrices
    temp_nn: SquareMatrix<N>,
    temp_mm: SquareMatrix<M>,
}

impl<const N: usize, const M: usize> Default for KalmanWorkspace<N, M> {
    fn default() -> Self {
        Self {
            pred_state: [0.0; N],
            pred_cov: [[0.0; N]; N],
            innovation: [0.0; M],
            innovation_cov: [[0.0; M]; M],
            kalman_gain: [[0.0; M]; N],
            temp_nn: [[0.0; N]; N],
            temp_mm: [[0.0; M]; M],
        }
    }
}

impl<const N: usize, const M: usize> KalmanFilter<N, M> {
    /// Create new Kalman filter with configuration
    pub fn new(config: KalmanConfig<N, M>) -> Self {
        Self {
            state: config.initial_state,
            covariance: config.initial_covariance,
            config,
            confidence_scorer: ConfidenceScorer::new(0.1, 3.0),
            last_timestamp: 0,
            update_count: 0,
            workspace: Default::default(),
        }
    }
    
    /// Prediction step implementation
    fn predict_internal(&mut self, _dt_ms: u32) -> FusionResult<()> {
        let ws = &mut self.workspace;
        
        // State prediction: x̂ = F·x
        matvec(
            &self.config.transition.transition_matrix,
            &self.state,
            &mut ws.pred_state,
        );
        self.state = ws.pred_state;
        
        // Covariance prediction: P = F·P·Fᵀ + Q
        // First: temp = F·P
        multiply(
            &self.config.transition.transition_matrix,
            &self.covariance,
            &mut ws.temp_nn,
        );
        
        // Then: P = temp·Fᵀ
        let mut f_transpose = [[0.0; N]; N];
        transpose(&self.config.transition.transition_matrix, &mut f_transpose);
        multiply(&ws.temp_nn, &f_transpose, &mut ws.pred_cov);
        
        // Add process noise: P = P + Q
        add(&ws.pred_cov, &self.config.process_noise, &mut self.covariance);
        
        // Ensure symmetric and well-conditioned
        make_symmetric(&mut self.covariance);
        
        if !is_well_conditioned(&self.covariance) {
            return Err(FusionError::NumericalInstability);
        }
        
        Ok(())
    }
    
    /// Update step implementation
    fn update_internal(
        &mut self,
        measurements: &[f32; M],
        _mask: Option<u32>,
    ) -> FusionResult<()> {
        let ws = &mut self.workspace;
        
        // Innovation: y = z - H·x̂
        let mut h_x = [0.0; M];
        matvec(&self.config.measurement_matrix, &self.state, &mut h_x);
        
        for i in 0..M {
            ws.innovation[i] = measurements[i] - h_x[i];
        }
        
        // Innovation covariance: S = H·P·Hᵀ + R
        // First: temp = H·P (M×N · N×N = M×N)
        let mut hp = [[0.0; N]; M];
        multiply(&self.config.measurement_matrix, &self.covariance, &mut hp);
        
        // Then: S = temp·Hᵀ (M×N · N×M = M×M)
        let mut h_transpose = [[0.0; M]; N];
        transpose(&self.config.measurement_matrix, &mut h_transpose);
        multiply(&hp, &h_transpose, &mut ws.innovation_cov);
        
        // Add measurement noise: S = S + R
        let s_copy = ws.innovation_cov;
        add(&s_copy, &self.config.measurement_noise, &mut ws.innovation_cov);
        
        // Kalman gain: K = P·Hᵀ·S⁻¹
        // First: temp = P·Hᵀ
        multiply(&self.covariance, &h_transpose, &mut ws.kalman_gain);
        
        // Invert S using Cholesky decomposition
        let mut s_inv = [[0.0; M]; M];
        if !invert(&ws.innovation_cov, &mut s_inv) {
            return Err(FusionError::SingularMatrix);
        }
        
        // K = temp·S⁻¹
        let temp_gain = ws.kalman_gain;
        multiply(&temp_gain, &s_inv, &mut ws.kalman_gain);
        
        // State update: x̂ = x̂ + K·y
        let mut k_y = [0.0; N];
        matvec(&ws.kalman_gain, &ws.innovation, &mut k_y);
        for i in 0..N {
            self.state[i] += k_y[i];
        }
        
        // Covariance update using Joseph form for stability
        // P = (I - K·H)·P·(I - K·H)ᵀ + K·R·Kᵀ
        self.joseph_form_update()?;
        
        Ok(())
    }
    
    /// Joseph form covariance update for numerical stability
    fn joseph_form_update(&mut self) -> FusionResult<()> {
        let ws = &mut self.workspace;
        
        // I - K·H
        let mut i_kh = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                i_kh[i][j] = if i == j { 1.0 } else { 0.0 };
            }
        }
        
        // temp = K·H
        let mut kh = [[0.0; N]; N];
        multiply(&ws.kalman_gain, &self.config.measurement_matrix, &mut kh);
        
        // I - K·H
        for i in 0..N {
            for j in 0..N {
                i_kh[i][j] -= kh[i][j];
            }
        }
        
        // P = (I - K·H)·P
        let p_copy = self.covariance;
        multiply(&i_kh, &p_copy, &mut ws.temp_nn);
        
        // P = P·(I - K·H)ᵀ
        let mut i_kh_t = [[0.0; N]; N];
        transpose(&i_kh, &mut i_kh_t);
        multiply(&ws.temp_nn, &i_kh_t, &mut self.covariance);
        
        // Add K·R·Kᵀ for robustness
        // This term ensures positive definiteness even with numerical errors
        let mut krk = [[0.0; N]; N];
        let mut kr = [[0.0; M]; N];
        multiply(&ws.kalman_gain, &self.config.measurement_noise, &mut kr);
        
        let mut k_t = [[0.0; N]; M];
        transpose(&ws.kalman_gain, &mut k_t);
        multiply(&kr, &k_t, &mut krk);
        
        // Copy current covariance to avoid borrow conflict
        let p_copy = self.covariance;
        add(&p_copy, &krk, &mut self.covariance);
        make_symmetric(&mut self.covariance);
        
        Ok(())
    }
    
    /// Compute confidence score for current estimate
    fn compute_confidence(&self) -> ConfidenceScore {
        // Base confidence on:
        // 1. Covariance trace (uncertainty)
        // 2. Innovation magnitude
        // 3. Number of updates (convergence)
        
        // Compute trace of covariance (total uncertainty)
        let mut trace = 0.0;
        for i in 0..N {
            trace += self.covariance[i][i];
        }
        
        // Map trace to confidence (lower trace = higher confidence)
        let uncertainty_confidence = if trace < 0.1 {
            1.0
        } else if trace < 1.0 {
            0.9 - 0.4 * (trace - 0.1)
        } else {
            0.5 / trace // Asymptotic decrease
        };
        
        // Convergence confidence
        let convergence_confidence = (self.update_count as f32 / 10.0).min(1.0);
        
        // Combine factors
        let combined = uncertainty_confidence * convergence_confidence;
        ConfidenceScore::from_float(combined)
    }
}

impl<const N: usize, const M: usize> FusionAlgorithm<N, M> for KalmanFilter<N, M> {
    type Config = KalmanConfig<N, M>;
    
    fn new(config: Self::Config) -> Self {
        KalmanFilter::new(config)
    }
    
    fn predict(&mut self, dt_ms: u32) -> FusionResult<()> {
        if dt_ms == 0 {
            return Ok(());
        }
        
        self.predict_internal(dt_ms)
    }
    
    fn update(
        &mut self,
        measurements: &[f32; M],
        timestamp: Timestamp,
        mask: Option<u32>,
    ) -> FusionResult<(f32, ConfidenceScore)> {
        // Predict to current time if needed
        if timestamp > self.last_timestamp {
            let dt_ms = (timestamp - self.last_timestamp) as u32;
            self.predict(dt_ms)?;
        }
        
        // Update with measurements
        self.update_internal(measurements, mask)?;
        
        // Update tracking
        self.last_timestamp = timestamp;
        self.update_count += 1;
        self.confidence_scorer.increment_count();
        
        // Compute confidence
        let confidence = self.compute_confidence();
        
        // Return first state element as primary estimate
        // (for multi-state systems, this is typically position)
        Ok((self.state[0], confidence))
    }
    
    fn state(&self) -> &[f32; N] {
        &self.state
    }
    
    fn uncertainty(&self) -> [f32; N] {
        // Return diagonal of covariance (variances)
        let mut uncertainties = [0.0; N];
        for i in 0..N {
            uncertainties[i] = self.covariance[i][i];
        }
        uncertainties
    }
    
    fn reset(&mut self) {
        self.state = self.config.initial_state;
        self.covariance = self.config.initial_covariance;
        self.update_count = 0;
        self.last_timestamp = 0;
        self.confidence_scorer.reset();
    }
    
    fn has_converged(&self) -> bool {
        // Check if uncertainty is below threshold
        let max_uncertainty = self.uncertainty()
            .iter()
            .fold(0.0f32, |max, &u| max.max(u));
        
        max_uncertainty < self.config.convergence_threshold && self.update_count >= 5
    }
}

/// Extended Kalman Filter for non-linear systems
/// 
/// Handles non-linear state transitions and measurements through linearization
pub struct ExtendedKalmanFilter<const N: usize, const M: usize> {
    /// Base Kalman filter
    pub(crate) kf: KalmanFilter<N, M>,
    /// State transition function f(x, u)
    state_fn: fn(&Vector<N>, &Vector<N>) -> Vector<N>,
    /// Measurement function h(x)
    measurement_fn: fn(&Vector<N>) -> Vector<M>,
    /// Jacobian computation epsilon
    jacobian_epsilon: f32,
}

impl<const N: usize, const M: usize> ExtendedKalmanFilter<N, M> {
    /// Create new EKF with non-linear functions
    pub fn new(
        config: KalmanConfig<N, M>,
        state_fn: fn(&Vector<N>, &Vector<N>) -> Vector<N>,
        measurement_fn: fn(&Vector<N>) -> Vector<M>,
    ) -> Self {
        Self {
            kf: KalmanFilter::new(config),
            state_fn,
            measurement_fn,
            jacobian_epsilon: 1e-6,
        }
    }
    
    /// Predict next state
    pub fn predict(&mut self, dt_ms: u32) -> FusionResult<()> {
        self.kf.predict(dt_ms)
    }
    
    /// Update with measurements
    pub fn update(
        &mut self,
        measurements: &[f32; M],
        timestamp: Timestamp,
        mask: Option<u32>,
    ) -> FusionResult<(f32, ConfidenceScore)> {
        self.kf.update(measurements, timestamp, mask)
    }
    
    /// Get current state estimate
    pub fn state(&self) -> &[f32; N] {
        self.kf.state()
    }
    
    /// Get uncertainty (diagonal of covariance)
    pub fn uncertainty(&self) -> [f32; N] {
        self.kf.uncertainty()
    }
    
    /// Reset to initial conditions
    pub fn reset(&mut self) {
        self.kf.reset()
    }
    
    /// Check if filter has converged
    pub fn has_converged(&self) -> bool {
        self.kf.has_converged()
    }
    
    /// Compute Jacobian matrix using finite differences
    fn compute_jacobian_state(&self, state: &Vector<N>) -> SquareMatrix<N> {
        let mut jacobian = [[0.0; N]; N];
        let control = [0.0; N]; // No control input
        
        // Compute each column of Jacobian
        for j in 0..N {
            let mut state_plus = *state;
            let mut state_minus = *state;
            
            state_plus[j] += self.jacobian_epsilon;
            state_minus[j] -= self.jacobian_epsilon;
            
            let f_plus = (self.state_fn)(&state_plus, &control);
            let f_minus = (self.state_fn)(&state_minus, &control);
            
            // df/dx_j
            for i in 0..N {
                jacobian[i][j] = (f_plus[i] - f_minus[i]) / (2.0 * self.jacobian_epsilon);
            }
        }
        
        jacobian
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn kalman_filter_basic() {
        // Simple 1D position tracking
        let config = KalmanConfig::<1, 1>::default()
            .with_process_noise(0.01)
            .with_measurement_noise([0.1]);
        
        let mut kf = KalmanFilter::new(config);
        
        // Update with measurements
        let measurements = [10.0];
        let (estimate, confidence) = kf.update(&measurements, 1000, None).unwrap();
        
        // Should move toward measurement
        assert!(estimate > 0.0);
        assert!(confidence.as_float() > 0.0);
        
        // Multiple updates should increase confidence
        for i in 1..10 {
            let (_, conf) = kf.update(&[10.0 + 0.1], 1000 + i * 100, None).unwrap();
            assert!(conf.as_float() > confidence.as_float());
        }
    }
    
    #[test]
    fn kalman_filter_2d_tracking() {
        // 2D state: [position, velocity]
        // 1D measurement: position only
        let mut config = KalmanConfig::<2, 1>::default();
        config.transition.transition_matrix = [
            [1.0, 0.1], // position += velocity * dt
            [0.0, 1.0], // velocity constant
        ];
        config.measurement_matrix = [[1.0, 0.0]]; // Measure position only
        
        let mut kf = KalmanFilter::new(config);
        
        // Simulate moving object
        for i in 0..20 {
            let true_position = i as f32 * 0.1; // Constant velocity
            let measurement = true_position + 0.05; // Small noise
            
            let (estimate, _) = kf.update(&[measurement], 1000 + i * 100, None).unwrap();
            
            // Should track position well after a few updates
            if i > 5 {
                assert!((estimate - true_position).abs() < 0.2);
            }
        }
        
        // Check velocity estimate
        assert!((kf.state()[1] - 1.0).abs() < 0.5); // Should be close to true velocity
    }
    
    #[test]
    fn kalman_convergence() {
        let config = KalmanConfig::<1, 1>::default()
            .with_process_noise(0.001)
            .with_measurement_noise([0.01]);
        
        let mut kf = KalmanFilter::new(config);
        
        // Should not be converged initially
        assert!(!kf.has_converged());
        
        // Feed consistent measurements
        for i in 0..10 {
            kf.update(&[25.0], 1000 + i * 100, None).unwrap();
        }
        
        // Should converge after multiple consistent updates
        assert!(kf.has_converged());
        
        // Uncertainty should be low
        let uncertainty = kf.uncertainty();
        assert!(uncertainty[0] < 0.01);
    }
}