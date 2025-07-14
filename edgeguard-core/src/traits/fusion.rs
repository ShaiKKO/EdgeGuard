//! Sensor Fusion Algorithm Traits
//!
//! This module defines traits for implementing sensor fusion algorithms that
//! combine multiple sensor readings to produce more accurate estimates.
//!
//! ## Fusion Concepts
//!
//! Sensor fusion addresses several challenges:
//! - **Noise reduction**: Combining multiple noisy sensors
//! - **Redundancy**: Handling sensor failures gracefully
//! - **Complementary data**: Merging sensors with different characteristics
//! - **Temporal filtering**: Smoothing over time
//!
//! ## Algorithm Types
//!
//! - **Kalman Filter**: Optimal for linear systems with Gaussian noise
//! - **Extended Kalman Filter**: Handles mild nonlinearities
//! - **Weighted Average**: Simple and efficient for similar sensors
//! - **Complementary Filter**: Good for combining high/low frequency sensors

use crate::fusion::{FusionResult, FusionError, ConfidenceScore};
use crate::time::Timestamp;
use crate::events::SensorType;
use crate::errors::ValidationError;

/// Generic fusion algorithm trait
///
/// This trait provides a generic interface for different fusion algorithms
/// while maintaining zero-allocation guarantees through const generics:
/// - `N`: State vector dimension (e.g., [position, velocity, acceleration])
/// - `M`: Measurement vector dimension (number of sensors)
///
/// ## Implementation Guidelines
///
/// 1. **Numerical Stability**: Check for ill-conditioned matrices
/// 2. **Bounded Time**: Ensure O(1) time complexity
/// 3. **Error Handling**: Never panic, return errors instead
/// 4. **Memory**: Use only stack allocation
///
/// ## Example Implementation
///
/// ```rust
/// use edgeguard_core::traits::FusionAlgorithm;
/// use edgeguard_core::fusion::{FusionResult, ConfidenceScore};
/// use edgeguard_core::time::Timestamp;
///
/// struct SimpleAverage;
///
/// impl FusionAlgorithm<1, 3> for SimpleAverage {
///     type Config = ();
///     
///     fn new(_: Self::Config) -> Self {
///         SimpleAverage
///     }
///     
///     fn predict(&mut self, _dt_ms: u32) -> FusionResult<()> {
///         Ok(()) // No prediction needed for simple average
///     }
///     
///     fn update(
///         &mut self,
///         measurements: &[f32; 3],
///         _timestamp: Timestamp,
///         _mask: Option<u32>,
///     ) -> FusionResult<(f32, ConfidenceScore)> {
///         let avg = measurements.iter().sum::<f32>() / 3.0;
///         Ok((avg, ConfidenceScore::new(0.8)))
///     }
///     
///     fn state(&self) -> &[f32; 1] {
///         &[0.0] // Placeholder
///     }
///     
///     fn uncertainty(&self) -> [f32; 1] {
///         [0.1]
///     }
///     
///     fn reset(&mut self) {}
///     
///     fn has_converged(&self) -> bool {
///         true
///     }
/// }
/// ```
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
    /// - `mask`: Optional bit mask for available sensors (bit i = sensor i available)
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

/// Dynamic fusion algorithm trait
/// 
/// This trait allows different fusion algorithms with varying dimensions
/// to be used polymorphically through dynamic dispatch. Useful for:
/// - Runtime algorithm selection
/// - Heterogeneous sensor configurations
/// - Plugin architectures
///
/// ## Trade-offs
///
/// - **Flexibility**: Can switch algorithms at runtime
/// - **Performance**: Dynamic dispatch overhead
/// - **Memory**: Requires heap allocation for trait objects
pub trait FusionAlgorithmDyn: Send {
    /// Predict next state based on time delta
    fn predict(&mut self, dt_ms: u32) -> Result<(), FusionError>;
    
    /// Update state with new measurements
    fn update(
        &mut self,
        measurements: &[f32],
        timestamp: Timestamp,
    ) -> Result<(f32, ConfidenceScore), FusionError>;
    
    /// Check if fusion has converged
    fn has_converged(&self) -> bool;
    
    /// Reset fusion to initial state
    fn reset(&mut self);
}

/// Sensor model for fusion algorithms
/// 
/// Defines the interface for sensor-specific physics models used in
/// fusion algorithms. Models capture:
/// - Sensor characteristics (noise, drift, response time)
/// - Environmental effects (temperature, pressure compensation)
/// - Measurement physics (how true state maps to sensor reading)
///
/// ## Example: Temperature Sensor Model
///
/// ```rust
/// use edgeguard_core::traits::SensorModel;
/// use edgeguard_core::events::SensorType;
/// use edgeguard_core::errors::ValidationError;
/// 
/// struct ThermistorModel {
///     sensor_id: String,
///     beta: f32,  // Thermistor beta parameter
///     r0: f32,    // Reference resistance
/// }
/// 
/// impl SensorModel for ThermistorModel {
///     fn sensor_type(&self) -> SensorType {
///         SensorType::Temperature
///     }
///     
///     fn sensor_id(&self) -> &str {
///         &self.sensor_id
///     }
///     
///     fn noise_variance(&self) -> f32 {
///         0.5 // ±0.5°C typical noise
///     }
///     
///     fn predict_state(&self, current_state: f32, dt_ms: u32) -> f32 {
///         // Simple exponential cooling model
///         let tau = 10000.0; // 10 second time constant
///         let alpha = (dt_ms as f32) / tau;
///         current_state * (1.0 - alpha) + 20.0 * alpha // Approaches ambient
///     }
///     
///     fn measurement_model(&self, true_state: f32) -> f32 {
///         // Thermistor has nonlinear response
///         // Simplified model here
///         true_state // Direct mapping for simplicity
///     }
///     
///     fn validate(&self, measurement: f32) -> Result<(), ValidationError> {
///         if measurement < -50.0 || measurement > 150.0 {
///             Err(ValidationError::OutOfRange {
///                 value: measurement,
///                 min: -50.0,
///                 max: 150.0,
///             })
///         } else {
///             Ok(())
///         }
///     }
///     
///     // ... other methods
/// }
/// ```
pub trait SensorModel: Send {
    /// Sensor type this model represents
    fn sensor_type(&self) -> SensorType;
    
    /// Sensor identifier
    fn sensor_id(&self) -> &str;
    
    /// Get measurement noise variance
    /// 
    /// Used by fusion algorithms to weight sensor contributions.
    /// Lower variance = higher weight in fusion.
    fn noise_variance(&self) -> f32;
    
    /// State transition function
    /// 
    /// Predicts how the state evolves over time based on physics.
    /// For example:
    /// - Temperature: Heat transfer equations
    /// - Pressure: Atmospheric dynamics
    /// - Humidity: Evaporation/condensation models
    fn predict_state(&self, current_state: f32, dt_ms: u32) -> f32;
    
    /// Measurement model
    /// 
    /// Maps true state to expected sensor reading, accounting for:
    /// - Sensor transfer function
    /// - Nonlinearities
    /// - Systematic biases
    fn measurement_model(&self, true_state: f32) -> f32;
    
    /// Validate measurement against physical constraints
    fn validate(&self, measurement: f32) -> Result<(), ValidationError>;
    
    /// Environmental compensation
    /// 
    /// Corrects measurement based on environmental conditions.
    /// The `env` parameter type would need to be defined based on
    /// the specific environmental factors relevant to the sensor.
    fn compensate(&self, measurement: f32, _env: &dyn core::any::Any) -> f32 {
        // Default: no compensation
        measurement
    }
    
    /// Compute confidence factors for this measurement
    /// 
    /// Returns factors that affect measurement confidence:
    /// - Innovation (difference from prediction)
    /// - Environmental conditions
    /// - Sensor age/drift
    fn confidence_factors(
        &self,
        measurement: f32,
        prediction: f32,
        _env: &dyn core::any::Any,
    ) -> f32 {
        // Default: base confidence on innovation
        let innovation = (measurement - prediction).abs();
        1.0 / (1.0 + innovation * 0.1)
    }
}