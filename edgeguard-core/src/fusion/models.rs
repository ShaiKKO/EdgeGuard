//! Sensor Models for Physics-Aware Fusion
//!
//! ## Overview
//!
//! This module provides physics-based models for different sensor types, enabling
//! accurate state prediction and measurement updates in fusion algorithms. Each model
//! captures the unique characteristics, constraints, and dynamics of its sensor type.
//!
//! ## Model Components
//!
//! Each sensor model includes:
//!
//! 1. **State Transition**: How the measured quantity evolves over time
//! 2. **Measurement Model**: Relationship between true state and sensor reading
//! 3. **Noise Characteristics**: Sensor-specific noise and uncertainty
//! 4. **Physical Constraints**: Limits based on physics laws
//! 5. **Environmental Compensation**: Corrections for operating conditions
//!
//! ## Physics Models
//!
//! ### Temperature
//!
//! State evolution follows Newton's law of cooling:
//! ```text
//! dT/dt = -k(T - T_ambient)
//! 
//! Where:
//! - k = cooling constant (depends on thermal mass)
//! - T_ambient = environmental temperature
//! ```
//!
//! ### Pressure
//!
//! Pressure changes follow the barometric formula with temperature compensation:
//! ```text
//! P = P₀ × (1 - L×h/T₀)^(g×M/R×L)
//! 
//! With temperature correction:
//! P_corrected = P × (T/T_ref)
//! ```
//!
//! ### Humidity
//!
//! Humidity is constrained by saturation physics:
//! ```text
//! RH_max = 100% × (e_s(T_dew) / e_s(T))
//! 
//! Where e_s(T) is saturation vapor pressure
//! ```
//!
//! ## Usage Example
//!
//! ```rust
//! use edgeguard_core::fusion::models::{SensorModel, TemperatureModel};
//! 
//! // Create temperature sensor model
//! let model = TemperatureModel::new()
//!     .with_thermal_mass(100.0)  // J/K
//!     .with_noise_std(0.1)       // °C
//!     .with_range(-40.0, 85.0);  // °C
//! 
//! // Use in Kalman filter
//! let kalman = KalmanFilter::new()
//!     .with_sensor_model(model);
//! ```

use crate::{
    errors::ValidationError,
    events::SensorType,
    fusion::confidence::{ConfidenceScore, ConfidenceFactors},
};

/// Generic sensor model trait
/// 
/// Defines the interface for sensor-specific physics models
/// used in fusion algorithms.
pub trait SensorModel: Send {
    /// Sensor type this model represents
    fn sensor_type(&self) -> SensorType;
    
    /// Sensor identifier
    fn sensor_id(&self) -> &str;
    
    /// Get measurement noise variance
    fn noise_variance(&self) -> f32;
    
    /// State transition function
    /// 
    /// Predicts how the state evolves over time based on physics
    fn predict_state(&self, current_state: f32, dt_ms: u32) -> f32;
    
    /// Measurement model
    /// 
    /// Maps true state to expected sensor reading
    fn measurement_model(&self, true_state: f32) -> f32;
    
    /// Validate measurement against physical constraints
    fn validate(&self, measurement: f32) -> Result<(), ValidationError>;
    
    /// Environmental compensation
    /// 
    /// Corrects measurement based on environmental conditions
    fn compensate(&self, measurement: f32, env: &EnvironmentalConditions) -> f32;
    
    /// Compute confidence factors for this measurement
    fn confidence_factors(
        &self,
        measurement: f32,
        prediction: f32,
        env: &EnvironmentalConditions,
    ) -> ConfidenceFactors;
}

/// Environmental conditions for compensation
#[derive(Debug, Clone, Copy)]
pub struct EnvironmentalConditions {
    /// Ambient temperature (°C)
    pub temperature: f32,
    /// Atmospheric pressure (hPa)
    pub pressure: f32,
    /// Relative humidity (%)
    pub humidity: f32,
    /// Altitude above sea level (m)
    pub altitude: f32,
}

impl Default for EnvironmentalConditions {
    fn default() -> Self {
        Self {
            temperature: 20.0,  // Room temperature
            pressure: 1013.25,  // Sea level
            humidity: 50.0,     // Moderate humidity
            altitude: 0.0,      // Sea level
        }
    }
}

/// Temperature sensor model
/// 
/// Models thermal dynamics including:
/// - Thermal mass effects
/// - Ambient heat exchange
/// - Sensor self-heating
pub struct TemperatureModel {
    sensor_id: String,
    /// Thermal mass (J/K) - affects response time
    thermal_mass: f32,
    /// Heat transfer coefficient (W/K)
    heat_transfer: f32,
    /// Measurement noise standard deviation (°C)
    noise_std: f32,
    /// Valid measurement range (°C)
    min_temp: f32,
    max_temp: f32,
    /// Self-heating offset (°C)
    self_heating: f32,
}

impl TemperatureModel {
    /// Create with default parameters
    pub fn new(sensor_id: &str) -> Self {
        Self {
            sensor_id: sensor_id.to_string(),
            thermal_mass: 10.0,    // Small sensor
            heat_transfer: 0.1,    // Moderate insulation
            noise_std: 0.1,        // ±0.1°C noise
            min_temp: -40.0,       // Typical range
            max_temp: 85.0,
            self_heating: 0.0,     // No self-heating
        }
    }
    
    /// Set thermal mass (affects response time)
    pub fn with_thermal_mass(mut self, mass_j_per_k: f32) -> Self {
        self.thermal_mass = mass_j_per_k;
        self
    }
    
    /// Set measurement noise
    pub fn with_noise_std(mut self, std_dev_celsius: f32) -> Self {
        self.noise_std = std_dev_celsius;
        self
    }
    
    /// Set valid temperature range
    pub fn with_range(mut self, min: f32, max: f32) -> Self {
        self.min_temp = min;
        self.max_temp = max;
        self
    }
}

impl SensorModel for TemperatureModel {
    fn sensor_type(&self) -> SensorType {
        SensorType::Temperature
    }
    
    fn sensor_id(&self) -> &str {
        &self.sensor_id
    }
    
    fn noise_variance(&self) -> f32 {
        self.noise_std * self.noise_std
    }
    
    fn predict_state(&self, current_state: f32, dt_ms: u32) -> f32 {
        // Newton's law of cooling: dT/dt = -k(T - T_ambient)
        // Discrete approximation: T_new = T + dt × (-k(T - T_ambient))
        
        let dt_s = dt_ms as f32 / 1000.0;
        let k = self.heat_transfer / self.thermal_mass;
        
        // Assume ambient temperature of 20°C if not specified
        let t_ambient = 20.0;
        
        // Exponential decay toward ambient
        // Using Taylor series approximation for exp(-x)
        // exp(-x) ≈ 1 - x + x²/2 - x³/6 for small x
        let x = k * dt_s;
        let decay_factor = if x < 0.5 {
            1.0 - x + x * x * 0.5 - x * x * x / 6.0
        } else {
            // For larger x, use simpler approximation
            1.0 / (1.0 + x + 0.5 * x * x)
        };
        t_ambient + (current_state - t_ambient) * decay_factor
    }
    
    fn measurement_model(&self, true_state: f32) -> f32 {
        // Simple linear model with self-heating offset
        true_state + self.self_heating
    }
    
    fn validate(&self, measurement: f32) -> Result<(), ValidationError> {
        if measurement < self.min_temp || measurement > self.max_temp {
            return Err(ValidationError::OutOfRange {
                value: measurement,
                min: self.min_temp,
                max: self.max_temp,
            });
        }
        
        if !measurement.is_finite() {
            return Err(ValidationError::InvalidValue);
        }
        
        Ok(())
    }
    
    fn compensate(&self, measurement: f32, _env: &EnvironmentalConditions) -> f32 {
        // Temperature sensors typically don't need environmental compensation
        // Just remove self-heating if known
        measurement - self.self_heating
    }
    
    fn confidence_factors(
        &self,
        measurement: f32,
        prediction: f32,
        _env: &EnvironmentalConditions,
    ) -> ConfidenceFactors {
        let mut factors = ConfidenceFactors::moderate();
        
        // Statistical confidence from prediction error
        let error = (measurement - prediction).abs();
        let normalized_error = error / self.noise_std;
        factors.statistical = if normalized_error < 1.0 {
            ConfidenceScore::from_float(1.0 - normalized_error * 0.2)
        } else if normalized_error < 3.0 {
            ConfidenceScore::from_float(0.8 - (normalized_error - 1.0) * 0.3)
        } else {
            ConfidenceScore::MIN_CONFIDENCE
        };
        
        // Environmental confidence (operating range)
        let range_factor = if measurement > self.min_temp + 5.0 && measurement < self.max_temp - 5.0 {
            1.0 // Well within range
        } else if measurement > self.min_temp && measurement < self.max_temp {
            0.7 // Near limits
        } else {
            0.3 // Outside range
        };
        factors.environmental = ConfidenceScore::from_float(range_factor);
        
        // Cross-validation placeholder (would check against other temperature sensors)
        factors.cross_validation = ConfidenceScore::from_float(0.8);
        
        factors
    }
}

/// Pressure sensor model
/// 
/// Models atmospheric pressure with:
/// - Altitude compensation
/// - Temperature effects
/// - Weather variations
pub struct PressureModel {
    sensor_id: String,
    /// Measurement noise standard deviation (hPa)
    noise_std: f32,
    /// Valid measurement range (hPa)
    min_pressure: f32,
    max_pressure: f32,
    /// Reference altitude for calibration (m)
    reference_altitude: f32,
    /// Temperature coefficient (hPa/°C)
    temp_coefficient: f32,
}

impl PressureModel {
    /// Create with default parameters
    pub fn new(sensor_id: &str) -> Self {
        Self {
            sensor_id: sensor_id.to_string(),
            noise_std: 0.1,         // ±0.1 hPa noise
            min_pressure: 300.0,    // ~9000m altitude
            max_pressure: 1100.0,   // Storm conditions
            reference_altitude: 0.0, // Sea level
            temp_coefficient: 0.0,   // No temperature dependence
        }
    }
    
    /// Set reference altitude for local calibration
    pub fn with_altitude(mut self, altitude_m: f32) -> Self {
        self.reference_altitude = altitude_m;
        self
    }
}

impl SensorModel for PressureModel {
    fn sensor_type(&self) -> SensorType {
        SensorType::Pressure
    }
    
    fn sensor_id(&self) -> &str {
        &self.sensor_id
    }
    
    fn noise_variance(&self) -> f32 {
        self.noise_std * self.noise_std
    }
    
    fn predict_state(&self, current_state: f32, dt_ms: u32) -> f32 {
        // Atmospheric pressure changes slowly
        // Model as random walk with small variance
        let dt_hours = dt_ms as f32 / 3_600_000.0;
        
        // Typical pressure change rate: ~1 hPa/hour during weather fronts
        let max_change_rate = 1.0; // hPa/hour
        // Approximate sqrt using Newton's method
        let mut sqrt_dt = dt_hours;
        for _ in 0..2 { // 2 iterations sufficient for this use case
            sqrt_dt = 0.5 * (sqrt_dt + dt_hours / sqrt_dt);
        }
        let _process_noise = 0.1 * max_change_rate * sqrt_dt;
        
        // Bounded random walk (pressure tends to revert to mean)
        let mean_pressure = 1013.25; // Standard atmosphere
        let reversion_rate = 0.01; // Weak mean reversion
        
        let predicted = current_state + reversion_rate * (mean_pressure - current_state) * dt_hours;
        
        // Add bounds to prevent unrealistic values
        predicted.max(self.min_pressure).min(self.max_pressure)
    }
    
    fn measurement_model(&self, true_state: f32) -> f32 {
        // Linear measurement model
        true_state
    }
    
    fn validate(&self, measurement: f32) -> Result<(), ValidationError> {
        if measurement < self.min_pressure || measurement > self.max_pressure {
            return Err(ValidationError::OutOfRange {
                value: measurement,
                min: self.min_pressure,
                max: self.max_pressure,
            });
        }
        
        if !measurement.is_finite() {
            return Err(ValidationError::InvalidValue);
        }
        
        Ok(())
    }
    
    fn compensate(&self, measurement: f32, env: &EnvironmentalConditions) -> f32 {
        // Temperature compensation
        let temp_correction = self.temp_coefficient * (env.temperature - 20.0);
        
        // Altitude compensation using barometric formula
        // P = P0 * (1 - 0.0065*h/T0)^5.255
        let altitude_diff = env.altitude - self.reference_altitude;
        let base = 1.0 - 0.0065 * altitude_diff / 288.15;
        
        // Approximate pow(base, 5.255) using exp(5.255 * ln(base))
        // For base close to 1, ln(base) ≈ base - 1
        // So pow(base, 5.255) ≈ exp(5.255 * (base - 1))
        let exponent = 5.255 * (base - 1.0);
        let altitude_factor = if exponent.abs() < 0.5 {
            // Taylor expansion of exp(x)
            1.0 + exponent + exponent * exponent * 0.5
        } else {
            // For larger values, use simpler approximation
            if exponent > 0.0 {
                1.0 + exponent * (1.0 + 0.5 * exponent)
            } else {
                1.0 / (1.0 - exponent * (1.0 - 0.5 * exponent))
            }
        };
        
        (measurement + temp_correction) / altitude_factor
    }
    
    fn confidence_factors(
        &self,
        measurement: f32,
        prediction: f32,
        env: &EnvironmentalConditions,
    ) -> ConfidenceFactors {
        let mut factors = ConfidenceFactors::moderate();
        
        // Statistical confidence
        let error = (measurement - prediction).abs();
        let normalized_error = error / self.noise_std;
        factors.statistical = ConfidenceScore::from_float((1.0 - normalized_error * 0.1).max(0.2));
        
        // Environmental confidence based on altitude
        let altitude_confidence = if env.altitude < 3000.0 {
            1.0 // Normal operating altitude
        } else if env.altitude < 5000.0 {
            0.7 // High altitude, less accurate
        } else {
            0.4 // Very high altitude
        };
        factors.environmental = ConfidenceScore::from_float(altitude_confidence);
        
        factors
    }
}

/// Humidity sensor model
/// 
/// Models relative humidity with:
/// - Temperature dependence
/// - Saturation constraints
/// - Hysteresis effects
pub struct HumidityModel {
    sensor_id: String,
    /// Measurement noise standard deviation (%)
    noise_std: f32,
    /// Hysteresis width (%)
    hysteresis: f32,
    /// Response time constant (ms)
    response_time: f32,
}

impl HumidityModel {
    /// Create with default parameters
    pub fn new(sensor_id: &str) -> Self {
        Self {
            sensor_id: sensor_id.to_string(),
            noise_std: 2.0,      // ±2% RH noise
            hysteresis: 1.0,     // ±1% RH hysteresis
            response_time: 8000.0, // 8 second response
        }
    }
}

impl SensorModel for HumidityModel {
    fn sensor_type(&self) -> SensorType {
        SensorType::Humidity
    }
    
    fn sensor_id(&self) -> &str {
        &self.sensor_id
    }
    
    fn noise_variance(&self) -> f32 {
        self.noise_std * self.noise_std
    }
    
    fn predict_state(&self, current_state: f32, dt_ms: u32) -> f32 {
        // Humidity changes follow exponential approach to equilibrium
        // Limited by sensor response time
        let tau = self.response_time;
        let x = (dt_ms as f32) / tau;
        
        // Approximate exp(-x) using Taylor series
        let decay = if x < 0.5 {
            1.0 - x + x * x * 0.5 - x * x * x / 6.0
        } else if x < 2.0 {
            // Padé approximation for medium range
            (1.0 - 0.5 * x) / (1.0 + 0.5 * x)
        } else {
            // For large x, exp(-x) ≈ 0
            0.1
        };
        
        // Assume slow drift toward 50% RH (typical indoor)
        let equilibrium = 50.0;
        equilibrium + (current_state - equilibrium) * decay
    }
    
    fn measurement_model(&self, true_state: f32) -> f32 {
        // Add hysteresis effect (simplified)
        true_state
    }
    
    fn validate(&self, measurement: f32) -> Result<(), ValidationError> {
        if measurement < 0.0 || measurement > 100.0 {
            return Err(ValidationError::OutOfRange {
                value: measurement,
                min: 0.0,
                max: 100.0,
            });
        }
        
        if !measurement.is_finite() {
            return Err(ValidationError::InvalidValue);
        }
        
        Ok(())
    }
    
    fn compensate(&self, measurement: f32, _env: &EnvironmentalConditions) -> f32 {
        // Humidity sensors typically provide temperature-compensated readings
        // Just ensure physical constraints
        measurement.max(0.0).min(100.0)
    }
    
    fn confidence_factors(
        &self,
        measurement: f32,
        prediction: f32,
        env: &EnvironmentalConditions,
    ) -> ConfidenceFactors {
        let mut factors = ConfidenceFactors::moderate();
        
        // Statistical confidence
        let error = (measurement - prediction).abs();
        let normalized_error = error / self.noise_std;
        factors.statistical = ConfidenceScore::from_float((1.0 - normalized_error * 0.1).max(0.3));
        
        // Environmental confidence based on temperature
        // Humidity sensors less accurate at temperature extremes
        let temp_confidence = if env.temperature > 0.0 && env.temperature < 50.0 {
            1.0
        } else if env.temperature > -20.0 && env.temperature < 70.0 {
            0.7
        } else {
            0.4
        };
        factors.environmental = ConfidenceScore::from_float(temp_confidence);
        
        // Cross-validation with dew point constraint
        let dew_point_valid = measurement <= 100.0; // Simplified check
        factors.cross_validation = ConfidenceScore::from_float(if dew_point_valid { 0.9 } else { 0.3 });
        
        factors
    }
}

/// State transition model for Kalman filter
/// 
/// Defines how system state evolves over time
#[derive(Debug, Clone)]
pub struct StateTransition<const N: usize> {
    /// State transition matrix F
    pub transition_matrix: [[f32; N]; N],
    /// Process noise covariance Q
    pub process_noise: [[f32; N]; N],
}

impl<const N: usize> StateTransition<N> {
    /// Create identity transition (no dynamics)
    pub fn identity() -> Self {
        let mut transition_matrix = [[0.0; N]; N];
        let mut process_noise = [[0.0; N]; N];
        
        for i in 0..N {
            transition_matrix[i][i] = 1.0;
            process_noise[i][i] = 0.01; // Small process noise
        }
        
        Self {
            transition_matrix,
            process_noise,
        }
    }
    
    /// Create constant velocity model (position + velocity)
    pub fn constant_velocity(dt: f32, process_noise_std: f32) -> StateTransition<2> {
        StateTransition {
            transition_matrix: [
                [1.0, dt],  // position += velocity * dt
                [0.0, 1.0], // velocity stays constant
            ],
            process_noise: [
                [0.25 * dt * dt * dt * dt * process_noise_std, 0.5 * dt * dt * dt * process_noise_std],
                [0.5 * dt * dt * dt * process_noise_std, dt * dt * process_noise_std],
            ],
        }
    }
}

/// Factory functions for common sensor models
pub mod sensor_models {
    use super::*;
    
    /// Create temperature sensor model with standard parameters
    pub fn temperature(sensor_id: &str, noise_std: f32) -> TemperatureModel {
        TemperatureModel::new(sensor_id).with_noise_std(noise_std)
    }
    
    /// Create pressure sensor model with standard parameters
    pub fn pressure(sensor_id: &str, _noise_std: f32) -> PressureModel {
        PressureModel::new(sensor_id)
    }
    
    /// Create humidity sensor model with standard parameters
    pub fn humidity(sensor_id: &str, _noise_std: f32) -> HumidityModel {
        HumidityModel::new(sensor_id)
    }
}

/// Extended Kalman Filter models for specific sensor types
/// 
/// These models provide non-linear state transition and measurement functions
/// for use with the EKF implementation.
pub mod ekf_models {
    use super::*;
    use crate::fusion::kalman::{ExtendedKalmanFilter, KalmanConfig};
    use crate::fusion::matrix::Vector;
    use crate::fusion::FusionAlgorithm;
    
    /// Temperature EKF model with thermal dynamics
    /// 
    /// State vector: [temperature, ambient_temp, cooling_rate]
    /// - temperature: Current sensor temperature (°C)
    /// - ambient_temp: Estimated ambient temperature (°C)  
    /// - cooling_rate: Heat transfer coefficient (1/s)
    pub struct TemperatureEKF {
        /// Thermal mass affects how quickly temperature changes
        thermal_mass: f32,
        /// Measurement noise standard deviation
        measurement_noise: f32,
        /// Process noise for temperature evolution
        process_noise: f32,
    }
    
    impl TemperatureEKF {
        /// Create new temperature EKF model
        pub fn new(thermal_mass: f32, measurement_noise: f32) -> Self {
            Self {
                thermal_mass,
                measurement_noise,
                process_noise: 0.01, // Default process noise
            }
        }
        
        /// State transition function for temperature dynamics
        /// 
        /// Implements Newton's law of cooling:
        /// dT/dt = -k(T - T_ambient)
        pub fn state_transition(state: &Vector<3>, _control: &Vector<3>) -> Vector<3> {
            let temperature = state[0];
            let ambient = state[1];
            let cooling_rate = state[2];
            
            // Time step (assumed 0.1s for now - would be passed via control)
            let dt = 0.1;
            
            // Newton's cooling law
            let temp_change = -cooling_rate * (temperature - ambient) * dt;
            
            [
                temperature + temp_change,     // New temperature
                ambient,                       // Ambient slowly drifts
                cooling_rate,                  // Cooling rate constant
            ]
        }
        
        /// Measurement function - sensor measures temperature directly
        pub fn measurement_function(state: &Vector<3>) -> Vector<1> {
            [state[0]] // Direct temperature measurement
        }
        
        /// Create configured EKF for temperature tracking
        pub fn create_filter(&self) -> ExtendedKalmanFilter<3, 1> {
            let mut config = KalmanConfig::<3, 1>::default();
            
            // Initial state: [25°C, 20°C ambient, 0.1 cooling rate]
            config.initial_state = [25.0, 20.0, 0.1];
            
            // Initial covariance (uncertainty)
            config.initial_covariance = [
                [1.0, 0.0, 0.0],   // Temperature uncertainty
                [0.0, 4.0, 0.0],   // Ambient uncertainty  
                [0.0, 0.0, 0.01],  // Cooling rate uncertainty
            ];
            
            // Process noise (how much state can change)
            config.process_noise = [
                [self.process_noise, 0.0, 0.0],
                [0.0, 0.001, 0.0],  // Ambient drifts slowly
                [0.0, 0.0, 0.0001], // Cooling rate nearly constant
            ];
            
            // Measurement noise
            config.measurement_noise = [[self.measurement_noise * self.measurement_noise]];
            
            // Measurement matrix H: we only observe temperature
            config.measurement_matrix = [[1.0, 0.0, 0.0]];
            
            ExtendedKalmanFilter::new(
                config,
                Self::state_transition,
                Self::measurement_function,
            )
        }
    }
    
    /// Pressure-Altitude EKF model
    /// 
    /// State vector: [pressure, altitude, vertical_velocity]
    /// - pressure: Atmospheric pressure (hPa)
    /// - altitude: Estimated altitude (m)
    /// - vertical_velocity: Rate of altitude change (m/s)
    pub struct PressureAltitudeEKF {
        /// Sea level pressure for calibration
        sea_level_pressure: f32,
        /// Measurement noise for pressure sensor
        pressure_noise: f32,
        /// Temperature for pressure calculations
        temperature: f32,
    }
    
    impl PressureAltitudeEKF {
        /// Create new pressure-altitude EKF model
        pub fn new(sea_level_pressure: f32, pressure_noise: f32) -> Self {
            Self {
                sea_level_pressure,
                pressure_noise,
                temperature: 288.15, // Standard temperature (15°C)
            }
        }
        
        /// State transition with barometric physics
        pub fn state_transition(state: &Vector<3>, _control: &Vector<3>) -> Vector<3> {
            let _pressure = state[0];
            let altitude = state[1];
            let velocity = state[2];
            
            let dt = 0.1; // Time step
            
            // Update altitude based on velocity
            let new_altitude = altitude + velocity * dt;
            
            // Pressure follows barometric formula
            // P = P0 * (1 - 0.0065*h/T)^5.255
            // Simplified: P ≈ P0 * exp(-h/H) where H ≈ 8400m
            let scale_height = 8400.0;
            let pressure_ratio = (-new_altitude / scale_height).min(10.0).max(-10.0);
            
            // Approximate exp using Taylor series
            let exp_factor = if pressure_ratio.abs() < 0.5 {
                1.0 + pressure_ratio + pressure_ratio * pressure_ratio * 0.5
            } else {
                // For larger values, use Padé approximation
                let x = pressure_ratio;
                (1.0 + x * 0.5 + x * x / 12.0) / (1.0 - x * 0.5 + x * x / 12.0)
            };
            
            let new_pressure = 1013.25 * exp_factor; // Sea level reference
            
            [
                new_pressure,
                new_altitude,
                velocity * 0.95, // Velocity decays slightly
            ]
        }
        
        /// Measurement function - we measure pressure
        pub fn measurement_function(state: &Vector<3>) -> Vector<1> {
            [state[0]] // Direct pressure measurement
        }
        
        /// Create configured EKF for pressure-altitude fusion
        pub fn create_filter(&self) -> ExtendedKalmanFilter<3, 1> {
            let mut config = KalmanConfig::<3, 1>::default();
            
            // Initial state: [sea level pressure, 0m altitude, 0 m/s]
            config.initial_state = [self.sea_level_pressure, 0.0, 0.0];
            
            // Initial covariance
            config.initial_covariance = [
                [1.0, 0.0, 0.0],    // Pressure uncertainty
                [0.0, 100.0, 0.0],  // Altitude uncertainty (10m)
                [0.0, 0.0, 0.1],    // Velocity uncertainty
            ];
            
            // Process noise
            config.process_noise = [
                [0.1, 0.0, 0.0],    // Pressure variation
                [0.0, 0.01, 0.0],   // Altitude drift
                [0.0, 0.0, 0.1],    // Velocity changes
            ];
            
            // Measurement noise
            config.measurement_noise = [[self.pressure_noise * self.pressure_noise]];
            
            // We only measure pressure
            config.measurement_matrix = [[1.0, 0.0, 0.0]];
            
            ExtendedKalmanFilter::new(
                config,
                Self::state_transition,
                Self::measurement_function,
            )
        }
        
        /// Extract altitude estimate from filter state
        pub fn get_altitude(ekf: &ExtendedKalmanFilter<3, 1>) -> f32 {
            // Access state through trait method
            FusionAlgorithm::state(&ekf.kf)[1]
        }
    }
    
    /// Humidity EKF model with hysteresis compensation
    /// 
    /// State vector: [true_humidity, sensor_reading, hysteresis_state]
    /// - true_humidity: Actual relative humidity (%)
    /// - sensor_reading: What the sensor currently reads (%)
    /// - hysteresis_state: Memory effect from previous readings
    pub struct HumidityEKF {
        /// Hysteresis time constant (seconds)
        hysteresis_tau: f32,
        /// Measurement noise
        measurement_noise: f32,
        /// Maximum humidity (constrained by physics)
        max_humidity: f32,
    }
    
    impl HumidityEKF {
        /// Create new humidity EKF model
        pub fn new(measurement_noise: f32) -> Self {
            Self {
                hysteresis_tau: 5.0, // 5 second hysteresis
                measurement_noise,
                max_humidity: 100.0,
            }
        }
        
        /// State transition with hysteresis modeling
        pub fn state_transition(state: &Vector<3>, _control: &Vector<3>) -> Vector<3> {
            let true_humidity = state[0];
            let sensor_reading = state[1];
            let hysteresis = state[2];
            
            let dt = 0.1;
            let tau = 5.0; // Hysteresis time constant
            
            // Sensor reading approaches true value with hysteresis
            let reading_rate = (true_humidity - sensor_reading - hysteresis) / tau;
            let new_reading = sensor_reading + reading_rate * dt;
            
            // Hysteresis decays over time
            let hysteresis_decay = -hysteresis / (tau * 2.0);
            let new_hysteresis = hysteresis + hysteresis_decay * dt;
            
            // Constrain to physical limits
            let constrained_reading = new_reading.max(0.0).min(100.0);
            
            [
                true_humidity,       // True value evolves slowly
                constrained_reading, // Sensor reading with hysteresis
                new_hysteresis,     // Hysteresis state
            ]
        }
        
        /// Measurement function - we measure sensor reading
        pub fn measurement_function(state: &Vector<3>) -> Vector<1> {
            [state[1]] // Sensor reading (includes hysteresis)
        }
        
        /// Create configured EKF for humidity tracking
        pub fn create_filter(&self) -> ExtendedKalmanFilter<3, 1> {
            let mut config = KalmanConfig::<3, 1>::default();
            
            // Initial state: [50% true, 50% reading, 0 hysteresis]
            config.initial_state = [50.0, 50.0, 0.0];
            
            // Initial covariance
            config.initial_covariance = [
                [4.0, 2.0, 0.0],   // True humidity uncertainty
                [2.0, 4.0, 0.0],   // Reading uncertainty (correlated)
                [0.0, 0.0, 1.0],   // Hysteresis uncertainty
            ];
            
            // Process noise
            config.process_noise = [
                [0.1, 0.0, 0.0],   // True humidity changes
                [0.0, 0.01, 0.0],  // Sensor drift
                [0.0, 0.0, 0.05],  // Hysteresis variation
            ];
            
            // Measurement noise
            config.measurement_noise = [[self.measurement_noise * self.measurement_noise]];
            
            // We measure the sensor reading
            config.measurement_matrix = [[0.0, 1.0, 0.0]];
            
            ExtendedKalmanFilter::new(
                config,
                Self::state_transition,
                Self::measurement_function,
            )
        }
        
        /// Get compensated humidity (true value estimate)
        pub fn get_true_humidity(ekf: &ExtendedKalmanFilter<3, 1>) -> f32 {
            // Access state through trait method
            FusionAlgorithm::state(&ekf.kf)[0]
        }
    }
    
    /// Multi-sensor fusion EKF for environmental monitoring
    /// 
    /// Fuses temperature, humidity, and pressure for better estimates
    /// State vector: [temp, humidity, pressure, dew_point, altitude]
    pub struct EnvironmentalEKF {
        /// Individual sensor noise levels
        temp_noise: f32,
        humidity_noise: f32,
        pressure_noise: f32,
    }
    
    impl EnvironmentalEKF {
        /// Create new environmental fusion EKF
        pub fn new(temp_noise: f32, humidity_noise: f32, pressure_noise: f32) -> Self {
            Self {
                temp_noise,
                humidity_noise,
                pressure_noise,
            }
        }
        
        /// State transition with coupled environmental physics
        pub fn state_transition(state: &Vector<5>, _control: &Vector<5>) -> Vector<5> {
            let temp = state[0];
            let humidity = state[1];
            let pressure = state[2];
            let _dew_point = state[3];
            let _altitude = state[4];
            
            let _dt = 0.1;
            
            // Temperature and humidity affect each other slowly
            // Higher temperature allows more moisture
            let max_humidity = 100.0; // Simplified - would use Magnus formula
            let humidity_constrained = humidity.min(max_humidity);
            
            // Dew point calculation (simplified Magnus formula)
            // Td = T - (100 - RH) / 5
            let new_dew_point = temp - (100.0 - humidity_constrained) / 5.0;
            
            // Altitude from pressure (simplified barometric formula)
            let pressure_ratio = pressure / 1013.25;
            let ln_ratio = if (pressure_ratio - 1.0).abs() < 0.3 {
                // Taylor series for ln(1 + x)
                let x = pressure_ratio - 1.0;
                x - x * x * 0.5 + x * x * x / 3.0
            } else {
                // Newton's method for ln(pressure_ratio)
                let mut ln_est = pressure_ratio - 1.0;
                for _ in 0..3 {
                    let exp_ln = if ln_est.abs() < 0.5 {
                        1.0 + ln_est + ln_est * ln_est * 0.5
                    } else {
                        1.0 + ln_est
                    };
                    ln_est = ln_est - (exp_ln - pressure_ratio) / exp_ln;
                }
                ln_est
            };
            let new_altitude = -8400.0 * ln_ratio;
            
            [
                temp,                  // Temperature evolves slowly
                humidity_constrained,  // Humidity constrained by physics
                pressure,             // Pressure from measurements
                new_dew_point,        // Calculated dew point
                new_altitude,         // Calculated altitude
            ]
        }
        
        /// Measurement function - we measure temp, humidity, pressure
        pub fn measurement_function(state: &Vector<5>) -> Vector<3> {
            [
                state[0], // Temperature
                state[1], // Humidity
                state[2], // Pressure
            ]
        }
        
        /// Create configured EKF for environmental monitoring
        pub fn create_filter(&self) -> ExtendedKalmanFilter<5, 3> {
            let mut config = KalmanConfig::<5, 3>::default();
            
            // Initial state: typical indoor conditions
            config.initial_state = [20.0, 50.0, 1013.25, 10.0, 0.0];
            
            // Initial covariance
            config.initial_covariance = [
                [1.0, 0.0, 0.0, 0.5, 0.0],   // Temp uncertainty
                [0.0, 4.0, 0.0, 0.5, 0.0],   // Humidity uncertainty
                [0.0, 0.0, 1.0, 0.0, 0.5],   // Pressure uncertainty
                [0.5, 0.5, 0.0, 2.0, 0.0],   // Dew point (derived)
                [0.0, 0.0, 0.5, 0.0, 100.0], // Altitude (derived)
            ];
            
            // Process noise
            config.process_noise = [
                [0.01, 0.0, 0.0, 0.0, 0.0],  // Temp variation
                [0.0, 0.1, 0.0, 0.0, 0.0],   // Humidity variation
                [0.0, 0.0, 0.1, 0.0, 0.0],   // Pressure variation
                [0.0, 0.0, 0.0, 0.01, 0.0],  // Dew point (calculated)
                [0.0, 0.0, 0.0, 0.0, 1.0],   // Altitude (calculated)
            ];
            
            // Measurement noise
            config.measurement_noise = [
                [self.temp_noise * self.temp_noise, 0.0, 0.0],
                [0.0, self.humidity_noise * self.humidity_noise, 0.0],
                [0.0, 0.0, self.pressure_noise * self.pressure_noise],
            ];
            
            // Measurement matrix - we observe first 3 states
            config.measurement_matrix = [
                [1.0, 0.0, 0.0, 0.0, 0.0], // Temperature
                [0.0, 1.0, 0.0, 0.0, 0.0], // Humidity
                [0.0, 0.0, 1.0, 0.0, 0.0], // Pressure
            ];
            
            ExtendedKalmanFilter::new(
                config,
                Self::state_transition,
                Self::measurement_function,
            )
        }
        
        /// Get derived environmental parameters
        pub fn get_derived_values(ekf: &ExtendedKalmanFilter<5, 3>) -> (f32, f32) {
            // Access state through trait method
            let state = FusionAlgorithm::state(&ekf.kf);
            (state[3], state[4]) // (dew_point, altitude)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::FusionAlgorithm;
    
    #[test]
    fn temperature_model_validation() {
        let model = TemperatureModel::new("temp1")
            .with_range(-40.0, 85.0);
        
        // Valid measurements
        assert!(model.validate(25.0).is_ok());
        assert!(model.validate(-20.0).is_ok());
        
        // Out of range
        assert!(model.validate(-50.0).is_err());
        assert!(model.validate(100.0).is_err());
        
        // Invalid values
        assert!(model.validate(f32::NAN).is_err());
        assert!(model.validate(f32::INFINITY).is_err());
    }
    
    #[test]
    fn pressure_compensation() {
        let model = PressureModel::new("pressure1")
            .with_altitude(0.0); // Sea level reference
        
        let env = EnvironmentalConditions {
            altitude: 1000.0, // 1000m altitude
            ..Default::default()
        };
        
        // Pressure at 1000m should be ~900 hPa
        let measured = 900.0;
        let compensated = model.compensate(measured, &env);
        
        // Should be adjusted to sea level equivalent (~1013 hPa)
        assert!(compensated > measured);
        assert!((compensated - 1013.0).abs() < 20.0);
    }
    
    #[test]
    fn humidity_constraints() {
        let model = HumidityModel::new("humidity1");
        
        // Test saturation constraints
        assert!(model.validate(50.0).is_ok());
        assert!(model.validate(100.0).is_ok());
        assert!(model.validate(101.0).is_err());
        assert!(model.validate(-1.0).is_err());
    }
    
    #[test]
    fn state_transition_models() {
        // Test constant velocity model
        let dt = 0.1; // 100ms
        let cv_model = StateTransition::constant_velocity(dt, 0.1);
        
        // Check transition matrix
        assert_eq!(cv_model.transition_matrix[0][0], 1.0);
        assert_eq!(cv_model.transition_matrix[0][1], dt);
        assert_eq!(cv_model.transition_matrix[1][0], 0.0);
        assert_eq!(cv_model.transition_matrix[1][1], 1.0);
        
        // Process noise should be positive definite
        assert!(cv_model.process_noise[0][0] > 0.0);
        assert!(cv_model.process_noise[1][1] > 0.0);
    }
    
    #[test]
    fn temperature_ekf_model() {
        use ekf_models::TemperatureEKF;
        
        let model = TemperatureEKF::new(10.0, 0.1);
        let filter = model.create_filter();
        
        // Test state transition
        let state = [25.0, 20.0, 0.1]; // 25°C, 20°C ambient, 0.1 cooling rate
        let control = [0.0, 0.0, 0.0];
        let new_state = TemperatureEKF::state_transition(&state, &control);
        
        // Temperature should decrease toward ambient
        assert!(new_state[0] < state[0]);
        assert!(new_state[0] > state[1]); // But still above ambient
        
        // Measurement function should return temperature
        let measurement = TemperatureEKF::measurement_function(&state);
        assert_eq!(measurement[0], state[0]);
    }
    
    #[test]
    fn pressure_altitude_ekf() {
        use ekf_models::PressureAltitudeEKF;
        
        let model = PressureAltitudeEKF::new(1013.25, 0.1);
        let mut filter = model.create_filter();
        
        // Test altitude calculation
        // At 1000m, pressure should be ~900 hPa
        let state = [900.0, 1000.0, 0.0]; // 900 hPa, 1000m, 0 m/s
        let new_state = PressureAltitudeEKF::state_transition(&state, &[0.0, 0.0, 0.0]);
        
        // Pressure and altitude should be consistent
        assert!((new_state[0] - 900.0).abs() < 50.0); // Reasonable pressure
        assert!((new_state[1] - 1000.0).abs() < 10.0); // Altitude stable
    }
    
    #[test]
    fn humidity_ekf_hysteresis() {
        use ekf_models::HumidityEKF;
        
        let model = HumidityEKF::new(2.0);
        let filter = model.create_filter();
        
        // Test hysteresis effect
        let state = [70.0, 50.0, 5.0]; // 70% true, 50% reading, 5% hysteresis
        let new_state = HumidityEKF::state_transition(&state, &[0.0, 0.0, 0.0]);
        
        // Reading should move toward true value
        assert!(new_state[1] > state[1]); // Increasing toward 70%
        assert!(new_state[1] < state[0]); // But not there yet
        
        // Hysteresis should decay
        assert!(new_state[2].abs() < state[2].abs());
    }
    
    #[test]
    fn environmental_ekf_fusion() {
        use ekf_models::EnvironmentalEKF;
        
        let model = EnvironmentalEKF::new(0.1, 2.0, 0.1);
        let mut filter = model.create_filter();
        
        // Test state relationships
        let state = [20.0, 60.0, 1013.25, 0.0, 0.0]; // 20°C, 60% RH, sea level
        let new_state = EnvironmentalEKF::state_transition(&state, &[0.0; 5]);
        
        // Dew point should be calculated correctly
        // Approximation: Td ≈ T - (100 - RH) / 5
        let expected_dew = 20.0 - (100.0 - 60.0) / 5.0;
        assert!((new_state[3] - expected_dew).abs() < 1.0);
        
        // Altitude should be near 0 at sea level pressure
        assert!(new_state[4].abs() < 10.0);
    }
}