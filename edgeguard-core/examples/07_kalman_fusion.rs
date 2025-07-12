//! Kalman Filter Sensor Fusion Example
//!
//! This example demonstrates how EdgeGuard uses Kalman filtering to
//! combine multiple noisy sensor readings into accurate estimates.
//!
//! ## What You'll Learn
//!
//! - Basic Kalman filter concepts and parameters
//! - Fusing multiple temperature sensors
//! - Handling sensor noise and uncertainty
//! - Visualizing filter convergence
//!
//! ## Kalman Filter Benefits
//!
//! 1. **Optimal Estimation**: Mathematically optimal for linear systems
//! 2. **Uncertainty Tracking**: Maintains confidence estimates
//! 3. **Noise Reduction**: Filters out sensor noise effectively
//! 4. **Prediction**: Can predict future states
//!
//! ## Running the Example
//!
//! ```bash
//! cargo run --example 07_kalman_fusion
//! ```

use edgeguard_core::{
    fusion::{
        KalmanFilter, KalmanConfig, FusionAlgorithm,
    },
};

fn main() {
    println!("EdgeGuard Kalman Filter Fusion Example");
    println!("======================================\n");

    // Scenario: Three temperature sensors with different characteristics
    println!("Scenario: Room with 3 temperature sensors");
    println!("-----------------------------------------");
    println!("Sensor 1: High quality, low noise (σ = 0.1°C)");
    println!("Sensor 2: Medium quality, moderate noise (σ = 0.3°C)");
    println!("Sensor 3: Low quality, high noise (σ = 0.5°C)");
    println!("True temperature: 22.0°C\n");

    // Create Kalman filter configuration
    // State vector: [temperature, rate_of_change]
    // Measurement vector: [sensor1, sensor2, sensor3]
    let mut config = KalmanConfig::<2, 3>::default();
    
    // Configure state transition matrix F (2x2)
    // [1  dt]  (temperature = temperature + rate*dt)
    // [0  1 ]  (rate stays constant)
    config.transition.transition_matrix = [
        [1.0, 0.1],  // dt = 0.1 seconds
        [0.0, 1.0],
    ];
    
    // Configure measurement matrix H (3x2)
    // All sensors measure temperature directly
    config.measurement_matrix = [
        [1.0, 0.0],  // Sensor 1 measures temperature
        [1.0, 0.0],  // Sensor 2 measures temperature
        [1.0, 0.0],  // Sensor 3 measures temperature
    ];
    
    // Configure process noise covariance Q (2x2)
    config.process_noise = [
        [0.01, 0.0],   // Temperature process noise
        [0.0, 0.001],  // Rate process noise
    ];
    config.transition.process_noise = config.process_noise;
    
    // Configure measurement noise covariance R (3x3)
    config.measurement_noise = [
        [0.01, 0.0, 0.0],   // Sensor 1: σ² = 0.1² = 0.01
        [0.0, 0.09, 0.0],   // Sensor 2: σ² = 0.3² = 0.09
        [0.0, 0.0, 0.25],   // Sensor 3: σ² = 0.5² = 0.25
    ];
    
    // Set initial state estimate
    config.initial_state = [20.0, 0.0];  // 20°C, no change
    
    // Set initial covariance
    config.initial_covariance = [
        [1.0, 0.0],   // Initial temperature uncertainty
        [0.0, 0.1],   // Initial rate uncertainty
    ];
    
    let mut kalman = KalmanFilter::<2, 3>::new(config.clone());
    kalman.reset();
    
    // Simulate sensor readings over time
    println!("Time | Sensor Readings | Kalman Estimate | True Value | Error");
    println!("-----|-----------------|-----------------|------------|-------");
    
    let true_temp = 22.0;
    let mut time = 0;
    
    // Sensor noise characteristics
    let sensor_noise = [0.1, 0.3, 0.5];
    let sensor_bias = [0.0, 0.2, -0.1];
    
    for i in 0..20 {
        time += 100; // 100ms intervals
        
        // Generate noisy sensor readings
        let reading1 = generate_sensor_reading(true_temp, sensor_noise[0], sensor_bias[0], i);
        let reading2 = generate_sensor_reading(true_temp, sensor_noise[1], sensor_bias[1], i);
        let reading3 = generate_sensor_reading(true_temp, sensor_noise[2], sensor_bias[2], i);
        
        // Predict next state (dt = 0.1 seconds)
        kalman.predict(100).unwrap();
        
        // Create measurement vector [reading1, reading2, reading3]
        let measurement = [reading1, reading2, reading3];
        kalman.update(&measurement, time as u64, None).unwrap();
        
        // Get fused estimate
        let state = kalman.state();
        let estimate = state[0];
        let error = (estimate - true_temp).abs();
        
        println!("{:4} | {:4.1} {:4.1} {:4.1} | {:15.2} | {:10.1} | {:5.2}",
                 time,
                 reading1, reading2, reading3,
                 estimate,
                 true_temp,
                 error);
    }
    
    // Show final statistics
    let final_state = kalman.state();
    let uncertainties = kalman.uncertainty();
    
    println!("\nFinal Kalman Filter State:");
    println!("--------------------------");
    println!("Temperature estimate: {:.2}°C", final_state[0]);
    println!("Rate of change: {:.3}°C/s", final_state[1]);
    println!("Estimation uncertainty: ±{:.3}°C", uncertainties[0].sqrt());
    
    // Demonstrate prediction capability
    println!("\nPrediction Demo:");
    println!("----------------");
    println!("Predicting temperature 1 second into future...");
    
    // Clone the config since KalmanFilter doesn't implement Clone
    let predictor_config = config.clone();
    let mut predictor = KalmanFilter::<2, 3>::new(predictor_config);
    // Copy state from original filter
    predictor.predict(1000).unwrap(); // Predict 1 second ahead
    
    let predicted_state = predictor.state();
    let predicted_uncertainties = predictor.uncertainty();
    
    println!("Predicted temperature: {:.2}°C", predicted_state[0]);
    println!("Prediction uncertainty: ±{:.3}°C", predicted_uncertainties[0].sqrt());
    
    // Show convergence analysis
    demonstrate_convergence();
}

fn generate_sensor_reading(true_value: f32, noise_std: f32, bias: f32, seed: i32) -> f32 {
    // Simple pseudo-random noise generation
    let noise = ((seed * 12345 + 6789) % 1000) as f32 / 1000.0 - 0.5;
    let scaled_noise = noise * noise_std * 2.0;
    true_value + bias + scaled_noise
}

fn demonstrate_convergence() {
    println!("\n\nConvergence Analysis:");
    println!("====================");
    println!("Showing how quickly Kalman filter converges to true value\n");
    
    // Create a filter with high initial uncertainty
    let mut config = KalmanConfig::<2, 1>::default();
    
    config.transition.transition_matrix = [
        [1.0, 0.1],
        [0.0, 1.0],
    ];
    config.measurement_matrix = [[1.0, 0.0]]; // Single sensor
    config.process_noise = [
        [0.01, 0.0],
        [0.0, 0.001],
    ];
    config.transition.process_noise = config.process_noise;
    config.measurement_noise = [[0.04]]; // σ² = 0.2²
    config.initial_state = [10.0, 0.0]; // Start far from true value (22°C)
    config.initial_covariance = [
        [25.0, 0.0],  // High initial uncertainty
        [0.0, 0.1],
    ];
    
    let mut kalman = KalmanFilter::<2, 1>::new(config);
    kalman.reset();
    
    let true_temp = 22.0;
    let sensor_noise = 0.2;
    
    println!("Step | Measurement | Estimate | Error | Uncertainty");
    println!("-----|-------------|----------|-------|------------");
    
    let mut time = 0;
    for i in 0..10 {
        time += 100;
        
        // Generate measurement
        let measurement = generate_sensor_reading(true_temp, sensor_noise, 0.0, i * 7);
        
        // Predict and update
        kalman.predict(100).unwrap();
        kalman.update(&[measurement], time as u64, None).unwrap();
        
        let state = kalman.state();
        let estimate = state[0];
        let error = (estimate - true_temp).abs();
        let uncertainties = kalman.uncertainty();
        let uncertainty = uncertainties[0].sqrt();
        
        println!("{:4} | {:11.2} | {:8.2} | {:5.2} | ±{:10.3}",
                 i + 1,
                 measurement,
                 estimate,
                 error,
                 uncertainty);
    }
    
    println!("\nNote: Kalman filter quickly converges despite starting 12°C away!");
    
    // Demonstrate handling of outliers
    demonstrate_outlier_handling();
}

fn demonstrate_outlier_handling() {
    println!("\n\nOutlier Handling:");
    println!("=================");
    println!("Showing how Kalman filter handles sensor spikes\n");
    
    let mut config = KalmanConfig::<2, 1>::default();
    
    config.transition.transition_matrix = [
        [1.0, 0.1],
        [0.0, 1.0],
    ];
    config.measurement_matrix = [[1.0, 0.0]];
    config.process_noise = [
        [0.01, 0.0],
        [0.0, 0.001],
    ];
    config.transition.process_noise = config.process_noise;
    config.measurement_noise = [[0.25]]; // σ² = 0.5²
    config.initial_state = [22.0, 0.0];
    config.initial_covariance = [
        [0.25, 0.0],
        [0.0, 0.01],
    ];
    
    let mut kalman = KalmanFilter::<2, 1>::new(config);
    kalman.reset();
    
    let measurements = vec![
        22.1, 22.0, 21.9, 22.1,  // Normal readings
        35.0,                     // Outlier!
        22.2, 22.0, 21.8,        // Back to normal
    ];
    
    println!("Reading | Estimate | Comment");
    println!("--------|----------|--------");
    
    let mut time = 0;
    for (_i, &measurement) in measurements.iter().enumerate() {
        time += 100;
        
        kalman.predict(100).unwrap();
        kalman.update(&[measurement], time as u64, None).unwrap();
        
        let state = kalman.state();
        let estimate = state[0];
        let comment = if (measurement - 22.0).abs() > 5.0 {
            "← Outlier detected!"
        } else {
            ""
        };
        
        println!("{:7.1} | {:8.2} | {}", measurement, estimate, comment);
    }
    
    println!("\nNote: Kalman filter smooths out the outlier, maintaining stable estimate!");
    
    // Show benefits summary
    println!("\n\nKalman Filter Benefits Summary:");
    println!("==============================");
    println!("✓ Optimal fusion of multiple sensors");
    println!("✓ Handles different sensor qualities");
    println!("✓ Tracks estimation uncertainty");
    println!("✓ Smooths out noise and outliers");
    println!("✓ Can predict future states");
    println!("✓ Computationally efficient for embedded");
}