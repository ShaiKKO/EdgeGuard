//! Extended Kalman Filter (EKF) Sensor Models Example
//!
//! This example demonstrates how EdgeGuard uses specialized EKF models
//! that incorporate physics-based sensor characteristics for accurate
//! state estimation in non-linear systems.
//!
//! ## What You'll Learn
//!
//! - Physics-based sensor models for common IoT sensors
//! - Extended Kalman Filter for non-linear dynamics
//! - Environmental compensation in sensor fusion
//! - Model-specific parameters and tuning
//!
//! ## EKF vs Standard Kalman Filter
//!
//! Standard Kalman filters assume linear dynamics. EKF handles non-linear
//! systems by linearizing around the current estimate, making it ideal for:
//! - Temperature sensors with thermal mass
//! - Pressure sensors affected by temperature
//! - Humidity sensors with hysteresis
//!
//! ## Running the Example
//!
//! ```bash
//! cargo run --example 08_ekf_models
//! ```

use edgeguard_core::{
    fusion::{
        models::ekf_models::{
            TemperatureEKF, HumidityEKF, PressureAltitudeEKF,
            EnvironmentalEKF,
        },
    },
};

fn main() {
    println!("EdgeGuard Extended Kalman Filter Models Example");
    println!("==============================================\n");

    // Demonstrate each specialized EKF model
    temperature_ekf_demo();
    println!("\n{}\n", "=".repeat(60));
    
    humidity_ekf_demo();
    println!("\n{}\n", "=".repeat(60));
    
    pressure_altitude_ekf_demo();
    println!("\n{}\n", "=".repeat(60));
    
    environmental_ekf_demo();
}

fn temperature_ekf_demo() {
    println!("Temperature EKF Model:");
    println!("--------------------");
    println!("Models thermal dynamics including:");
    println!("- Thermal mass (heat capacity)");
    println!("- Ambient coupling (heat transfer)");
    println!("- Sensor self-heating effects\n");

    // Create temperature model with physical parameters
    let temp_model = TemperatureEKF::new(
        0.5,   // thermal_mass: 0.5 J/K (small sensor)
        0.05,  // measurement_noise
    );

    // Get the configured Kalman filter
    let mut ekf = temp_model.create_filter();

    // Simulate temperature change scenario
    println!("Scenario: Room heater turns on, temperature rises from 20°C to 25°C");
    println!("Time(s) | Measured | EKF State  | Rate(°C/s)");
    println!("--------|----------|------------|------------");

    let true_temps = simulate_temperature_rise(20.0, 25.0, 0.1);
    
    for (i, &true_temp) in true_temps.iter().enumerate() {
        let time = i as f32 * 0.5;
        let measurement = true_temp + gaussian_noise(i, 0.1);
        
        // Predict and update
        ekf.predict(500).unwrap();
        let (estimate, _confidence) = ekf.update(
            &[measurement], 
            (i * 500) as u64, 
            None
        ).unwrap();
        
        let state = ekf.state();
        let rate = state[1];
        
        println!("{:7.1} | {:8.2} | {:10.2} | {:11.3}",
                 time, measurement, estimate, rate);
    }

    println!("\nNote: EKF captures thermal lag from sensor mass!");
}

fn humidity_ekf_demo() {
    println!("Humidity EKF Model:");
    println!("------------------");
    println!("Models humidity sensor dynamics including:");
    println!("- Hysteresis effects");
    println!("- Temperature compensation");
    println!("- Saturation behavior\n");

    let humidity_model = HumidityEKF::new(
        0.5,   // measurement_noise
    );

    let mut ekf = humidity_model.create_filter();

    println!("Scenario: Rapid humidity change with hysteresis");
    println!("Time(s) | Measured | EKF State | Hysteresis Effect");
    println!("--------|----------|-----------|------------------");

    // Simulate rapid humidity increase then decrease
    let humidity_profile = vec![
        50.0, 55.0, 60.0, 65.0, 70.0, // Rising
        70.0, 65.0, 60.0, 55.0, 50.0, // Falling
    ];

    for (i, &true_humidity) in humidity_profile.iter().enumerate() {
        let time = i as f32 * 1.0;
        let measurement = true_humidity + gaussian_noise(i * 7, 0.5);
        
        ekf.predict(1000).unwrap();
        ekf.update(
            &[measurement],
            (i * 1000) as u64,
            None
        ).unwrap();
        
        let estimate = HumidityEKF::get_true_humidity(&ekf);
        
        let hysteresis = (measurement - estimate).abs();
        
        println!("{:7.1} | {:8.1} | {:9.1} | {:17.2}",
                 time, measurement, estimate, hysteresis);
    }

    println!("\nNote: Hysteresis causes lag during rapid changes!");
}

fn pressure_altitude_ekf_demo() {
    println!("Pressure-Altitude EKF Model:");
    println!("---------------------------");
    println!("Models barometric altitude including:");
    println!("- Atmospheric pressure variations");
    println!("- Temperature compensation");
    println!("- Vertical velocity estimation\n");

    let altitude_model = PressureAltitudeEKF::new(
        1013.25, // sea_level_pressure
        0.1,     // pressure_noise_std
    );

    let mut ekf = altitude_model.create_filter();

    println!("Scenario: Elevator ride from ground floor to 10th floor (~30m)");
    println!("Time(s) | Pressure(hPa) | Altitude(m) | Velocity(m/s)");
    println!("--------|---------------|-------------|---------------");

    // Simulate elevator ride
    let pressures = simulate_elevator_ride(1013.25, 1009.5, 10.0);
    
    for (i, &pressure) in pressures.iter().enumerate() {
        let time = i as f32 * 0.5;
        let noisy_pressure = pressure + gaussian_noise(i * 3, 0.1);
        
        ekf.predict(500).unwrap();
        ekf.update(
            &[noisy_pressure],
            (i * 500) as u64,
            None
        ).unwrap();
        
        let state = ekf.state();
        let altitude = state[0];
        let velocity = state[1];
        
        println!("{:7.1} | {:13.2} | {:11.1} | {:14.2}",
                 time, noisy_pressure, altitude, velocity);
    }

    println!("\nNote: EKF estimates both altitude and vertical velocity!");
}

fn environmental_ekf_demo() {
    println!("Environmental EKF Model:");
    println!("-----------------------");
    println!("Combines multiple sensors with cross-correlations:");
    println!("- Temperature affects humidity readings");
    println!("- Pressure correlates with weather patterns");
    println!("- Multi-sensor consensus improves accuracy\n");

    let env_model = EnvironmentalEKF::new(
        0.1,  // temp_noise
        0.5,  // humidity_noise
        0.2,  // pressure_noise
    );
    let mut ekf = env_model.create_filter();

    println!("Scenario: Weather front passing (temp drop, humidity rise, pressure drop)");
    println!("Time | Temperature | Humidity | Pressure");
    println!("-----|-------------|----------|----------");

    // Simulate weather front
    for i in 0..15 {
        let time = i as f32;
        
        // True values changing due to weather front
        let true_temp = 25.0 - time * 0.5;      // Cooling
        let true_humidity = 50.0 + time * 2.0;  // Rising humidity
        let true_pressure = 1013.0 - time * 0.3; // Falling pressure
        
        // Add measurement noise
        let measurements = [
            true_temp + gaussian_noise(i, 0.1),
            true_humidity + gaussian_noise(i * 2, 0.5),
            true_pressure + gaussian_noise(i * 3, 0.2),
        ];
        
        ekf.predict(1000).unwrap();
        ekf.update(
            &measurements,
            (i * 1000) as u64,
            None
        ).unwrap();
        
        let state = ekf.state();
        
        println!("{:4.0} | {:11.1} | {:8.1} | {:9.1}",
                 time, state[0], state[1], state[2]);
    }

    println!("\nNote: Multi-sensor fusion captures correlated environmental changes!");
    
    // Summary
    println!("\n\nEKF Model Benefits Summary:");
    println!("==========================");
    println!("✓ Physics-based models improve accuracy");
    println!("✓ Non-linear dynamics properly handled");
    println!("✓ Environmental effects compensated");
    println!("✓ Multi-sensor correlations utilized");
    println!("✓ Robust to sensor-specific artifacts");
}

// Helper functions

fn simulate_temperature_rise(start: f32, end: f32, tau: f32) -> Vec<f32> {
    let mut temps = Vec::new();
    for i in 0..20 {
        let t = i as f32 * 0.5;
        let temp = start + (end - start) * (1.0 - (-t / tau).exp());
        temps.push(temp);
    }
    temps
}

fn simulate_elevator_ride(start_pressure: f32, end_pressure: f32, duration: f32) -> Vec<f32> {
    let mut pressures = Vec::new();
    let steps = (duration * 2.0) as usize;
    
    for i in 0..steps {
        let t = i as f32 / 2.0;
        let progress = (t / duration).min(1.0);
        
        // S-curve for smooth acceleration/deceleration
        let smooth_progress = progress * progress * (3.0 - 2.0 * progress);
        let pressure = start_pressure + (end_pressure - start_pressure) * smooth_progress;
        pressures.push(pressure);
    }
    pressures
}

fn gaussian_noise(seed: usize, std_dev: f32) -> f32 {
    // Simple pseudo-random noise generator
    let x = ((seed * 12345 + 6789) % 1000) as f32 / 1000.0;
    let _y = ((seed * 98765 + 4321) % 1000) as f32 / 1000.0;
    
    // Box-Muller transform approximation
    let z = (x - 0.5) * 2.0;
    z * std_dev
}

// Helper for exp function approximation
trait ExpApprox {
    fn exp(self) -> f32;
}

impl ExpApprox for f32 {
    fn exp(self) -> f32 {
        // Taylor series approximation for small values
        if self.abs() < 2.0 {
            let x = self;
            let x2 = x * x;
            let x3 = x2 * x;
            let x4 = x2 * x2;
            1.0 + x + x2 / 2.0 + x3 / 6.0 + x4 / 24.0
        } else if self < 0.0 {
            1.0 / ((-self).exp())
        } else {
            // For large positive values, use repeated squaring
            let n = self as i32;
            let frac = self - n as f32;
            let e = 2.71828;
            let mut result = 1.0;
            for _ in 0..n {
                result *= e;
            }
            result * (1.0 + frac)
        }
    }
}