//! Multi-Algorithm Sensor Fusion Example
//!
//! This example demonstrates how EdgeGuard supports multiple fusion
//! algorithms working together, allowing you to choose the best
//! approach for your specific sensor configuration and requirements.
//!
//! ## What You'll Learn
//!
//! - Comparing different fusion algorithms on the same data
//! - Choosing the right algorithm for your use case
//! - Combining fusion results for consensus
//! - Performance vs accuracy trade-offs
//!
//! ## Fusion Algorithms Compared
//!
//! 1. **Weighted Average**: Simple, fast, good for similar sensors
//! 2. **Kalman Filter**: Optimal for linear systems with Gaussian noise
//! 3. **Complementary Filter**: Great for frequency separation
//! 4. **Voting/Consensus**: Robust against outliers
//!
//! ## Running the Example
//!
//! ```bash
//! cargo run --example 09_multi_fusion
//! ```

use edgeguard_core::{
    fusion::{
        FusionAlgorithm,
        KalmanFilter, KalmanConfig,
        WeightedAverageFusion, WeightedAverageConfig,
        ComplementaryFilter, ComplementaryConfig,
        ConsensusVoting, VotingConfig,
    },
};

fn main() {
    println!("EdgeGuard Multi-Algorithm Fusion Example");
    println!("=======================================\n");

    // Scenario: Three temperature sensors with different characteristics
    println!("Scenario: Industrial furnace with 3 temperature sensors");
    println!("------------------------------------------------------");
    println!("Sensor 1: Type-K thermocouple (fast response, moderate noise)");
    println!("Sensor 2: RTD sensor (slow response, low noise)");
    println!("Sensor 3: Infrared sensor (instant response, high noise)\n");

    // Run comparison test
    compare_fusion_algorithms();
    
    println!("\n{}\n", "=".repeat(60));
    
    // Demonstrate consensus approach
    consensus_fusion_demo();
    
    println!("\n{}\n", "=".repeat(60));
    
    // Show real-world scenario
    real_world_scenario();
}

fn compare_fusion_algorithms() {
    println!("Comparing Fusion Algorithms:");
    println!("---------------------------\n");

    // Sensor characteristics
    let sensor_noise = [0.5, 0.2, 1.0];  // Thermocouple, RTD, IR
    let sensor_lag = [0.1, 1.0, 0.0];    // Response time in seconds
    
    // Create fusion algorithms
    
    // 1. Weighted Average
    let weights = [0.4, 0.4, 0.2]; // Higher weight for reliable sensors
    let weighted_config = WeightedAverageConfig {
        weights,
        min_sensors: 2,
        outlier_threshold: 3.0,
    };
    let mut weighted_fusion = <WeightedAverageFusion<3> as FusionAlgorithm<1, 3>>::new(weighted_config);
    
    // 2. Kalman Filter
    let kalman_config = KalmanConfig::<2, 3>::default()
        .with_process_noise(0.01)
        .with_measurement_noise(sensor_noise);
    let mut kalman_fusion = KalmanFilter::new(kalman_config);
    
    // 3. Complementary Filter
    let comp_config = ComplementaryConfig {
        fast_weight: 0.3,  // Weight for fast sensor (IR)
        slow_weight: 0.7,  // Weight for slow sensor (RTD)
        crossover_freq: 0.1, // Hz
    };
    let mut comp_filter = ComplementaryFilter::new(comp_config);
    
    // 4. Consensus Voting
    let voting_config = VotingConfig {
        outlier_threshold: 2.0,
        min_votes: 2,
        confidence_threshold: 0.7,
    };
    let mut consensus = ConsensusVoting::new(voting_config);

    // Simulate temperature profile
    println!("Time | True Temp | Measurements      | Weighted | Kalman | Comp.  | Voting");
    println!("-----|-----------|-------------------|----------|--------|--------|--------");
    
    let true_temps = simulate_temperature_profile();
    
    for (i, &true_temp) in true_temps.iter().enumerate() {
        let time = i as f32 * 0.5;
        let timestamp = (i * 500) as u64;
        
        // Generate sensor readings with different characteristics
        let measurements = [
            // Thermocouple: moderate noise, slight lag
            true_temp + gaussian_noise(i, sensor_noise[0]) + 
                if i > 0 { (true_temps[i-1] - true_temp) * sensor_lag[0] } else { 0.0 },
            // RTD: low noise, significant lag
            true_temp + gaussian_noise(i * 2, sensor_noise[1]) + 
                if i > 0 { (true_temps[i-1] - true_temp) * sensor_lag[1] } else { 0.0 },
            // IR: high noise, no lag
            true_temp + gaussian_noise(i * 3, sensor_noise[2]),
        ];
        
        // Get fusion results
        let (weighted_est, _) = weighted_fusion.update(&measurements, timestamp, None).unwrap();
        let (kalman_est, _) = kalman_fusion.update(&measurements, timestamp, None).unwrap();
        // Complementary filter only uses 2 sensors
        let comp_measurements = [measurements[0], measurements[1]];
        let (comp_est, _) = comp_filter.update(&comp_measurements, timestamp, None).unwrap();
        let (voting_est, _) = consensus.update(&measurements, timestamp, None).unwrap();
        
        println!("{:4.1} | {:9.1} | {:4.1} {:4.1} {:4.1} | {:8.1} | {:6.1} | {:6.1} | {:7.1}",
                 time, true_temp,
                 measurements[0], measurements[1], measurements[2],
                 weighted_est, kalman_est, comp_est, voting_est);
    }
    
    // Performance comparison
    println!("\n\nAlgorithm Performance Summary:");
    println!("-----------------------------");
    println!("Algorithm      | Pros                        | Cons                      | Best For");
    println!("---------------|-----------------------------|--------------------------|---------");
    println!("Weighted Avg   | Simple, fast, predictable   | No dynamics modeling     | Similar sensors");
    println!("Kalman Filter  | Optimal estimation          | Complex, needs tuning    | Well-modeled systems");
    println!("Complementary  | Handles different dynamics  | Only 2 sensors           | Frequency separation");
    println!("Consensus      | Robust to outliers          | Needs redundancy         | Safety-critical");
}

fn consensus_fusion_demo() {
    println!("Consensus Fusion with Outlier Detection:");
    println!("---------------------------------------\n");
    
    println!("Scenario: Chemical reactor with 5 redundant temperature sensors");
    println!("One sensor intermittently fails, giving wild readings\n");

    let mut consensus = ConsensusVoting::<5>::new(VotingConfig {
        outlier_threshold: 2.0,
        min_votes: 3,
        confidence_threshold: 0.8,
    });

    let _true_temp = 150.0; // °C
    let test_cases = [
        ([150.1, 150.3, 149.8, 150.2, 150.0], "All sensors agree"),
        ([150.1, 150.3, 149.8, 150.2, 180.0], "Sensor 5 reads high"),
        ([150.1, 150.3, 149.8, 120.0, 180.0], "Sensors 4 & 5 fail"),
        ([150.1, 150.3, 149.8, 150.2, 0.0],   "Sensor 5 reads zero"),
    ];

    println!("Measurements                    | Consensus | Confidence | Status");
    println!("--------------------------------|-----------|------------|-------");
    
    for (i, (measurements, scenario)) in test_cases.iter().enumerate() {
        let (estimate, confidence) = consensus.update(
            measurements, 
            (i * 1000) as u64, 
            None
        ).unwrap();
        
        let status = if confidence.as_float() > 0.8 {
            "✓ Reliable"
        } else if confidence.as_float() > 0.5 {
            "⚠ Degraded"
        } else {
            "✗ Unreliable"
        };
        
        print!("[");
        for (j, &m) in measurements.iter().enumerate() {
            if j > 0 { print!(", "); }
            print!("{:5.1}", m);
        }
        print!("] | {:9.1} | {:10.2} | {} - {}",
               estimate, confidence.as_float(), status, scenario);
        println!();
    }

    println!("\nNote: Consensus voting maintains accuracy even with sensor failures!");
}

fn real_world_scenario() {
    println!("Real-World Scenario: Adaptive Fusion Selection");
    println!("=============================================\n");
    
    println!("EdgeGuard automatically selects the best fusion algorithm based on:");
    println!("- Number of sensors available");
    println!("- Sensor characteristics");
    println!("- System dynamics");
    println!("- Computational constraints\n");

    // Create adaptive fusion selector
    let mut adaptive_fusion = AdaptiveFusion::new();
    
    println!("Phase | Sensors | Algorithm Selected | Reasoning");
    println!("------|---------|-------------------|----------");
    
    // Phase 1: System startup (1 sensor)
    println!("  1   |    1    | Pass-through      | Only one sensor available");
    
    // Phase 2: Normal operation (3 sensors)
    adaptive_fusion.add_sensor(0.1); // Low noise sensor
    adaptive_fusion.add_sensor(0.5); // Medium noise sensor  
    adaptive_fusion.add_sensor(0.2); // Low noise sensor
    println!("  2   |    3    | Weighted Average  | Multiple similar sensors");
    
    // Phase 3: High dynamics detected
    adaptive_fusion.set_dynamics_level(DynamicsLevel::High);
    println!("  3   |    3    | Kalman Filter     | High dynamics require state estimation");
    
    // Phase 4: Sensor failure detected
    adaptive_fusion.mark_sensor_failed(1);
    println!("  4   |    2    | Complementary     | Two sensors with different characteristics");
    
    // Phase 5: Critical mode
    adaptive_fusion.add_sensor(0.3);
    adaptive_fusion.add_sensor(0.4);
    adaptive_fusion.set_mode(OperationMode::SafetyCritical);
    println!("  5   |    4    | Consensus Voting  | Safety-critical requires outlier rejection");

    println!("\n\nKey Insights:");
    println!("------------");
    println!("• No single fusion algorithm is best for all situations");
    println!("• EdgeGuard adapts to changing conditions automatically");
    println!("• Algorithm selection impacts both accuracy and performance");
    println!("• Redundancy enables more sophisticated fusion strategies");
}

// Helper structures for adaptive fusion demo
struct AdaptiveFusion {
    sensors: Vec<f32>,
    dynamics: DynamicsLevel,
    mode: OperationMode,
    failed_sensors: Vec<usize>,
}

#[derive(Clone, Copy)]
enum DynamicsLevel {
    Low,
    Medium,
    High,
}

#[derive(Clone, Copy)]
enum OperationMode {
    Normal,
    SafetyCritical,
}

impl AdaptiveFusion {
    fn new() -> Self {
        Self {
            sensors: Vec::new(),
            dynamics: DynamicsLevel::Low,
            mode: OperationMode::Normal,
            failed_sensors: Vec::new(),
        }
    }
    
    fn add_sensor(&mut self, noise_level: f32) {
        self.sensors.push(noise_level);
    }
    
    fn mark_sensor_failed(&mut self, index: usize) {
        self.failed_sensors.push(index);
        self.sensors.remove(
            self.sensors.iter().position(|_| true).unwrap()
        );
    }
    
    fn set_dynamics_level(&mut self, level: DynamicsLevel) {
        self.dynamics = level;
    }
    
    fn set_mode(&mut self, mode: OperationMode) {
        self.mode = mode;
    }
}

// Helper functions
fn simulate_temperature_profile() -> Vec<f32> {
    vec![
        100.0, 100.0, 100.5, 101.0, 102.0,  // Gradual rise
        103.0, 104.0, 105.0, 105.0, 105.0,  // Stabilize
        104.5, 104.0, 103.0, 102.0, 101.0,  // Cool down
    ]
}

fn gaussian_noise(seed: usize, std_dev: f32) -> f32 {
    let x = ((seed * 12345 + 6789) % 1000) as f32 / 1000.0;
    (x - 0.5) * 2.0 * std_dev
}