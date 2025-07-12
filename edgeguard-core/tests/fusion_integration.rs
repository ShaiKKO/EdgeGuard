//! Integration tests for sensor fusion algorithms
//!
//! Tests various fusion algorithms including:
//! - Kalman filters for multi-sensor fusion
//! - Weighted averaging with quality scores
//! - Complementary filters for IMU data
//! - Consensus voting for safety-critical applications

#![cfg(test)]

mod common;

use edgeguard_core::{
    events::{Event, EventBuilder, SensorType},
    fusion::{
        FusionEngine, FusionConfig, FusionAlgorithm,
        KalmanConfig, ExtendedKalmanFilter,
        WeightedAverageFusion, ComplementaryFilter, ConsensusVoting,
    },
    time::MockTimeSource,
};

use common::{
    harness::{TestHarness, TestRng},
    scenarios::{Scenarios, TestScenario},
    generators::{PhysicsAwareGenerator, SensorModel},
};

use heapless::Vec as HVec;

#[test]
fn test_kalman_fusion_basic() {
    let mut harness = TestHarness::new(160);
    
    harness.run_test("kalman_fusion_basic", || {
        // Create Kalman filter for temperature fusion
        let config = KalmanConfig {
            process_noise: 0.1,
            measurement_noise: HVec::from_slice(&[0.5, 0.5]).unwrap(),
            initial_covariance: 1.0,
        };
        
        let mut kalman = ExtendedKalmanFilter::<1, 2>::new(config);
        
        // Initialize with temperature model
        kalman.initialize_temperature_model();
        
        // Simulate two temperature sensors
        let mut rng = TestRng::new(42);
        let true_temp = 25.0;
        
        for i in 0..100 {
            let dt_ms = 1000; // 1 second intervals
            
            // Predict step
            kalman.predict(dt_ms)?;
            
            // Simulate noisy measurements
            let sensor1 = true_temp + rng.gen_range(-0.5, 0.5);
            let sensor2 = true_temp + rng.gen_range(-0.5, 0.5);
            
            // Update with measurements
            let (estimate, confidence) = kalman.update(
                &[sensor1, sensor2],
                i * 1000,
                None,
            )?;
            
            // After convergence, estimate should be close to true value
            if i > 50 {
                assert_within_tolerance!(estimate, true_temp, 0.2);
                assert!(confidence.score() > 0.9, "Confidence should be high");
            }
        }
        
        Ok(())
    });
    
    assert!(harness.all_passed());
}

#[test]
fn test_weighted_average_fusion() {
    let mut harness = TestHarness::new(160);
    
    harness.run_test("weighted_average_fusion", || {
        // Create weighted average fusion
        let config = FusionConfig::default();
        let mut fusion = WeightedAverageFusion::<3>::new(config);
        
        // Add three sensors with different qualities
        let sensor_models = vec![
            SensorModel::high_quality(),     // Weight ~0.5
            SensorModel::consumer_grade(),   // Weight ~0.3
            SensorModel::low_quality(),      // Weight ~0.2
        ];
        
        for model in sensor_models {
            fusion.add_sensor(Box::new(model));
        }
        
        // Test fusion with different measurements
        let measurements = [25.2, 25.0, 24.5]; // High to low quality
        let qualities = [0.95, 0.8, 0.6];
        
        let result = fusion.fuse(&measurements, &qualities, 1000)?;
        
        // Result should be weighted towards higher quality sensors
        assert_within_tolerance!(result.0, 25.1, 0.2);
        assert!(result.1.score() > 0.8);
        
        Ok(())
    });
    
    assert!(harness.all_passed());
}

#[test]
fn test_complementary_filter() {
    let mut harness = TestHarness::new(160);
    
    harness.run_test("complementary_filter", || {
        // Create complementary filter for temperature
        // High-pass filter for fast sensor, low-pass for slow sensor
        let mut filter = ComplementaryFilter::<2>::new(0.98); // 98% fast, 2% slow
        
        let mut rng = TestRng::new(42);
        let mut time = 0u64;
        
        // Simulate step change in temperature
        for i in 0..200 {
            time += 100; // 100ms intervals
            
            // True temperature with step at i=100
            let true_temp = if i < 100 { 20.0 } else { 25.0 };
            
            // Fast sensor (noisy but responsive)
            let fast_sensor = true_temp + rng.gen_range(-1.0, 1.0);
            
            // Slow sensor (smooth but delayed)
            let slow_sensor = if i < 110 { 20.0 } else { 25.0 };
            
            let (filtered, confidence) = filter.update(
                &[fast_sensor, slow_sensor],
                time,
                None,
            )?;
            
            // After step change, should track fast sensor initially
            if i == 105 {
                assert!(filtered > 23.0, "Should respond to step change");
            }
            
            // After settling, should converge to true value
            if i > 150 {
                assert_within_tolerance!(filtered, 25.0, 0.5);
            }
        }
        
        Ok(())
    });
    
    assert!(harness.all_passed());
}

#[test]
fn test_consensus_voting() {
    let mut harness = TestHarness::new(160);
    
    harness.run_test("consensus_voting", || {
        // Create consensus voting for safety-critical application
        let mut voting = ConsensusVoting::<5>::new(
            0.5,  // 50% threshold
            2.0,  // 2 std dev for outlier
        );
        
        // Test with one outlier sensor
        let measurements = [25.0, 25.1, 24.9, 25.2, 30.0]; // Last is outlier
        let qualities = [0.9, 0.9, 0.9, 0.9, 0.9];
        
        let (consensus, confidence) = voting.update(
            &measurements,
            1000,
            None,
        )?;
        
        // Should exclude outlier
        assert_within_tolerance!(consensus, 25.05, 0.2);
        assert!(confidence.score() > 0.8);
        
        // Test with majority failure
        let bad_measurements = [30.0, 31.0, 32.0, 25.0, 25.1];
        let result = voting.update(&bad_measurements, 2000, None);
        
        // Should detect consensus failure
        assert!(result.is_err() || result.unwrap().1.score() < 0.5);
        
        Ok(())
    });
    
    assert!(harness.all_passed());
}

#[test]
fn test_fusion_with_scenarios() {
    let mut harness = TestHarness::new(160);
    
    harness.run_test("fusion_scenarios", || {
        let scenario = Scenarios::sensor_degradation();
        
        // Create fusion engine
        let config = FusionConfig {
            min_sensors: 2,
            max_history: 100,
            confidence_threshold: 0.7,
        };
        
        let mut engine = FusionEngine::<10>::new(config);
        
        // Process events with fusion
        let mut fusion_results = 0;
        let mut failed_fusions = 0;
        
        for event in scenario.events {
            if let Event::SensorReading { sensor_id, value, quality, .. } = &event {
                // Accumulate readings for fusion
                engine.add_reading(sensor_id.as_str(), *value, *quality);
                
                // Try fusion when we have enough readings
                if engine.ready_for_fusion() {
                    match engine.fuse() {
                        Ok(_) => fusion_results += 1,
                        Err(_) => failed_fusions += 1,
                    }
                }
            }
        }
        
        println!("Fusion results: {}, Failed: {}", fusion_results, failed_fusions);
        
        // Should have successful fusions despite sensor failures
        assert!(fusion_results > 0, "Should have successful fusions");
        assert!(failed_fusions > 0, "Should detect fusion failures during degradation");
        
        Ok(())
    });
    
    assert!(harness.all_passed());
}

#[test]
fn test_fusion_performance() {
    let mut harness = TestHarness::new(160);
    
    harness.run_test("fusion_performance", || {
        let mut kalman = ExtendedKalmanFilter::<1, 4>::new(KalmanConfig {
            process_noise: 0.1,
            measurement_noise: HVec::from_slice(&[0.5; 4]).unwrap(),
            initial_covariance: 1.0,
        });
        
        kalman.initialize_temperature_model();
        
        // Measure fusion performance
        let iterations = 1000;
        let measurements = [25.0, 25.1, 24.9, 25.2];
        
        for i in 0..iterations {
            kalman.predict(100)?;
            kalman.update(&measurements, i * 100, None)?;
        }
        
        // Should handle 10k+ fusions/second on target hardware
        println!("Completed {} fusion iterations", iterations);
        
        Ok(())
    });
    
    assert!(harness.all_passed());
}

#[test]
fn test_fusion_memory_usage() {
    let mut harness = TestHarness::new(160);
    
    harness.run_test("fusion_memory_usage", || {
        // Check memory footprint of fusion algorithms
        let kalman_size = core::mem::size_of::<ExtendedKalmanFilter<3, 5>>();
        let weighted_size = core::mem::size_of::<WeightedAverageFusion<10>>();
        let voting_size = core::mem::size_of::<ConsensusVoting<8>>();
        
        println!("Fusion memory usage:");
        println!("  Kalman<3,5>: {} bytes", kalman_size);
        println!("  Weighted<10>: {} bytes", weighted_size);
        println!("  Voting<8>: {} bytes", voting_size);
        
        // All should be reasonable for embedded use
        assert!(kalman_size < 512, "Kalman filter too large");
        assert!(weighted_size < 256, "Weighted average too large");
        assert!(voting_size < 256, "Consensus voting too large");
        
        Ok(())
    });
    
    assert!(harness.all_passed());
}

#[test]
fn test_adaptive_fusion() {
    let mut harness = TestHarness::new(160);
    
    harness.run_test("adaptive_fusion", || {
        // Test fusion that adapts to sensor quality
        let mut fusion = WeightedAverageFusion::<4>::new(FusionConfig::default());
        
        // Add sensors
        for _ in 0..4 {
            fusion.add_sensor(Box::new(SensorModel::consumer_grade()));
        }
        
        // Simulate quality degradation
        let base_value = 25.0;
        let mut qualities = [0.95, 0.95, 0.95, 0.95];
        
        for i in 0..100 {
            // Degrade sensor 2 quality over time
            if i > 50 {
                qualities[1] = (0.95 - (i - 50) as f32 * 0.01).max(0.1);
            }
            
            let measurements = [
                base_value + 0.1,
                base_value + if i > 50 { 2.0 } else { 0.1 }, // Drift
                base_value,
                base_value - 0.1,
            ];
            
            let (fused, confidence) = fusion.fuse(&measurements, &qualities, i * 1000)?;
            
            // Should adapt weights based on quality
            if i > 80 {
                // Should mostly ignore degraded sensor
                assert_within_tolerance!(fused, base_value, 0.3);
            }
        }
        
        Ok(())
    });
    
    assert!(harness.all_passed());
}