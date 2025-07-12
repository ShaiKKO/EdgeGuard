//! Integration tests for streaming data processing
//!
//! Tests:
//! - Streaming window operations
//! - Adaptive sampling based on signal characteristics
//! - Backpressure handling
//! - Stream fusion and aggregation

#![cfg(test)]

mod common;

use edgeguard_core::{
    events::{Event, EventBuilder, SensorType},
    stream::{Stream, StreamConfig, Window, AdaptiveSampler, StreamProcessor},
    pipeline::Pipeline,
    time::MockTimeSource,
};

use common::{
    harness::{TestHarness, TestTimer},
    scenarios::Scenarios,
    generators::{PhysicsAwareGenerator, SensorModel, EnvironmentScenario},
};

use core::task::Poll;

#[test]
fn test_basic_streaming() {
    let mut harness = TestHarness::new(160);
    
    harness.run_test("basic_streaming", || {
        // Create stream processor
        let config = StreamConfig {
            window_size: 10,
            window_overlap: 5,
            ..Default::default()
        };
        
        let mut stream = StreamProcessor::<100>::new(config);
        
        // Generate streaming data
        let mut generator = PhysicsAwareGenerator::new(1000);
        let events = generator.generate_temperature_with_thermal_mass(
            "stream_sensor",
            25.0,
            &[(0, 25.0), (1, 30.0)],
            1.0,
            1,
            60,
            &SensorModel::consumer_grade(),
        );
        
        let mut windows_processed = 0;
        
        for event in events {
            stream.push(event)?;
            
            // Process any complete windows
            while let Some(window) = stream.next_window() {
                assert_eq!(window.len(), 10, "Window should have correct size");
                windows_processed += 1;
            }
        }
        
        assert!(windows_processed > 0, "Should process some windows");
        
        Ok(())
    });
    
    assert!(harness.all_passed());
}

#[test]
fn test_adaptive_sampling() {
    let mut harness = TestHarness::new(160);
    
    harness.run_test("adaptive_sampling", || {
        // Create adaptive sampler
        let mut sampler = AdaptiveSampler::new(
            1.0,   // Base rate: 1 Hz
            0.1,   // Min rate: 0.1 Hz
            10.0,  // Max rate: 10 Hz
            0.5,   // Change threshold
        );
        
        // Generate data with varying rates of change
        let scenario = Scenarios::industrial_process();
        
        let mut sampled = 0;
        let mut skipped = 0;
        
        for event in scenario.events {
            if let Event::SensorReading { value, timestamp, .. } = event {
                if sampler.should_sample(value, timestamp) {
                    sampled += 1;
                } else {
                    skipped += 1;
                }
            }
        }
        
        println!("Sampled: {}, Skipped: {}", sampled, skipped);
        
        // Should adapt sampling rate
        assert!(sampled > 0, "Should sample some events");
        assert!(skipped > 0, "Should skip some events when signal is stable");
        
        // Efficiency ratio
        let efficiency = skipped as f32 / (sampled + skipped) as f32;
        assert!(efficiency > 0.3, "Should achieve >30% reduction");
        
        Ok(())
    });
    
    assert!(harness.all_passed());
}

#[test]
fn test_windowed_aggregation() {
    let mut harness = TestHarness::new(160);
    
    harness.run_test("windowed_aggregation", || {
        let config = StreamConfig {
            window_size: 30,
            window_overlap: 0,
            ..Default::default()
        };
        
        let mut stream = StreamProcessor::<1000>::new(config);
        
        // Generate hourly data
        let scenario = Scenarios::home_environment();
        
        let mut window_stats = Vec::new();
        
        for event in scenario.events {
            stream.push(event)?;
            
            while let Some(window) = stream.next_window() {
                // Calculate window statistics
                let mut sum = 0.0;
                let mut count = 0;
                let mut min = f32::MAX;
                let mut max = f32::MIN;
                
                for event in &window {
                    if let Event::SensorReading { value, .. } = event {
                        sum += value;
                        count += 1;
                        min = min.min(*value);
                        max = max.max(*value);
                    }
                }
                
                if count > 0 {
                    let avg = sum / count as f32;
                    window_stats.push((avg, min, max));
                }
            }
        }
        
        // Should have aggregated windows
        assert!(!window_stats.is_empty(), "Should have window statistics");
        
        // Verify statistics are reasonable
        for (avg, min, max) in &window_stats {
            assert!(min <= avg && avg <= max, "Statistics should be consistent");
            assert!(*min >= 15.0 && *max <= 30.0, "Values should be in expected range");
        }
        
        Ok(())
    });
    
    assert!(harness.all_passed());
}

#[test]
fn test_stream_backpressure() {
    let mut harness = TestHarness::new(160);
    
    harness.run_test("stream_backpressure", || {
        let config = StreamConfig {
            window_size: 10,
            max_buffer_size: 50,
            ..Default::default()
        };
        
        let mut stream = StreamProcessor::<50>::new(config);
        
        // Generate more data than buffer can hold
        let mut generator = PhysicsAwareGenerator::new(1000);
        let events = generator.generate_temperature_with_thermal_mass(
            "sensor",
            25.0,
            &[(0, 25.0)],
            1.0,
            1,
            100, // 100 events
            &SensorModel::consumer_grade(),
        );
        
        let mut accepted = 0;
        let mut rejected = 0;
        
        for event in events {
            match stream.push(event) {
                Ok(_) => accepted += 1,
                Err(_) => rejected += 1,
            }
        }
        
        println!("Accepted: {}, Rejected: {}", accepted, rejected);
        
        // Should handle backpressure gracefully
        assert!(accepted > 0, "Should accept some events");
        assert!(rejected > 0, "Should reject some events due to backpressure");
        
        Ok(())
    });
    
    assert!(harness.all_passed());
}

#[test]
fn test_multi_stream_fusion() {
    let mut harness = TestHarness::new(160);
    
    harness.run_test("multi_stream_fusion", || {
        // Create multiple streams for different sensors
        let config = StreamConfig {
            window_size: 20,
            window_overlap: 10,
            ..Default::default()
        };
        
        let mut temp_stream = StreamProcessor::<100>::new(config.clone());
        let mut humidity_stream = StreamProcessor::<100>::new(config);
        
        // Generate correlated data
        let scenario = Scenarios::weather_station();
        
        // Separate events by sensor type
        for event in scenario.events {
            match event.sensor_type() {
                Some(SensorType::Temperature) => temp_stream.push(event)?,
                Some(SensorType::Humidity) => humidity_stream.push(event)?,
                _ => {},
            }
        }
        
        // Process windows from both streams
        let mut temp_windows = 0;
        let mut humidity_windows = 0;
        
        while temp_stream.next_window().is_some() {
            temp_windows += 1;
        }
        
        while humidity_stream.next_window().is_some() {
            humidity_windows += 1;
        }
        
        assert!(temp_windows > 0, "Should have temperature windows");
        assert!(humidity_windows > 0, "Should have humidity windows");
        
        Ok(())
    });
    
    assert!(harness.all_passed());
}

#[test]
fn test_stream_memory_efficiency() {
    let mut harness = TestHarness::new(160);
    
    harness.run_test("stream_memory_efficiency", || {
        // Test memory usage of stream processors
        let small_stream_size = core::mem::size_of::<StreamProcessor<100>>();
        let large_stream_size = core::mem::size_of::<StreamProcessor<1000>>();
        
        println!("Stream memory usage:");
        println!("  StreamProcessor<100>: {} bytes", small_stream_size);
        println!("  StreamProcessor<1000>: {} bytes", large_stream_size);
        
        // Verify linear scaling with buffer size
        let size_per_event = (large_stream_size - small_stream_size) / 900;
        println!("  Per-event overhead: {} bytes", size_per_event);
        
        assert!(size_per_event < 100, "Per-event overhead should be reasonable");
        
        Ok(())
    });
    
    assert!(harness.all_passed());
}

#[test]
fn test_stream_latency() {
    let mut harness = TestHarness::new(160);
    
    harness.run_test("stream_latency", || {
        let config = StreamConfig {
            window_size: 10,
            window_overlap: 0,
            max_latency_ms: 1000, // 1 second max latency
            ..Default::default()
        };
        
        let mut stream = StreamProcessor::<100>::new(config);
        let mut time_source = MockTimeSource::new(0);
        
        // Add events slowly
        for i in 0..5 {
            let event = EventBuilder::new(time_source.now())
                .sensor("sensor", SensorType::Temperature)
                .reading(25.0, 0.95)
                .unwrap();
            
            stream.push(event)?;
            time_source.advance(500); // 500ms between events
        }
        
        // Should not have window yet (only 5 events)
        assert!(stream.next_window().is_none());
        
        // Advance time past max latency
        time_source.advance(2000);
        
        // Now should flush partial window due to latency
        stream.check_latency(time_source.now());
        
        let window = stream.next_window();
        assert!(window.is_some(), "Should flush window due to latency");
        assert_eq!(window.unwrap().len(), 5, "Should have partial window");
        
        Ok(())
    });
    
    assert!(harness.all_passed());
}

#[test]
fn test_stream_pipeline_integration() {
    let mut harness = TestHarness::new(160);
    
    harness.run_test("stream_pipeline_integration", || {
        // Test stream feeding into validation pipeline
        let stream_config = StreamConfig {
            window_size: 30,
            window_overlap: 0,
            ..Default::default()
        };
        
        let mut stream = StreamProcessor::<500>::new(stream_config);
        let mut pipeline = Pipeline::<10, 100>::new(Default::default());
        
        // Add validators
        pipeline.add_validator(
            Box::new(edgeguard_core::validators::TemperatureValidator::default()),
            SensorType::Temperature,
        )?;
        
        // Generate streaming data
        let scenario = Scenarios::home_environment();
        
        let mut windows_validated = 0;
        
        for event in scenario.events {
            stream.push(event)?;
            
            // Process complete windows through pipeline
            while let Some(window) = stream.next_window() {
                for event in window {
                    pipeline.process(event)?;
                }
                windows_validated += 1;
                
                // Drain validation results
                while pipeline.next_output().is_some() {}
            }
        }
        
        assert!(windows_validated > 0, "Should validate some windows");
        println!("Validated {} windows", windows_validated);
        
        Ok(())
    });
    
    assert!(harness.all_passed());
}