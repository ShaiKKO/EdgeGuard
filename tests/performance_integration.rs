//! Performance Integration Tests for EdgeGuard
//!
//! These tests measure and validate performance characteristics across
//! integrated components under realistic workloads.
//!
//! ## Performance Targets
//!
//! Based on typical embedded IoT deployments:
//! - ESP32 (240MHz dual-core): 1-10k events/sec
//! - Cortex-M4 (168MHz): 10-50k events/sec  
//! - Raspberry Pi (1.5GHz quad-core): 50-100k events/sec
//!
//! These tests ensure EdgeGuard meets performance requirements
//! across the spectrum of target devices.

use edgeguard_core::{
    buffer::CircularBuffer,
    events::{Event, EventBuilder, SensorType},
    pipeline::{Pipeline, PipelineBuilder, ValidationStage, AggregationStage, RouterStage},
    stream::{MemoryStream, Stream, StreamProcessor, RateLimitedStream},
    time::{MonotonicTime, TimeSource},
    validators::{TemperatureValidator, HumidityValidator, PressureValidator},
    fusion::{KalmanFilter, KalmanConfig, StateTransition, WeightedAverageFusion},
};

#[cfg(feature = "ml")]
use edgeguard_ml::{MLAnomalyStage, MLConfig, FeatureComplexity};

use std::time::{Duration, Instant};

// ===== PERFORMANCE CONSTANTS =====

/// Target throughput for basic validation on mid-range MCU (events/sec).
/// Based on Cortex-M4 @ 168MHz with optimized code.
const TARGET_VALIDATION_THROUGHPUT: f64 = 50_000.0;

/// Target throughput for multi-stage pipeline (events/sec).
/// Accounts for routing, validation, and aggregation overhead.
const TARGET_PIPELINE_THROUGHPUT: f64 = 20_000.0;

/// Target fusion rate for Kalman filter (measurements/sec).
/// Based on 3x3 matrix operations on embedded processors.
const TARGET_FUSION_RATE: f64 = 10_000.0;

/// Target ML pipeline throughput (events/sec).
/// Isolation Forest with feature extraction is compute-intensive.
const TARGET_ML_THROUGHPUT: f64 = 5_000.0;

/// Maximum acceptable latency for single event (microseconds).
/// Critical for real-time control applications.
const MAX_EVENT_LATENCY_US: f64 = 100.0;

/// Test event counts - scaled for meaningful measurements.
const PERF_TEST_EVENTS: usize = 10_000;
const PIPELINE_TEST_EVENTS: usize = 5_000;
const FUSION_TEST_MEASUREMENTS: usize = 1_000;
const ML_TEST_EVENTS: usize = 2_000;

/// Standard batch size for performance tests.
/// Balances throughput with latency.
const PERF_BATCH_SIZE: usize = 100;

/// Measure basic validation throughput.
///
/// This test establishes the baseline performance for simple temperature
/// validation, which should be the fastest operation in EdgeGuard.
#[test]
fn test_validation_throughput() {
    // Test parameters
    const PIPELINE_BUFFER: usize = 1024;  // Standard buffer size
    const TEMP_BASE: f32 = 20.0;         // Base temperature
    const TEMP_VARIATION: f32 = 5.0;     // ±5°C variation
    const SENSOR_QUALITY: f32 = 0.95;    // High quality sensor
    
    // Pre-allocate events to exclude allocation time from measurement
    let mut events = Vec::with_capacity(PERF_TEST_EVENTS);
    for i in 0..PERF_TEST_EVENTS {
        // Generate realistic temperature pattern
        let temp = TEMP_BASE + (i as f32 * 0.01).sin() * TEMP_VARIATION;
        events.push(
            EventBuilder::new(i as u64)
                .sensor("perf_sensor", SensorType::Temperature)
                .reading(temp, SENSOR_QUALITY)
                .unwrap()
        );
    }
    
    // Create minimal pipeline with just validation
    let mut pipeline = Pipeline::<PIPELINE_BUFFER>::builder()
        .add_stage(ValidationStage::new(
            TemperatureValidator::new(),
            SensorType::Temperature,
        ))
        .build();
    
    // Warm up the pipeline (important for accurate measurements)
    for event in events.iter().take(100) {
        pipeline.push_event(event.clone());
    }
    pipeline.process_batch(100).unwrap();
    while pipeline.pop_result().is_some() {}
    
    // Measure actual processing time
    let start = Instant::now();
    
    // Push all events
    for event in &events {
        pipeline.push_event(event.clone());
    }
    
    // Process in batches
    let mut processed = 0;
    while processed < PERF_TEST_EVENTS {
        let batch_processed = pipeline.process_batch(PERF_BATCH_SIZE).unwrap();
        processed += batch_processed;
        
        // Drain results to prevent queue backup
        while pipeline.pop_result().is_some() {}
    }
    
    let elapsed = start.elapsed();
    let throughput = PERF_TEST_EVENTS as f64 / elapsed.as_secs_f64();
    let latency_us = elapsed.as_micros() as f64 / PERF_TEST_EVENTS as f64;
    
    // Report results
    println!("\nValidation Performance Metrics:");
    println!("  Test events: {}", PERF_TEST_EVENTS);
    println!("  Processing time: {:?}", elapsed);
    println!("  Throughput: {:.0} events/sec", throughput);
    println!("  Average latency: {:.2} μs/event", latency_us);
    println!("  Target throughput: {:.0} events/sec", TARGET_VALIDATION_THROUGHPUT);
    println!("  Performance ratio: {:.1}x", throughput / TARGET_VALIDATION_THROUGHPUT);
    
    // Performance assertions
    assert!(
        throughput > TARGET_VALIDATION_THROUGHPUT,
        "Throughput {:.0} events/sec should exceed target of {:.0} events/sec",
        throughput, TARGET_VALIDATION_THROUGHPUT
    );
    
    assert!(
        latency_us < MAX_EVENT_LATENCY_US,
        "Latency {:.2} μs should be under {:.0} μs for real-time requirements",
        latency_us, MAX_EVENT_LATENCY_US
    );
    
    // Ensure we can process 10k events in reasonable time for embedded systems
    assert!(
        elapsed < Duration::from_millis(500),
        "Should complete {} events in <500ms for embedded compatibility",
        PERF_TEST_EVENTS
    );
}

/// Measure multi-stage pipeline performance
#[test]
fn test_multi_stage_pipeline_performance() {
    const EVENT_COUNT: usize = 5_000;
    
    // Create complex pipeline
    let mut router = RouterStage::new();
    router.add_route(
        SensorType::Temperature,
        Box::new(ValidationStage::new(
            TemperatureValidator::new(),
            SensorType::Temperature,
        )),
    ).unwrap();
    router.add_route(
        SensorType::Humidity,
        Box::new(ValidationStage::new(
            HumidityValidator::new(),
            SensorType::Humidity,
        )),
    ).unwrap();
    
    let mut pipeline = Pipeline::<512>::builder()
        .add_stage(router)
        .add_stage(AggregationStage::new(
            edgeguard_core::pipeline::WindowSpec::Count { size: 10 },
            edgeguard_core::pipeline::AggregationMethod::Mean,
            SensorType::Temperature,
        ))
        .build();
    
    // Generate mixed sensor events
    let mut events = Vec::with_capacity(EVENT_COUNT);
    for i in 0..EVENT_COUNT {
        let sensor_type = if i % 2 == 0 {
            SensorType::Temperature
        } else {
            SensorType::Humidity
        };
        
        let value = match sensor_type {
            SensorType::Temperature => 22.0 + (i as f32 * 0.1).sin() * 3.0,
            SensorType::Humidity => 55.0 + (i as f32 * 0.15).cos() * 10.0,
            _ => 0.0,
        };
        
        events.push(
            EventBuilder::new(i as u64 * 100)
                .sensor(&format!("sensor_{}", i % 4), sensor_type)
                .reading(value, 0.92)
                .unwrap()
        );
    }
    
    // Measure processing
    let start = Instant::now();
    
    for event in &events {
        pipeline.push_event(event.clone());
    }
    
    let mut total_processed = 0;
    while total_processed < EVENT_COUNT {
        let processed = pipeline.process_batch(50).unwrap();
        if processed == 0 {
            break;
        }
        total_processed += processed;
        
        // Drain results
        while let Some(_) = pipeline.pop_result() {}
    }
    
    let elapsed = start.elapsed();
    let throughput = EVENT_COUNT as f64 / elapsed.as_secs_f64();
    
    println!("\nMulti-Stage Pipeline Performance:");
    println!("  Stages: Router + Validation + Aggregation");
    println!("  Events: {}", EVENT_COUNT);
    println!("  Time: {:?}", elapsed);
    println!("  Throughput: {:.0} events/sec", throughput);
    
    // Check metrics
    let metrics = pipeline.metrics();
    println!("  Events processed: {}", metrics.events_processed);
    println!("  Events dropped: {}", metrics.events_dropped);
    
    assert!(throughput > 20_000.0, "Multi-stage should handle >20k events/sec");
    assert_eq!(metrics.events_dropped, 0, "Should not drop events");
}

/// Measure fusion algorithm performance
#[test]
fn test_fusion_performance() {
    const MEASUREMENT_COUNT: usize = 1_000;
    const SENSOR_COUNT: usize = 3;
    
    // Configure Kalman filter for multi-sensor fusion
    let config = KalmanConfig {
        initial_state: [20.0],
        initial_covariance: [[1.0]],
        process_noise: [[0.01]],
        measurement_noise: [[0.5]],
        transition: StateTransition {
            transition_matrix: [[1.0]],
            control_matrix: None,
        },
        measurement_matrix: [[1.0]; SENSOR_COUNT],
        control_matrix: None,
        convergence_threshold: 0.01,
    };
    
    let mut kalman = KalmanFilter::<1, SENSOR_COUNT>::new(config);
    
    // Generate measurements
    let mut measurements = Vec::with_capacity(MEASUREMENT_COUNT);
    for i in 0..MEASUREMENT_COUNT {
        let true_value = 22.0 + (i as f32 * 0.05).sin() * 2.0;
        let measurement = [
            true_value + 0.1 * ((i * 7) as f32 * 0.01).sin(),
            true_value + 0.3 * ((i * 13) as f32 * 0.01).cos(),
            true_value + 0.2 * ((i * 23) as f32 * 0.01).sin() - 0.1,
        ];
        measurements.push(measurement);
    }
    
    // Measure fusion performance
    let start = Instant::now();
    
    for (i, measurement) in measurements.iter().enumerate() {
        kalman.predict(100).unwrap(); // 100ms prediction step
        kalman.update(measurement, (i * 100) as u64, None).unwrap();
    }
    
    let elapsed = start.elapsed();
    let fusion_rate = MEASUREMENT_COUNT as f64 / elapsed.as_secs_f64();
    
    println!("\nKalman Fusion Performance:");
    println!("  Measurements: {}", MEASUREMENT_COUNT);
    println!("  Sensors: {}", SENSOR_COUNT);
    println!("  Time: {:?}", elapsed);
    println!("  Fusion rate: {:.0} measurements/sec", fusion_rate);
    println!("  Latency: {:.2} μs/fusion", elapsed.as_micros() as f64 / MEASUREMENT_COUNT as f64);
    
    assert!(fusion_rate > 10_000.0, "Should fuse >10k measurements/sec");
}

/// Measure streaming performance with rate limiting
#[test]
fn test_streaming_performance() {
    const EVENT_COUNT: usize = 10_000;
    const TARGET_RATE: u32 = 1000; // events/sec
    
    // Generate events
    let mut events = Vec::with_capacity(EVENT_COUNT);
    for i in 0..EVENT_COUNT {
        events.push(
            EventBuilder::new(i as u64)
                .sensor("stream_sensor", SensorType::Temperature)
                .reading(25.0, 0.95)
                .unwrap()
        );
    }
    
    // Create rate-limited stream
    let base_stream = MemoryStream::new(&events);
    let limited_stream = RateLimitedStream::new(base_stream, TARGET_RATE, MonotonicTime::new());
    
    // Simple pipeline
    let pipeline = Pipeline::<256>::builder()
        .add_stage(ValidationStage::new(
            TemperatureValidator::new(),
            SensorType::Temperature,
        ))
        .build();
    
    let mut processor = StreamProcessor::new(limited_stream, pipeline);
    
    // Measure streaming
    let start = Instant::now();
    let mut total_processed = 0;
    let mut iterations = 0;
    
    // Process for up to 2 seconds
    while start.elapsed() < Duration::from_secs(2) && total_processed < EVENT_COUNT {
        match processor.process_batch(100) {
            Ok(0) => break,
            Ok(n) => {
                total_processed += n;
                iterations += 1;
                
                // Drain results
                while let Some(_) = processor.pipeline_mut().pop_result() {}
            }
            Err(_) => break,
        }
    }
    
    let elapsed = start.elapsed();
    let actual_rate = total_processed as f64 / elapsed.as_secs_f64();
    
    println!("\nStreaming Performance:");
    println!("  Target rate: {} events/sec", TARGET_RATE);
    println!("  Actual rate: {:.0} events/sec", actual_rate);
    println!("  Events processed: {}", total_processed);
    println!("  Iterations: {}", iterations);
    println!("  Time: {:?}", elapsed);
    
    // Rate limiting should keep us close to target
    let rate_error = (actual_rate - TARGET_RATE as f64).abs() / TARGET_RATE as f64;
    assert!(rate_error < 0.1, "Rate limiting error should be <10%");
}

/// Measure memory allocation patterns
#[test]
fn test_memory_efficiency() {
    // Test different buffer sizes
    let sizes = [4, 16, 64, 256, 1024];
    
    println!("\nMemory Usage Analysis:");
    println!("  Component sizes:");
    
    for &size in &sizes {
        let pipeline_size = std::mem::size_of_val(&Pipeline::<16>::builder().build()) 
            * size / 16; // Approximate scaling
        println!("    Pipeline<{}>: ~{} bytes", size, pipeline_size);
    }
    
    // Component sizes
    println!("  Individual components:");
    println!("    Event: {} bytes", std::mem::size_of::<Event>());
    println!("    CircularBuffer<100>: {} bytes", std::mem::size_of::<CircularBuffer<100>>());
    println!("    TemperatureValidator: {} bytes", std::mem::size_of::<TemperatureValidator>());
    println!("    KalmanFilter<1,3>: {} bytes", 
        std::mem::size_of::<KalmanFilter<1,3>>());
    
    // Stack usage test
    let validator = TemperatureValidator::new();
    let stack_usage = std::mem::size_of_val(&validator);
    assert!(stack_usage < 1024, "Validator should use <1KB stack");
    
    // Event allocation test
    let event = EventBuilder::new(0)
        .sensor("test", SensorType::Temperature)
        .reading(25.0, 0.95)
        .unwrap();
    let event_size = std::mem::size_of_val(&event);
    assert!(event_size < 128, "Event should be <128 bytes");
}

/// Measure ML pipeline performance
#[cfg(feature = "ml")]
#[test]
fn test_ml_pipeline_performance() {
    const EVENT_COUNT: usize = 2_000;
    const WINDOW_SIZE: usize = 50;
    
    let ml_config = MLConfig {
        window_size: WINDOW_SIZE,
        forest_config: edgeguard_ml::ForestConfig {
            num_trees: 3,
            sample_size: 30,
            max_depth: 3,
            seed: 42,
            anomaly_threshold: 0.7,
        },
        sensor_types: vec![SensorType::Temperature],
        feature_window_ms: 5000,
        retrain_interval: Some(500),
        anomaly_threshold: 0.7,
        feature_complexity: FeatureComplexity::Basic,
        enable_correlation: false,
    };
    
    let mut pipeline = Pipeline::<512>::builder()
        .add_stage(ValidationStage::new(
            TemperatureValidator::new(),
            SensorType::Temperature,
        ))
        .add_stage(MLAnomalyStage::<WINDOW_SIZE>::new(ml_config))
        .build();
    
    // Generate events with some anomalies
    let mut events = Vec::with_capacity(EVENT_COUNT);
    for i in 0..EVENT_COUNT {
        let value = if i > 1000 && i < 1100 {
            35.0 + (i as f32 * 0.1).sin() * 10.0 // Anomalous pattern
        } else {
            22.0 + (i as f32 * 0.05).sin() * 2.0 // Normal pattern
        };
        
        events.push(
            EventBuilder::new(i as u64 * 100)
                .sensor("ml_sensor", SensorType::Temperature)
                .reading(value, 0.95)
                .unwrap()
        );
    }
    
    // Measure ML processing
    let start = Instant::now();
    
    for event in &events {
        pipeline.push_event(event.clone());
    }
    
    let mut processed = 0;
    let mut anomalies_detected = 0;
    
    while processed < EVENT_COUNT {
        let batch = pipeline.process_batch(50).unwrap();
        if batch == 0 {
            break;
        }
        processed += batch;
        
        // Count anomalies
        while let Some(result) = pipeline.pop_result() {
            if let Event::SystemEvent { event_type, .. } = result {
                if event_type == edgeguard_core::events::SystemEventType::ValidatorError {
                    anomalies_detected += 1;
                }
            }
        }
    }
    
    let elapsed = start.elapsed();
    let throughput = EVENT_COUNT as f64 / elapsed.as_secs_f64();
    
    println!("\nML Pipeline Performance:");
    println!("  Events: {}", EVENT_COUNT);
    println!("  Window size: {}", WINDOW_SIZE);
    println!("  Time: {:?}", elapsed);
    println!("  Throughput: {:.0} events/sec", throughput);
    println!("  Anomalies detected: {}", anomalies_detected);
    
    assert!(throughput > 5_000.0, "ML pipeline should handle >5k events/sec");
    assert!(anomalies_detected > 50, "Should detect anomalies in test data");
}

/// Stress test with maximum load
#[test]
fn test_stress_maximum_load() {
    const DURATION: Duration = Duration::from_secs(1);
    const PIPELINE_SIZE: usize = 1024;
    
    // Create pipeline with multiple stages
    let mut pipeline = Pipeline::<PIPELINE_SIZE>::builder()
        .add_stage(ValidationStage::new(
            TemperatureValidator::new(),
            SensorType::Temperature,
        ))
        .add_stage(AggregationStage::new(
            edgeguard_core::pipeline::WindowSpec::Time(1000),
            edgeguard_core::pipeline::AggregationMethod::Mean,
            SensorType::Temperature,
        ))
        .backpressure(edgeguard_core::pipeline::BackpressureStrategy::DropOldest)
        .build();
    
    // Generate events as fast as possible
    let start = Instant::now();
    let mut events_sent = 0;
    let mut events_accepted = 0;
    let mut timestamp = 0u64;
    
    while start.elapsed() < DURATION {
        let event = EventBuilder::new(timestamp)
            .sensor("stress_sensor", SensorType::Temperature)
            .reading(20.0 + (timestamp as f32 * 0.001).sin() * 5.0, 0.9)
            .unwrap();
        
        events_sent += 1;
        if pipeline.push_event(event) {
            events_accepted += 1;
        }
        
        // Process periodically
        if events_sent % 100 == 0 {
            pipeline.process_batch(50).unwrap();
            
            // Drain some results
            for _ in 0..10 {
                if pipeline.pop_result().is_none() {
                    break;
                }
            }
        }
        
        timestamp += 10;
    }
    
    // Final processing
    while pipeline.process_batch(100).unwrap() > 0 {}
    
    let elapsed = start.elapsed();
    let send_rate = events_sent as f64 / elapsed.as_secs_f64();
    let accept_rate = events_accepted as f64 / elapsed.as_secs_f64();
    
    println!("\nStress Test Results:");
    println!("  Duration: {:?}", elapsed);
    println!("  Events sent: {}", events_sent);
    println!("  Events accepted: {}", events_accepted);
    println!("  Send rate: {:.0} events/sec", send_rate);
    println!("  Accept rate: {:.0} events/sec", accept_rate);
    println!("  Drop rate: {:.1}%", 
        (events_sent - events_accepted) as f64 / events_sent as f64 * 100.0);
    
    let metrics = pipeline.metrics();
    println!("  Pipeline processed: {}", metrics.events_processed);
    println!("  Pipeline dropped: {}", metrics.events_dropped);
    
    assert!(accept_rate > 100_000.0, "Should handle >100k events/sec under stress");
}