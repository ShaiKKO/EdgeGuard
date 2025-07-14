//! End-to-End Integration Tests for EdgeGuard
//!
//! These tests simulate complete IoT scenarios from sensor input to final output,
//! validating the entire data pipeline under realistic conditions.
//!
//! ## Test Philosophy
//!
//! Each test represents a real-world scenario that EdgeGuard users might encounter.
//! The constants and parameters are chosen based on:
//! - Industry standards (e.g., ASHRAE for HVAC)
//! - Physical constraints (e.g., thermal mass, sensor response times)
//! - Practical IoT deployment patterns
//! - Memory constraints of target embedded systems

use edgeguard_core::{
    buffer::CircularBuffer,
    events::{Event, EventBuilder, SensorType},
    pipeline::{Pipeline, PipelineBuilder, ValidationStage, CrossValidationStage, AggregationStage},
    stream::{MemoryStream, Stream, StreamProcessor},
    time::{FixedTime, TimeSource, Timestamp},
    traits::TimestampedReading,
    validators::{TemperatureValidator, HumidityValidator, PressureValidator},
};

#[cfg(feature = "ml")]
use edgeguard_ml::{MLAnomalyStage, MLConfig, FeatureComplexity};

#[cfg(feature = "schemas")]
use edgeguard_schemas::{SchemaRegistry, physics::SensorConstraints};

#[cfg(feature = "connectors")]
use edgeguard_connectors::{mqtt::MqttConnector, AsyncConnector};

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

// ===== TEST CONSTANTS =====
// These constants represent realistic operational parameters based on
// industry experience and IoT deployment best practices.

/// Pipeline buffer size suitable for ESP32 (4MB RAM) or similar embedded devices.
/// 1024 events × ~64 bytes/event = ~64KB, leaving plenty of headroom.
const PIPELINE_BUFFER_SIZE: usize = 1024;

/// Smaller buffer for memory-constrained scenarios (e.g., Arduino, bare-metal MCUs).
/// 256 events × ~64 bytes/event = ~16KB.
const SMALL_PIPELINE_BUFFER: usize = 256;

/// Critical buffer size for ultra-constrained devices.
/// 128 events × ~64 bytes/event = ~8KB.
const CRITICAL_BUFFER_SIZE: usize = 128;

/// Standard MQTT/sensor reporting interval in milliseconds.
/// 1 second is typical for home automation systems.
const SENSOR_REPORT_INTERVAL_MS: u64 = 1000;

/// High-frequency sampling for industrial applications.
/// 100ms allows for 10Hz sampling, suitable for vibration monitoring.
const HIGH_FREQ_INTERVAL_MS: u64 = 100;

/// Environmental monitoring interval.
/// 60 seconds is standard for HVAC and weather stations.
const ENV_MONITOR_INTERVAL_SEC: u32 = 60;

/// Indoor temperature comfort range per ASHRAE 55-2020.
/// 20-24°C (68-75°F) is the standard comfort zone.
const INDOOR_TEMP_MIN: f32 = 20.0;
const INDOOR_TEMP_MAX: f32 = 24.0;
const INDOOR_TEMP_NOMINAL: f32 = 22.0;

/// Indoor humidity comfort range per ASHRAE.
/// 30-60% RH is recommended for human comfort and mold prevention.
const INDOOR_HUMIDITY_MIN: f32 = 30.0;
const INDOOR_HUMIDITY_MAX: f32 = 60.0;
const INDOOR_HUMIDITY_NOMINAL: f32 = 45.0;

/// Sensor quality thresholds.
/// 0.95 = Professional grade sensor (±0.5°C accuracy)
/// 0.93 = Consumer grade sensor (±1°C accuracy)
/// 0.90 = Budget sensor (±2°C accuracy)
const HIGH_QUALITY_SENSOR: f32 = 0.95;
const MEDIUM_QUALITY_SENSOR: f32 = 0.93;
const LOW_QUALITY_SENSOR: f32 = 0.90;

/// Maximum rate of temperature change in indoor environments.
/// 10°C/second would indicate sensor failure or fire.
const MAX_INDOOR_TEMP_RATE: f32 = 10.0;

/// Maximum rate of humidity change.
/// 20%/second is physically impossible in normal conditions.
const MAX_HUMIDITY_RATE: f32 = 20.0;

/// Aggregation window for smart home applications.
/// 60 seconds provides good balance between responsiveness and data reduction.
const SMART_HOME_AGG_WINDOW_MS: u32 = 60000;

/// Test duration constants.
const MINUTES_PER_HOUR: u32 = 60;
const SECONDS_PER_MINUTE: u32 = 60;
const MS_PER_SECOND: u64 = 1000;

/// Standard batch processing size for tests.
/// 100 events balances processing efficiency with memory usage.
const BATCH_SIZE: usize = 100;

/// Simulate a complete smart home monitoring scenario.
///
/// This test models a typical residential HVAC monitoring system with:
/// - Temperature and humidity sensors reporting every second
/// - Cross-validation for dew point constraints
/// - 1-minute aggregation windows for trend analysis
/// - Simulation of AC failure after 3 minutes
#[test]
fn test_smart_home_end_to_end() {
    // Test configuration
    const TEST_DURATION_MINUTES: u32 = 5;
    const AC_FAILURE_AT_MINUTE: u32 = 3;
    const NORMAL_TEMP_C: f32 = INDOOR_TEMP_NOMINAL;  // 22°C
    const FAILURE_TEMP_C: f32 = 28.0;  // AC failure leads to temperature rise
    const NORMAL_HUMIDITY_PCT: f32 = 55.0;  // Comfortable middle range
    const FAILURE_HUMIDITY_PCT: f32 = 65.0;  // Humidity rises with temperature
    
    // Sensor variation amplitudes - based on typical sensor noise
    const TEMP_NOISE_AMPLITUDE: f32 = 0.5;  // ±0.5°C variation
    const HUMIDITY_NOISE_AMPLITUDE: f32 = 2.0;  // ±2% RH variation
    
    // Create time source
    let mut time_source = FixedTime::new(0);
    
    // Build complete pipeline with validation, cross-validation, and aggregation
    let mut pipeline = Pipeline::<PIPELINE_BUFFER_SIZE>::builder()
        // Temperature validation with indoor parameters
        .add_stage(ValidationStage::new(
            TemperatureValidator::indoor(),
            SensorType::Temperature,
        ))
        // Humidity validation with wider range for real-world conditions
        .add_stage(ValidationStage::new(
            HumidityValidator::new_with_limits(
                INDOOR_HUMIDITY_MIN - 10.0,  // 20% - allow for dry winter conditions
                INDOOR_HUMIDITY_MAX + 20.0,  // 80% - allow for humid summer conditions
                MAX_HUMIDITY_RATE / 2.0,     // 10%/s - reasonable for HVAC systems
            ),
            SensorType::Humidity,
        ))
        // Cross-validation ensures dew point < air temperature (prevents condensation)
        .add_stage({
            let mut cross_val = CrossValidationStage::new();
            cross_val.add_pair(
                SensorType::Temperature,
                SensorType::Humidity,
                edgeguard_core::events::CrossValidationType::DewPoint,
            ).unwrap();
            cross_val
        })
        // Aggregation for trend analysis and bandwidth reduction
        .add_stage(AggregationStage::new(
            edgeguard_core::pipeline::WindowSpec::Time(SMART_HOME_AGG_WINDOW_MS),
            edgeguard_core::pipeline::AggregationMethod::Mean,
            SensorType::Temperature,
        ))
        .build();
    
    // Simulate sensor data
    let mut events = Vec::new();
    let mut timestamp = 0u64;
    
    for minute in 0..TEST_DURATION_MINUTES {
        // Model AC failure scenario
        let base_temp = if minute < AC_FAILURE_AT_MINUTE { 
            NORMAL_TEMP_C 
        } else { 
            FAILURE_TEMP_C 
        };
        let base_humidity = if minute < AC_FAILURE_AT_MINUTE { 
            NORMAL_HUMIDITY_PCT 
        } else { 
            FAILURE_HUMIDITY_PCT 
        };
        
        // Generate readings at 1Hz (typical for home automation)
        for second in 0..SECONDS_PER_MINUTE {
            // Add realistic sensor noise using sine waves at different frequencies
            // This simulates thermal mass and air circulation effects
            let temp_variation = (second as f32 * 0.01).sin() * TEMP_NOISE_AMPLITUDE;
            let humidity_variation = (second as f32 * 0.02).cos() * HUMIDITY_NOISE_AMPLITUDE;
            
            let temp = base_temp + temp_variation;
            let humidity = base_humidity + humidity_variation;
            
            timestamp = (minute * SECONDS_PER_MINUTE + second) as u64 * MS_PER_SECOND;
            time_source.set(timestamp);
            
            // Temperature sensor reading
            events.push(
                EventBuilder::new(timestamp)
                    .sensor("living_room_temp", SensorType::Temperature)
                    .reading(temp, HIGH_QUALITY_SENSOR)
                    .unwrap()
            );
            
            // Humidity sensor reading (slightly lower quality typical for humidity sensors)
            events.push(
                EventBuilder::new(timestamp)
                    .sensor("living_room_humidity", SensorType::Humidity)
                    .reading(humidity, MEDIUM_QUALITY_SENSOR)
                    .unwrap()
            );
        }
    }
    
    // Process all events
    let mut validation_failures = 0;
    let mut cross_validation_failures = 0;
    let mut aggregated_readings = 0;
    
    for event in &events {
        pipeline.push_event(event.clone());
    }
    
    // Process in batches (100 events per batch for efficiency)
    const BATCH_SIZE: usize = 100;
    while pipeline.process_batch(BATCH_SIZE).unwrap() > 0 {
        // Collect results
        while let Some(result) = pipeline.pop_result() {
            match &result {
                Event::ValidationResult { status, .. } => {
                    if *status != edgeguard_core::events::ValidationStatus::Valid {
                        validation_failures += 1;
                    }
                }
                Event::CrossValidationResult { status, .. } => {
                    if *status != edgeguard_core::events::ValidationStatus::Valid {
                        cross_validation_failures += 1;
                    }
                }
                Event::BatchReading { sensor_type, .. } => {
                    if *sensor_type == SensorType::Temperature {
                        aggregated_readings += 1;
                    }
                }
                _ => {}
            }
        }
    }
    
    // Verify results - all readings should be valid in this scenario
    assert_eq!(validation_failures, 0, 
        "Temperature rise from AC failure should still be within valid range");
    assert_eq!(cross_validation_failures, 0, 
        "Dew point should remain below air temperature throughout test");
    assert!(aggregated_readings >= TEST_DURATION_MINUTES as usize - 1, 
        "Should have approximately one aggregation per minute");
    
    // Check pipeline metrics
    let metrics = pipeline.metrics();
    assert_eq!(metrics.events_processed, events.len() as u64,
        "All events should be processed");
    assert_eq!(metrics.events_dropped, 0,
        "No events should be dropped with adequate buffer size");
}

/// Test sensor failure detection and recovery.
///
/// This test simulates a common field scenario where an outdoor temperature
/// sensor gradually degrades due to environmental factors (moisture ingress,
/// corrosion, etc.) before failing completely and then recovering after
/// maintenance or self-healing.
#[test]
fn test_sensor_failure_recovery() {
    // Test phases based on typical sensor lifecycle
    const PHASE_1_NORMAL_SECONDS: u32 = 30;      // Healthy sensor operation
    const PHASE_2_DEGRADATION_SECONDS: u32 = 30; // Gradual degradation
    const PHASE_3_FAILURE_SECONDS: u32 = 10;     // Complete failure
    const PHASE_4_RECOVERY_SECONDS: u32 = 30;    // Post-maintenance recovery
    
    // Outdoor temperature range for temperate climate
    const OUTDOOR_TEMP_MIN: f32 = -10.0;  // Winter minimum
    const OUTDOOR_TEMP_MAX: f32 = 50.0;   // Summer maximum  
    const OUTDOOR_TEMP_NOMINAL: f32 = 15.0;  // Typical spring/fall temperature
    const MAX_OUTDOOR_TEMP_RATE: f32 = 5.0;  // °C/s - faster changes outdoors
    
    // Quality degradation parameters
    const QUALITY_DEGRADATION_RATE: f32 = 0.02;  // 2% quality loss per second
    const NOISE_GROWTH_RATE: f32 = 0.5;  // 0.5°C additional noise per second
    const QUALITY_RECOVERY_RATE: f32 = 0.015;  // 1.5% quality gain per second
    const MIN_QUALITY_THRESHOLD: f32 = 0.5;  // Below this, sensor is "bad"
    
    let mut time_source = FixedTime::new(0);
    
    // Configure validator for outdoor conditions
    let mut pipeline = Pipeline::<SMALL_PIPELINE_BUFFER>::builder()
        .add_stage(ValidationStage::new(
            TemperatureValidator::new_with_limits(
                OUTDOOR_TEMP_MIN,
                OUTDOOR_TEMP_MAX,
                MAX_OUTDOOR_TEMP_RATE
            ),
            SensorType::Temperature,
        ))
        .build();
    
    let mut events = Vec::new();
    let mut timestamp = 0u64;
    
    // Phase 1: Normal operation - sensor working correctly
    // Simulates typical diurnal temperature variation
    for i in 0..PHASE_1_NORMAL_SECONDS {
        timestamp = i as u64 * MS_PER_SECOND;
        let diurnal_variation = (i as f32 * 0.1).sin();  // Smooth temperature changes
        events.push(
            EventBuilder::new(timestamp)
                .sensor("outdoor_temp", SensorType::Temperature)
                .reading(OUTDOOR_TEMP_NOMINAL + diurnal_variation, HIGH_QUALITY_SENSOR)
                .unwrap()
        );
    }
    
    // Phase 2: Degrading sensor - quality drops, noise increases
    // This models water ingress or electrical interference
    for i in 0..PHASE_2_DEGRADATION_SECONDS {
        timestamp = (PHASE_1_NORMAL_SECONDS + i) as u64 * MS_PER_SECOND;
        
        // Quality degrades linearly
        let quality = HIGH_QUALITY_SENSOR - (i as f32 * QUALITY_DEGRADATION_RATE);
        
        // Noise increases as sensor degrades
        let noise = i as f32 * NOISE_GROWTH_RATE;
        
        // Temperature reading becomes increasingly erratic
        let reading = OUTDOOR_TEMP_NOMINAL + noise;
        
        events.push(
            EventBuilder::new(timestamp)
                .sensor("outdoor_temp", SensorType::Temperature)
                .reading(reading, quality)
                .unwrap()
        );
    }
    
    // Phase 3: Complete sensor failure - returns NaN
    // This simulates short circuit or disconnection
    const FAILURE_START: u32 = PHASE_1_NORMAL_SECONDS + PHASE_2_DEGRADATION_SECONDS;
    for i in 0..PHASE_3_FAILURE_SECONDS {
        timestamp = (FAILURE_START + i) as u64 * MS_PER_SECOND;
        events.push(
            EventBuilder::new(timestamp)
                .sensor("outdoor_temp", SensorType::Temperature)
                .reading(f32::NAN, 0.1)  // NaN with very low quality
                .unwrap()
        );
    }
    
    // Phase 4: Recovery after maintenance
    // Sensor replaced or dried out, gradually stabilizes
    const RECOVERY_START: u32 = FAILURE_START + PHASE_3_FAILURE_SECONDS;
    for i in 0..PHASE_4_RECOVERY_SECONDS {
        timestamp = (RECOVERY_START + i) as u64 * MS_PER_SECOND;
        
        // Quality improves gradually as sensor stabilizes
        let quality = MIN_QUALITY_THRESHOLD + (i as f32 * QUALITY_RECOVERY_RATE);
        
        // Temperature readings stabilize at seasonal normal
        let recovery_temp = 16.0;  // Slightly warmer than before
        
        events.push(
            EventBuilder::new(timestamp)
                .sensor("outdoor_temp", SensorType::Temperature)
                .reading(recovery_temp, quality)
                .unwrap()
        );
    }
    
    // Process and analyze failure patterns
    let mut quality_failures = 0;
    let mut invalid_value_failures = 0;
    
    for event in &events {
        pipeline.push_event(event.clone());
    }
    
    while pipeline.process_batch(BATCH_SIZE).unwrap() > 0 {
        while let Some(result) = pipeline.pop_result() {
            if let Event::ValidationResult { status, .. } = result {
                match status {
                    edgeguard_core::events::ValidationStatus::SensorQualityBad => {
                        quality_failures += 1;
                    }
                    edgeguard_core::events::ValidationStatus::InvalidValue => {
                        invalid_value_failures += 1;
                    }
                    _ => {}
                }
            }
        }
    }
    
    // Verify detection capabilities
    assert!(quality_failures > 0, 
        "Pipeline should detect quality degradation during phase 2");
    assert_eq!(invalid_value_failures, PHASE_3_FAILURE_SECONDS as usize, 
        "Should detect all NaN readings during complete failure");
}

/// Test multi-sensor fusion with realistic physics.
///
/// This test demonstrates Kalman filter fusion of three temperature sensors
/// with different accuracy characteristics, as commonly found in industrial
/// installations where redundancy is critical for safety.
#[test]
fn test_multi_sensor_fusion() {
    use edgeguard_core::fusion::{
        KalmanFilter, KalmanConfig, StateTransition,
        WeightedAverageFusion, FusionAlgorithm,
    };
    
    // Fusion test parameters
    const NUM_SENSORS: usize = 3;
    const NUM_MEASUREMENTS: usize = 100;
    const MEASUREMENT_INTERVAL_MS: u32 = 1000;  // 1 Hz sampling
    
    // True temperature profile - simulating daily variation
    const BASE_TEMP: f32 = 22.0;
    const TEMP_VARIATION: f32 = 2.0;  // ±2°C variation
    const DIURNAL_FREQUENCY: f32 = 0.05;  // Complete cycle over test duration
    
    // Sensor characteristics based on real sensor datasheets
    // Sensor 1: High-end PT100 RTD (e.g., Class A)
    const SENSOR1_NOISE_STD: f32 = 0.1;   // ±0.1°C std dev
    const SENSOR1_BIAS: f32 = 0.0;        // No bias
    const SENSOR1_FREQ_MOD: u32 = 7;      // Prime number for uncorrelated noise
    
    // Sensor 2: Standard thermistor (e.g., 10K NTC)
    const SENSOR2_NOISE_STD: f32 = 0.3;   // ±0.3°C std dev
    const SENSOR2_BIAS: f32 = 0.1;        // Small positive bias
    const SENSOR2_FREQ_MOD: u32 = 13;     // Different prime for uncorrelated noise
    
    // Sensor 3: Low-cost semiconductor (e.g., LM35)
    const SENSOR3_NOISE_STD: f32 = 0.5;   // ±0.5°C std dev
    const SENSOR3_BIAS: f32 = -0.2;       // Negative bias (self-heating)
    const SENSOR3_FREQ_MOD: u32 = 23;     // Another prime
    
    // Kalman filter configuration
    const INITIAL_TEMP_ESTIMATE: f32 = 20.0;  // Conservative initial guess
    const INITIAL_UNCERTAINTY: f32 = 1.0;     // 1°C² initial covariance
    const PROCESS_NOISE: f32 = 0.01;          // Small process noise for indoor environment
    const MEASUREMENT_NOISE: f32 = 0.5;       // Conservative measurement noise estimate
    const CONVERGENCE_THRESHOLD: f32 = 0.01;  // Convergence criterion
    
    let mut time_source = FixedTime::new(0);
    
    // Configure Kalman filter for multi-sensor fusion
    let config = KalmanConfig {
        initial_state: [INITIAL_TEMP_ESTIMATE],
        initial_covariance: [[INITIAL_UNCERTAINTY]],
        process_noise: [[PROCESS_NOISE]],
        measurement_noise: [[MEASUREMENT_NOISE]],  // Will be overridden per sensor
        transition: StateTransition {
            transition_matrix: [[1.0]], // Static model (temperature doesn't change on its own)
            control_matrix: None,       // No control inputs
        },
        measurement_matrix: [[1.0], [1.0], [1.0]], // Direct measurement from 3 sensors
        control_matrix: None,
        convergence_threshold: CONVERGENCE_THRESHOLD,
    };
    
    let mut kalman = KalmanFilter::<1, NUM_SENSORS>::new(config);
    
    // Generate realistic sensor measurements
    let mut measurements = Vec::new();
    
    for i in 0..NUM_MEASUREMENTS {
        // True temperature with slow sinusoidal variation
        let true_temp = BASE_TEMP + (i as f32 * DIURNAL_FREQUENCY).sin() * TEMP_VARIATION;
        
        // Sensor 1: High accuracy RTD
        let sensor1_noise = ((i * SENSOR1_FREQ_MOD) as f32 * 0.01).sin() * SENSOR1_NOISE_STD;
        let sensor1 = true_temp + sensor1_noise + SENSOR1_BIAS;
        
        // Sensor 2: Medium accuracy thermistor
        let sensor2_noise = ((i * SENSOR2_FREQ_MOD) as f32 * 0.01).cos() * SENSOR2_NOISE_STD;
        let sensor2 = true_temp + sensor2_noise + SENSOR2_BIAS;
        
        // Sensor 3: Low accuracy semiconductor
        let sensor3_noise = ((i * SENSOR3_FREQ_MOD) as f32 * 0.01).sin() * SENSOR3_NOISE_STD;
        let sensor3 = true_temp + sensor3_noise + SENSOR3_BIAS;
        
        measurements.push([sensor1, sensor2, sensor3]);
    }
    
    // Perform fusion and collect results
    let mut fused_values = Vec::new();
    let mut confidence_scores = Vec::new();
    
    for (i, measurement) in measurements.iter().enumerate() {
        let timestamp = (i as u64) * MEASUREMENT_INTERVAL_MS as u64;
        
        // Predict step (time update)
        kalman.predict(MEASUREMENT_INTERVAL_MS).unwrap();
        
        // Update step (measurement update)
        let (fused_value, confidence) = kalman.update(measurement, timestamp, None).unwrap();
        fused_values.push(fused_value);
        confidence_scores.push(confidence.as_f32());
    }
    
    // Verify fusion quality metrics
    assert_eq!(fused_values.len(), NUM_MEASUREMENTS, 
        "Should produce one fused value per measurement");
    
    // Calculate variance reduction
    let fused_variance = calculate_variance(&fused_values);
    let sensor3_variance = calculate_variance(
        &measurements.iter().map(|m| m[2]).collect::<Vec<_>>()
    );
    
    // Fusion should significantly reduce noise from worst sensor
    let variance_reduction = sensor3_variance / fused_variance;
    assert!(variance_reduction > 2.0, 
        "Fusion should reduce variance by at least 2x compared to worst sensor. \
         Got reduction factor of {:.2}", variance_reduction);
    
    // Verify confidence improves as filter converges
    const EARLY_SAMPLE: usize = 10;
    const LATE_SAMPLE: usize = 90;
    let early_confidence = confidence_scores[EARLY_SAMPLE];
    let late_confidence = confidence_scores[LATE_SAMPLE];
    let confidence_improvement = late_confidence / early_confidence;
    
    assert!(confidence_improvement > 1.1, 
        "Confidence should improve by at least 10% as filter converges. \
         Early: {:.3}, Late: {:.3}", early_confidence, late_confidence);
}

/// Test memory-bounded operation for embedded systems.
///
/// This test validates EdgeGuard's behavior under memory constraints typical
/// of embedded IoT devices. It simulates a data burst scenario where sensor
/// readings arrive faster than they can be processed, requiring the system
/// to gracefully handle buffer overflow.
#[test]
fn test_memory_bounded_operation() {
    // ESP32 memory constraints (typical IoT device)
    // ESP32 has 520KB SRAM, but much is used by WiFi/BLE stack
    // Realistic available memory for application: 100-200KB
    const ESP32_QUEUE_SIZE: usize = 128;  // 128 × 64 bytes = 8KB for events
    
    // Burst scenario parameters
    const BURST_SIZE: usize = 1000;  // Large burst to trigger overflow
    const BURST_RATE_HZ: u32 = 1000;  // 1kHz burst (faster than processing)
    const NORMAL_RATE_HZ: u32 = 10;   // Normal 10Hz sensor rate
    
    // Processing constraints
    const PROCESS_BATCH_SIZE: usize = 50;  // Process in small batches
    const EMBEDDED_TEMP_MIN: f32 = -20.0;  // Industrial temperature range
    const EMBEDDED_TEMP_MAX: f32 = 60.0;   // Extended for outdoor use
    const EMBEDDED_TEMP_RATE: f32 = 10.0;  // Allow faster changes outdoors
    
    // Configure pipeline with backpressure handling
    let mut pipeline = Pipeline::<ESP32_QUEUE_SIZE>::builder()
        .add_stage(ValidationStage::new(
            TemperatureValidator::new_with_limits(
                EMBEDDED_TEMP_MIN,
                EMBEDDED_TEMP_MAX,
                EMBEDDED_TEMP_RATE
            ),
            SensorType::Temperature,
        ))
        // Drop oldest events when buffer full (FIFO behavior)
        .backpressure(edgeguard_core::pipeline::BackpressureStrategy::DropOldest)
        .build();
    
    // Track metrics
    let mut events_sent = 0;
    let mut events_accepted = 0;
    
    // Simulate burst of sensor data (e.g., buffered readings after reconnection)
    println!("Simulating burst of {} events...", BURST_SIZE);
    for i in 0..BURST_SIZE {
        // Generate realistic temperature with some variation
        let temp_variation = (i as f32 * 0.1).sin();
        let temperature = 20.0 + temp_variation;
        
        let event = EventBuilder::new(i as u64)
            .sensor("embedded_sensor", SensorType::Temperature)
            .reading(temperature, LOW_QUALITY_SENSOR)
            .unwrap();
        
        events_sent += 1;
        if pipeline.push_event(event) {
            events_accepted += 1;
        }
    }
    
    println!("Burst complete: {} sent, {} accepted", events_sent, events_accepted);
    
    // Process events in batches (simulating limited CPU time)
    let mut events_processed = 0;
    let mut validation_results = 0;
    
    // Process until pipeline is empty
    loop {
        let batch_processed = pipeline.process_batch(PROCESS_BATCH_SIZE).unwrap();
        if batch_processed == 0 {
            break;
        }
        
        // Collect results (simulating result handling)
        while let Some(result) = pipeline.pop_result() {
            events_processed += 1;
            if matches!(result, Event::ValidationResult { .. }) {
                validation_results += 1;
            }
        }
    }
    
    // Calculate drop rate
    let drop_rate = (events_sent - events_accepted) as f32 / events_sent as f32 * 100.0;
    println!("Drop rate: {:.1}%", drop_rate);
    println!("Events processed: {}", events_processed);
    
    // Verify memory bounds were respected
    assert!(events_accepted <= ESP32_QUEUE_SIZE * 2, 
        "Should not accept more than 2x queue size even with pipelining");
    
    assert!(events_processed > 0, 
        "Should successfully process some events despite overflow");
    
    assert!(validation_results > 0,
        "Should produce validation results for processed events");
    
    // Verify metrics accuracy
    let metrics = pipeline.metrics();
    assert_eq!(
        metrics.events_dropped, 
        (events_sent - events_accepted) as u64,
        "Drop count should match sent-accepted difference"
    );
    
    // In embedded systems, some data loss is acceptable if system remains stable
    assert!(drop_rate < 95.0, 
        "Should process at least 5% of events even under extreme load");
}

/// Test streaming with network simulation.
///
/// This test simulates a real-world IoT scenario where sensor data is
/// transmitted over an unreliable network connection (e.g., LoRaWAN, NB-IoT).
/// Network delays, jitter, and occasional hiccups are modeled to ensure
/// EdgeGuard can handle typical connectivity issues.
#[cfg(feature = "std")]
#[test]
fn test_streaming_with_network_delays() {
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;
    
    // Network simulation parameters
    const TOTAL_EVENTS: usize = 100;
    const SENSOR_INTERVAL_MS: u64 = 100;  // 10Hz sensor rate
    const NETWORK_HICCUP_INTERVAL: usize = 10;  // Every 10th packet
    const NETWORK_HICCUP_MS: u64 = 50;  // 50ms delay spike
    const NETWORK_BASE_LATENCY_MS: u64 = 5;  // Typical IoT network latency
    const CONSUMER_POLL_INTERVAL_MS: u64 = 10;  // How often to check for data
    const PROCESSING_TIMEOUT_SEC: u64 = 2;  // Total test timeout
    
    // Remote sensor configuration (outdoor weather station)
    const REMOTE_TEMP_MIN: f32 = 0.0;   // Minimum expected temperature
    const REMOTE_TEMP_MAX: f32 = 50.0;  // Maximum expected temperature  
    const REMOTE_TEMP_BASE: f32 = 25.0;  // Base temperature
    const REMOTE_TEMP_VARIATION: f32 = 3.0;  // ±3°C variation
    const REMOTE_SENSOR_QUALITY: f32 = 0.92;  // Slightly degraded due to remote location
    
    // Create channel to simulate network transport
    let (tx, rx) = mpsc::channel();
    
    // Producer thread simulates remote sensor with network delays
    let producer = thread::spawn(move || {
        let mut timestamp = 0u64;
        
        for i in 0..TOTAL_EVENTS {
            // Generate realistic temperature reading
            let temp_variation = (i as f32 * 0.1).sin() * REMOTE_TEMP_VARIATION;
            let temperature = REMOTE_TEMP_BASE + temp_variation;
            
            let event = EventBuilder::new(timestamp)
                .sensor("remote_weather_station", SensorType::Temperature)
                .reading(temperature, REMOTE_SENSOR_QUALITY)
                .unwrap();
            
            // Simulate network behavior
            // Base latency for all packets
            thread::sleep(Duration::from_millis(NETWORK_BASE_LATENCY_MS));
            
            // Periodic network congestion/interference
            if i % NETWORK_HICCUP_INTERVAL == 0 {
                println!("Network hiccup at event {}", i);
                thread::sleep(Duration::from_millis(NETWORK_HICCUP_MS));
            }
            
            // Send event (ignoring send errors as network might drop packets)
            let _ = tx.send(event);
            
            // Next timestamp
            timestamp += SENSOR_INTERVAL_MS;
            
            // Small sleep to simulate sensor interval
            thread::sleep(Duration::from_millis(
                SENSOR_INTERVAL_MS.saturating_sub(NETWORK_BASE_LATENCY_MS)
            ));
        }
        
        println!("Producer sent {} events", TOTAL_EVENTS);
    });
    
    // Consumer simulates edge device receiving and processing data
    let mut pipeline = Pipeline::<SMALL_PIPELINE_BUFFER>::builder()
        .add_stage(ValidationStage::new(
            TemperatureValidator::new_with_limits(
                REMOTE_TEMP_MIN,
                REMOTE_TEMP_MAX,
                MAX_OUTDOOR_TEMP_RATE
            ),
            SensorType::Temperature,
        ))
        .build();
    
    let mut received = 0;
    let mut processed = 0;
    let mut validation_results = 0;
    let mut sensor_readings = 0;
    
    // Process with timeout to handle network delays gracefully
    let start = std::time::Instant::now();
    while start.elapsed() < Duration::from_secs(PROCESSING_TIMEOUT_SEC) {
        // Non-blocking receive to simulate async network handling
        while let Ok(event) = rx.try_recv() {
            received += 1;
            pipeline.push_event(event);
        }
        
        // Process accumulated events in small batches
        const NETWORK_BATCH_SIZE: usize = 10;
        if pipeline.process_batch(NETWORK_BATCH_SIZE).unwrap() > 0 {
            // Collect and categorize results
            while let Some(result) = pipeline.pop_result() {
                processed += 1;
                match result {
                    Event::ValidationResult { .. } => validation_results += 1,
                    Event::SensorReading { .. } => sensor_readings += 1,
                    _ => {}
                }
            }
        }
        
        // Small sleep to prevent CPU spinning
        thread::sleep(Duration::from_millis(CONSUMER_POLL_INTERVAL_MS));
    }
    
    // Wait for producer to finish
    producer.join().unwrap();
    
    // Drain any remaining events
    while let Ok(event) = rx.try_recv() {
        received += 1;
        pipeline.push_event(event);
    }
    
    // Final processing
    while pipeline.process_batch(BATCH_SIZE).unwrap() > 0 {
        while let Some(result) = pipeline.pop_result() {
            processed += 1;
            match result {
                Event::ValidationResult { .. } => validation_results += 1,
                Event::SensorReading { .. } => sensor_readings += 1,
                _ => {}
            }
        }
    }
    
    println!("Network streaming results:");
    println!("  Events sent: {}", TOTAL_EVENTS);
    println!("  Events received: {}", received);
    println!("  Events processed: {}", processed);
    println!("  Validation results: {}", validation_results);
    println!("  Sensor readings forwarded: {}", sensor_readings);
    
    // Verify streaming resilience
    assert_eq!(received, TOTAL_EVENTS, 
        "Should receive all events despite network delays");
    
    // Each event produces both a validation result and forwarded reading
    assert!(validation_results >= TOTAL_EVENTS, 
        "Should validate all received events");
    
    assert!(sensor_readings >= TOTAL_EVENTS,
        "Should forward all valid sensor readings");
    
    // Total processed should be at least 2x events (validation + forwarding)
    assert!(processed >= TOTAL_EVENTS * 2, 
        "Should process both validation results and forwarded readings");
}

// Helper function to calculate variance
fn calculate_variance(values: &[f32]) -> f32 {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    values.iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f32>() / values.len() as f32
}

/// Test complete ML-enhanced pipeline.
///
/// This test demonstrates EdgeGuard's machine learning capabilities for
/// detecting complex anomalies that physics-based validation might miss.
/// The scenario simulates a subtle sensor failure where readings remain
/// within valid ranges but lose their expected correlation patterns.
#[cfg(feature = "ml")]
#[test]
fn test_ml_enhanced_pipeline() {
    use edgeguard_ml::{MLAnomalyStage, MLConfig, ForestConfig};
    
    // ML configuration tuned for environmental monitoring
    const ML_WINDOW_SIZE: usize = 50;  // 50 samples for pattern learning
    const ML_TRAINING_SAMPLES: usize = 200;  // Initial training period
    const ML_ANOMALY_SAMPLES: usize = 50;   // Anomaly duration
    
    // Isolation Forest parameters (balanced for accuracy vs performance)
    const FOREST_NUM_TREES: usize = 5;     // Enough for good detection
    const FOREST_SAMPLE_SIZE: usize = 30;  // Subsample size per tree
    const FOREST_MAX_DEPTH: usize = 4;    // Limit tree depth for speed
    const ANOMALY_THRESHOLD: f32 = 0.7;   // Conservative threshold
    
    // Environmental parameters
    const NORMAL_TEMP_BASE: f32 = 22.0;    // Comfortable room temperature
    const NORMAL_TEMP_VARIATION: f32 = 1.0; // ±1°C variation
    const NORMAL_HUMIDITY_BASE: f32 = 50.0; // Comfortable humidity
    const NORMAL_HUMIDITY_VARIATION: f32 = 5.0; // ±5% RH variation
    const STUCK_HUMIDITY_VALUE: f32 = 70.0;  // Sensor stuck at 70%
    
    // Feature extraction parameters
    const FEATURE_WINDOW_MS: u32 = 5000;  // 5-second feature window
    const RETRAIN_INTERVAL: usize = 300;   // Retrain every 300 samples
    
    let ml_config = MLConfig {
        window_size: ML_WINDOW_SIZE,
        forest_config: ForestConfig {
            num_trees: FOREST_NUM_TREES,
            sample_size: FOREST_SAMPLE_SIZE,
            max_depth: FOREST_MAX_DEPTH,
            seed: 42,  // Reproducible results
            anomaly_threshold: ANOMALY_THRESHOLD,
        },
        sensor_types: vec![SensorType::Temperature, SensorType::Humidity],
        feature_window_ms: FEATURE_WINDOW_MS,
        retrain_interval: Some(RETRAIN_INTERVAL),
        anomaly_threshold: ANOMALY_THRESHOLD,
        feature_complexity: FeatureComplexity::Extended,  // Use advanced features
        enable_correlation: true,  // Critical for detecting correlation breaks
    };
    
    // Build pipeline with physics validation followed by ML
    let mut pipeline = Pipeline::<512>::builder()
        // Traditional physics-based validation
        .add_stage(ValidationStage::new(
            TemperatureValidator::indoor(),
            SensorType::Temperature,
        ))
        .add_stage(ValidationStage::new(
            HumidityValidator::new_with_limits(
                INDOOR_HUMIDITY_MIN,  // 30%
                INDOOR_HUMIDITY_MAX + 10.0,  // 70% - allow stuck value
                MAX_HUMIDITY_RATE / 2.0,  // 10%/s
            ),
            SensorType::Humidity,
        ))
        // ML-based anomaly detection
        .add_stage(MLAnomalyStage::<ML_WINDOW_SIZE>::new(ml_config))
        .build();
    
    // Generate training data with normal correlated patterns
    let mut events = Vec::new();
    
    println!("Generating {} normal samples for ML training...", ML_TRAINING_SAMPLES);
    for i in 0..ML_TRAINING_SAMPLES {
        let timestamp = i as u64 * MS_PER_SECOND;
        
        // Temperature and humidity follow natural correlation
        // Higher temperature typically means lower relative humidity
        let temp_phase = (i as f32 * 0.05).sin();
        let temp = NORMAL_TEMP_BASE + temp_phase * NORMAL_TEMP_VARIATION;
        
        // Humidity inversely correlated with temperature
        let humidity_phase = (i as f32 * 0.03).cos();
        let humidity = NORMAL_HUMIDITY_BASE + humidity_phase * NORMAL_HUMIDITY_VARIATION 
                      - temp_phase * 2.0;  // Inverse correlation
        
        events.push(EventBuilder::new(timestamp)
            .sensor("room_temp", SensorType::Temperature)
            .reading(temp, HIGH_QUALITY_SENSOR)
            .unwrap());
        
        events.push(EventBuilder::new(timestamp)
            .sensor("room_humidity", SensorType::Humidity)
            .reading(humidity, MEDIUM_QUALITY_SENSOR)
            .unwrap());
    }
    
    // Introduce anomaly: humidity sensor gets stuck while temperature varies normally
    // This simulates a common failure mode where sensor electronics fail but
    // still report "valid" values
    println!("Introducing stuck sensor anomaly for {} samples...", ML_ANOMALY_SAMPLES);
    for i in ML_TRAINING_SAMPLES..(ML_TRAINING_SAMPLES + ML_ANOMALY_SAMPLES) {
        let timestamp = i as u64 * MS_PER_SECOND;
        
        // Temperature continues normal variation
        let temp = NORMAL_TEMP_BASE + (i as f32 * 0.05).sin() * NORMAL_TEMP_VARIATION;
        
        // Humidity stuck at fixed value - sensor failure
        let humidity = STUCK_HUMIDITY_VALUE;
        
        events.push(EventBuilder::new(timestamp)
            .sensor("room_temp", SensorType::Temperature)
            .reading(temp, HIGH_QUALITY_SENSOR)
            .unwrap());
        
        events.push(EventBuilder::new(timestamp)
            .sensor("room_humidity", SensorType::Humidity)
            .reading(humidity, HIGH_QUALITY_SENSOR)  // Quality appears normal
            .unwrap());
    }
    
    // Process all events and count anomalies
    let mut ml_anomalies = 0;
    let mut physics_failures = 0;
    
    for event in &events {
        pipeline.push_event(event.clone());
    }
    
    while pipeline.process_batch(BATCH_SIZE).unwrap() > 0 {
        while let Some(result) = pipeline.pop_result() {
            match result {
                Event::SystemEvent { event_type, .. } => {
                    if event_type == edgeguard_core::events::SystemEventType::ValidatorError {
                        ml_anomalies += 1;
                    }
                }
                Event::ValidationResult { status, .. } => {
                    if status != edgeguard_core::events::ValidationStatus::Valid {
                        physics_failures += 1;
                    }
                }
                _ => {}
            }
        }
    }
    
    println!("ML anomaly detection results:");
    println!("  Physics validation failures: {}", physics_failures);
    println!("  ML anomalies detected: {}", ml_anomalies);
    
    // Verify ML detected the subtle anomaly that physics validation missed
    assert_eq!(physics_failures, 0, 
        "Physics validation should pass since values are within range");
    
    assert!(ml_anomalies > ML_ANOMALY_SAMPLES / 5, 
        "ML should detect at least 20% of anomalous samples. \
         Expected > {}, got {}", ML_ANOMALY_SAMPLES / 5, ml_anomalies);
    
    assert!(ml_anomalies < ML_TRAINING_SAMPLES / 10,
        "ML should not flag too many normal samples as anomalous. \
         Should be < {}, got {}", ML_TRAINING_SAMPLES / 10, ml_anomalies);
}

/// Test schema-based validation.
///
/// This test demonstrates how EdgeGuard can extract validation constraints
/// from Apache Avro schemas, enabling configuration-driven validation that
/// can be updated without recompiling the edge device firmware.
#[cfg(feature = "schemas")]
#[test]
fn test_schema_based_validation() {
    use edgeguard_schemas::physics::{SensorConstraints, create_validator_from_constraints};
    
    // Physical constants for validation
    const ABSOLUTE_ZERO_CELSIUS: f32 = -273.15;  // Theoretical minimum
    const ROOM_TEMPERATURE: f32 = 25.0;          // Typical indoor temp
    const EXTREME_HOT: f32 = 300.0;              // Beyond any sensor range
    
    // Get standard temperature constraints from schema
    let constraints = SensorConstraints::temperature_celsius();
    
    // Verify constraint values match expectations
    assert_eq!(constraints.min_value, -80.0, 
        "Minimum should be industrial sensor limit");
    assert_eq!(constraints.max_value, 125.0, 
        "Maximum should be typical sensor upper limit");
    assert_eq!(constraints.max_rate_change, 10.0,
        "Rate limit should prevent impossible temperature swings");
    
    // Create validator from constraints - this simulates runtime
    // configuration from a schema file or configuration service
    let validator = create_validator_from_constraints(&constraints, "temperature")
        .expect("Should create validator from well-formed constraints");
    
    // Create validation context with sensor history
    let mut context = edgeguard_core::traits::ValidationContext {
        history: CircularBuffer::new(),
        timestamp: MS_PER_SECOND,
        ambient_temp: None,
        ambient_humidity: None,
        sensor_quality: HIGH_QUALITY_SENSOR,
    };
    
    // Add some historical readings for rate-of-change validation
    context.history.push(24.0);  // Previous reading
    context.history.push(24.5);  // Gradual warming
    
    // Test 1: Valid room temperature
    assert!(validator.validate(&ROOM_TEMPERATURE, &context).is_ok(),
        "Room temperature should validate successfully");
    
    // Test 2: Below absolute zero (physics violation)
    let absolute_zero_result = validator.validate(&ABSOLUTE_ZERO_CELSIUS, &context);
    assert!(absolute_zero_result.is_err(),
        "Temperature below absolute zero should fail validation");
    
    // Test 3: Extreme temperature beyond sensor capabilities
    let extreme_result = validator.validate(&EXTREME_HOT, &context);
    assert!(extreme_result.is_err(),
        "Temperature beyond sensor range should fail validation");
    
    // Test 4: Rate of change validation
    // Simulate impossible temperature jump
    context.history.push(ROOM_TEMPERATURE);
    context.timestamp += MS_PER_SECOND;
    
    let rapid_change = ROOM_TEMPERATURE + 15.0;  // 15°C in 1 second!
    let rate_result = validator.validate(&rapid_change, &context);
    assert!(rate_result.is_err(),
        "Rapid temperature change should fail rate validation");
    
    // Test 5: Edge case - exactly at limits
    let min_edge = validator.validate(&constraints.min_value, &context);
    let max_edge = validator.validate(&constraints.max_value, &context);
    
    assert!(min_edge.is_ok(), "Minimum edge value should be valid");
    assert!(max_edge.is_ok(), "Maximum edge value should be valid");
    
    // Demonstrate how schemas enable runtime reconfiguration
    println!("Schema-based validation test results:");
    println!("  Constraint source: SensorConstraints::temperature_celsius()");
    println!("  Valid range: [{}, {}]°C", constraints.min_value, constraints.max_value);
    println!("  Max rate: {}°C/s", constraints.max_rate_change);
    println!("  All physics-based tests passed ✓");
}