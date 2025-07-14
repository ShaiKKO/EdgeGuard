//! Cross-Module Integration Tests for EdgeGuard
//!
//! These tests verify the integration between different EdgeGuard modules,
//! ensuring they work together correctly in various configurations.
//!
//! ## Test Scope
//!
//! - Core + Fusion: Validation feeding into sensor fusion algorithms
//! - Core + ML: Physics validation combined with anomaly detection
//! - Core + Schemas: Runtime configuration from schema definitions
//! - Core + Connectors: Data ingestion and result publishing
//! - Multi-module: Complex scenarios using 3+ modules together

use edgeguard_core::{
    events::{Event, EventBuilder, SensorType, ValidationStatus},
    pipeline::{Pipeline, ValidationStage, FusionStage},
    validators::{TemperatureValidator, HumidityValidator},
    fusion::{WeightedAverageFusion, FusionAlgorithm},
    time::{FixedTime, TimeSource},
};

#[cfg(feature = "ml")]
use edgeguard_ml::{MLAnomalyStage, MLConfig, FeatureComplexity};

#[cfg(feature = "schemas")]
use edgeguard_schemas::{
    SchemaRegistry,
    physics::{SensorConstraints, create_validator_from_constraints},
};

#[cfg(feature = "connectors")]
use edgeguard_connectors::{
    mqtt::{MqttConnector, MqttConfig},
    AsyncConnector,
};

use std::sync::{Arc, Mutex};

// ===== INTEGRATION TEST CONSTANTS =====

/// Standard pipeline buffer size for integration tests.
/// Sufficient for multi-module processing without excessive memory.
const INTEGRATION_PIPELINE_SIZE: usize = 256;

/// Number of sensor readings to generate for fusion tests.
/// Ensures algorithms have enough data to converge.
const FUSION_TEST_SAMPLES: usize = 10;

/// Sensor quality values for different grades of sensors.
const HIGH_QUALITY: f32 = 0.95;   // ±0.5°C accuracy
const MEDIUM_QUALITY: f32 = 0.93; // ±1.0°C accuracy  
const LOW_QUALITY: f32 = 0.90;    // ±2.0°C accuracy

/// Temperature test values.
const ROOM_TEMP: f32 = 22.0;      // Standard room temperature
const TEMP_SENSOR1: f32 = 22.0;   // First sensor reading
const TEMP_SENSOR2: f32 = 22.5;   // Second sensor (slight variation)

/// Humidity test values.
const NORMAL_HUMIDITY: f32 = 55.0; // Comfortable indoor humidity

/// ML test parameters.
#[cfg(feature = "ml")]
const ML_WINDOW_SIZE: usize = 20;
#[cfg(feature = "ml")]
const ML_TRAINING_SAMPLES: usize = 50;

/// Test core validation with fusion integration.
///
/// Verifies that validated sensor readings can be successfully fused
/// using weighted averaging, demonstrating the data flow from raw
/// readings through validation to fusion output.
#[test]
fn test_validation_fusion_integration() {
    // Constants for this test
    const FUSION_GROUP_SIZE: usize = 2;  // Two temperature sensors
    const EXPECTED_FUSED_TEMP: f32 = 22.25; // Expected weighted average
    
    // Create validators with standard indoor parameters
    let temp_validator = TemperatureValidator::new();
    let humidity_validator = HumidityValidator::new();
    
    // Create weighted average fusion for temperature sensors
    let fusion = WeightedAverageFusion::<FUSION_GROUP_SIZE>::new();
    
    // Build pipeline: Validation → Fusion
    let mut fusion_stage = FusionStage::<FUSION_GROUP_SIZE>::new(Box::new(fusion));
    fusion_stage.add_sensor_group("temp_group", ["temp1", "temp2"])
        .expect("Should add sensor group successfully");
    
    let mut pipeline = Pipeline::<INTEGRATION_PIPELINE_SIZE>::builder()
        .add_stage(ValidationStage::new(temp_validator, SensorType::Temperature))
        .add_stage(ValidationStage::new(humidity_validator, SensorType::Humidity))
        .add_stage(fusion_stage)
        .build();
    
    // Generate test events at same timestamp (simultaneous readings)
    const TEST_TIMESTAMP: u64 = 1000;
    let events = vec![
        // Temperature sensor 1: nominal reading
        EventBuilder::new(TEST_TIMESTAMP)
            .sensor("temp1", SensorType::Temperature)
            .reading(TEMP_SENSOR1, HIGH_QUALITY)
            .unwrap(),
        // Temperature sensor 2: slightly higher reading
        EventBuilder::new(TEST_TIMESTAMP)
            .sensor("temp2", SensorType::Temperature)
            .reading(TEMP_SENSOR2, MEDIUM_QUALITY)
            .unwrap(),
        // Humidity sensor: independent reading
        EventBuilder::new(TEST_TIMESTAMP)
            .sensor("hum1", SensorType::Humidity)
            .reading(NORMAL_HUMIDITY, MEDIUM_QUALITY)
            .unwrap(),
    ];
    
    // Process all events
    for event in &events {
        pipeline.push_event(event.clone());
    }
    
    const BATCH_SIZE: usize = 10;
    pipeline.process_batch(BATCH_SIZE).unwrap();
    
    // Analyze results
    let mut validation_count = 0;
    let mut fusion_count = 0;
    let mut sensor_readings = 0;
    
    while let Some(result) = pipeline.pop_result() {
        match result {
            Event::ValidationResult { status, sensor_id, .. } => {
                assert_eq!(status, ValidationStatus::Valid,
                    "All readings should pass validation");
                validation_count += 1;
                
                println!("Validated sensor: {}", sensor_id.as_str());
            }
            Event::SystemEvent { event_type, .. } => {
                // Fusion stage generates system events
                fusion_count += 1;
            }
            Event::SensorReading { .. } => {
                // Validated readings are forwarded
                sensor_readings += 1;
            }
            _ => {}
        }
    }
    
    // Verify integration worked correctly
    assert_eq!(validation_count, events.len(), 
        "Should validate all {} input readings", events.len());
    
    assert!(fusion_count > 0, 
        "Fusion stage should produce output when group is complete");
    
    assert!(sensor_readings > 0,
        "Valid readings should be forwarded through pipeline");
    
    println!("\nValidation-Fusion Integration Results:");
    println!("  Input events: {}", events.len());
    println!("  Validations: {}", validation_count);
    println!("  Fusion outputs: {}", fusion_count);
    println!("  Forwarded readings: {}", sensor_readings);
}

/// Test ML integration with validation
#[cfg(feature = "ml")]
#[test]
fn test_ml_validation_integration() {
    // Create ML configuration
    let ml_config = MLConfig {
        window_size: 20,
        forest_config: edgeguard_ml::ForestConfig {
            num_trees: 3,
            sample_size: 15,
            max_depth: 3,
            seed: 42,
            anomaly_threshold: 0.6,
        },
        sensor_types: vec![SensorType::Temperature],
        feature_window_ms: 5000,
        retrain_interval: Some(100),
        anomaly_threshold: 0.6,
        feature_complexity: FeatureComplexity::Basic,
        enable_correlation: false,
    };
    
    // Build pipeline with validation and ML
    let mut pipeline = Pipeline::<256>::builder()
        .add_stage(ValidationStage::new(
            TemperatureValidator::new(),
            SensorType::Temperature,
        ))
        .add_stage(MLAnomalyStage::<20>::new(ml_config))
        .build();
    
    // Generate normal pattern followed by anomaly
    let mut events = Vec::new();
    
    // Normal pattern
    for i in 0..50 {
        events.push(
            EventBuilder::new(i as u64 * 1000)
                .sensor("ml_temp", SensorType::Temperature)
                .reading(22.0 + (i as f32 * 0.1).sin(), 0.95)
                .unwrap()
        );
    }
    
    // Anomalous pattern
    for i in 50..60 {
        events.push(
            EventBuilder::new(i as u64 * 1000)
                .sensor("ml_temp", SensorType::Temperature)
                .reading(35.0 + (i as f32).sin() * 5.0, 0.95)
                .unwrap()
        );
    }
    
    // Process all events
    for event in &events {
        pipeline.push_event(event.clone());
    }
    
    let mut anomalies = 0;
    let mut valid_readings = 0;
    
    while pipeline.process_batch(10).unwrap() > 0 {
        while let Some(result) = pipeline.pop_result() {
            match result {
                Event::ValidationResult { status, .. } => {
                    if status == ValidationStatus::Valid {
                        valid_readings += 1;
                    }
                }
                Event::SystemEvent { event_type, .. } => {
                    if event_type == edgeguard_core::events::SystemEventType::ValidatorError {
                        anomalies += 1;
                    }
                }
                _ => {}
            }
        }
    }
    
    assert!(valid_readings > 50, "Most readings should be valid");
    assert!(anomalies > 0, "ML should detect anomalies");
}

/// Test schema-based validation with pipeline
#[cfg(feature = "schemas")]
#[test]
fn test_schema_validation_integration() {
    use apache_avro::Schema;
    
    // Create schema with physics constraints
    let schema_str = r#"{
        "type": "record",
        "name": "TemperatureSensor",
        "fields": [
            {
                "name": "value",
                "type": "float",
                "edgeguard.min": -40.0,
                "edgeguard.max": 85.0,
                "edgeguard.max_rate": 5.0
            }
        ]
    }"#;
    
    let schema = Schema::parse_str(schema_str).unwrap();
    
    // Extract constraints and create validator
    let constraints = SensorConstraints::extract_from_schema(&schema).unwrap();
    let validator = create_validator_from_constraints(&constraints, "temperature").unwrap();
    
    // Use in pipeline
    let mut pipeline = Pipeline::<128>::builder()
        .add_stage(ValidationStage::new(
            validator,
            SensorType::Temperature,
        ))
        .build();
    
    // Test events
    let events = vec![
        EventBuilder::new(1000)
            .sensor("schema_temp", SensorType::Temperature)
            .reading(25.0, 0.95)
            .unwrap(), // Valid
        EventBuilder::new(2000)
            .sensor("schema_temp", SensorType::Temperature)
            .reading(90.0, 0.95)
            .unwrap(), // Too high
        EventBuilder::new(3000)
            .sensor("schema_temp", SensorType::Temperature)
            .reading(-50.0, 0.95)
            .unwrap(), // Too low
    ];
    
    for event in &events {
        pipeline.push_event(event.clone());
    }
    
    pipeline.process_batch(10).unwrap();
    
    let mut valid = 0;
    let mut invalid = 0;
    
    while let Some(result) = pipeline.pop_result() {
        if let Event::ValidationResult { status, .. } = result {
            match status {
                ValidationStatus::Valid => valid += 1,
                ValidationStatus::OutOfRange => invalid += 1,
                _ => {}
            }
        }
    }
    
    assert_eq!(valid, 1, "One reading should be valid");
    assert_eq!(invalid, 2, "Two readings should be out of range");
}

/// Test connector integration with pipeline
#[cfg(all(feature = "connectors", feature = "std"))]
#[test]
fn test_connector_pipeline_integration() {
    use tokio::runtime::Runtime;
    use std::time::Duration;
    
    // Create MQTT connector
    let mqtt_config = MqttConfig {
        broker_url: "mqtt://test.mosquitto.org:1883".to_string(),
        client_id: "edgeguard_test".to_string(),
        topic: "edgeguard/test".to_string(),
        qos: 0,
        keep_alive_secs: 60,
        username: None,
        password: None,
        clean_session: true,
        reconnect_delay_ms: 5000,
    };
    
    // Create pipeline
    let mut pipeline = Pipeline::<128>::builder()
        .add_stage(ValidationStage::new(
            TemperatureValidator::new(),
            SensorType::Temperature,
        ))
        .build();
    
    // Shared result storage
    let results = Arc::new(Mutex::new(Vec::new()));
    let results_clone = results.clone();
    
    // Process pipeline results
    std::thread::spawn(move || {
        loop {
            if let Some(result) = pipeline.pop_result() {
                results_clone.lock().unwrap().push(result);
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    });
    
    // Test event
    let event = EventBuilder::new(1000)
        .sensor("mqtt_temp", SensorType::Temperature)
        .reading(25.0, 0.95)
        .unwrap();
    
    // Note: Actual MQTT connection would require a running broker
    // This test demonstrates the integration pattern
    println!("MQTT connector integration configured (broker connection not tested)");
    
    // Simulate processing
    pipeline.push_event(event);
    pipeline.process_batch(1).unwrap();
    
    std::thread::sleep(Duration::from_millis(100));
    
    let stored_results = results.lock().unwrap();
    assert!(!stored_results.is_empty(), "Should have processed results");
}

/// Test cross-validation between modules
#[test]
fn test_cross_module_validation() {
    use edgeguard_core::pipeline::CrossValidationStage;
    use edgeguard_core::events::CrossValidationType;
    
    // Create pipeline with cross-validation
    let mut cross_val = CrossValidationStage::new();
    cross_val.add_pair(
        SensorType::Temperature,
        SensorType::Humidity,
        CrossValidationType::DewPoint,
    ).unwrap();
    
    let mut pipeline = Pipeline::<256>::builder()
        .add_stage(ValidationStage::new(
            TemperatureValidator::new(),
            SensorType::Temperature,
        ))
        .add_stage(ValidationStage::new(
            HumidityValidator::new(),
            SensorType::Humidity,
        ))
        .add_stage(cross_val)
        .build();
    
    // Test events that should pass cross-validation
    let events = vec![
        EventBuilder::new(1000)
            .sensor("temp", SensorType::Temperature)
            .reading(25.0, 0.95)
            .unwrap(),
        EventBuilder::new(1000)
            .sensor("hum", SensorType::Humidity)
            .reading(60.0, 0.95)
            .unwrap(), // Dew point ~16.5°C < 25°C, valid
    ];
    
    for event in &events {
        pipeline.push_event(event.clone());
    }
    
    pipeline.process_batch(10).unwrap();
    
    let mut cross_val_results = 0;
    let mut all_valid = true;
    
    while let Some(result) = pipeline.pop_result() {
        if let Event::CrossValidationResult { status, .. } = result {
            cross_val_results += 1;
            if status != ValidationStatus::Valid {
                all_valid = false;
            }
        }
    }
    
    assert!(cross_val_results > 0, "Should have cross-validation results");
    assert!(all_valid, "All cross-validations should pass");
}

/// Test memory efficiency across modules
#[test]
fn test_cross_module_memory_efficiency() {
    // Measure combined module sizes
    let sizes = [
        ("Pipeline<128>", std::mem::size_of::<Pipeline<128>>()),
        ("TemperatureValidator", std::mem::size_of::<TemperatureValidator>()),
        ("WeightedAverageFusion<3>", std::mem::size_of::<WeightedAverageFusion<3>>()),
    ];
    
    println!("\nCross-Module Memory Analysis:");
    for (name, size) in &sizes {
        println!("  {}: {} bytes", name, size);
    }
    
    // Test complete system memory usage
    let total_size: usize = sizes.iter().map(|(_, s)| s).sum();
    println!("  Total core components: {} bytes", total_size);
    
    // Verify it fits in embedded constraints
    assert!(total_size < 8192, "Core components should fit in 8KB");
    
    #[cfg(feature = "ml")]
    {
        let ml_size = std::mem::size_of::<MLAnomalyStage<50>>();
        println!("  MLAnomalyStage<50>: {} bytes", ml_size);
        assert!(ml_size < 16384, "ML stage should fit in 16KB");
    }
    
    #[cfg(feature = "schemas")]
    {
        let registry_size = std::mem::size_of::<SchemaRegistry>();
        println!("  SchemaRegistry: {} bytes", registry_size);
        assert!(registry_size < 4096, "Schema registry should fit in 4KB");
    }
}

/// Test error propagation across modules
#[test]
fn test_cross_module_error_handling() {
    // Create pipeline with multiple validation stages
    let mut pipeline = Pipeline::<128>::builder()
        .add_stage(ValidationStage::new(
            TemperatureValidator::new_with_limits(-10.0, 40.0, 5.0),
            SensorType::Temperature,
        ))
        .add_stage(ValidationStage::new(
            HumidityValidator::new_with_limits(20.0, 80.0, 10.0),
            SensorType::Humidity,
        ))
        .build();
    
    // Test events that will fail validation
    let events = vec![
        EventBuilder::new(1000)
            .sensor("temp", SensorType::Temperature)
            .reading(-20.0, 0.95)
            .unwrap(), // Below minimum
        EventBuilder::new(1000)
            .sensor("hum", SensorType::Humidity)
            .reading(90.0, 0.95)
            .unwrap(), // Above maximum
        EventBuilder::new(1000)
            .sensor("temp2", SensorType::Temperature)
            .reading(f32::NAN, 0.95)
            .unwrap(), // Invalid value
    ];
    
    for event in &events {
        pipeline.push_event(event.clone());
    }
    
    pipeline.process_batch(10).unwrap();
    
    let mut error_types = std::collections::HashMap::new();
    
    while let Some(result) = pipeline.pop_result() {
        if let Event::ValidationResult { status, .. } = result {
            *error_types.entry(status).or_insert(0) += 1;
        }
    }
    
    assert_eq!(error_types[&ValidationStatus::OutOfRange], 2, "Should have range errors");
    assert_eq!(error_types[&ValidationStatus::InvalidValue], 1, "Should have invalid value error");
}

/// Test module lifecycle and resource management
#[test]
fn test_module_lifecycle() {
    // Test creating and destroying modules repeatedly
    for i in 0..10 {
        let mut pipeline = Pipeline::<64>::builder()
            .add_stage(ValidationStage::new(
                TemperatureValidator::new(),
                SensorType::Temperature,
            ))
            .build();
        
        // Process some events
        let event = EventBuilder::new(i as u64 * 1000)
            .sensor("lifecycle", SensorType::Temperature)
            .reading(25.0, 0.95)
            .unwrap();
        
        pipeline.push_event(event);
        pipeline.process_batch(1).unwrap();
        
        // Pipeline and stages should be properly cleaned up
    }
    
    // Test resetting modules
    let mut fusion = WeightedAverageFusion::<3>::new();
    
    // Add some data
    for i in 0..10 {
        fusion.update(&[20.0, 21.0, 22.0], i as u64 * 1000).unwrap();
    }
    
    // Reset and verify clean state
    fusion.reset();
    let (value, confidence) = fusion.update(&[25.0, 25.0, 25.0], 0).unwrap();
    assert!((value - 25.0).abs() < 0.1, "Should reset to new values");
}

/// Test module compatibility with different configurations
#[test]
fn test_module_configuration_compatibility() {
    // Test different validator configurations
    let validators = vec![
        TemperatureValidator::new(),
        TemperatureValidator::new_with_limits(-50.0, 50.0, 2.0),
        TemperatureValidator::indoor(),
        TemperatureValidator::outdoor(),
    ];
    
    for (i, validator) in validators.into_iter().enumerate() {
        let mut pipeline = Pipeline::<64>::builder()
            .add_stage(ValidationStage::new(validator, SensorType::Temperature))
            .build();
        
        let event = EventBuilder::new(1000)
            .sensor(&format!("config_test_{}", i), SensorType::Temperature)
            .reading(20.0, 0.95)
            .unwrap();
        
        pipeline.push_event(event);
        assert_eq!(pipeline.process_batch(1).unwrap(), 1, "Should process event");
    }
    
    // Test different fusion configurations
    let fusion_configs = vec![
        WeightedAverageFusion::<2>::new(),
        WeightedAverageFusion::<2>::with_weights([0.7, 0.3]),
    ];
    
    for mut fusion in fusion_configs {
        let (value, _) = fusion.update(&[20.0, 22.0], 1000).unwrap();
        assert!(value >= 20.0 && value <= 22.0, "Fusion should produce valid result");
    }
}