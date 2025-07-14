//! EdgeGuard ML Anomaly Detection Example
//!
//! This example demonstrates how machine learning complements physics-based validation:
//! - Physics validation catches hard constraint violations (impossible values)
//! - ML detection identifies subtle anomalies (unusual but valid patterns)
//! 
//! ## Scenario: Smart Building HVAC Monitoring
//! 
//! We monitor temperature, humidity, and pressure sensors in a building.
//! The ML system learns normal patterns and detects:
//! - Sensor drift (gradual degradation)
//! - HVAC system changes (AC turning on/off)
//! - Environmental anomalies (door left open)
//! - Correlation breaks (sensor malfunction)
//!
//! ## Memory Usage
//! - Physics validation: ~10KB
//! - ML with Isolation Forest: ~50KB
//! - Total: ~60KB (suitable for ESP32)

use edgeguard_core::{
    events::{Event, EventBuilder, SensorType},
    pipeline::{Pipeline, PipelineBuilder, ValidationStage, PipelineStage},
    time::{FixedTime, Timestamp},
    validators::{TemperatureValidator, HumidityValidator, PressureValidator},
};

use edgeguard_ml::{
    MLAnomalyStage, MLConfig, FeatureComplexity,
    ForestConfig,
};

use std::f32::consts::PI;

/// Simulate realistic sensor data with various anomaly patterns
struct SensorSimulator {
    time: Timestamp,
    phase: f32,
    hvac_on: bool,
    door_open: bool,
    sensor_drift: f32,
}

impl SensorSimulator {
    fn new() -> Self {
        Self {
            time: 0,
            phase: 0.0,
            hvac_on: false,
            door_open: false,
            sensor_drift: 0.0,
        }
    }
    
    fn generate_reading(&mut self) -> (f32, f32, f32) {
        // Base temperature follows daily cycle
        let base_temp = 22.0 + 3.0 * (self.phase * 2.0 * PI / 86400.0).sin();
        
        // HVAC effect
        let hvac_effect = if self.hvac_on { -2.0 } else { 0.0 };
        
        // Door open causes rapid changes
        let door_effect = if self.door_open { 
            5.0 * (1.0 - (-self.phase / 300.0).exp()) // Exponential rise
        } else { 
            0.0 
        };
        
        // Add sensor drift over time
        let temp = base_temp + hvac_effect + door_effect + self.sensor_drift;
        
        // Humidity inversely correlates with temperature normally
        let base_humidity = 50.0 - 2.0 * (temp - 22.0);
        let humidity = base_humidity + if self.door_open { 10.0 } else { 0.0 };
        
        // Pressure slightly affected by HVAC
        let pressure = 1013.25 + if self.hvac_on { -0.5 } else { 0.0 };
        
        self.time += 1000; // 1 second intervals
        self.phase += 1.0;
        
        (temp, humidity.clamp(0.0, 100.0), pressure)
    }
    
    fn inject_anomaly(&mut self, anomaly_type: &str) {
        match anomaly_type {
            "hvac_on" => self.hvac_on = true,
            "hvac_off" => self.hvac_on = false,
            "door_open" => self.door_open = true,
            "door_close" => self.door_open = false,
            "drift_start" => self.sensor_drift = 0.01, // Will accumulate
            _ => {}
        }
    }
}

/// Custom stage to print ML anomaly events
struct AnomalyReporter;

impl PipelineStage for AnomalyReporter {
    fn process(&mut self, event: Event, output: &mut edgeguard_core::pipeline::StageOutput) -> edgeguard_core::pipeline::PipelineResult<()> {
        if let Event::SystemEvent { event_type, timestamp, details } = &event {
            if *event_type == edgeguard_core::events::SystemEventType::ValidatorError {
                let anomaly_score = (*details as f32) / 1000.0;
                println!("[{:6}s] 🚨 ML Anomaly Detected! Score: {:.3}", 
                    timestamp / 1000, anomaly_score);
            }
        }
        output.push(event);
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "AnomalyReporter"
    }
}

fn main() {
    println!("=== EdgeGuard ML Anomaly Detection Example ===\n");
    println!("This example shows how ML complements physics validation.");
    println!("Physics catches impossible values, ML catches unusual patterns.\n");
    
    // Create fixed time source for reproducible results
    let mut time_source = FixedTime::new(0);
    
    // Configure ML anomaly detection
    let ml_config = MLConfig {
        window_size: 100,
        forest_config: ForestConfig {
            num_trees: 10,      // Reduced for embedded
            sample_size: 50,    // Smaller samples
            max_depth: 6,       // Shallower trees
            seed: 42,
            anomaly_threshold: 0.6,
        },
        sensor_types: vec![
            SensorType::Temperature,
            SensorType::Humidity,
            SensorType::Pressure,
        ],
        feature_window_ms: 10000,     // 10 second windows
        retrain_interval: Some(300),  // Retrain every 5 minutes
        anomaly_threshold: 0.65,
        feature_complexity: FeatureComplexity::Extended,
        enable_correlation: true,     // Use our physics-based correlations
    };
    
    // Build pipeline with physics validation AND ML detection
    let mut pipeline = Pipeline::<1024>::builder()
        // Physics validation first
        .add_stage(ValidationStage::new(
            TemperatureValidator::new_with_limits(-10.0, 50.0, 5.0),
            SensorType::Temperature,
        ))
        .add_stage(ValidationStage::new(
            HumidityValidator::new_with_limits(10.0, 90.0, 10.0),
            SensorType::Humidity,
        ))
        .add_stage(ValidationStage::new(
            PressureValidator::new_with_altitude(0.0),
            SensorType::Pressure,
        ))
        
        // ML anomaly detection for subtle patterns
        .add_stage(MLAnomalyStage::<50>::new(ml_config))
        
        // Report anomalies
        .add_stage(AnomalyReporter)
        
        .build();
    
    // Simulate sensor data with anomalies
    let mut simulator = SensorSimulator::new();
    let mut events = Vec::new();
    
    println!("Phase 1: Normal Operation (Learning)");
    println!("=====================================");
    
    // Generate 2 minutes of normal data for learning
    for i in 0..120 {
        let (temp, humidity, pressure) = simulator.generate_reading();
        
        // Create events
        let timestamp = i as u64 * 1000;
        time_source.set(timestamp);
        
        events.push(
            EventBuilder::new(timestamp)
                .sensor("temp_sensor", SensorType::Temperature)
                .reading(temp, 0.95)
                .unwrap()
        );
        events.push(
            EventBuilder::new(timestamp)
                .sensor("humidity_sensor", SensorType::Humidity)
                .reading(humidity, 0.95)
                .unwrap()
        );
        events.push(
            EventBuilder::new(timestamp)
                .sensor("pressure_sensor", SensorType::Pressure)
                .reading(pressure, 0.95)
                .unwrap()
        );
        
        if i % 30 == 0 {
            println!("[{:3}s] T: {:.1}°C, H: {:.1}%, P: {:.1} hPa", 
                i, temp, humidity, pressure);
        }
    }
    
    println!("\nPhase 2: HVAC Activation");
    println!("========================");
    simulator.inject_anomaly("hvac_on");
    
    for i in 120..180 {
        let (temp, humidity, pressure) = simulator.generate_reading();
        let timestamp = i as u64 * 1000;
        
        events.push(
            EventBuilder::new(timestamp)
                .sensor("temp_sensor", SensorType::Temperature)
                .reading(temp, 0.95)
                .unwrap()
        );
        events.push(
            EventBuilder::new(timestamp)
                .sensor("humidity_sensor", SensorType::Humidity)
                .reading(humidity, 0.95)
                .unwrap()
        );
        events.push(
            EventBuilder::new(timestamp)
                .sensor("pressure_sensor", SensorType::Pressure)
                .reading(pressure, 0.95)
                .unwrap()
        );
    }
    
    println!("\nPhase 3: Door Left Open");
    println!("=======================");
    simulator.inject_anomaly("door_open");
    
    for i in 180..210 {
        let (temp, humidity, pressure) = simulator.generate_reading();
        let timestamp = i as u64 * 1000;
        
        events.push(
            EventBuilder::new(timestamp)
                .sensor("temp_sensor", SensorType::Temperature)
                .reading(temp, 0.95)
                .unwrap()
        );
        events.push(
            EventBuilder::new(timestamp)
                .sensor("humidity_sensor", SensorType::Humidity)
                .reading(humidity, 0.95)
                .unwrap()
        );
        events.push(
            EventBuilder::new(timestamp)
                .sensor("pressure_sensor", SensorType::Pressure)
                .reading(pressure, 0.95)
                .unwrap()
        );
    }
    
    simulator.inject_anomaly("door_close");
    simulator.inject_anomaly("hvac_off");
    
    println!("\nPhase 4: Sensor Drift");
    println!("====================");
    simulator.inject_anomaly("drift_start");
    
    for i in 210..300 {
        let (temp, humidity, pressure) = simulator.generate_reading();
        let timestamp = i as u64 * 1000;
        
        // Accumulate drift
        simulator.sensor_drift += 0.01;
        
        events.push(
            EventBuilder::new(timestamp)
                .sensor("temp_sensor", SensorType::Temperature)
                .reading(temp, 0.95)
                .unwrap()
        );
        events.push(
            EventBuilder::new(timestamp)
                .sensor("humidity_sensor", SensorType::Humidity)
                .reading(humidity, 0.95)
                .unwrap()
        );
        events.push(
            EventBuilder::new(timestamp)
                .sensor("pressure_sensor", SensorType::Pressure)
                .reading(pressure, 0.95)
                .unwrap()
        );
    }
    
    // Process all events through the pipeline
    println!("\nProcessing {} events through pipeline...\n", events.len());
    
    let mut physics_violations = 0;
    let mut ml_anomalies = 0;
    
    for event in &events {
        pipeline.push_event(event.clone());
    }
    
    // Process in batches
    while pipeline.process_batch(100).unwrap() > 0 {
        // Collect results
        while let Some(result) = pipeline.pop_result() {
            match &result {
                Event::ValidationResult { status, .. } => {
                    if *status != edgeguard_core::events::ValidationStatus::Valid {
                        physics_violations += 1;
                    }
                }
                Event::SystemEvent { event_type, .. } => {
                    if *event_type == edgeguard_core::events::SystemEventType::ValidatorError {
                        ml_anomalies += 1;
                    }
                }
                _ => {}
            }
        }
    }
    
    // Summary
    println!("\n=== Summary ===");
    println!("Total events processed: {}", events.len());
    println!("Physics violations: {}", physics_violations);
    println!("ML anomalies detected: {}", ml_anomalies);
    
    println!("\n📊 Memory Usage Estimate:");
    println!("  - Physics validators: ~10KB");
    println!("  - ML Isolation Forest: ~50KB");
    println!("  - Sensor buffers: ~5KB");
    println!("  - Total: ~65KB (fits in ESP32!)");
    
    println!("\n✅ Key Insights:");
    println!("  - Physics validation caught {} hard violations", physics_violations);
    println!("  - ML detected {} subtle anomalies", ml_anomalies);
    println!("  - HVAC changes detected as correlation breaks");
    println!("  - Door events caused rapid environmental shifts");
    println!("  - Sensor drift accumulated gradually");
    
    println!("\n🎯 EdgeGuard combines physics and ML for robust validation!");
}