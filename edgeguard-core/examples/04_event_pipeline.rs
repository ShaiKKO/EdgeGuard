//! Event-Driven Validation Pipeline Example
//!
//! This example demonstrates EdgeGuard's powerful event pipeline system,
//! which processes sensor data through customizable stages.
//!
//! ## What You'll Learn
//!
//! - Building validation pipelines with multiple stages
//! - Using pre-built stages (validation, filtering, buffering)
//! - Processing events through the pipeline
//! - Handling pipeline metrics and monitoring
//!
//! ## Pipeline Architecture
//!
//! ```text
//! Input Queue → Stage 1 → Stage 2 → ... → Stage N → Output Queue
//!      ↓           ↓         ↓               ↓           ↓
//!   Events    Validate   Filter      Aggregate      Results
//! ```
//!
//! ## Running the Example
//!
//! ```bash
//! cargo run --example 04_event_pipeline
//! ```

use edgeguard_core::{
    events::{Event, SensorType, ValidationStatus, EventBuilder},
    pipeline::{Pipeline, ValidationStage, FilterStage, AggregationStage, 
               BackpressureStrategy, WindowSpec, AggregationMethod},
    validators::{TemperatureValidator, HumidityValidator},
};

fn main() {
    println!("EdgeGuard Event Pipeline Example");
    println!("================================\n");

    // Build the event processing pipeline
    println!("Building event pipeline with stages:");
    println!("1. Temperature Validation - Physics constraints");
    println!("2. Humidity Validation - Range checks");
    println!("3. Filter Stage - Remove out-of-range readings");
    println!("4. Aggregation - Batch similar events\n");
    
    let mut pipeline = Pipeline::<8>::builder()
        .add_stage(ValidationStage::new(
            TemperatureValidator::new_with_limits(-20.0, 45.0, 3.0),
            SensorType::Temperature
        ))
        .add_stage(ValidationStage::new(
            HumidityValidator::default(),
            SensorType::Humidity
        ))
        .add_stage(FilterStage::new(
            |event: &Event| {
                // Filter out invalid validation results
                match event {
                    Event::ValidationResult { status, .. } => 
                        *status == ValidationStatus::Valid,
                    _ => true, // Pass through non-validation events
                }
            },
            "ValidOnlyFilter"
        ))
        .add_stage(AggregationStage::new(
            WindowSpec::Time { duration_ms: 1000 }, // 1 second window
            AggregationMethod::Mean,
            SensorType::Temperature
        ))
        .backpressure(BackpressureStrategy::DropOldest)
        .build();

    // Simulate sensor events
    println!("Processing sensor events:");
    println!("------------------------\n");
    
    let events = generate_test_events();
    let _total_events = events.len();
    
    // Push events to input queue
    for (i, event) in events.into_iter().enumerate() {
        println!("Event {}: {}", i + 1, format_event(&event));
        
        if !pipeline.push_event(event) {
            println!("  ⚠ Input queue full!");
        }
    }
    
    // Process the batch
    println!("\nProcessing batch...");
    match pipeline.process_batch(20) {
        Ok(processed) => println!("Processed {} events", processed),
        Err(e) => println!("Processing error: {:?}", e),
    }
    
    // Collect results
    println!("\nCollecting results:");
    println!("------------------");
    let mut results = 0;
    while let Some(event) = pipeline.pop_result() {
        results += 1;
        println!("  Result {}: {}", results, format_event(&event));
    }
    
    // Show pipeline metrics
    let metrics = pipeline.metrics();
    println!("\n\nPipeline Metrics:");
    println!("----------------");
    println!("Events processed by stage:");
    for (i, &count) in metrics.events_processed.iter().enumerate() {
        if count > 0 {
            println!("  Stage {}: {} events", i, count);
        }
    }
    println!("Events dropped by stage:");
    for (i, &count) in metrics.events_dropped.iter().enumerate() {
        if count > 0 {
            println!("  Stage {}: {} events", i, count);
        }
    }
    
    // Demonstrate cross-sensor validation pipeline
    println!("\n\nCross-Sensor Validation Pipeline:");
    println!("--------------------------------");
    demonstrate_cross_validation_pipeline();
}

fn generate_test_events() -> Vec<Event> {
    let mut events = Vec::new();
    let base_time = 1000;
    
    // Temperature readings - normal progression
    for i in 0..5 {
        events.push(
            EventBuilder::new(base_time + i * 100)
                .sensor("temp_01", SensorType::Temperature)
                .reading(20.0 + i as f32 * 0.5, 0.95)
                .unwrap()
        );
    }
    
    // Humidity readings
    events.push(
        EventBuilder::new(base_time + 200)
            .sensor("hum_01", SensorType::Humidity)
            .reading(55.0, 0.90)
            .unwrap()
    );
    
    // Temperature spike (rate violation)
    events.push(
        EventBuilder::new(base_time + 500)
            .sensor("temp_01", SensorType::Temperature)
            .reading(35.0, 0.95) // 12.5°C jump!
            .unwrap()
    );
    
    // Out of range temperature
    events.push(
        EventBuilder::new(base_time + 600)
            .sensor("temp_02", SensorType::Temperature)
            .reading(50.0, 0.95) // Too hot!
            .unwrap()
    );
    
    // Humidity out of range
    events.push(
        EventBuilder::new(base_time + 700)
            .sensor("hum_01", SensorType::Humidity)
            .reading(105.0, 0.90) // Impossible!
            .unwrap()
    );
    
    // Rapid fire temperature events
    for i in 0..3 {
        events.push(
            EventBuilder::new(base_time + 800 + i * 10)
                .sensor("temp_03", SensorType::Temperature)
                .reading(25.0 + i as f32 * 0.1, 0.85)
                .unwrap()
        );
    }
    
    events
}

fn demonstrate_cross_validation_pipeline() {
    use edgeguard_core::pipeline::CrossValidationStage;
    use edgeguard_core::events::CrossValidationType;
    
    println!("Building pipeline with cross-sensor validation:");
    
    let mut cross_stage = CrossValidationStage::new();
    cross_stage.add_pair(
        SensorType::Temperature,
        SensorType::Humidity,
        CrossValidationType::DewPoint
    );
    
    let mut pipeline = Pipeline::<8>::builder()
        .add_stage(ValidationStage::new(
            TemperatureValidator::default(),
            SensorType::Temperature
        ))
        .add_stage(ValidationStage::new(
            HumidityValidator::default(),
            SensorType::Humidity
        ))
        .add_stage(cross_stage)
        .build();
    
    println!("  ✓ Temperature validation");
    println!("  ✓ Humidity validation");
    println!("  ✓ Dew point cross-validation");
    
    // Generate paired sensor readings
    let events = vec![
        EventBuilder::new(1000)
            .sensor("temp_01", SensorType::Temperature)
            .reading(25.0, 0.95)
            .unwrap(),
        EventBuilder::new(1000)
            .sensor("hum_01", SensorType::Humidity)  
            .reading(90.0, 0.90)
            .unwrap(),
        EventBuilder::new(2000)
            .sensor("temp_01", SensorType::Temperature)
            .reading(5.0, 0.95)
            .unwrap(),
        EventBuilder::new(2000)
            .sensor("hum_01", SensorType::Humidity)
            .reading(100.0, 0.90) // Dew point would exceed temperature!
            .unwrap(),
    ];
    
    println!("\nProcessing paired sensor readings...");
    
    // Push and process events
    for event in events {
        pipeline.push_event(event);
    }
    
    match pipeline.process_batch(10) {
        Ok(processed) => println!("Processed {} events", processed),
        Err(e) => println!("Error: {:?}", e),
    }
    
    // Check results
    println!("\nCross-validation results:");
    while let Some(event) = pipeline.pop_result() {
        if let Event::CrossValidationResult { 
            primary_sensor, 
            related_sensor, 
            status, 
            details, 
            .. 
        } = event {
            println!("  {} + {} → {:?}", 
                     primary_sensor.as_str(), 
                     related_sensor.as_str(), 
                     status);
            if status != ValidationStatus::Valid {
                println!("    Expected: {:.1}, Actual: {:.1}, Deviation: {:.1}%",
                         details.expected_value,
                         details.actual_value,
                         details.deviation_percent);
            }
        }
    }
}

fn format_event(event: &Event) -> String {
    match event {
        Event::SensorReading { sensor_id, sensor_type, value, timestamp, .. } => {
            format!("{} [{}]: {:.1}{} at t={}",
                    sensor_id.as_str(),
                    sensor_type.name(),
                    value,
                    sensor_type.unit(),
                    timestamp)
        }
        Event::ValidationResult { sensor_id, status, timestamp, .. } => {
            format!("{} validation: {:?} at t={}",
                    sensor_id.as_str(),
                    status,
                    timestamp)
        }
        Event::CrossValidationResult { primary_sensor, related_sensor, status, .. } => {
            format!("{} × {} cross-validation: {:?}",
                    primary_sensor.as_str(),
                    related_sensor.as_str(),
                    status)
        }
        Event::BatchReading { sensor_id, count, mean_value, .. } => {
            format!("{} batch: {} readings, avg={:.1}",
                    sensor_id.as_str(),
                    count,
                    mean_value)
        }
        Event::SystemEvent { event_type, .. } => {
            format!("System event: {:?}", event_type)
        }
        _ => format!("{:?}", event),
    }
}