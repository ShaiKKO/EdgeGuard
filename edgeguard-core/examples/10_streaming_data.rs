//! Example 10: Streaming Data from Files
//!
//! This example demonstrates how to:
//! - Read sensor data from CSV and JSON files
//! - Process streaming data through validation pipeline
//! - Handle parse errors gracefully
//! - Collect statistics on data processing

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(not(feature = "std"), no_main)]

#[cfg(feature = "std")]
use edgeguard_core::{
    events::{Event, SensorType},
    pipeline::{Pipeline, ValidationStage, AggregationStage, WindowSpec, AggregationMethod},
    stream::{FileStream, Stream, StreamError, FileFormat},
    validators::{TemperatureValidator, HumidityValidator, PressureValidator},
};

#[cfg(feature = "std")]
fn main() {
    println!("=== EdgeGuard Streaming Data Example ===\n");
    
    // Example 1: Process CSV file
    if let Err(e) = process_csv_file() {
        println!("CSV processing error: {:?}", e);
    }
    
    println!("\n{}\n", "=".repeat(50));
    
    // Example 2: Process JSON Lines file
    if let Err(e) = process_json_file() {
        println!("JSON processing error: {:?}", e);
    }
    
    println!("\n{}\n", "=".repeat(50));
    
    // Example 3: Stream to pipeline with validation
    if let Err(e) = stream_to_pipeline() {
        println!("Pipeline processing error: {:?}", e);
    }
    
    println!("\n{}\n", "=".repeat(50));
    
    // Example 4: Batch processing with aggregation
    if let Err(e) = batch_processing() {
        println!("Batch processing error: {:?}", e);
    }
}

#[cfg(feature = "std")]
fn process_csv_file() -> Result<(), StreamError<std::io::Error>> {
    println!("Example 1: Processing CSV File");
    println!("------------------------------");
    
    // Open CSV file and skip header
    let mut stream = FileStream::from_csv("examples/data/sensors.csv")?
        .with_skip_lines(1); // Skip header row
    
    let mut event_count = 0;
    let mut temp_count = 0;
    let mut humid_count = 0;
    let mut press_count = 0;
    
    // Process all events
    let _stats = stream.process_all(|event| {
        event_count += 1;
        
        // Count events by sensor type
        if let Event::SensorReading { sensor_type, value, sensor_id, .. } = &event {
            match sensor_type {
                SensorType::Temperature => temp_count += 1,
                SensorType::Humidity => humid_count += 1,
                SensorType::Pressure => press_count += 1,
                _ => {},
            }
            
            // Show first few events
            if event_count <= 3 {
                println!("  {} ({}): {:.2}", sensor_id.as_str(), 
                    match sensor_type {
                        SensorType::Temperature => "temp",
                        SensorType::Humidity => "humid",
                        SensorType::Pressure => "press",
                        _ => "other",
                    }, value);
            }
        }
        Ok(())
    })?;
    
    println!("\nCSV Processing Summary:");
    println!("  Total events: {}", event_count);
    println!("  Temperature: {}", temp_count);
    println!("  Humidity: {}", humid_count);
    println!("  Pressure: {}", press_count);
    
    Ok(())
}

#[cfg(feature = "std")]
fn process_json_file() -> Result<(), StreamError<std::io::Error>> {
    println!("Example 2: Processing JSON Lines File");
    println!("------------------------------------");
    
    let mut stream = FileStream::new("examples/data/sensors.jsonl", FileFormat::JsonLines)?;
    
    let mut event_count = 0;
    let mut anomaly_count = 0;
    
    stream.process_all(|event| {
        event_count += 1;
        
        if let Event::SensorReading { value, sensor_type, sensor_id, .. } = &event {
            // Check for anomalies
            let is_anomaly = match sensor_type {
                SensorType::Temperature => *value < -20.0 || *value > 50.0,
                SensorType::Humidity => *value < 0.0 || *value > 100.0,
                SensorType::Pressure => *value < 900.0 || *value > 1100.0,
                _ => false,
            };
            
            if is_anomaly {
                anomaly_count += 1;
                println!("  ANOMALY: {} = {:.2}", sensor_id.as_str(), value);
            }
        }
        Ok(())
    })?;
    
    println!("\nJSON Processing Summary:");
    println!("  Total events: {}", event_count);
    println!("  Anomalies detected: {}", anomaly_count);
    
    Ok(())
}

#[cfg(feature = "std")]
fn stream_to_pipeline() -> Result<(), StreamError<std::io::Error>> {
    println!("Example 3: Stream to Validation Pipeline");
    println!("---------------------------------------");
    
    // Create validation pipeline
    let mut pipeline = Pipeline::<16>::builder()
        .add_stage(ValidationStage::new(
            TemperatureValidator::indoor(),
            SensorType::Temperature
        ))
        .add_stage(ValidationStage::new(
            HumidityValidator::strict(),
            SensorType::Humidity
        ))
        .add_stage(ValidationStage::new(
            PressureValidator::new_with_altitude(0.0), // Sea level
            SensorType::Pressure
        ))
        .build();
    
    // Stream CSV data
    let mut stream = FileStream::from_csv("examples/data/sensors.csv")?
        .with_skip_lines(1);
    
    // Process through pipeline
    stream.process_all(|event| {
        pipeline.push_event(event.clone());
        Ok(())
    })?;
    
    // Process pipeline events
    let _ = pipeline.process_batch(100);
    
    // Collect validation results
    let mut valid_count = 0;
    let mut invalid_count = 0;
    
    while let Some(result) = pipeline.pop_result() {
        if let Event::ValidationResult { status, sensor_id, .. } = result {
            if matches!(status, edgeguard_core::events::ValidationStatus::Valid) {
                valid_count += 1;
            } else {
                invalid_count += 1;
                println!("  Invalid: {} - {:?}", sensor_id.as_str(), status);
            }
        }
    }
    
    println!("\nValidation Summary:");
    println!("  Valid readings: {}", valid_count);
    println!("  Invalid readings: {}", invalid_count);
    
    Ok(())
}

#[cfg(feature = "std")]
fn batch_processing() -> Result<(), StreamError<std::io::Error>> {
    println!("Example 4: Batch Processing with Aggregation");
    println!("-------------------------------------------");
    
    // Create aggregation pipeline
    let mut pipeline = Pipeline::<16>::builder()
        .add_stage(AggregationStage::new(
            WindowSpec::Time { duration_ms: 60000 }, // 1 minute windows
            AggregationMethod::Mean,
            SensorType::Temperature
        ))
        .build();
    
    let mut stream = FileStream::from_csv("examples/data/sensors.csv")?
        .with_skip_lines(1);
    
    let mut batch_count = 0;
    
    // Process in batches
    stream.process_batch(10, |batch| {
        println!("  Processing batch of {} events", batch.len());
        
        for event in batch {
            pipeline.push_event(event.clone());
        }
        
        // Process any events in the pipeline (ignore pipeline errors)
        let _ = pipeline.process_batch(20);
        
        // Check for aggregated results
        while let Some(result) = pipeline.pop_result() {
            if let Event::BatchReading { mean_value, count, sensor_type, .. } = result {
                if matches!(sensor_type, SensorType::Temperature) {
                    println!("    Aggregated: {} readings, mean = {:.2}°C", count, mean_value);
                }
            }
        }
        
        batch_count += 1;
        Ok(())
    })?;
    
    println!("\nBatch Processing Summary:");
    println!("  Total batches processed: {}", batch_count);
    
    Ok(())
}

#[cfg(not(feature = "std"))]
fn main() {
    // No-std placeholder
}