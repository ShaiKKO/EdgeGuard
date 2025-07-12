//! Error Handling and Recovery Example
//!
//! This example demonstrates EdgeGuard's comprehensive error handling
//! capabilities and recovery strategies for production deployments.
//!
//! ## What You'll Learn
//!
//! - Different types of validation errors
//! - Error recovery strategies
//! - Graceful degradation patterns
//! - Logging and monitoring integration
//!
//! ## Error Categories
//!
//! 1. **Validation Errors** - Out of range, rate exceeded, invalid values
//! 2. **System Errors** - Queue overflow, resource exhaustion
//! 3. **Sensor Errors** - Quality degradation, missing data
//! 4. **Pipeline Errors** - Stage failures, backpressure
//!
//! ## Running the Example
//!
//! ```bash
//! cargo run --example 06_error_handling
//! ```

use edgeguard_core::{
    events::{Event, SensorType, ValidationStatus, EventBuilder, SystemEventType, InlineString},
    pipeline::{Pipeline, PipelineStage, StageOutput, PipelineResult, PipelineError,
               ValidationStage, BackpressureStrategy},
    validators::{TemperatureValidator, HumidityValidator},
    errors::ValidationError,
};

/// Error recovery stage that handles validation failures
struct ErrorRecoveryStage {
    /// Count of consecutive errors per sensor
    error_counts: heapless::FnvIndexMap<u64, u8, 16>,
    /// Last known good values per sensor
    last_good_values: heapless::FnvIndexMap<u64, (f32, u64), 16>,
    /// Maximum consecutive errors before sensor marked bad
    max_errors: u8,
}

impl ErrorRecoveryStage {
    fn new(max_errors: u8) -> Self {
        Self {
            error_counts: heapless::FnvIndexMap::new(),
            last_good_values: heapless::FnvIndexMap::new(),
            max_errors,
        }
    }
    
    fn sensor_id_hash(sensor_id: &str) -> u64 {
        let bytes = sensor_id.as_bytes();
        let mut hash = 0u64;
        for &b in bytes {
            hash = hash.wrapping_mul(31).wrapping_add(b as u64);
        }
        hash
    }
}

impl PipelineStage for ErrorRecoveryStage {
    fn process(&mut self, event: Event, output: &mut StageOutput) -> PipelineResult<()> {
        match &event {
            Event::ValidationResult { sensor_id, status, timestamp, .. } => {
                let id_hash = Self::sensor_id_hash(sensor_id.as_str());
                
                match status {
                    ValidationStatus::Valid => {
                        // Reset error count on successful validation
                        let _ = self.error_counts.remove(&id_hash);
                        println!("  ✓ {} validation successful", sensor_id.as_str());
                        output.forward(event)?;
                    }
                    ValidationStatus::OutOfRange | 
                    ValidationStatus::RateExceeded |
                    ValidationStatus::InvalidValue => {
                        // Increment error count
                        let count = self.error_counts.get(&id_hash).copied().unwrap_or(0);
                        let new_count = count + 1;
                        let _ = self.error_counts.insert(id_hash, new_count);
                        
                        println!("  ⚠ {} validation failed: {:?} (count: {})", 
                                 sensor_id.as_str(), status, new_count);
                        
                        if new_count >= self.max_errors {
                            // Mark sensor as bad
                            output.emit(Event::SystemEvent {
                                event_type: SystemEventType::ValidatorError,
                                timestamp: *timestamp,
                                details: new_count as u32,
                            })?;
                            
                            println!("  ✗ {} marked as bad after {} errors", 
                                     sensor_id.as_str(), new_count);
                            
                            // Try to use last known good value
                            if let Some((last_value, last_ts)) = self.last_good_values.get(&id_hash) {
                                println!("  → Using last good value: {:.1} from t={}", 
                                         last_value, last_ts);
                                
                                // Create substitute reading
                                output.emit(
                                    EventBuilder::new(*timestamp)
                                        .sensor(sensor_id.as_str(), SensorType::Temperature)
                                        .reading(*last_value, 0.5) // Lower quality
                                        .unwrap()
                                )?;
                            }
                        } else {
                            // Forward the error for logging
                            output.forward(event)?;
                        }
                    }
                    _ => {
                        // Other validation statuses
                        output.forward(event)?;
                    }
                }
            }
            Event::SensorReading { sensor_id, value, timestamp, quality, .. } => {
                let id_hash = Self::sensor_id_hash(sensor_id.as_str());
                
                // Store as last known good if quality is acceptable
                if *quality > 0.7 {
                    let _ = self.last_good_values.insert(id_hash, (*value, *timestamp));
                }
                
                output.forward(event)?;
            }
            _ => output.forward(event)?,
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "ErrorRecoveryStage"
    }
}

/// Fallback stage that provides default values when all else fails
struct FallbackStage {
    default_temp: f32,
    default_humidity: f32,
    sensor_timeouts: heapless::FnvIndexMap<u64, u64, 16>, // Last seen timestamp
    timeout_ms: u64,
}

impl FallbackStage {
    fn new(default_temp: f32, default_humidity: f32, timeout_ms: u64) -> Self {
        Self {
            default_temp,
            default_humidity,
            sensor_timeouts: heapless::FnvIndexMap::new(),
            timeout_ms,
        }
    }
}

impl PipelineStage for FallbackStage {
    fn process(&mut self, event: Event, output: &mut StageOutput) -> PipelineResult<()> {
        if let Event::SensorReading { sensor_id, timestamp, .. } = &event {
            let id_hash = ErrorRecoveryStage::sensor_id_hash(sensor_id.as_str());
            let _ = self.sensor_timeouts.insert(id_hash, *timestamp);
        }
        
        // Check for timed-out sensors
        let current_time = match &event {
            Event::SensorReading { timestamp, .. } |
            Event::ValidationResult { timestamp, .. } |
            Event::CrossValidationResult { timestamp, .. } |
            Event::SystemEvent { timestamp, .. } => *timestamp,
            Event::BatchReading { base_timestamp, .. } => *base_timestamp,
        };
        
        // Generate fallback values for timed-out sensors
        for (id_hash, last_seen) in self.sensor_timeouts.iter() {
            if current_time - last_seen > self.timeout_ms {
                println!("  ⏱ Sensor timeout detected (last seen: {}ms ago)", 
                         current_time - last_seen);
                
                // Generate fallback reading
                let fallback_value = match id_hash % 2 {
                    0 => self.default_temp,
                    _ => self.default_humidity,
                };
                
                output.emit(Event::SystemEvent {
                    event_type: SystemEventType::MemoryWarning,
                    timestamp: current_time,
                    details: (current_time - last_seen) as u32,
                })?;
                
                println!("  → Using fallback value: {:.1}", fallback_value);
            }
        }
        
        output.forward(event)?;
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "FallbackStage"
    }
}

fn main() {
    println!("EdgeGuard Error Handling Example");
    println!("================================\n");
    
    // Build pipeline with error handling stages
    println!("Building pipeline with error recovery:");
    println!("1. Validation - Detect errors");
    println!("2. Error Recovery - Handle failures gracefully");
    println!("3. Fallback - Provide defaults when needed\n");
    
    let mut pipeline = Pipeline::<8>::builder()
        .add_stage(ValidationStage::new(
            TemperatureValidator::new_with_limits(-10.0, 40.0, 3.0),
            SensorType::Temperature
        ))
        .add_stage(ValidationStage::new(
            HumidityValidator::new_with_limits(0.0, 100.0, 20.0),
            SensorType::Humidity
        ))
        .add_stage(ErrorRecoveryStage::new(3))
        .add_stage(FallbackStage::new(20.0, 50.0, 5000))
        .backpressure(BackpressureStrategy::DropOldest)
        .build();
    
    // Generate events with various error conditions
    let events = generate_error_scenarios();
    
    println!("Processing events with error scenarios:\n");
    
    for event in events {
        match &event {
            Event::SensorReading { sensor_id, sensor_type, value, quality, .. } => {
                println!("Reading: {} [{}] = {:.1} (quality: {:.0}%)",
                         sensor_id.as_str(),
                         sensor_type.name(),
                         value,
                         quality * 100.0);
            }
            _ => {}
        }
        
        pipeline.push_event(event);
        
        // Process and check results
        match pipeline.process_batch(10) {
            Ok(_) => {
                while let Some(result) = pipeline.pop_result() {
                    match result {
                        Event::SystemEvent { event_type, details, .. } => {
                            println!("  💡 System: {:?} (details: {})", event_type, details);
                        }
                        Event::ValidationResult { status, .. } if status != ValidationStatus::Valid => {
                            // Already logged by recovery stage
                        }
                        _ => {}
                    }
                }
            }
            Err(e) => {
                println!("  Pipeline error: {:?}", e);
                demonstrate_error_recovery(&e);
            }
        }
        
        println!();
    }
    
    // Demonstrate different error types
    println!("\nError Type Reference:");
    println!("--------------------");
    demonstrate_error_types();
}

fn generate_error_scenarios() -> std::vec::Vec<Event> {
    let mut events = std::vec::Vec::new();
    let base_time = 1000;
    
    // Scenario 1: Good readings
    for i in 0..3 {
        events.push(
            EventBuilder::new(base_time + i * 1000)
                .sensor("temp_sensor", SensorType::Temperature)
                .reading(20.0 + i as f32, 0.95)
                .unwrap()
        );
    }
    
    // Scenario 2: Out of range values
    events.push(
        EventBuilder::new(base_time + 3000)
            .sensor("temp_sensor", SensorType::Temperature)
            .reading(50.0, 0.95) // Too hot!
            .unwrap()
    );
    
    // Scenario 3: Invalid value (NaN)
    events.push(Event::SensorReading {
        sensor_id: InlineString::new("temp_sensor").unwrap(),
        sensor_type: SensorType::Temperature,
        value: f32::NAN,
        timestamp: base_time + 4000,
        quality: 0.95,
    });
    
    // Scenario 4: Rate violation
    events.push(
        EventBuilder::new(base_time + 4100)
            .sensor("temp_sensor", SensorType::Temperature)
            .reading(35.0, 0.95) // Big jump!
            .unwrap()
    );
    
    // Scenario 5: Low quality sensor
    events.push(
        EventBuilder::new(base_time + 5000)
            .sensor("temp_sensor", SensorType::Temperature)
            .reading(25.0, 0.3) // Poor quality
            .unwrap()
    );
    
    // Scenario 6: Multiple consecutive errors
    for i in 0..5 {
        events.push(
            EventBuilder::new(base_time + 6000 + i * 100)
                .sensor("humidity_sensor", SensorType::Humidity)
                .reading(110.0 + i as f32, 0.9) // All invalid
                .unwrap()
        );
    }
    
    // Scenario 7: Recovery after errors
    events.push(
        EventBuilder::new(base_time + 7000)
            .sensor("temp_sensor", SensorType::Temperature)
            .reading(22.0, 0.95)
            .unwrap()
    );
    
    events
}

fn demonstrate_error_recovery(error: &PipelineError) {
    match error {
        PipelineError::QueueFull => {
            println!("  Recovery: Implement backpressure or increase queue size");
        }
        PipelineError::StageError { stage, error } => {
            println!("  Recovery: Stage {} failed with {:?}", stage, error);
            match error {
                ValidationError::OutOfRange { .. } => {
                    println!("    → Check sensor calibration");
                    println!("    → Use last known good value");
                }
                ValidationError::RateExceeded { .. } => {
                    println!("    → Apply low-pass filter");
                    println!("    → Check for electrical interference");
                }
                ValidationError::InvalidValue => {
                    println!("    → Check sensor connection");
                    println!("    → Replace with interpolated value");
                }
                _ => {}
            }
        }
        PipelineError::ResourceExhausted => {
            println!("  Recovery: Free memory, reduce window sizes");
        }
        _ => {}
    }
}

fn demonstrate_error_types() {
    // Show all error types and their typical causes
    println!("1. OutOfRange - Value exceeds physical limits");
    println!("   Causes: Sensor drift, calibration error, hardware fault");
    println!("   Recovery: Recalibrate, use backup sensor, apply bounds");
    
    println!("\n2. RateExceeded - Change too rapid");
    println!("   Causes: Electrical noise, sensor glitch, real spike");
    println!("   Recovery: Filter, increase sampling, check shielding");
    
    println!("\n3. InvalidValue - NaN or infinity");
    println!("   Causes: Division by zero, sensor disconnect, math error");
    println!("   Recovery: Check connections, reset sensor, use default");
    
    println!("\n4. SensorQualityBad - Low confidence");
    println!("   Causes: Aging sensor, environmental interference");
    println!("   Recovery: Schedule maintenance, sensor fusion");
    
    println!("\n5. InsufficientData - Not enough history");
    println!("   Causes: Startup, reset, data loss");
    println!("   Recovery: Wait for more data, use defaults");
}