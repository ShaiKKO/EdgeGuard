//! Custom Pipeline Stages Example
//!
//! This example demonstrates how to create custom pipeline stages for
//! specialized processing needs that aren't covered by the built-in stages.
//!
//! ## What You'll Learn
//!
//! - Implementing the PipelineStage trait
//! - Creating domain-specific processing logic
//! - Integrating custom stages with built-in ones
//! - Advanced event transformation and enrichment
//!
//! ## Custom Stages Demonstrated
//!
//! 1. **Anomaly Detection Stage** - Statistical outlier detection
//! 2. **Data Enrichment Stage** - Add computed fields to events
//! 3. **Alert Generation Stage** - Create alerts from patterns
//! 4. **Compression Stage** - Reduce data for transmission
//!
//! ## Running the Example
//!
//! ```bash
//! cargo run --example 05_custom_stages
//! ```

use edgeguard_core::{
    events::{Event, SensorType, ValidationStatus, EventBuilder, SystemEventType},
    pipeline::{Pipeline, PipelineStage, StageOutput, PipelineResult, PipelineError,
               ValidationStage, BackpressureStrategy},
    validators::TemperatureValidator,
};
use heapless::Vec as HeaplessVec;

/// Anomaly detection stage using Z-score method
/// 
/// This stage maintains a rolling window of sensor values and
/// flags readings that deviate significantly from the mean.
struct AnomalyDetectionStage {
    window_size: usize,
    z_threshold: f32,
    sensor_type: SensorType,
    // Store recent values for each sensor
    value_windows: heapless::FnvIndexMap<u64, HeaplessVec<f32, 20>, 8>,
}

impl AnomalyDetectionStage {
    fn new(window_size: usize, z_threshold: f32, sensor_type: SensorType) -> Self {
        Self {
            window_size,
            z_threshold,
            sensor_type,
            value_windows: heapless::FnvIndexMap::new(),
        }
    }
    
    fn calculate_z_score(&self, values: &[f32], current: f32) -> Option<f32> {
        if values.len() < 3 {
            return None; // Need at least 3 values for meaningful stats
        }
        
        // Calculate mean
        let sum: f32 = values.iter().sum();
        let mean = sum / values.len() as f32;
        
        // Calculate standard deviation
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        
        // Simple integer square root approximation
        let std_dev = {
            let mut x = variance;
            let mut x1 = (x + 1.0) / 2.0;
            while (x1 - x).abs() > 0.01 {
                x = x1;
                x1 = (x + variance / x) / 2.0;
            }
            x1
        };
        
        if std_dev < 0.001 {
            return None; // No variation
        }
        
        Some((current - mean).abs() / std_dev)
    }
}

impl PipelineStage for AnomalyDetectionStage {
    fn process(&mut self, event: Event, output: &mut StageOutput) -> PipelineResult<()> {
        // First, forward the original event
        output.forward(event.clone())?;
        
        // Then check for anomalies in sensor readings
        if let Event::SensorReading { sensor_id, sensor_type, value, timestamp, .. } = &event {
            if *sensor_type != self.sensor_type {
                return Ok(()); // Not our sensor type
            }
            
            // Get sensor ID hash for indexing
            let id_hash = {
                let bytes = sensor_id.as_str().as_bytes();
                let mut hash = 0u64;
                for &b in bytes {
                    hash = hash.wrapping_mul(31).wrapping_add(b as u64);
                }
                hash
            };
            
            // Calculate Z-score if we have existing data
            let z_score = if let Some(window) = self.value_windows.get(&id_hash) {
                self.calculate_z_score(window, *value)
            } else {
                None
            };
            
            // Get or create window for this sensor
            let window = match self.value_windows.get_mut(&id_hash) {
                Some(w) => w,
                None => {
                    self.value_windows.insert(id_hash, HeaplessVec::new())
                        .map_err(|_| PipelineError::ResourceExhausted)?;
                    self.value_windows.get_mut(&id_hash).unwrap()
                }
            };
            
            // Add value to window
            if window.len() >= self.window_size {
                window.remove(0); // Remove oldest
            }
            let _ = window.push(*value);
            
            // Check if anomaly detected
            if let Some(z_score) = z_score {
                if z_score > self.z_threshold {
                    // Emit anomaly event
                    output.emit(Event::SystemEvent {
                        event_type: SystemEventType::PerformanceWarning,
                        timestamp: *timestamp,
                        details: (z_score * 100.0) as u32, // Store z-score * 100 in details
                    })?;
                    
                    println!("  🚨 Anomaly detected: Z-score = {:.2}", z_score);
                }
            }
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "AnomalyDetectionStage"
    }
}

/// Data enrichment stage that adds computed fields
/// 
/// This stage calculates additional metrics like:
/// - Rate of change
/// - Moving average
/// - Trend direction
struct DataEnrichmentStage {
    sensor_type: SensorType,
    ma_window: usize,
    // Track last values for rate calculation
    last_values: heapless::FnvIndexMap<u64, (f32, u64), 8>, // (value, timestamp)
    // Moving average windows
    ma_windows: heapless::FnvIndexMap<u64, HeaplessVec<f32, 10>, 8>,
}

impl DataEnrichmentStage {
    fn new(sensor_type: SensorType, ma_window: usize) -> Self {
        Self {
            sensor_type,
            ma_window,
            last_values: heapless::FnvIndexMap::new(),
            ma_windows: heapless::FnvIndexMap::new(),
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

impl PipelineStage for DataEnrichmentStage {
    fn process(&mut self, event: Event, output: &mut StageOutput) -> PipelineResult<()> {
        if let Event::SensorReading { sensor_id, sensor_type, value, timestamp, .. } = &event {
            if *sensor_type != self.sensor_type {
                output.forward(event)?;
                return Ok(());
            }
            
            let id_hash = Self::sensor_id_hash(sensor_id.as_str());
            
            // Calculate rate of change
            let rate_of_change = if let Some((last_value, last_ts)) = self.last_values.get(&id_hash) {
                let time_diff = (*timestamp - last_ts) as f32 / 1000.0; // Convert to seconds
                if time_diff > 0.0 {
                    Some((value - last_value) / time_diff)
                } else {
                    None
                }
            } else {
                None
            };
            
            // Update last value
            let _ = self.last_values.insert(id_hash, (*value, *timestamp));
            
            // Calculate moving average
            let ma_window = match self.ma_windows.get_mut(&id_hash) {
                Some(w) => w,
                None => {
                    let _ = self.ma_windows.insert(id_hash, HeaplessVec::new());
                    self.ma_windows.get_mut(&id_hash).unwrap()
                }
            };
            
            if ma_window.len() >= self.ma_window {
                ma_window.remove(0);
            }
            let _ = ma_window.push(*value);
            
            let moving_avg = ma_window.iter().sum::<f32>() / ma_window.len() as f32;
            
            // Determine trend
            let trend = if let Some(rate) = rate_of_change {
                if rate > 0.5 {
                    "rising"
                } else if rate < -0.5 {
                    "falling"
                } else {
                    "stable"
                }
            } else {
                "unknown"
            };
            
            // Log enriched data
            println!("  📊 Enriched: MA={:.1}, Rate={:.2}/s, Trend={}", 
                     moving_avg,
                     rate_of_change.unwrap_or(0.0),
                     trend);
            
            // Forward original event (in real use, might create enriched event type)
            output.forward(event)?;
        } else {
            output.forward(event)?;
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "DataEnrichmentStage"
    }
}

/// Alert generation stage that watches for specific patterns
struct AlertGenerationStage {
    high_temp_threshold: f32,
    low_temp_threshold: f32,
    rapid_change_threshold: f32,
    // Track consecutive violations
    violation_counts: heapless::FnvIndexMap<u64, u8, 8>,
}

impl AlertGenerationStage {
    fn new(high: f32, low: f32, rapid: f32) -> Self {
        Self {
            high_temp_threshold: high,
            low_temp_threshold: low,
            rapid_change_threshold: rapid,
            violation_counts: heapless::FnvIndexMap::new(),
        }
    }
}

impl PipelineStage for AlertGenerationStage {
    fn process(&mut self, event: Event, output: &mut StageOutput) -> PipelineResult<()> {
        // Forward original event
        output.forward(event.clone())?;
        
        match &event {
            Event::SensorReading { sensor_id, sensor_type, value, timestamp, .. } => {
                if *sensor_type != SensorType::Temperature {
                    return Ok(());
                }
                
                let id_hash = DataEnrichmentStage::sensor_id_hash(sensor_id.as_str());
                let mut alert_triggered = false;
                let mut alert_message = "";
                
                // Check temperature thresholds
                if *value > self.high_temp_threshold {
                    alert_triggered = true;
                    alert_message = "High temperature alert";
                } else if *value < self.low_temp_threshold {
                    alert_triggered = true;
                    alert_message = "Low temperature alert";
                }
                
                if alert_triggered {
                    // Track consecutive violations
                    let count = self.violation_counts.get(&id_hash).copied().unwrap_or(0);
                    let _ = self.violation_counts.insert(id_hash, count + 1);
                    
                    if count >= 2 {
                        // Generate alert after 3 consecutive violations
                        output.emit(Event::SystemEvent {
                            event_type: SystemEventType::ValidatorError,
                            timestamp: *timestamp,
                            details: (*value as u32), // Store temperature in details
                        })?;
                        
                        println!("  🚨 ALERT: {} ({}°C)", alert_message, value);
                    }
                } else {
                    // Reset violation count
                    let _ = self.violation_counts.remove(&id_hash);
                }
            }
            Event::ValidationResult { sensor_id, status, .. } => {
                if *status == ValidationStatus::RateExceeded {
                    let id_hash = DataEnrichmentStage::sensor_id_hash(sensor_id.as_str());
                    let count = self.violation_counts.get(&id_hash).copied().unwrap_or(0);
                    
                    if count >= 1 {
                        println!("  ⚡ Rapid change alert for {}", sensor_id.as_str());
                    }
                }
            }
            _ => {}
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "AlertGenerationStage"
    }
}

fn main() {
    println!("EdgeGuard Custom Pipeline Stages Example");
    println!("=======================================\n");
    
    // Build pipeline with custom stages
    println!("Building pipeline with custom stages:");
    println!("1. Validation - Standard physics checks");
    println!("2. Anomaly Detection - Statistical outlier detection");
    println!("3. Data Enrichment - Add computed metrics");
    println!("4. Alert Generation - Pattern-based alerts\n");
    
    let mut pipeline = Pipeline::<8>::builder()
        .add_stage(ValidationStage::new(
            TemperatureValidator::new_with_limits(-10.0, 40.0, 5.0),
            SensorType::Temperature
        ))
        .add_stage(AnomalyDetectionStage::new(10, 2.5, SensorType::Temperature))
        .add_stage(DataEnrichmentStage::new(SensorType::Temperature, 5))
        .add_stage(AlertGenerationStage::new(35.0, 5.0, 5.0))
        .backpressure(BackpressureStrategy::Error)
        .build();
    
    // Generate test events including anomalies
    let events = generate_test_events_with_anomalies();
    
    println!("Processing events with anomalies:\n");
    
    // Process events
    for event in events {
        match &event {
            Event::SensorReading { sensor_id, value, .. } => {
                println!("Processing: {} = {:.1}°C", sensor_id.as_str(), value);
            }
            _ => {}
        }
        
        pipeline.push_event(event);
        
        // Process batch
        match pipeline.process_batch(10) {
            Ok(_) => {
                // Collect any system events generated
                while let Some(result) = pipeline.pop_result() {
                    if let Event::SystemEvent { event_type, details, .. } = result {
                        println!("  → System: {:?} (details: {})", event_type, details);
                    }
                }
            }
            Err(e) => println!("  Error: {:?}", e),
        }
        
        println!();
    }
    
    // Show final metrics
    let metrics = pipeline.metrics();
    println!("\nPipeline Metrics:");
    println!("----------------");
    for (i, &count) in metrics.events_processed.iter().enumerate() {
        if count > 0 {
            let stage_name = match i {
                0 => "Validation",
                1 => "Anomaly Detection",
                2 => "Data Enrichment",
                3 => "Alert Generation",
                _ => "Unknown",
            };
            println!("{}: {} events", stage_name, count);
        }
    }
}

fn generate_test_events_with_anomalies() -> std::vec::Vec<Event> {
    let mut events = std::vec::Vec::new();
    let base_time = 1000;
    
    // Normal temperature progression
    for i in 0..10 {
        events.push(
            EventBuilder::new(base_time + i * 1000)
                .sensor("outdoor_temp", SensorType::Temperature)
                .reading(20.0 + (i as f32 * 0.3).sin() * 2.0, 0.95)
                .unwrap()
        );
    }
    
    // Sudden spike (anomaly)
    events.push(
        EventBuilder::new(base_time + 10000)
            .sensor("outdoor_temp", SensorType::Temperature)
            .reading(28.0, 0.95) // Big jump
            .unwrap()
    );
    
    // Continue with slightly elevated readings
    for i in 11..15 {
        events.push(
            EventBuilder::new(base_time + i * 1000)
                .sensor("outdoor_temp", SensorType::Temperature)
                .reading(26.0 + (i as f32 * 0.2).cos(), 0.95)
                .unwrap()
        );
    }
    
    // Gradual rise to high temperature (should trigger alert)
    for i in 15..20 {
        events.push(
            EventBuilder::new(base_time + i * 1000)
                .sensor("outdoor_temp", SensorType::Temperature)
                .reading(30.0 + i as f32 * 0.5, 0.95)
                .unwrap()
        );
    }
    
    // Rapid drop (rate violation)
    events.push(
        EventBuilder::new(base_time + 20000)
            .sensor("outdoor_temp", SensorType::Temperature)
            .reading(20.0, 0.95) // Sudden drop
            .unwrap()
    );
    
    events
}