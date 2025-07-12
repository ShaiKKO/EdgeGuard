//! Streaming Data Processing Example
//!
//! This example demonstrates EdgeGuard's ability to process continuous
//! streams of sensor data efficiently, combining validation, fusion,
//! and adaptive processing in a streaming context.
//!
//! ## What You'll Learn
//!
//! - Processing continuous sensor data streams
//! - Handling backpressure and flow control
//! - Adaptive sampling based on data characteristics
//! - Real-time fusion with streaming data
//! - Memory-efficient batch processing
//!
//! ## Key Concepts
//!
//! 1. **Streaming Architecture**: Event-driven processing without buffering entire dataset
//! 2. **Backpressure**: Automatically slow down when pipeline can't keep up
//! 3. **Batch Processing**: Process multiple events efficiently
//! 4. **Adaptive Behavior**: Adjust processing based on data patterns
//!
//! ## Running the Example
//!
//! ```bash
//! cargo run --example 10_streaming_data
//! ```

use edgeguard_core::{
    events::{Event, SensorType, ValidationStatus, EventBuilder},
    pipeline::{Pipeline, PipelineStage, StageOutput, PipelineResult, ValidationStage, FilterStage},
    stream::{Stream, MemoryStream, BatchProcessorWithTime},
    validators::{TemperatureValidator, HumidityValidator},
    fusion::pipeline::FusionBuilder,
    time::{TimeSource, MockTimeSource},
};

fn main() {
    println!("EdgeGuard Streaming Data Processing Example");
    println!("==========================================\n");

    // Demonstrate different streaming scenarios
    memory_stream_demo();
    println!("\n{}\n", "=".repeat(60));
    
    batch_processing_demo();
    println!("\n{}\n", "=".repeat(60));
    
    adaptive_sampling_demo();
}

fn memory_stream_demo() {
    println!("Memory Stream Processing:");
    println!("-----------------------");
    println!("Processing sensor data from memory with validation\n");

    // Create a pipeline with validation
    let mut pipeline = Pipeline::<4>::builder()
        .add_stage(ValidationStage::new(
            TemperatureValidator::default(),
            SensorType::Temperature,
        ))
        .add_stage(ValidationStage::new(
            HumidityValidator::default(), 
            SensorType::Humidity,
        ))
        .build();

    // Generate sensor events
    let events = generate_sensor_events();
    println!("Generated {} sensor events", events.len());

    // Create memory stream
    let mut stream = MemoryStream::new(&events);
    
    // Process stream through pipeline using poll_next
    let mut processed = 0;
    let mut validated = 0;
    
    println!("\nProcessing events using stream poll...");
    println!("Time    | Event Type        | Sensor ID | Value  | Status");
    println!("--------|-------------------|-----------|--------|--------");
    
    // Process events one by one
    loop {
        match stream.poll_next() {
            Ok(event) => {
                // Push to pipeline
                pipeline.push_event(event.clone());
                
                // Process
                let _ = pipeline.process_batch(1).unwrap();
                
                // Collect results
                while let Some(out_event) = pipeline.pop_result() {
                    match &out_event {
                        Event::ValidationResult { timestamp, sensor_id, status, .. } => {
                            if processed < 10 { // Only print first few for brevity
                                println!("{:7} | ValidationResult  | {:9} | -      | {:?}",
                                         timestamp % 10000, sensor_id.as_str(), status);
                            }
                            if matches!(status, ValidationStatus::Valid) {
                                validated += 1;
                            }
                        }
                        _ => {}
                    }
                }
                processed += 1;
            }
            Err(nb::Error::Other(_)) => break, // End of stream
            Err(nb::Error::WouldBlock) => continue, // Would block, try again
        }
    }
    
    println!("\nProcessing Summary:");
    println!("- Total events processed: {}", processed);
    println!("- Valid measurements: {}", validated);
}

fn batch_processing_demo() {
    println!("Batch Processing:");
    println!("----------------");
    println!("Demonstrating efficient batch processing\n");

    // Create filter pipeline
    let mut pipeline = Pipeline::<4>::builder()
        .add_stage(FilterStage::new(
            |event| match event {
                Event::SensorReading { quality, .. } => *quality > 0.9,
                _ => true,
            },
            "QualityFilter",
        ))
        .build();

    // Generate a larger dataset
    let mut events = Vec::new();
    for i in 0..1000 {
        events.push(EventBuilder::new((i * 10) as u64)
            .sensor("sensor1", SensorType::Temperature)
            .reading(20.0 + (i as f32 * 0.01), 0.85 + (i as f32 * 0.0001))
            .unwrap());
    }
    
    let mut stream = MemoryStream::new(&events);
    let time_source = MockTimeSource::new(0);
    
    println!("Processing {} events in batches...", events.len());
    
    // Process in batches
    let start = time_source.now();
    let mut total_processed = 0;
    let mut total_passed = 0;
    
    let _ = stream.process_batch_timed::<_, 100>(
        1000,  // process up to 1000 events
        5000,  // 5 second timeout
        &time_source,
        |batch| {
            // Process batch through pipeline
            for event in batch {
                pipeline.push_event(event.clone());
            }
            
            let processed = pipeline.process_batch(batch.len()).unwrap();
            total_processed += processed;
            
            // Count output
            while let Some(_) = pipeline.pop_result() {
                total_passed += 1;
            }
            
            // Show progress
            if total_processed % 200 == 0 {
                println!("Processed {} events, {} passed filter", total_processed, total_passed);
            }
        }
    );
    
    let elapsed = time_source.now() - start;
    println!("\nBatch processing complete:");
    println!("- Total events: {}", total_processed);
    println!("- Events passed filter: {} ({:.1}%)", 
             total_passed, 
             (total_passed as f32 / total_processed as f32) * 100.0);
    println!("- Processing time: {}ms", elapsed);
}

fn adaptive_sampling_demo() {
    println!("Adaptive Sampling:");
    println!("-----------------");
    println!("Demonstrating adaptive sampling based on data characteristics\n");

    // Simple adaptive sampler
    struct AdaptiveSampler {
        last_value: Option<f32>,
        sample_count: u32,
        skip_count: u32,
    }
    
    impl AdaptiveSampler {
        fn new() -> Self {
            Self {
                last_value: None,
                sample_count: 0,
                skip_count: 0,
            }
        }
        
        fn should_sample(&mut self, event: &Event) -> bool {
            if let Event::SensorReading { value, .. } = event {
                if let Some(last) = self.last_value {
                    let change = (value - last).abs();
                    
                    // Sample more frequently when changes are large
                    let should_sample = if change > 1.0 {
                        true // Always sample large changes
                    } else if change > 0.1 {
                        self.skip_count % 2 == 0 // Sample every other for moderate changes
                    } else {
                        self.skip_count % 10 == 0 // Sample 1 in 10 for small changes
                    };
                    
                    if should_sample {
                        self.last_value = Some(*value);
                        self.sample_count += 1;
                    }
                    self.skip_count += 1;
                    
                    should_sample
                } else {
                    // First sample
                    self.last_value = Some(*value);
                    self.sample_count = 1;
                    true
                }
            } else {
                true // Pass through non-sensor events
            }
        }
    }
    
    // Generate events with varying dynamics
    let mut events = Vec::new();
    
    // Period 1: Stable (small changes)
    for i in 0..100 {
        events.push(EventBuilder::new((i * 10) as u64)
            .sensor("temp1", SensorType::Temperature)
            .reading(25.0 + (i as f32 * 0.01), 0.95)
            .unwrap());
    }
    
    // Period 2: Rapid change
    for i in 0..100 {
        events.push(EventBuilder::new((1000 + i * 10) as u64)
            .sensor("temp1", SensorType::Temperature)
            .reading(26.0 + (i as f32 * 0.5), 0.95)
            .unwrap());
    }
    
    // Period 3: Stable again
    for i in 0..100 {
        events.push(EventBuilder::new((2000 + i * 10) as u64)
            .sensor("temp1", SensorType::Temperature)
            .reading(75.0 + (i as f32 * 0.01), 0.95)
            .unwrap());
    }
    
    // Process with adaptive sampling
    let mut sampler = AdaptiveSampler::new();
    let mut stream = MemoryStream::new(&events);
    let mut total = 0;
    
    println!("Period    | Total | Sampled | Reduction");
    println!("----------|-------|---------|----------");
    
    // Process each period
    for period in 0..3 {
        let period_name = match period {
            0 => "Stable 1  ",
            1 => "Rapid     ",
            2 => "Stable 2  ",
            _ => unreachable!(),
        };
        
        let start_samples = sampler.sample_count;
        let mut period_count = 0;
        
        // Process 100 events per period
        for _ in 0..100 {
            match stream.poll_next() {
                Ok(event) => {
                    total += 1;
                    period_count += 1;
                    sampler.should_sample(&event);
                }
                Err(_) => break,
            }
        }
        
        let period_samples = sampler.sample_count - start_samples;
        let reduction = 100.0 - (period_samples as f32 / period_count as f32 * 100.0);
        
        println!("{} | {:5} | {:7} | {:8.1}%",
                 period_name, period_count, period_samples, reduction);
    }
    
    println!("\nAdaptive Sampling Results:");
    println!("- Total events: {}", total);
    println!("- Sampled events: {}", sampler.sample_count);
    println!("- Overall reduction: {:.1}%", 
             100.0 - (sampler.sample_count as f32 / total as f32 * 100.0));
    
    // Summary
    println!("\n\nStreaming Processing Benefits:");
    println!("============================");
    println!("✓ Memory efficient - process unlimited data");
    println!("✓ Low latency - events processed as they arrive");
    println!("✓ Adaptive - adjust to data characteristics");
    println!("✓ Batch processing - efficient for high throughput");
    println!("✓ Composable - combine with validation and fusion");
}

// Helper functions

fn generate_sensor_events() -> Vec<Event> {
    let mut events = Vec::new();
    
    for i in 0..30 {
        let time = (1000 + i * 100) as u64;
        
        // Temperature sensors
        events.push(EventBuilder::new(time)
            .sensor("temp1", SensorType::Temperature)
            .reading(25.0 + (i as f32 * 0.1), 0.95)
            .unwrap());
            
        // Humidity sensor
        events.push(EventBuilder::new(time)
            .sensor("hum1", SensorType::Humidity)
            .reading(50.0 + (i as f32 * 0.5), 0.9)
            .unwrap());
    }
    
    events
}