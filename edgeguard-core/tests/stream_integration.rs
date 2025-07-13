//! Integration tests for streaming data processing
//!
//! Tests the integration between streams and pipelines, including:
//! - Basic stream-to-pipeline processing
//! - Rate limiting and backpressure
//! - Stream adapters (batching, merging)
//! - Memory efficiency
//! - Error handling

#![cfg(test)]

mod common;

use edgeguard_core::{
    events::{Event, EventBuilder, SensorType},
    stream::{Stream, MemoryStream, RateLimitedStream, BackpressureControl, BackpressureWrapper},
    pipeline::{Pipeline, StreamProcessor, ValidationStage, AggregationStage, WindowSpec, AggregationMethod},
    validators::TemperatureValidator,
    time::{MockTimeSource, TimeSource, MonotonicTime},
};

use common::{
    harness::{TestHarness, TestRng},
    scenarios::Scenarios,
    generators::{PhysicsAwareGenerator, SensorModel},
};

#[test]
fn test_basic_stream_to_pipeline() {
    let mut harness = TestHarness::new(160);
    
    harness.run_test("basic_stream_to_pipeline", || {
        // Generate test events
        let mut generator = PhysicsAwareGenerator::new(1000);
        let events = generator.generate_temperature_with_thermal_mass(
            "temp1",
            25.0,
            &[(0, 25.0), (1, 30.0), (2, 20.0)],
            1.0,
            1,
            60,
            &SensorModel::consumer_grade(),
        );
        
        // Create stream from events
        let stream = MemoryStream::new(&events);
        
        // Create pipeline with validation
        let pipeline = Pipeline::<4>::builder()
            .add_stage(ValidationStage::new(
                TemperatureValidator::default(),
                SensorType::Temperature,
            ))
            .build();
        
        // Create stream processor
        let mut processor = StreamProcessor::new(stream, pipeline);
        
        // Process all events
        let processed = processor.process_batch(100)
            .map_err(|e| format!("Processing failed: {:?}", e))?;
        
        assert_eq!(processed, events.len(), "Should process all events");
        
        // Check for validation results in the output queue
        let mut validation_results = 0;
        let mut sensor_readings = 0;
        
        while let Some(event) = processor.pipeline_mut().pop_result() {
            match event {
                Event::ValidationResult { .. } => validation_results += 1,
                Event::SensorReading { .. } => sensor_readings += 1,
                _ => {}
            }
        }
        
        println!("Processed {} events", processed);
        println!("Output: {} validation results, {} sensor readings", validation_results, sensor_readings);
        
        // ValidationStage should emit both ValidationResult AND forward valid SensorReading
        assert!(validation_results > 0, "Should have validation results");
        assert!(sensor_readings > 0, "Should have forwarded valid sensor readings");
        
        Ok(())
    });
    
    assert!(harness.all_passed());
}

#[test]
fn test_rate_limited_stream() {
    let mut harness = TestHarness::new(160);
    
    harness.run_test("rate_limited_stream", || {
        // Create events at high frequency
        let mut time_source = MockTimeSource::new(0);
        let mut events = Vec::new();
        
        for i in 0..50 {
            events.push(
                EventBuilder::new(time_source.now())
                    .sensor("sensor", SensorType::Temperature)
                    .reading(25.0 + (i as f32 * 0.1), 0.95)
                    .ok_or("Failed to build event")?
            );
            time_source.advance(10); // 100Hz
        }
        
        // Create rate-limited stream (10 events/sec)
        let base_stream = MemoryStream::new(&events);
        let limited = RateLimitedStream::new(base_stream, 10, MonotonicTime::new());
        
        // Simple pipeline
        let pipeline = Pipeline::<4>::builder().build();
        let mut processor = StreamProcessor::new(limited, pipeline);
        
        // Process should be rate-limited
        let processed = processor.process_batch(100)
            .map_err(|e| format!("Processing failed: {:?}", e))?;
        
        // With rate limiting, we shouldn't process all events immediately
        assert!(processed < events.len(), "Rate limiting should limit processing");
        println!("Processed {} of {} events with rate limiting", processed, events.len());
        
        Ok(())
    });
    
    assert!(harness.all_passed());
}

#[test]
fn test_stream_with_aggregation() {
    let mut harness = TestHarness::new(160);
    
    harness.run_test("stream_with_aggregation", || {
        // Generate scenario data
        let scenario = Scenarios::home_environment();
        
        // Debug: check scenario events
        let temp_events = scenario.events.iter()
            .filter(|e| matches!(e, Event::SensorReading { sensor_type, .. } if *sensor_type == SensorType::Temperature))
            .count();
        println!("Scenario has {} temperature events", temp_events);
        
        // Create pipeline with aggregation
        let pipeline = Pipeline::<4>::builder()
            .add_stage(AggregationStage::new(
                WindowSpec::Count { size: 20 },
                AggregationMethod::Mean,
                SensorType::Temperature,
            ))
            .build();
        
        // Process through stream
        let stream = MemoryStream::new(&scenario.events);
        let mut processor = StreamProcessor::new(stream, pipeline);
        
        // Count aggregated results
        let mut batch_count = 0;
        let mut total_processed = 0;
        let total_events = scenario.events.len();
        
        // Process in batches to avoid queue overflow
        while total_processed < total_events {
            let batch_size = 30.min(total_events - total_processed);
            let processed = processor.process_batch(batch_size)
                .map_err(|e| format!("Processing failed: {:?}", e))?;
            total_processed += processed;
            
            // Collect batch results
            while let Some(event) = processor.pipeline_mut().pop_result() {
                if let Event::BatchReading { count, mean_value, min_value, max_value, .. } = event {
                    batch_count += 1;
                    assert_eq!(count, 20, "Each batch should have 20 events");
                    assert!(min_value <= mean_value && mean_value <= max_value, 
                        "Statistics should be consistent");
                }
            }
            
            if processed == 0 {
                break;
            }
        }
        
        println!("Processed {} events", total_processed);
        
        assert!(batch_count > 0, "Should have aggregated batches");
        println!("Created {} aggregated batches", batch_count);
        
        Ok(())
    });
    
    assert!(harness.all_passed());
}

#[test]
fn test_stream_backpressure() {
    let mut harness = TestHarness::new(160);
    
    harness.run_test("stream_backpressure", || {
        // Generate many events
        let mut generator = PhysicsAwareGenerator::new(1000);
        let events = generator.generate_temperature_with_thermal_mass(
            "sensor",
            25.0,
            &[(0, 25.0)],
            1.0,
            1,
            100,
            &SensorModel::consumer_grade(),
        );
        
        // Create stream with backpressure control
        let base_stream = MemoryStream::new(&events);
        let backpressure = BackpressureControl::new(20, 10); // Small buffer
        let stream = BackpressureWrapper::new(base_stream, backpressure);
        
        // Pipeline with slow processing (validation)
        let pipeline = Pipeline::<4>::builder()
            .add_stage(ValidationStage::new(
                TemperatureValidator::default(),
                SensorType::Temperature,
            ))
            .build();
        
        let mut processor = StreamProcessor::new(stream, pipeline);
        
        // Try to process in small batches
        let mut total_processed = 0;
        let mut iterations = 0;
        
        while total_processed < events.len() && iterations < 20 {
            match processor.process_batch(10) {
                Ok(0) => break,
                Ok(n) => {
                    total_processed += n;
                    // Simulate slow consumption
                    for _ in 0..5 {
                        let _ = processor.pipeline_mut().pop_result();
                    }
                }
                Err(e) => return Err(format!("Processing error: {:?}", e)),
            }
            iterations += 1;
        }
        
        println!("Processed {} events in {} iterations", total_processed, iterations);
        assert!(total_processed > 0, "Should process some events");
        assert!(iterations > 1, "Should take multiple iterations due to backpressure");
        
        Ok(())
    });
    
    assert!(harness.all_passed());
}

#[test]
fn test_multi_sensor_streaming() {
    let mut harness = TestHarness::new(160);
    
    harness.run_test("multi_sensor_streaming", || {
        // Generate events from multiple sensors
        let mut generator = PhysicsAwareGenerator::new(1000);
        let mut all_events = Vec::new();
        
        // Generate from 3 temperature sensors
        let sensor_ids = ["temp1", "temp2", "temp3"];
        for (i, sensor_id) in sensor_ids.iter().enumerate() {
            let base_temp = 22.0 + (i as f32 * 2.0);
            let events = generator.generate_temperature_with_thermal_mass(
                sensor_id,
                base_temp,
                &[(0, base_temp), (1, base_temp + 5.0)],
                1.0,
                1,
                10,  // Reduced from 30 to 10 events per sensor
                &SensorModel::consumer_grade(),
            );
            println!("Generated {} events for sensor {}", events.len(), sensor_id);
            all_events.extend(events);
        }
        
        // Sort by timestamp
        all_events.sort_by_key(|e| e.timestamp());
        
        // Debug: print first few events
        println!("Total events generated: {}", all_events.len());
        for (i, event) in all_events.iter().take(5).enumerate() {
            println!("Event {}: sensor_id = {:?}", i, event.sensor_id());
        }
        
        // Create pipeline with validation
        let pipeline = Pipeline::<4>::builder()
            .add_stage(ValidationStage::new(
                TemperatureValidator::default(),
                SensorType::Temperature,
            ))
            .build();
        
        // Process all sensors in batches to avoid queue overflow
        let stream = MemoryStream::new(&all_events);
        let mut processor = StreamProcessor::new(stream, pipeline);
        
        // Count results per sensor
        let mut temp1_results = 0;
        let mut temp2_results = 0;
        let mut temp3_results = 0;
        let mut total_results = 0;
        let mut total_processed = 0;
        
        // Process in batches of 10 to avoid overflowing the 16-element queues
        while total_processed < all_events.len() {
            let batch_size = 10.min(all_events.len() - total_processed);
            let processed = processor.process_batch(batch_size)
                .map_err(|e| format!("Processing failed: {:?}", e))?;
            total_processed += processed;
            
            // Collect results after each batch
            while let Some(event) = processor.pipeline_mut().pop_result() {
                total_results += 1;
                if let Some(sensor_id) = event.sensor_id() {
                    match sensor_id {
                        "temp1" => temp1_results += 1,
                        "temp2" => temp2_results += 1,
                        "temp3" => temp3_results += 1,
                        other => println!("Unknown sensor: '{}'", other),
                    }
                }
            }
            
            if processed == 0 {
                break; // No more events
            }
        }
        
        assert_eq!(total_processed, all_events.len(), "Should process all events");
        
        println!("Total results: {}", total_results);
        println!("Sensor temp1: {} results", temp1_results);
        println!("Sensor temp2: {} results", temp2_results);
        println!("Sensor temp3: {} results", temp3_results);
        
        assert!(total_results > 0, "Should have some results");
        assert!(temp1_results > 0, "temp1 should produce results");
        assert!(temp2_results > 0, "temp2 should produce results");
        assert!(temp3_results > 0, "temp3 should produce results");
        
        Ok(())
    });
    
    assert!(harness.all_passed());
}

#[test]
fn test_stream_memory_efficiency() {
    let mut harness = TestHarness::new(160);
    
    harness.run_test("stream_memory_efficiency", || {
        // Check memory usage of stream types
        let memory_stream_size = core::mem::size_of::<MemoryStream>();
        let rate_limited_size = core::mem::size_of::<RateLimitedStream<MemoryStream, MonotonicTime>>();
        let backpressure_wrapper_size = core::mem::size_of::<BackpressureWrapper<MemoryStream>>();
        
        // Check pipeline sizes
        let small_pipeline = core::mem::size_of::<Pipeline<4>>();
        let large_pipeline = core::mem::size_of::<Pipeline<16>>();
        
        // Check processor size
        let processor_size = core::mem::size_of::<StreamProcessor<MemoryStream, 8>>();
        
        println!("Memory usage:");
        println!("  MemoryStream: {} bytes", memory_stream_size);
        println!("  RateLimitedStream: {} bytes", rate_limited_size);
        println!("  BackpressureWrapper: {} bytes", backpressure_wrapper_size);
        println!("  Pipeline<4>: {} bytes", small_pipeline);
        println!("  Pipeline<16>: {} bytes", large_pipeline);
        println!("  StreamProcessor: {} bytes", processor_size);
        
        // Verify sizes are reasonable for embedded use
        assert!(memory_stream_size < 64, "MemoryStream should be small");
        assert!(processor_size < 8192, "StreamProcessor should fit in embedded RAM");
        
        // Verify linear scaling
        let pipeline_overhead = (large_pipeline - small_pipeline) / 12;
        println!("  Pipeline overhead per stage: ~{} bytes", pipeline_overhead);
        assert!(pipeline_overhead < 512, "Pipeline should scale efficiently");
        
        Ok(())
    });
    
    assert!(harness.all_passed());
}

#[test]
fn test_stream_error_recovery() {
    let mut harness = TestHarness::new(160);
    
    harness.run_test("stream_error_recovery", || {
        // Create events with some invalid values
        let mut time_source = MockTimeSource::new(1000);
        let mut events = Vec::new();
        
        for i in 0..20 {
            let value = if i % 5 == 0 {
                -100.0 // Invalid temperature
            } else {
                25.0 + (i as f32 * 0.5)
            };
            
            events.push(
                EventBuilder::new(time_source.now())
                    .sensor("sensor", SensorType::Temperature)
                    .reading(value, 0.95)
                    .ok_or("Failed to build event")?
            );
            time_source.advance(1000);
        }
        
        // Pipeline with validation (will reject invalid values)
        let pipeline = Pipeline::<4>::builder()
            .add_stage(ValidationStage::new(
                TemperatureValidator::default(),
                SensorType::Temperature,
            ))
            .build();
        
        let stream = MemoryStream::new(&events);
        let mut processor = StreamProcessor::new(stream, pipeline);
        
        // Count valid and invalid results
        let mut valid_count = 0;
        let mut invalid_count = 0;
        let mut total_processed = 0;
        
        // Process in small batches
        while total_processed < events.len() {
            let batch_size = 5.min(events.len() - total_processed);
            let processed = processor.process_batch(batch_size)
                .map_err(|e| format!("Processing failed: {:?}", e))?;
            total_processed += processed;
            
            // Collect results after each batch
            while let Some(event) = processor.pipeline_mut().pop_result() {
                if let Event::ValidationResult { status, .. } = event {
                    match status {
                        edgeguard_core::events::ValidationStatus::Valid => valid_count += 1,
                        _ => invalid_count += 1,
                    }
                }
            }
            
            if processed == 0 {
                break;
            }
        }
        
        assert_eq!(total_processed, events.len(), "Should attempt to process all events");
        
        println!("Valid: {}, Invalid: {}", valid_count, invalid_count);
        assert_eq!(valid_count + invalid_count, events.len(), "Should have result for each event");
        assert!(invalid_count == 4, "Should have 4 invalid temperatures");
        
        Ok(())
    });
    
    assert!(harness.all_passed());
}

#[test]
fn test_stream_to_pipeline_with_routing() {
    let mut harness = TestHarness::new(160);
    
    harness.run_test("stream_routing", || {
        // Generate mixed sensor data
        let mut generator = PhysicsAwareGenerator::new(1000);
        let mut all_events = Vec::new();
        
        // Temperature events
        all_events.extend(generator.generate_temperature_with_thermal_mass(
            "temp1",
            25.0,
            &[(0, 25.0)],
            1.0,
            1,
            20,
            &SensorModel::consumer_grade(),
        ));
        
        // Pressure events
        all_events.extend(generator.generate_pressure_with_weather(
            "press1",
            1013.25,
            &[], // No weather systems
            1,
            20,
            &SensorModel::consumer_grade(),
        ));
        
        // Sort by timestamp
        all_events.sort_by_key(|e| e.timestamp());
        
        // Create pipeline with routing
        use edgeguard_core::pipeline::RouterStage;
        use edgeguard_core::validators::PressureValidator;
        
        let mut router = RouterStage::new();
        router.add_route(
            SensorType::Temperature,
            Box::new(ValidationStage::new(
                TemperatureValidator::default(),
                SensorType::Temperature,
            )),
        ).map_err(|_| "Failed to add temperature route")?;
        
        router.add_route(
            SensorType::Pressure,
            Box::new(ValidationStage::new(
                PressureValidator::default(),
                SensorType::Pressure,
            )),
        ).map_err(|_| "Failed to add pressure route")?;
        
        let pipeline = Pipeline::<4>::builder()
            .add_stage(router)
            .build();
        
        // Process mixed stream in batches
        let stream = MemoryStream::new(&all_events);
        let mut processor = StreamProcessor::new(stream, pipeline);
        
        // Count results by type
        let mut temp_results = 0;
        let mut pressure_results = 0;
        let mut total_processed = 0;
        
        // Process in small batches to avoid queue overflow
        while total_processed < all_events.len() {
            let batch_size = 10.min(all_events.len() - total_processed);
            let processed = processor.process_batch(batch_size)
                .map_err(|e| format!("Processing failed: {:?}", e))?;
            total_processed += processed;
            
            // Collect results after each batch
            while let Some(event) = processor.pipeline_mut().pop_result() {
                if let Event::ValidationResult { sensor_id, .. } = event {
                    if sensor_id.as_str().contains("temp") {
                        temp_results += 1;
                    } else if sensor_id.as_str().contains("press") {
                        pressure_results += 1;
                    }
                }
            }
            
            if processed == 0 {
                break;
            }
        }
        
        assert_eq!(total_processed, all_events.len(), "Should process all events");
        
        println!("Temperature validations: {}, Pressure validations: {}", 
                 temp_results, pressure_results);
        assert!(temp_results > 0 && pressure_results > 0, 
                "Should have results from both sensor types");
        
        Ok(())
    });
    
    assert!(harness.all_passed());
}