//! Pipeline builder and core implementation
//!
//! This module provides the main Pipeline struct and its builder
//! for constructing event processing pipelines.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

#[cfg(feature = "std")]
use std::boxed::Box;

use heapless::Vec;

use crate::{
    events::Event,
    queue::EventQueue,
};

use super::{
    PipelineStage, PipelineResult, StageOutput,
    BackpressureStrategy, PipelineMetrics, MAX_PIPELINE_STAGES,
};

/// Event processing pipeline
/// 
/// ## Design Goals
/// 
/// 1. **Fixed Memory**: All allocations happen at construction time
/// 2. **Composable**: Stages can be mixed and matched
/// 3. **Efficient**: Minimal overhead between stages
/// 4. **Observable**: Built-in metrics for monitoring
/// 
/// ## Memory Layout
/// 
/// The pipeline uses a fixed-size event queue (N events) which determines
/// the maximum number of events that can be buffered between processing cycles.
/// Common sizes:
/// - N=256: Low-memory devices (32KB systems)
/// - N=1024: Standard IoT devices  
/// - N=4096: High-throughput gateways
pub struct Pipeline<const N: usize> {
    /// Processing stages
    stages: Vec<Box<dyn PipelineStage>, MAX_PIPELINE_STAGES>,
    /// Input event queue
    input_queue: EventQueue<N>,
    /// Output event queue
    output_queue: EventQueue<N>,
    /// Backpressure strategy
    backpressure: BackpressureStrategy,
    /// Pipeline metrics
    metrics: PipelineMetrics,
}

impl<const N: usize> Pipeline<N> {
    /// Create a new pipeline builder
    pub fn builder() -> PipelineBuilder<N> {
        PipelineBuilder::new()
    }
    
    /// Push an event into the pipeline
    pub fn push_event(&mut self, event: Event) -> bool {
        if self.input_queue.push(event.clone()) {
            true
        } else {
            match self.backpressure {
                BackpressureStrategy::DropOldest => {
                    // Pop oldest and retry
                    let _ = self.input_queue.pop();
                    self.input_queue.push(event)
                }
                BackpressureStrategy::DropNewest => {
                    // Just drop the new event
                    false
                }
                BackpressureStrategy::Error => {
                    // Return error (already false)
                    false
                }
            }
        }
    }
    
    /// Process a batch of events through the pipeline
    pub fn process_batch(&mut self, max_events: usize) -> PipelineResult<usize> {
        let mut processed = 0;
        let mut stage_output = StageOutput::new();
        
        for _ in 0..max_events {
            // Get next event from input queue
            let event = match self.input_queue.pop() {
                Some(e) => e,
                None => break,
            };
            
            // Pass through all stages
            let mut current_events = Vec::<Event, 16>::new();
            current_events.push(event).ok();
            
            for (stage_idx, stage) in self.stages.iter_mut().enumerate() {
                let mut next_events = Vec::<Event, 16>::new();
                
                // Process each event through this stage
                for event in &current_events {
                    stage_output.take(); // Clear output buffer
                    
                    // Process event
                    match stage.process(event.clone(), &mut stage_output) {
                        Ok(()) => {
                            // Collect output events
                            for output_event in stage_output.take() {
                                next_events.push(output_event).ok();
                            }
                            self.metrics.events_processed[stage_idx] += 1;
                        }
                        Err(_e) => {
                            self.metrics.events_dropped[stage_idx] += 1;
                        }
                    }
                }
                
                current_events = next_events;
                
                // If no events left, stop processing
                if current_events.is_empty() {
                    break;
                }
            }
            
            // Push final events to output queue
            for event in current_events {
                if !self.output_queue.push(event) {
                    self.metrics.events_dropped[self.stages.len()] += 1;
                }
            }
            
            processed += 1;
        }
        
        self.metrics.current_depth = self.input_queue.len() as u16;
        
        Ok(processed)
    }
    
    /// Get the next result event
    pub fn pop_result(&mut self) -> Option<Event> {
        self.output_queue.pop()
    }
    
    /// Get pipeline metrics
    pub fn metrics(&self) -> &PipelineMetrics {
        &self.metrics
    }
    
    /// Reset all pipeline stages
    pub fn reset(&mut self) {
        for stage in &mut self.stages {
            stage.reset();
        }
        self.metrics = PipelineMetrics::new();
    }
    
    /// Get current input queue depth
    pub fn input_depth(&self) -> usize {
        self.input_queue.len()
    }
    
    /// Get current output queue depth  
    pub fn output_depth(&self) -> usize {
        self.output_queue.len()
    }
}

/// Pipeline builder for constructing pipelines
pub struct PipelineBuilder<const N: usize> {
    stages: Vec<Box<dyn PipelineStage>, MAX_PIPELINE_STAGES>,
    backpressure: BackpressureStrategy,
}

impl<const N: usize> PipelineBuilder<N> {
    /// Create a new pipeline builder
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            backpressure: BackpressureStrategy::DropOldest,
        }
    }
    
    /// Add a processing stage
    pub fn add_stage<S: PipelineStage + 'static>(mut self, stage: S) -> Self {
        self.stages.push(Box::new(stage)).ok();
        self
    }
    
    /// Set backpressure strategy
    pub fn backpressure(mut self, strategy: BackpressureStrategy) -> Self {
        self.backpressure = strategy;
        self
    }
    
    /// Build the pipeline
    pub fn build(self) -> Pipeline<N> {
        Pipeline {
            stages: self.stages,
            input_queue: EventQueue::new(),
            output_queue: EventQueue::new(),
            backpressure: self.backpressure,
            metrics: PipelineMetrics::new(),
        }
    }
}

impl<const N: usize> Default for PipelineBuilder<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{EventBuilder, SensorType};
    use crate::pipeline::stages::FilterStage;
    
    #[test]
    fn test_pipeline_builder() {
        let pipeline: Pipeline<8> = Pipeline::builder()
            .add_stage(FilterStage::new(|event| {
                matches!(event, Event::SensorReading { .. })
            }))
            .backpressure(BackpressureStrategy::DropNewest)
            .build();
        
        assert_eq!(pipeline.stages.len(), 1);
        assert_eq!(pipeline.input_depth(), 0);
        assert_eq!(pipeline.output_depth(), 0);
    }
    
    #[test]
    fn test_pipeline_processing() {
        let mut pipeline: Pipeline<8> = Pipeline::builder()
            .add_stage(FilterStage::new(|event| {
                if let Event::SensorReading { value, .. } = event {
                    *value > 20.0
                } else {
                    false
                }
            }))
            .build();
        
        // Add some events
        let event1 = EventBuilder::new(1000)
            .sensor("temp1", SensorType::Temperature)
            .reading(25.0, 0.95)
            .unwrap();
            
        let event2 = EventBuilder::new(2000)
            .sensor("temp2", SensorType::Temperature)
            .reading(15.0, 0.95)
            .unwrap();
        
        assert!(pipeline.push_event(event1));
        assert!(pipeline.push_event(event2));
        
        // Process events
        let processed = pipeline.process_batch(10).unwrap();
        assert_eq!(processed, 2);
        
        // Check results - only event1 should pass filter
        assert!(pipeline.pop_result().is_some());
        assert!(pipeline.pop_result().is_none());
    }
}