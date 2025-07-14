//! Event Processing Pipeline with Composable Validation Stages
//!
//! ## Overview
//!
//! This module implements a flexible, composable pipeline for processing sensor events
//! through multiple stages of validation, filtering, transformation, and aggregation.
//! The pipeline design enables complex data processing workflows while maintaining
//! EdgeGuard's strict performance and memory constraints.
//!
//! ## Architecture
//!
//! The pipeline follows a staged architecture where events flow through a series
//! of processing stages:
//!
//! ```text
//! Source → Queue → Stage 1 → Stage 2 → ... → Stage N → Sink
//!          ↓         ↓         ↓               ↓        ↓
//!       Events   Validate   Filter      Aggregate   Output
//! ```
//!
//! ## Module Organization
//!
//! The pipeline module is split into several submodules:
//! - Core traits and types (this file)
//! - `stages` - Built-in pipeline stages (validation, filtering, etc.)
//! - `builder` - Pipeline builder and configuration
//! - `stream` - Stream integration and processing

use crate::{
    events::Event,
    errors::ValidationError,
};
use heapless::Vec;

// Re-export submodules
#[cfg(feature = "pipeline-stages")]
pub mod stages;

#[cfg(feature = "pipeline-core")]
pub mod builder;

#[cfg(all(feature = "pipeline-stream", feature = "stream-memory"))]
pub mod stream;

// Re-export commonly used types
#[cfg(feature = "pipeline-stages")]
pub use stages::{
    ValidationStage, FilterStage, RouterStage, 
    CrossValidationStage, AggregationStage,
    WindowSpec, AggregationMethod,
};

#[cfg(feature = "pipeline-core")]
pub use builder::{Pipeline, PipelineBuilder};

#[cfg(all(feature = "pipeline-stream", feature = "stream-memory"))]
pub use stream::{StreamProcessor, ProcessingStats, SensorStreamAdapter};

/// Maximum number of stages in a pipeline
pub const MAX_PIPELINE_STAGES: usize = 16;

/// Maximum number of routes in router stage
pub const MAX_ROUTES: usize = 8;

/// Maximum number of sensor pairs for cross-validation
pub const MAX_SENSOR_PAIRS: usize = 4;

/// Maximum window size for aggregation
pub const MAX_AGGREGATION_WINDOW: usize = 100;

/// Pipeline processing error
#[derive(Debug)]
pub enum PipelineError {
    /// Stage processing failed
    StageError { stage: usize, error: ValidationError },
    /// Queue overflow
    QueueFull,
    /// Invalid configuration
    InvalidConfig(&'static str),
    /// Resource exhaustion
    ResourceExhausted,
}

/// Result type for pipeline operations
pub type PipelineResult<T> = Result<T, PipelineError>;

/// Backpressure handling strategy
#[derive(Debug, Clone, Copy)]
pub enum BackpressureStrategy {
    /// Drop oldest events when queue full
    DropOldest,
    /// Drop newest events when queue full
    DropNewest,
    /// Return error when queue full
    Error,
}

/// Pipeline metrics for monitoring
pub struct PipelineMetrics {
    /// Events processed per stage
    pub events_processed: [u32; MAX_PIPELINE_STAGES],
    /// Events dropped per stage
    pub events_dropped: [u32; MAX_PIPELINE_STAGES],
    /// Processing time per stage (microseconds)
    pub processing_time_us: [u32; MAX_PIPELINE_STAGES],
    /// Current pipeline depth
    pub current_depth: u16,
}

impl PipelineMetrics {
    pub const fn new() -> Self {
        Self {
            events_processed: [0; MAX_PIPELINE_STAGES],
            events_dropped: [0; MAX_PIPELINE_STAGES],
            processing_time_us: [0; MAX_PIPELINE_STAGES],
            current_depth: 0,
        }
    }
}

// Re-export trait for convenience
pub use crate::traits::PipelineStage;

/// Output buffer for stage processing
/// 
/// ## Capacity Limit Rationale
/// 
/// The buffer is limited to 16 events to ensure bounded memory usage:
/// - 16 events × 128 bytes/event = 2KB maximum
/// - Prevents runaway memory growth from misbehaving stages
/// - Sufficient for most transformations (1:1 or 1:few mappings)
pub struct StageOutput {
    /// Fixed-capacity buffer for emitted events
    events: Vec<Event, 16>,
}

impl StageOutput {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
        }
    }
    
    /// Push an event to the output buffer
    pub fn push(&mut self, event: Event) -> bool {
        self.events.push(event).is_ok()
    }
    
    /// Push multiple events
    pub fn push_all(&mut self, events: &[Event]) -> usize {
        let mut pushed = 0;
        for event in events {
            if self.push(event.clone()) {
                pushed += 1;
            } else {
                break;
            }
        }
        pushed
    }
    
    /// Take all events from the output buffer
    pub fn take(&mut self) -> Vec<Event, 16> {
        let mut output = Vec::new();
        core::mem::swap(&mut self.events, &mut output);
        output
    }
    
    /// Check if output buffer is full
    pub fn is_full(&self) -> bool {
        self.events.is_full()
    }
    
    /// Get current output count
    pub fn len(&self) -> usize {
        self.events.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

impl Default for StageOutput {
    fn default() -> Self {
        Self::new()
    }
}

// Re-export streaming extension trait
#[cfg(all(feature = "pipeline-stream", feature = "stream-memory"))]
pub use crate::traits::StreamingPipelineExt;