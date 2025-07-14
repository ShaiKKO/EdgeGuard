//! Event Processing Pipeline Traits
//!
//! This module defines traits for building event processing pipelines that
//! transform, validate, aggregate, and route sensor events.
//!
//! ## Pipeline Architecture
//!
//! ```text
//! Events → [Stage 1] → [Stage 2] → ... → [Stage N] → Results
//!            ↓           ↓                  ↓
//!         [Output]    [Output]          [Output]
//! ```
//!
//! Each stage can:
//! - Transform events (1:1 mapping)
//! - Filter events (1:0 mapping)
//! - Generate events (1:N mapping)
//! - Aggregate events (N:1 mapping)
//!
//! ## Design Principles
//!
//! - **Composability**: Stages can be combined in any order
//! - **Bounded Resources**: Fixed-size buffers prevent memory issues
//! - **Back-pressure**: Stages can signal when overwhelmed
//! - **Observability**: Metrics for monitoring performance

use crate::events::Event;
use crate::pipeline::{PipelineResult, StageOutput};

/// Trait for pipeline stages
///
/// Each stage in a pipeline implements this trait to define how it
/// processes events. Stages are composable building blocks that can
/// be combined to create complex processing pipelines.
///
/// ## Implementation Guidelines
///
/// 1. **Stateless when possible**: Simplifies testing and reasoning
/// 2. **Bounded processing time**: Avoid blocking operations
/// 3. **Error propagation**: Return errors instead of panicking
/// 4. **Resource limits**: Respect output buffer capacity
///
/// ## Example: Filtering Stage
///
/// ```rust
/// use edgeguard_core::traits::PipelineStage;
/// use edgeguard_core::events::{Event, SensorType};
/// use edgeguard_core::pipeline::{PipelineResult, StageOutput};
///
/// struct TemperatureFilter {
///     min_value: f32,
///     max_value: f32,
/// }
///
/// impl PipelineStage for TemperatureFilter {
///     fn process(&mut self, event: Event, output: &mut StageOutput) -> PipelineResult<()> {
///         match &event {
///             Event::SensorReading { sensor_type, value, .. } => {
///                 if *sensor_type == SensorType::Temperature {
///                     if *value >= self.min_value && *value <= self.max_value {
///                         output.push(event);
///                     }
///                     // Filtered out events are simply not forwarded
///                 } else {
///                     // Pass through non-temperature events
///                     output.push(event);
///                 }
///             }
///             _ => {
///                 // Pass through non-sensor events
///                 output.push(event);
///             }
///         }
///         Ok(())
///     }
///     
///     fn name(&self) -> &'static str {
///         "TemperatureFilter"
///     }
/// }
/// ```
///
/// ## Example: Transformation Stage
///
/// ```rust
/// use edgeguard_core::traits::PipelineStage;
/// use edgeguard_core::events::{Event, SensorType};
/// use edgeguard_core::pipeline::{PipelineResult, StageOutput};
///
/// struct CelsiusToFahrenheit;
///
/// impl PipelineStage for CelsiusToFahrenheit {
///     fn process(&mut self, mut event: Event, output: &mut StageOutput) -> PipelineResult<()> {
///         if let Event::SensorReading { sensor_type, value, .. } = &mut event {
///             if *sensor_type == SensorType::Temperature {
///                 *value = (*value * 9.0 / 5.0) + 32.0;
///             }
///         }
///         output.push(event);
///         Ok(())
///     }
///     
///     fn name(&self) -> &'static str {
///         "CelsiusToFahrenheit"
///     }
/// }
/// ```
pub trait PipelineStage: Send {
    /// Process an event, potentially producing new events
    ///
    /// ## Parameters
    /// - `event`: The input event to process
    /// - `output`: Buffer for emitting output events
    ///
    /// ## Output Patterns
    ///
    /// - **Transform**: Modify and push the input event
    /// - **Filter**: Conditionally push the event
    /// - **Generate**: Push multiple derived events
    /// - **Aggregate**: Store event, push aggregate when ready
    ///
    /// ## Error Handling
    ///
    /// Return an error to signal processing failure. The pipeline
    /// will handle the error according to its configuration (log,
    /// skip, or propagate).
    fn process(&mut self, event: Event, output: &mut StageOutput) -> PipelineResult<()>;
    
    /// Get stage name for debugging
    ///
    /// Used in logs, metrics, and error messages to identify
    /// which stage produced an event or error.
    fn name(&self) -> &'static str;
    
    /// Check if stage can handle this event type
    ///
    /// Allows stages to declare which events they process,
    /// enabling optimizations like routing and short-circuiting.
    ///
    /// Default implementation returns `true` (handle all events).
    fn can_handle(&self, _event: &Event) -> bool {
        true
    }
    
    /// Reset stage state
    ///
    /// Called when the pipeline is reset or when error recovery
    /// requires clearing intermediate state. Stages should:
    /// - Clear any accumulated data
    /// - Reset counters and statistics
    /// - Return to initial configuration
    fn reset(&mut self) {}
}

/// Extension trait for creating streaming pipelines
///
/// This trait extends Pipeline with streaming capabilities, allowing
/// it to process events from a Stream source. Only available when
/// both `pipeline-stream` and `stream-memory` features are enabled.
///
/// ## Example
///
/// ```rust
/// use edgeguard_core::traits::StreamingPipelineExt;
/// use edgeguard_core::pipeline::Pipeline;
/// use edgeguard_core::stream::MemoryStream;
/// 
/// let pipeline = Pipeline::<8>::builder()
///     .add_stage(MyValidationStage::new())
///     .add_stage(MyAggregationStage::new())
///     .build();
///
/// let stream = MemoryStream::new(&events);
/// let mut processor = pipeline.into_stream_processor(stream);
///
/// // Process events from stream through pipeline
/// while let Ok(result) = processor.process_next() {
///     // Handle results
/// }
/// ```
#[cfg(all(feature = "pipeline-stream", feature = "stream-memory"))]
pub trait StreamingPipelineExt<const N: usize> {
    /// Create a stream processor from this pipeline
    ///
    /// Consumes the pipeline and creates a StreamProcessor that
    /// reads events from the provided stream and processes them
    /// through the pipeline stages.
    ///
    /// ## Type Parameters
    /// - `S`: Stream type that produces `Event` items
    /// - `N`: Pipeline buffer size (const generic)
    ///
    /// ## Returns
    ///
    /// A `StreamProcessor` that combines the stream and pipeline,
    /// providing methods to process events and retrieve results.
    fn into_stream_processor<S>(self, stream: S) -> crate::pipeline::stream::StreamProcessor<S, N>
    where
        S: crate::traits::Stream<Item = Event>;
}