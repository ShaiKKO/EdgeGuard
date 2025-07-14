//! Stream integration for pipeline processing
//!
//! This module provides integration between streams and pipelines,
//! enabling continuous processing of sensor data.

use crate::{
    events::Event,
    stream::Stream,
};

use super::Pipeline;

/// Statistics for stream processing
#[derive(Debug, Default, Clone)]
pub struct ProcessingStats {
    /// Total events processed from stream
    pub events_processed: usize,
    /// Events successfully written to pipeline
    pub events_written: usize,
    /// Events dropped due to pipeline backpressure
    pub events_dropped: usize,
    /// Stream errors encountered
    pub stream_errors: usize,
}

/// Stream processor that connects a stream to a pipeline
/// 
/// ## Usage
/// 
/// ```rust,no_run
/// use edgeguard_core::stream::MemoryStream;
/// use edgeguard_core::pipeline::{Pipeline, StreamProcessor};
/// 
/// let stream = MemoryStream::new(&events);
/// let pipeline = Pipeline::<256>::builder()
///     .add_stage(validation_stage)
///     .build();
///     
/// let mut processor = StreamProcessor::new(stream, pipeline);
/// 
/// // Process next event
/// match processor.process_next() {
///     Ok(Some(event)) => { /* Handle output event */ },
///     Ok(None) => { /* No output yet */ },
///     Err(e) => { /* Handle error */ },
/// }
/// ```
pub struct StreamProcessor<S: Stream, const N: usize> {
    stream: S,
    pipeline: Pipeline<N>,
    stats: ProcessingStats,
}

impl<S: Stream<Item = Event>, const N: usize> StreamProcessor<S, N> {
    /// Create a new stream processor
    pub fn new(stream: S, pipeline: Pipeline<N>) -> Self {
        Self {
            stream,
            pipeline,
            stats: ProcessingStats::default(),
        }
    }
    
    /// Process the next event from the stream
    pub fn process_next(&mut self) -> nb::Result<Option<Event>, S::Error> {
        // Try to get next event from stream
        match self.stream.poll_next() {
            Ok(event) => {
                self.stats.events_processed += 1;
                
                // Push to pipeline
                if self.pipeline.push_event(event) {
                    self.stats.events_written += 1;
                } else {
                    self.stats.events_dropped += 1;
                }
                
                // Process pipeline
                if let Err(_e) = self.pipeline.process_batch(1) {
                    // Log error but continue
                }
                
                // Check for output
                Ok(self.pipeline.pop_result())
            }
            Err(nb::Error::WouldBlock) => Err(nb::Error::WouldBlock),
            Err(nb::Error::Other(e)) => {
                self.stats.stream_errors += 1;
                Err(nb::Error::Other(e))
            }
        }
    }
    
    /// Process multiple events in a batch
    pub fn process_batch(&mut self, max_events: usize) -> Result<usize, S::Error> {
        let mut processed = 0;
        
        for _ in 0..max_events {
            match self.process_next() {
                Ok(_) => processed += 1,
                Err(nb::Error::WouldBlock) => break,
                Err(nb::Error::Other(e)) => return Err(e),
            }
        }
        
        Ok(processed)
    }
    
    /// Get processing statistics
    pub fn stats(&self) -> &ProcessingStats {
        &self.stats
    }
    
    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = ProcessingStats::default();
    }
    
    /// Get reference to pipeline
    pub fn pipeline(&self) -> &Pipeline<N> {
        &self.pipeline
    }
    
    /// Get mutable reference to pipeline
    pub fn pipeline_mut(&mut self) -> &mut Pipeline<N> {
        &mut self.pipeline
    }
}

/// Extension implementation for creating stream processors
#[cfg(feature = "pipeline-core")]
impl<const N: usize> crate::traits::StreamingPipelineExt<N> for Pipeline<N> {
    fn into_stream_processor<S>(self, stream: S) -> StreamProcessor<S, N>
    where
        S: Stream<Item = Event>,
    {
        StreamProcessor::new(stream, self)
    }
}

/// Adapter that wraps any sensor stream for pipeline processing
pub struct SensorStreamAdapter<S: Stream> {
    inner: S,
}

impl<S: Stream> SensorStreamAdapter<S> {
    /// Create a new adapter
    pub fn new(stream: S) -> Self {
        Self { inner: stream }
    }
}

impl<S> Stream for SensorStreamAdapter<S>
where
    S: Stream,
    S::Item: Into<Event>,
{
    type Item = Event;
    type Error = S::Error;
    
    fn poll_next(&mut self) -> nb::Result<Self::Item, Self::Error> {
        self.inner.poll_next().map(Into::into)
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::MemoryStream;
    use crate::events::{EventBuilder, SensorType};
    use crate::pipeline::stages::FilterStage;
    
    #[test]
    fn test_stream_processor() {
        // Create test events
        let events = vec![
            EventBuilder::new(1000)
                .sensor("temp1", SensorType::Temperature)
                .reading(25.0, 0.95)
                .unwrap(),
            EventBuilder::new(2000)
                .sensor("temp2", SensorType::Temperature)
                .reading(15.0, 0.95)
                .unwrap(),
        ];
        
        // Create stream and pipeline
        let stream = MemoryStream::new(&events);
        let pipeline = Pipeline::<8>::builder()
            .add_stage(FilterStage::new(|event| {
                if let Event::SensorReading { value, .. } = event {
                    *value > 20.0
                } else {
                    false
                }
            }))
            .build();
        
        // Create processor
        let mut processor = StreamProcessor::new(stream, pipeline);
        
        // Process all events
        let processed = processor.process_batch(10).unwrap();
        assert_eq!(processed, 2);
        
        // Check stats
        assert_eq!(processor.stats().events_processed, 2);
        assert_eq!(processor.stats().events_written, 2);
        assert_eq!(processor.stats().events_dropped, 0);
    }
}