//! Stream processing for sensor data
//!
//! This module provides the core streaming abstractions for EdgeGuard,
//! enabling efficient processing of sensor data with minimal memory overhead.
//!
//! ## Module Organization
//!
//! The stream module is split into several submodules for better modularity:
//! - Core traits and errors (this file)
//! - `memory` - In-memory streams for testing
//! - `file` - File-based streams (requires `std`)
//! - `adapters` - Stream transformers (rate limiting, batching, etc.)

use core::fmt;

// Re-export submodules based on features
#[cfg(feature = "stream-memory")]
pub mod memory;

#[cfg(feature = "std")]
pub mod file;

#[cfg(feature = "stream-adapters")]
pub mod adapters;

// Re-export commonly used types
#[cfg(feature = "stream-memory")]
pub use memory::MemoryStream;

#[cfg(feature = "std")]
pub use file::{FileStream, FileFormat, FileStreamStats};

#[cfg(feature = "stream-rate-limit")]
pub use adapters::RateLimitedStream;

#[cfg(feature = "stream-batch")]
pub use adapters::BatchingStream;

#[cfg(feature = "stream-backpressure")]
pub use adapters::{BackpressureWrapper, BackpressureControl};

#[cfg(feature = "stream-combined")]
pub use adapters::CombinedStream;

/// Errors that can occur during stream processing
#[derive(Debug, Clone, PartialEq)]
pub enum StreamError<E> {
    /// Transport-level error (e.g., I/O error)
    Transport(E),
    /// Data format error
    Format(&'static str),
    /// Schema validation failed
    SchemaViolation,
    /// End of stream reached
    EndOfStream,
    /// Buffer overflow
    Overflow,
    /// Backpressure - consumer can't keep up
    Backpressure,
}

impl<E: fmt::Display> fmt::Display for StreamError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(e) => write!(f, "Transport error: {}", e),
            Self::Format(msg) => write!(f, "Format error: {}", msg),
            Self::SchemaViolation => write!(f, "Schema violation"),
            Self::EndOfStream => write!(f, "End of stream"),
            Self::Overflow => write!(f, "Buffer overflow"),
            Self::Backpressure => write!(f, "Backpressure"),
        }
    }
}

// Re-export traits for convenience
pub use crate::traits::{Stream, BackpressureStream};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn stream_error_display() {
        let err: StreamError<&str> = StreamError::Transport("connection lost");
        assert_eq!(format!("{}", err), "Transport error: connection lost");
        
        let err: StreamError<&str> = StreamError::EndOfStream;
        assert_eq!(format!("{}", err), "End of stream");
    }
}