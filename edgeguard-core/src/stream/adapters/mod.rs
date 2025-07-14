//! Stream adapters for transformation and control
//!
//! This module provides various stream adapters that wrap other streams
//! to add functionality like rate limiting, batching, and backpressure.

// Feature-gated submodules
#[cfg(feature = "stream-rate-limit")]
pub mod rate_limited;

#[cfg(feature = "stream-batch")]
pub mod batching;

#[cfg(feature = "stream-backpressure")]
pub mod backpressure;

#[cfg(feature = "stream-combined")]
pub mod combined;

// Re-exports
#[cfg(feature = "stream-rate-limit")]
pub use rate_limited::RateLimitedStream;

#[cfg(feature = "stream-batch")]
pub use batching::BatchingStream;

#[cfg(feature = "stream-backpressure")]
pub use backpressure::{BackpressureWrapper, BackpressureControl};

#[cfg(feature = "stream-combined")]
pub use combined::CombinedStream;