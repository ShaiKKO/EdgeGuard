//! Core Traits and Abstractions for EdgeGuard
//!
//! This module provides the fundamental trait definitions that define EdgeGuard's
//! architecture and extension points. All traits are organized by functional area
//! to improve discoverability and maintainability.
//!
//! ## Module Organization
//!
//! - [`core`] - Core validation traits (Validator, CrossValidator, etc.)
//! - [`time`] - Time source abstraction for embedded systems
//! - [`stream`] - Stream processing traits for sensor data
//! - [`fusion`] - Sensor fusion algorithm traits
//! - [`pipeline`] - Event processing pipeline traits
//!
//! ## Design Philosophy
//!
//! EdgeGuard uses a trait-based architecture to provide flexibility while maintaining
//! zero-cost abstractions. The trait system allows:
//!
//! - **Pluggable Components**: Add new sensor types without modifying core code
//! - **Static Dispatch**: No runtime overhead from dynamic dispatch (where possible)
//! - **Compile-Time Optimization**: Monomorphization eliminates unused code
//! - **Type Safety**: Catch configuration errors at compile time
//!
//! ## Usage Example
//!
//! ```rust
//! use edgeguard_core::traits::{Validator, ValidationContext};
//! use edgeguard_core::validators::TemperatureValidator;
//!
//! // Create validator with physics constraints
//! let validator = TemperatureValidator::new()
//!     .with_range(-40.0, 125.0)
//!     .with_rate_limit(5.0);
//!
//! // Build context with history
//! let mut ctx = ValidationContext::default();
//! ctx.add_reading(20.0, 1000);
//! ctx.timestamp = 2000;
//!
//! // Validate new reading
//! match validator.validate(&21.0, &ctx) {
//!     Ok(()) => println!("Valid reading"),
//!     Err(e) => println!("Invalid: {:?}", e),
//! }
//! ```

pub mod core;
pub mod time;
pub mod stream;
pub mod fusion;
pub mod pipeline;

// Re-export commonly used traits at the module level for convenience
pub use core::{
    Validator, CrossValidator, Validatable, EnvironmentalCompensation,
    ValidationContext, ValidatorConstraints, TimestampedReading,
};

pub use time::TimeSource;

// Stream traits are always available when stream modules are used
pub use stream::{Stream, BackpressureStream};

// Fusion traits are always available when fusion modules are used
pub use fusion::{FusionAlgorithm, FusionAlgorithmDyn, SensorModel};

// Pipeline traits are always available when pipeline modules are used
pub use pipeline::PipelineStage;

// StreamingPipelineExt requires both pipeline-stream and stream-memory features
#[cfg(all(feature = "pipeline-stream", feature = "stream-memory"))]
pub use pipeline::StreamingPipelineExt;