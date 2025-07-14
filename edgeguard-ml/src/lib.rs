//! Lightweight Machine Learning for Anomaly Detection on Edge Devices
//!
//! ## Overview
//!
//! This module provides machine learning algorithms specifically optimized for
//! resource-constrained edge devices. Unlike traditional ML libraries that assume
//! abundant memory and compute, these implementations use fixed allocations,
//! integer arithmetic, and incremental learning.
//!
//! ## Why Isolation Forest?
//!
//! Isolation Forest was chosen as our primary algorithm because:
//!
//! 1. **Low Memory**: Only stores tree structures, not training data
//! 2. **Fast Inference**: O(log n) per prediction
//! 3. **Unsupervised**: No labeled data required
//! 4. **Interpretable**: Anomaly scores have physical meaning
//! 5. **Online Learning**: Can adapt to changing patterns
//!
//! ## Algorithm Overview
//!
//! ### How Isolation Forest Works
//!
//! The algorithm isolates anomalies by randomly partitioning data:
//! ```text
//! Normal points: Need many partitions to isolate
//! Anomalies: Isolated with few partitions
//! 
//! Anomaly Score = 2^(-path_length / average_path_length)
//! ```
//!
//! ### EdgeGuard Optimizations
//!
//! 1. **Fixed-Point Math**: Replace floating-point with Q16.16 fixed-point
//! 2. **Compact Trees**: Bit-packed node representation
//! 3. **Incremental Updates**: Learn without storing all data
//! 4. **Multi-Sensor**: Correlate anomalies across sensors
//!
//! ## Memory Model
//!
//! For a forest with 100 trees of depth 8:
//! ```text
//! Per tree: 2^8 nodes × 8 bytes = 2KB
//! Forest: 100 trees × 2KB = 200KB
//! Runtime: 10KB for scoring
//! Total: ~210KB RAM
//! ```
//!
//! Compare to sklearn: 5-10MB typical
//!
//! ## Implementation Strategy
//!
//! ### Phase 1: Basic Isolation Forest
//! ```rust
//! struct IsolationTree {
//!     // Compact node representation
//!     nodes: [Node; MAX_NODES],
//!     // Current tree depth
//!     depth: u8,
//! }
//! 
//! struct Node {
//!     // Split feature (sensor type)
//!     feature: u8,
//!     // Split value (quantized)
//!     split_value: i16,
//!     // Child indices (bit-packed)
//!     children: u16,
//! }
//! ```
//!
//! ### Phase 2: Online Learning
//! - Reservoir sampling for training data
//! - Periodic tree rebuilding
//! - Concept drift detection
//!
//! ### Phase 3: Multi-Sensor Correlation
//! - Joint probability estimation
//! - Causal relationship learning
//! - Temporal pattern detection
//!
//! ## Use Cases
//!
//! ### 1. Sensor Fault Detection
//! ```rust
//! // Detect when sensor readings deviate from normal
//! let score = forest.anomaly_score(&reading);
//! if score > 0.7 {
//!     // Likely sensor fault
//! }
//! ```
//!
//! ### 2. Environmental Anomalies
//! ```rust
//! // Detect unusual environmental conditions
//! let features = [temp, humidity, pressure];
//! if forest.is_anomaly(&features) {
//!     // Unusual combination (e.g., high temp + high pressure)
//! }
//! ```
//!
//! ### 3. Predictive Maintenance
//! ```rust
//! // Track anomaly scores over time
//! let trend = forest.anomaly_trend(&history);
//! if trend.increasing() {
//!     // Equipment degradation detected
//! }
//! ```
//!
//! ## Performance Characteristics
//!
//! | Operation       | Time       | Memory |
//! |----------------|------------|--------|
//! | Train tree     | O(n log n) | O(n)   |
//! | Score sample   | O(log n)   | O(1)   |
//! | Update tree    | O(log n)   | O(1)   |
//! | Serialize      | O(n)       | O(n)   |
//!
//! ## Integration with Validators
//!
//! The ML module complements physics-based validation:
//!
//! 1. **Physics First**: Hard constraints catch impossible values
//! 2. **ML Second**: Soft constraints catch unusual patterns
//! 3. **Fusion**: Combine both for robust anomaly detection
//!
//! ```rust
//! // Example integration
//! match validator.validate(reading) {
//!     Ok(_) => {
//!         // Physics OK, check ML
//!         let score = forest.anomaly_score(&reading);
//!         if score > threshold {
//!             log::warn!("Unusual but valid: score={}", score);
//!         }
//!     }
//!     Err(_) => {
//!         // Physics violation - definitely anomalous
//!     }
//! }
//! ```
//!
//! ## Future Enhancements
//!
//! 1. **Tiny Neural Networks**: For specific sensor types
//! 2. **Kalman Filters**: For time-series prediction
//! 3. **Clustering**: For pattern discovery
//! 4. **Federated Learning**: Privacy-preserving updates

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]  
use heapless::Vec;

// Module organization
pub mod forest;
pub mod tree;
pub mod node;
pub mod scoring;
pub mod pipeline;

// Re-export main types
pub use forest::{IsolationForest, ForestConfig};
pub use tree::{IsolationTree, TreeConfig};
pub use node::{Node, NodeType};
pub use scoring::{AnomalyScore, ScoreHistory, calculate_anomaly_score};
pub use pipeline::{MLAnomalyStage, MLConfig, FeatureComplexity};

/// Error types for ML operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MLError {
    /// Not enough samples to train
    InsufficientData,
    /// Tree depth exceeded maximum
    MaxDepthExceeded,
    /// Invalid configuration
    InvalidConfig(&'static str),
    /// Buffer full
    BufferFull,
    /// Invalid feature index
    InvalidFeature,
}

/// Result type for ML operations
pub type MLResult<T> = Result<T, MLError>;

/// Sample data point for training/scoring
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    /// Feature values (sensor readings)
    pub features: [f32; MAX_FEATURES],
    /// Number of valid features
    pub num_features: usize,
}

impl Sample {
    /// Create a new sample
    pub fn new(features: &[f32]) -> Option<Self> {
        if features.len() > MAX_FEATURES {
            return None;
        }
        
        let mut sample = Self {
            features: [0.0; MAX_FEATURES],
            num_features: features.len(),
        };
        
        sample.features[..features.len()].copy_from_slice(features);
        Some(sample)
    }
    
    /// Get feature value
    pub fn get_feature(&self, index: usize) -> Option<f32> {
        if index < self.num_features {
            Some(self.features[index])
        } else {
            None
        }
    }
}

/// Maximum number of features (sensor types)
pub const MAX_FEATURES: usize = 64;

/// Maximum tree depth to prevent stack overflow
pub const MAX_TREE_DEPTH: usize = 16;

/// Maximum nodes per tree (2^depth - 1)
pub const MAX_NODES: usize = (1 << MAX_TREE_DEPTH) - 1;

/// Default number of trees in forest
pub const DEFAULT_NUM_TREES: usize = 100;

/// Default sample size for each tree
pub const DEFAULT_SAMPLE_SIZE: usize = 256;

/// Anomaly threshold (scores above this are anomalies)
pub const DEFAULT_ANOMALY_THRESHOLD: f32 = 0.6;

/// Simple pseudo-random number generator for no_std
/// 
/// Uses a linear congruential generator (LCG)
#[derive(Debug, Clone, Copy)]
pub struct Rng {
    seed: u32,
}

impl Rng {
    /// Create new RNG with seed
    pub fn new(seed: u32) -> Self {
        Self { seed }
    }
    
    /// Get next random u32
    pub fn next_u32(&mut self) -> u32 {
        // LCG parameters from Numerical Recipes
        self.seed = self.seed.wrapping_mul(1664525).wrapping_add(1013904223);
        self.seed
    }
    
    /// Get random f32 in [0, 1)
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }
    
    /// Get random integer in range [0, max)
    pub fn next_range(&mut self, max: usize) -> usize {
        (self.next_f32() * max as f32) as usize
    }
    
    /// Random float in range [min, max)
    pub fn next_f32_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }
}

/// Calculate average path length for BST with n nodes
/// 
/// This is used to normalize anomaly scores
pub fn average_path_length(n: usize) -> f32 {
    if n <= 1 {
        return 0.0;
    }
    
    // Approximation: 2 * H(n-1) - 2*(n-1)/n
    // where H(n) is the harmonic number
    let harmonic = harmonic_number(n - 1);
    2.0 * harmonic - 2.0 * (n as f32 - 1.0) / (n as f32)
}

/// Calculate harmonic number H(n) = 1 + 1/2 + ... + 1/n
fn harmonic_number(n: usize) -> f32 {
    if n == 0 {
        return 0.0;
    }
    
    // Use approximation for large n: ln(n) + gamma + 1/(2n)
    // where gamma is Euler-Mascheroni constant
    if n > 20 {
        const GAMMA: f32 = 0.5772156649;
        ln_approx(n as f32) + GAMMA + 0.5 / (n as f32)
    } else {
        // Direct calculation for small n
        let mut sum = 0.0;
        for i in 1..=n {
            sum += 1.0 / (i as f32);
        }
        sum
    }
}

/// Natural logarithm approximation using Taylor series
pub(crate) fn ln_approx(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }
    
    // Reduce x to range [1, 2) for better convergence
    let mut log_sum = 0.0;
    let mut x = x;
    
    while x >= 2.0 {
        x /= 2.0;
        log_sum += 0.693147; // ln(2)
    }
    
    while x < 1.0 {
        x *= 2.0;
        log_sum -= 0.693147;
    }
    
    // Taylor series around 1: ln(x) = (x-1) - (x-1)^2/2 + (x-1)^3/3 - ...
    let y = x - 1.0;
    let mut term = y;
    let mut sum = term;
    
    for i in 2..8 {
        term *= -y;
        sum += term / (i as f32);
    }
    
    log_sum + sum
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rng() {
        let mut rng = Rng::new(42);
        
        // Test u32 generation
        let val1 = rng.next_u32();
        let val2 = rng.next_u32();
        assert_ne!(val1, val2);
        
        // Test f32 range
        for _ in 0..100 {
            let val = rng.next_f32();
            assert!(val >= 0.0 && val < 1.0);
        }
        
        // Test range generation
        for _ in 0..100 {
            let val = rng.next_range(10);
            assert!(val < 10);
        }
    }
    
    #[test]
    fn test_sample() {
        let features = [1.0, 2.0, 3.0];
        let sample = Sample::new(&features).unwrap();
        
        assert_eq!(sample.num_features, 3);
        assert_eq!(sample.get_feature(0), Some(1.0));
        assert_eq!(sample.get_feature(1), Some(2.0));
        assert_eq!(sample.get_feature(2), Some(3.0));
        assert_eq!(sample.get_feature(3), None);
    }
    
    #[test]
    fn test_average_path_length() {
        assert_eq!(average_path_length(0), 0.0);
        assert_eq!(average_path_length(1), 0.0);
        
        // Test that values are reasonable
        let apl_10 = average_path_length(10);
        assert!(apl_10 > 3.0 && apl_10 < 4.5); // Should be around 3.86
        
        let apl_100 = average_path_length(100);
        assert!(apl_100 > 7.0 && apl_100 < 9.0); // Should be around 8.38
        
        // Test that it increases with n
        assert!(apl_100 > apl_10);
    }
    
    #[test]
    fn test_ln_approx() {
        // Test known values
        assert!((ln_approx(1.0) - 0.0).abs() < 0.01);
        assert!((ln_approx(2.718282) - 1.0).abs() < 0.01);
        assert!((ln_approx(10.0) - 2.302585).abs() < 0.01);
    }
}