//! Isolation Forest implementation
//!
//! This module provides the main Isolation Forest algorithm that combines
//! multiple isolation trees for robust anomaly detection.

use crate::{
    IsolationTree, TreeConfig, Sample, AnomalyScore, ScoreHistory,
    calculate_anomaly_score, Rng, MLError, MLResult,
    DEFAULT_SAMPLE_SIZE, DEFAULT_ANOMALY_THRESHOLD,
};

#[cfg(not(feature = "std"))]
use heapless::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Configuration for Isolation Forest
#[derive(Debug, Clone)]
pub struct ForestConfig {
    /// Number of trees in the forest
    pub num_trees: usize,
    /// Sample size for each tree
    pub sample_size: usize,
    /// Maximum tree depth
    pub max_depth: usize,
    /// Random seed
    pub seed: u32,
    /// Anomaly threshold
    pub anomaly_threshold: f32,
}

impl Default for ForestConfig {
    fn default() -> Self {
        Self {
            num_trees: 10, // Default that works for tests
            sample_size: DEFAULT_SAMPLE_SIZE,
            max_depth: 8,
            seed: 42,
            anomaly_threshold: DEFAULT_ANOMALY_THRESHOLD,
        }
    }
}

/// Isolation Forest for anomaly detection
#[cfg(not(feature = "std"))]
pub struct IsolationForest<const T: usize = 10> {
    /// Individual trees
    trees: Vec<IsolationTree, T>,
    /// Configuration
    config: ForestConfig,
    /// Random number generator
    rng: Rng,
    /// Number of samples used for training
    num_samples: usize,
    /// Score history for trend analysis
    history: ScoreHistory<100>,
}

#[cfg(feature = "std")]
pub struct IsolationForest<const T: usize = 10> {
    /// Individual trees
    trees: Vec<IsolationTree>,
    /// Configuration
    config: ForestConfig,
    /// Random number generator
    rng: Rng,
    /// Number of samples used for training
    num_samples: usize,
    /// Score history for trend analysis
    history: ScoreHistory<100>,
}

impl<const T: usize> IsolationForest<T> {
    /// Create a new Isolation Forest
    pub fn new(config: ForestConfig) -> Self {
        if config.num_trees > T {
            panic!("num_trees exceeds maximum capacity");
        }
        
        let seed = config.seed;
        Self {
            trees: Vec::new(),
            config,
            rng: Rng::new(seed),
            num_samples: 0,
            history: ScoreHistory::new(0.1), // Default smoothing
        }
    }
    
    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(ForestConfig::default())
    }
    
    /// Train the forest on samples
    pub fn fit(&mut self, samples: &[Sample]) -> MLResult<()> {
        if samples.is_empty() {
            return Err(MLError::InsufficientData);
        }
        
        self.num_samples = samples.len();
        self.trees.clear();
        
        // Build each tree
        for i in 0..self.config.num_trees {
            // Create tree with unique seed
            let tree_config = TreeConfig {
                max_depth: self.config.max_depth,
                seed: self.config.seed.wrapping_add(i as u32),
            };
            
            let mut tree = IsolationTree::new(tree_config);
            
            // Sample subset for this tree
            let tree_samples = self.sample_subset(samples)?;
            
            // Train tree
            tree.fit(&tree_samples)?;
            
            // Add to forest
            #[cfg(not(feature = "std"))]
            self.trees.push(tree).map_err(|_| MLError::BufferFull)?;
            #[cfg(feature = "std")]
            self.trees.push(tree);
        }
        
        Ok(())
    }
    
    /// Sample a subset of data for tree training
    #[cfg(not(feature = "std"))]
    fn sample_subset(&mut self, samples: &[Sample]) -> MLResult<Vec<Sample, 256>> {
        let sample_size = self.config.sample_size.min(samples.len());
        let mut subset = Vec::new();
        
        // If sample size >= data size, use all data
        if sample_size >= samples.len() {
            for &sample in samples {
                subset.push(sample).map_err(|_| MLError::BufferFull)?;
            }
            return Ok(subset);
        }
        
        // Random sampling without replacement
        let mut indices: Vec<usize, 256> = Vec::new();
        for i in 0..samples.len() {
            indices.push(i).map_err(|_| MLError::BufferFull)?;
        }
        
        // Fisher-Yates shuffle
        for i in 0..sample_size {
            let j = i + self.rng.next_range(samples.len() - i);
            indices.swap(i, j);
        }
        
        // Take first sample_size elements
        for i in 0..sample_size {
            subset.push(samples[indices[i]]).map_err(|_| MLError::BufferFull)?;
        }
        
        Ok(subset)
    }
    
    #[cfg(feature = "std")]
    fn sample_subset(&mut self, samples: &[Sample]) -> MLResult<Vec<Sample>> {
        let sample_size = self.config.sample_size.min(samples.len());
        let mut subset = Vec::new();
        
        // If sample size >= data size, use all data
        if sample_size >= samples.len() {
            for &sample in samples {
                subset.push(sample);
            }
            return Ok(subset);
        }
        
        // Random sampling without replacement
        let mut indices: Vec<usize> = Vec::new();
        for i in 0..samples.len() {
            indices.push(i);
        }
        
        // Fisher-Yates shuffle
        for i in 0..sample_size {
            let j = i + self.rng.next_range(samples.len() - i);
            indices.swap(i, j);
        }
        
        // Take first sample_size elements
        for i in 0..sample_size {
            subset.push(samples[indices[i]]);
        }
        
        Ok(subset)
    }
    
    /// Calculate anomaly score for a sample
    pub fn anomaly_score(&self, sample: &Sample) -> AnomalyScore {
        if self.trees.is_empty() {
            return AnomalyScore::new(0.5, 0.0, 0);
        }
        
        // Calculate average path length across all trees
        let total_path_length: f32 = self.trees.iter()
            .map(|tree| tree.path_length(sample))
            .sum();
        
        let avg_path_length = total_path_length / self.trees.len() as f32;
        
        // Calculate anomaly score
        let score = calculate_anomaly_score(avg_path_length, self.num_samples);
        
        AnomalyScore::new(score, avg_path_length, self.trees.len())
    }
    
    /// Check if a sample is an anomaly
    pub fn is_anomaly(&self, sample: &Sample) -> bool {
        let score = self.anomaly_score(sample);
        score.is_anomaly(self.config.anomaly_threshold)
    }
    
    /// Score a sample and update history
    pub fn score_and_track(&mut self, sample: &Sample) -> MLResult<AnomalyScore> {
        let score = self.anomaly_score(sample);
        self.history.add(score)?;
        Ok(score)
    }
    
    /// Get anomaly trend from history
    pub fn anomaly_trend(&self) -> &ScoreHistory<100> {
        &self.history
    }
    
    /// Predict anomaly scores for multiple samples
    #[cfg(not(feature = "std"))]
    pub fn predict(&self, samples: &[Sample]) -> Vec<AnomalyScore, 256> {
        let mut scores = Vec::new();
        
        for sample in samples {
            if scores.push(self.anomaly_score(sample)).is_err() {
                break;
            }
        }
        
        scores
    }
    
    #[cfg(feature = "std")]
    pub fn predict(&self, samples: &[Sample]) -> Vec<AnomalyScore> {
        samples.iter()
            .map(|sample| self.anomaly_score(sample))
            .collect()
    }
    
    /// Update forest with new samples (incremental learning)
    /// 
    /// This rebuilds a subset of trees with new data
    pub fn partial_fit(&mut self, samples: &[Sample], fraction: f32) -> MLResult<()> {
        if samples.is_empty() || self.trees.is_empty() {
            return Err(MLError::InsufficientData);
        }
        
        let fraction = fraction.clamp(0.0, 1.0);
        let trees_to_update = ((self.trees.len() as f32 * fraction) as usize).max(1);
        
        // Update random subset of trees
        for _ in 0..trees_to_update {
            let tree_idx = self.rng.next_range(self.trees.len());
            let tree_samples = self.sample_subset(samples)?;
            self.trees[tree_idx].fit(&tree_samples)?;
        }
        
        Ok(())
    }
    
    /// Get forest statistics
    pub fn stats(&self) -> ForestStats {
        let total_nodes: usize = self.trees.iter()
            .map(|t| t.node_count())
            .sum();
            
        let max_depth = self.trees.iter()
            .map(|t| t.depth())
            .max()
            .unwrap_or(0);
            
        ForestStats {
            num_trees: self.trees.len(),
            total_nodes,
            max_depth,
            num_samples: self.num_samples,
            threshold: self.config.anomaly_threshold,
        }
    }
}

/// Forest statistics
#[derive(Debug, Clone, Copy)]
pub struct ForestStats {
    /// Number of trees
    pub num_trees: usize,
    /// Total nodes across all trees
    pub total_nodes: usize,
    /// Maximum tree depth
    pub max_depth: usize,
    /// Number of training samples
    pub num_samples: usize,
    /// Anomaly threshold
    pub threshold: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[cfg(not(feature = "std"))]
    fn create_test_data() -> Vec<Sample, 20> {
        let mut samples = Vec::new();
        
        // Normal data cluster
        for i in 0..15 {
            let temp = 20.0 + (i as f32 * 0.1);
            let humidity = 50.0 + (i as f32 * 0.2);
            let _ = samples.push(Sample::new(&[temp, humidity]).unwrap());
        }
        
        // Anomalies
        let _ = samples.push(Sample::new(&[35.0, 90.0]).unwrap());
        let _ = samples.push(Sample::new(&[5.0, 20.0]).unwrap());
        
        samples
    }
    
    #[cfg(feature = "std")]
    fn create_test_data() -> Vec<Sample> {
        let mut samples = Vec::new();
        
        // Normal data cluster
        for i in 0..15 {
            let temp = 20.0 + (i as f32 * 0.1);
            let humidity = 50.0 + (i as f32 * 0.2);
            samples.push(Sample::new(&[temp, humidity]).unwrap());
        }
        
        // Anomalies
        samples.push(Sample::new(&[35.0, 90.0]).unwrap());
        samples.push(Sample::new(&[5.0, 20.0]).unwrap());
        
        samples
    }
    
    #[test]
    fn test_forest_creation() {
        let config = ForestConfig {
            num_trees: 5,
            sample_size: 10,
            max_depth: 6,
            seed: 123,
            anomaly_threshold: 0.6,
        };
        
        let forest = IsolationForest::<10>::new(config);
        assert_eq!(forest.trees.len(), 0);
    }
    
    #[test]
    fn test_forest_fit() {
        let config = ForestConfig {
            num_trees: 5,
            sample_size: 10,
            max_depth: 6,
            seed: 123,
            anomaly_threshold: 0.6,
        };
        let mut forest = IsolationForest::<10>::new(config);
        let samples = create_test_data();
        
        forest.fit(&samples).unwrap();
        
        let stats = forest.stats();
        assert_eq!(stats.num_trees, forest.config.num_trees);
        assert!(stats.total_nodes > 0);
    }
    
    #[test]
    fn test_anomaly_detection() {
        let mut forest = IsolationForest::<10>::default();
        let samples = create_test_data();
        
        forest.fit(&samples).unwrap();
        
        // Test normal sample
        let normal = Sample::new(&[21.0, 52.0]).unwrap();
        let normal_score = forest.anomaly_score(&normal);
        assert!(normal_score.score < 0.6);
        assert!(!forest.is_anomaly(&normal));
        
        // Test anomaly
        let anomaly = Sample::new(&[35.0, 90.0]).unwrap();
        let anomaly_score = forest.anomaly_score(&anomaly);
        assert!(anomaly_score.score > 0.5);
    }
    
    #[test]
    fn test_score_tracking() {
        let config = ForestConfig {
            num_trees: 5,
            sample_size: 10,
            max_depth: 6,
            seed: 123,
            anomaly_threshold: 0.6,
        };
        let mut forest = IsolationForest::<10>::new(config);
        let samples = create_test_data();
        
        forest.fit(&samples).unwrap();
        
        // Track scores
        for i in 0..5 {
            let sample = Sample::new(&[20.0 + i as f32, 50.0]).unwrap();
            forest.score_and_track(&sample).unwrap();
        }
        
        let trend = forest.anomaly_trend();
        assert!(trend.stats().count > 0);
    }
}