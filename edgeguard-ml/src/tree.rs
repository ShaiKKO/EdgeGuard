//! Isolation tree implementation
//!
//! This module provides the core isolation tree data structure and algorithms.
//! Trees are built by recursively partitioning data until isolation is achieved
//! or maximum depth is reached.

use crate::{
    Node, NodeType, Sample, Rng, MLError, MLResult,
};

#[cfg(not(feature = "std"))]
use heapless::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Configuration for isolation tree
#[derive(Debug, Clone, Copy)]
pub struct TreeConfig {
    /// Maximum depth of tree
    pub max_depth: usize,
    /// Random seed for this tree
    pub seed: u32,
}

impl Default for TreeConfig {
    fn default() -> Self {
        Self {
            max_depth: 10, // Reasonable default for embedded
            seed: 42,
        }
    }
}

/// Isolation tree structure
#[cfg(not(feature = "std"))]
pub struct IsolationTree {
    /// Tree nodes in array representation
    pub nodes: Vec<Node, MAX_NODES>,
    /// Configuration
    pub config: TreeConfig,
    /// Random number generator
    rng: Rng,
    /// Current node count
    node_count: usize,
}

#[cfg(feature = "std")]
pub struct IsolationTree {
    /// Tree nodes in array representation
    pub nodes: Vec<Node>,
    /// Configuration
    pub config: TreeConfig,
    /// Random number generator
    rng: Rng,
    /// Current node count
    node_count: usize,
}

impl IsolationTree {
    /// Create a new isolation tree
    pub fn new(config: TreeConfig) -> Self {
        Self {
            nodes: Vec::new(),
            config,
            rng: Rng::new(config.seed),
            node_count: 0,
        }
    }
    
    /// Train the tree on samples
    pub fn fit(&mut self, samples: &[Sample]) -> MLResult<()> {
        if samples.is_empty() {
            return Err(MLError::InsufficientData);
        }
        
        // Clear existing tree
        self.nodes.clear();
        self.node_count = 0;
        
        // Build tree recursively
        self.build_tree(samples, 0)?;
        
        Ok(())
    }
    
    /// Build tree recursively
    fn build_tree(&mut self, samples: &[Sample], depth: u8) -> MLResult<u16> {
        let node_index = self.node_count as u16;
        
        // Check termination conditions
        if depth as usize >= self.config.max_depth || samples.len() <= 1 {
            // Create external node
            let node = Node::external(samples.len() as u16, depth);
            self.add_node(node)?;
            return Ok(node_index);
        }
        
        // All samples identical?
        if self.all_same(samples) {
            let node = Node::external(samples.len() as u16, depth);
            self.add_node(node)?;
            return Ok(node_index);
        }
        
        // Select random feature and split value
        let (feature, split_value) = self.select_split(samples)?;
        
        // Partition samples
        let (left_samples, right_samples) = self.partition(samples, feature, split_value);
        
        // Handle edge case where partition fails
        if left_samples.is_empty() || right_samples.is_empty() {
            let node = Node::external(samples.len() as u16, depth);
            self.add_node(node)?;
            return Ok(node_index);
        }
        
        // Reserve space for current node
        self.node_count += 1;
        
        // Build children
        let left_index = self.build_tree(&left_samples, depth + 1)?;
        let right_index = self.build_tree(&right_samples, depth + 1)?;
        
        // Create internal node
        let node = Node::internal(feature, split_value, left_index, right_index, depth);
        
        // Add node at reserved position
        if node_index as usize >= self.nodes.len() {
            #[cfg(not(feature = "std"))]
            self.nodes.push(node).map_err(|_| MLError::BufferFull)?;
            #[cfg(feature = "std")]
            self.nodes.push(node);
        } else {
            self.nodes[node_index as usize] = node;
        }
        
        Ok(node_index)
    }
    
    /// Add a node to the tree
    fn add_node(&mut self, node: Node) -> MLResult<()> {
        #[cfg(not(feature = "std"))]
        self.nodes.push(node).map_err(|_| MLError::BufferFull)?;
        #[cfg(feature = "std")]
        self.nodes.push(node);
        self.node_count += 1;
        Ok(())
    }
    
    /// Check if all samples are identical
    fn all_same(&self, samples: &[Sample]) -> bool {
        if samples.len() <= 1 {
            return true;
        }
        
        let first = &samples[0];
        samples[1..].iter().all(|s| {
            s.features[..s.num_features] == first.features[..first.num_features]
        })
    }
    
    /// Select random feature and split value
    fn select_split(&mut self, samples: &[Sample]) -> MLResult<(u8, f32)> {
        // Get number of features from first sample
        let num_features = samples[0].num_features;
        if num_features == 0 {
            return Err(MLError::InvalidFeature);
        }
        
        // Try multiple times to find a good split
        for _ in 0..10 {
            // Random feature
            let feature = self.rng.next_range(num_features) as u8;
            
            // Find min/max for this feature
            let (min_val, max_val) = self.feature_range(samples, feature)?;
            
            if (max_val - min_val).abs() < f32::EPSILON {
                continue; // Try another feature
            }
            
            // Random split value between min and max
            let split_value = self.rng.next_f32_range(min_val, max_val);
            
            return Ok((feature, split_value));
        }
        
        // Fallback: use first feature, median value
        Ok((0, self.median_value(samples, 0)?))
    }
    
    /// Get min/max range for a feature
    fn feature_range(&self, samples: &[Sample], feature: u8) -> MLResult<(f32, f32)> {
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        
        for sample in samples {
            let val = sample.get_feature(feature as usize)
                .ok_or(MLError::InvalidFeature)?;
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
        
        Ok((min_val, max_val))
    }
    
    /// Get median value for a feature
    #[cfg(not(feature = "std"))]
    fn median_value(&self, samples: &[Sample], feature: u8) -> MLResult<f32> {
        let mut values: Vec<f32, 256> = Vec::new();
        
        for sample in samples {
            let val = sample.get_feature(feature as usize)
                .ok_or(MLError::InvalidFeature)?;
            values.push(val).map_err(|_| MLError::BufferFull)?;
        }
        
        // Simple bubble sort for no_std
        for i in 0..values.len() {
            for j in 0..values.len() - i - 1 {
                if values[j] > values[j + 1] {
                    values.swap(j, j + 1);
                }
            }
        }
        
        // Return median
        Ok(values[values.len() / 2])
    }
    
    #[cfg(feature = "std")]
    fn median_value(&self, samples: &[Sample], feature: u8) -> MLResult<f32> {
        let mut values: Vec<f32> = Vec::new();
        
        for sample in samples {
            let val = sample.get_feature(feature as usize)
                .ok_or(MLError::InvalidFeature)?;
            values.push(val);
        }
        
        // Sort values
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        
        // Return median
        Ok(values[values.len() / 2])
    }
    
    /// Partition samples based on split
    #[cfg(not(feature = "std"))]
    fn partition(&self, samples: &[Sample], feature: u8, split_value: f32) 
        -> (Vec<Sample, 256>, Vec<Sample, 256>) 
    {
        let mut left = Vec::new();
        let mut right = Vec::new();
        
        for &sample in samples {
            if let Some(val) = sample.get_feature(feature as usize) {
                if val < split_value {
                    let _ = left.push(sample);
                } else {
                    let _ = right.push(sample);
                }
            }
        }
        
        (left, right)
    }
    
    #[cfg(feature = "std")]
    fn partition(&self, samples: &[Sample], feature: u8, split_value: f32) 
        -> (Vec<Sample>, Vec<Sample>) 
    {
        let mut left = Vec::new();
        let mut right = Vec::new();
        
        for &sample in samples {
            if let Some(val) = sample.get_feature(feature as usize) {
                if val < split_value {
                    left.push(sample);
                } else {
                    right.push(sample);
                }
            }
        }
        
        (left, right)
    }
    
    /// Calculate path length for a sample
    pub fn path_length(&self, sample: &Sample) -> f32 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        
        let mut current_index = 0;
        
        loop {
            let node = &self.nodes[current_index];
            
            match node.node_type {
                NodeType::External { .. } => {
                    return node.path_length(sample);
                }
                NodeType::Internal { .. } => {
                    match node.traverse(sample) {
                        Ok(next_index) => {
                            current_index = next_index as usize;
                            if current_index >= self.nodes.len() {
                                // Shouldn't happen with valid tree
                                return node.depth as f32;
                            }
                        }
                        Err(_) => {
                            // Feature not found, treat as external
                            return node.depth as f32;
                        }
                    }
                }
            }
        }
    }
    
    /// Get the number of nodes in the tree
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    
    /// Get tree depth
    pub fn depth(&self) -> usize {
        self.nodes.iter()
            .map(|n| n.depth as usize)
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[cfg(not(feature = "std"))]
    fn create_test_samples() -> Vec<Sample, 10> {
        let mut samples = Vec::new();
        
        // Normal samples
        let _ = samples.push(Sample::new(&[20.0, 50.0, 1013.0]).unwrap());
        let _ = samples.push(Sample::new(&[22.0, 55.0, 1012.0]).unwrap());
        let _ = samples.push(Sample::new(&[21.0, 52.0, 1014.0]).unwrap());
        let _ = samples.push(Sample::new(&[19.0, 48.0, 1013.0]).unwrap());
        
        // Anomaly
        let _ = samples.push(Sample::new(&[35.0, 90.0, 1000.0]).unwrap());
        
        samples
    }
    
    #[cfg(feature = "std")]
    fn create_test_samples() -> Vec<Sample> {
        let mut samples = Vec::new();
        
        // Normal samples
        samples.push(Sample::new(&[20.0, 50.0, 1013.0]).unwrap());
        samples.push(Sample::new(&[22.0, 55.0, 1012.0]).unwrap());
        samples.push(Sample::new(&[21.0, 52.0, 1014.0]).unwrap());
        samples.push(Sample::new(&[19.0, 48.0, 1013.0]).unwrap());
        
        // Anomaly
        samples.push(Sample::new(&[35.0, 90.0, 1000.0]).unwrap());
        
        samples
    }
    
    #[test]
    fn test_tree_creation() {
        let config = TreeConfig::default();
        let tree = IsolationTree::new(config);
        
        assert_eq!(tree.node_count(), 0);
        assert_eq!(tree.depth(), 0);
    }
    
    #[test]
    fn test_tree_fit() {
        let config = TreeConfig {
            max_depth: 5,
            seed: 123,
        };
        let mut tree = IsolationTree::new(config);
        
        let samples = create_test_samples();
        
        tree.fit(&samples).unwrap();
        
        assert!(tree.node_count() > 0);
        assert!(tree.depth() <= 5);
    }
    
    #[test]
    fn test_path_length() {
        let config = TreeConfig::default();
        let mut tree = IsolationTree::new(config);
        
        let samples = create_test_samples();
        
        tree.fit(&samples).unwrap();
        
        // Normal sample should have longer path
        let normal_path = tree.path_length(&samples[0]);
        
        // Anomaly should have shorter path
        let anomaly_path = tree.path_length(&samples[4]);
        
        assert!(normal_path > 0.0);
        assert!(anomaly_path > 0.0);
    }
}