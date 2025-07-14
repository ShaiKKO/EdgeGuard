//! Isolation tree node implementation
//!
//! This module provides a compact node representation for isolation trees.
//! Nodes are designed to minimize memory usage while supporting efficient
//! tree traversal for anomaly scoring.

use crate::{Sample, MLError, MLResult};

/// Node type in the isolation tree
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NodeType {
    /// Internal node with split condition
    Internal {
        /// Feature index to split on
        feature: u8,
        /// Split value
        split_value: f32,
        /// Left child index
        left: u16,
        /// Right child index  
        right: u16,
    },
    /// Leaf node (external)
    External {
        /// Number of samples that reached this leaf
        size: u16,
    },
}

/// Compact node representation
/// 
/// Memory layout optimized for cache efficiency
#[derive(Debug, Clone, Copy)]
pub struct Node {
    /// Node type and data
    pub node_type: NodeType,
    /// Path length from root
    pub depth: u8,
}

impl Node {
    /// Create an internal node
    pub fn internal(feature: u8, split_value: f32, left: u16, right: u16, depth: u8) -> Self {
        Self {
            node_type: NodeType::Internal {
                feature,
                split_value,
                left,
                right,
            },
            depth,
        }
    }
    
    /// Create an external (leaf) node
    pub fn external(size: u16, depth: u8) -> Self {
        Self {
            node_type: NodeType::External { size },
            depth,
        }
    }
    
    /// Check if node is a leaf
    pub fn is_leaf(&self) -> bool {
        matches!(self.node_type, NodeType::External { .. })
    }
    
    /// Get the path length for a sample
    /// 
    /// Returns the depth plus adjustment for external nodes
    pub fn path_length(&self, _sample: &Sample) -> f32 {
        match self.node_type {
            NodeType::External { size } => {
                // Add average path length within the leaf
                self.depth as f32 + c_factor(size as usize)
            }
            NodeType::Internal { .. } => {
                // Should not be called on internal nodes
                self.depth as f32
            }
        }
    }
    
    /// Traverse to next node based on sample
    /// 
    /// Returns the child index to visit next
    pub fn traverse(&self, sample: &Sample) -> MLResult<u16> {
        match self.node_type {
            NodeType::Internal { feature, split_value, left, right } => {
                let feature_value = sample.get_feature(feature as usize)
                    .ok_or(MLError::InvalidFeature)?;
                
                if feature_value < split_value {
                    Ok(left)
                } else {
                    Ok(right)
                }
            }
            NodeType::External { .. } => {
                // Leaf nodes don't have children
                Err(MLError::InvalidConfig("Cannot traverse from leaf node"))
            }
        }
    }
}

/// Calculate c(n) factor for path length adjustment
/// 
/// This represents the average path length of unsuccessful search in BST
pub fn c_factor(n: usize) -> f32 {
    if n <= 1 {
        return 0.0;
    }
    
    // 2 * H(n-1) - 2*(n-1)/n where H is harmonic number
    // For small n, use exact calculation
    match n {
        2 => 1.0,
        3 => 1.5,
        4 => 1.833,
        5 => 2.083,
        6 => 2.283,
        7 => 2.450,
        8 => 2.593,
        9 => 2.718,
        10 => 2.829,
        _ => {
            // For larger n, use approximation
            let h = harmonic_approx(n - 1);
            2.0 * h - 2.0 * (n as f32 - 1.0) / (n as f32)
        }
    }
}

/// Harmonic number approximation
fn harmonic_approx(n: usize) -> f32 {
    // ln(n) + Euler's constant + 1/(2n)
    const EULER: f32 = 0.5772156649;
    crate::ln_approx(n as f32) + EULER + 0.5 / (n as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_node_creation() {
        let internal = Node::internal(0, 25.0, 1, 2, 3);
        assert!(!internal.is_leaf());
        assert_eq!(internal.depth, 3);
        
        let external = Node::external(10, 5);
        assert!(external.is_leaf());
        assert_eq!(external.depth, 5);
    }
    
    #[test]
    fn test_node_traverse() {
        let node = Node::internal(0, 25.0, 1, 2, 0);
        
        let sample1 = Sample::new(&[20.0]).unwrap();
        assert_eq!(node.traverse(&sample1).unwrap(), 1);
        
        let sample2 = Sample::new(&[30.0]).unwrap();
        assert_eq!(node.traverse(&sample2).unwrap(), 2);
    }
    
    #[test]
    fn test_c_factor() {
        assert_eq!(c_factor(0), 0.0);
        assert_eq!(c_factor(1), 0.0);
        assert_eq!(c_factor(2), 1.0);
        assert!((c_factor(10) - 2.829).abs() < 0.001);
    }
}