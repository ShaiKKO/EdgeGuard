//! Anomaly scoring and history tracking
//!
//! This module provides anomaly score calculation and tracking functionality
//! for detecting trends and patterns in anomaly detection over time.

use crate::{average_path_length, MLResult};

#[cfg(not(feature = "std"))]
use heapless::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Anomaly score result
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnomalyScore {
    /// Raw anomaly score (0.0 = normal, 1.0 = anomaly)
    pub score: f32,
    /// Average path length across trees
    pub avg_path_length: f32,
    /// Number of trees used
    pub num_trees: usize,
}

impl AnomalyScore {
    /// Create a new anomaly score
    pub fn new(score: f32, avg_path_length: f32, num_trees: usize) -> Self {
        Self {
            score,
            avg_path_length,
            num_trees,
        }
    }
    
    /// Check if score indicates an anomaly
    pub fn is_anomaly(&self, threshold: f32) -> bool {
        self.score > threshold
    }
    
    /// Get confidence level (based on number of trees)
    pub fn confidence(&self) -> f32 {
        // More trees = higher confidence
        (self.num_trees as f32 / 100.0).min(1.0)
    }
}

/// Score history for trend analysis
#[cfg(not(feature = "std"))]
pub struct ScoreHistory<const N: usize = 100> {
    /// Historical scores
    scores: Vec<AnomalyScore, N>,
    /// Exponential moving average
    ema: f32,
    /// EMA smoothing factor (0.0 - 1.0)
    alpha: f32,
}

#[cfg(feature = "std")]
pub struct ScoreHistory<const N: usize = 100> {
    /// Historical scores
    scores: Vec<AnomalyScore>,
    /// Exponential moving average
    ema: f32,
    /// EMA smoothing factor (0.0 - 1.0)
    alpha: f32,
}

impl<const N: usize> ScoreHistory<N> {
    /// Create new score history
    pub fn new(alpha: f32) -> Self {
        Self {
            scores: Vec::new(),
            ema: 0.5, // Start at neutral
            alpha: alpha.clamp(0.0, 1.0),
        }
    }
    
    /// Add a new score to history
    pub fn add(&mut self, score: AnomalyScore) -> MLResult<()> {
        // Update EMA
        self.ema = self.alpha * score.score + (1.0 - self.alpha) * self.ema;
        
        // Add to history
        if self.scores.len() >= N {
            // Remove oldest
            self.scores.remove(0);
        }
        
        #[cfg(not(feature = "std"))]
        self.scores.push(score).map_err(|_| MLError::BufferFull)?;
        #[cfg(feature = "std")]
        self.scores.push(score);
        Ok(())
    }
    
    /// Get current trend (positive = increasing anomaly)
    pub fn trend(&self) -> f32 {
        if self.scores.len() < 2 {
            return 0.0;
        }
        
        // Simple linear regression on recent scores
        let n = self.scores.len().min(10); // Use last 10 scores
        let start = self.scores.len() - n;
        
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;
        
        for (i, score) in self.scores[start..].iter().enumerate() {
            let x = i as f32;
            let y = score.score;
            
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }
        
        let n_f = n as f32;
        let slope = (n_f * sum_xy - sum_x * sum_y) / (n_f * sum_xx - sum_x * sum_x);
        
        slope
    }
    
    /// Check if trend is increasing
    pub fn is_increasing(&self) -> bool {
        self.trend() > 0.01 // Small positive threshold
    }
    
    /// Get exponential moving average
    pub fn ema(&self) -> f32 {
        self.ema
    }
    
    /// Get recent average
    pub fn recent_average(&self, window: usize) -> Option<f32> {
        if self.scores.is_empty() {
            return None;
        }
        
        let n = window.min(self.scores.len());
        let start = self.scores.len() - n;
        
        let sum: f32 = self.scores[start..]
            .iter()
            .map(|s| s.score)
            .sum();
        
        Some(sum / n as f32)
    }
    
    /// Get statistics
    pub fn stats(&self) -> HistoryStats {
        if self.scores.is_empty() {
            return HistoryStats::default();
        }
        
        let mean = self.scores.iter()
            .map(|s| s.score)
            .sum::<f32>() / self.scores.len() as f32;
        
        let variance = self.scores.iter()
            .map(|s| (s.score - mean).powi(2))
            .sum::<f32>() / self.scores.len() as f32;
        
        let std_dev = variance.sqrt();
        
        let min = self.scores.iter()
            .map(|s| s.score)
            .fold(f32::INFINITY, |a, b| a.min(b));
        let max = self.scores.iter()
            .map(|s| s.score)
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));
        
        HistoryStats {
            count: self.scores.len(),
            mean,
            std_dev,
            min,
            max,
            trend: self.trend(),
            ema: self.ema,
        }
    }
}

/// Statistics for score history
#[derive(Debug, Clone, Copy, Default)]
pub struct HistoryStats {
    /// Number of scores
    pub count: usize,
    /// Mean score
    pub mean: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Minimum score
    pub min: f32,
    /// Maximum score  
    pub max: f32,
    /// Current trend
    pub trend: f32,
    /// Exponential moving average
    pub ema: f32,
}

/// Calculate anomaly score from path lengths
/// 
/// Uses the formula: score = 2^(-E(h(x))/c(n))
/// where E(h(x)) is expected path length and c(n) is average path length
pub fn calculate_anomaly_score(avg_path_length: f32, num_samples: usize) -> f32 {
    if num_samples <= 1 {
        return 0.5; // Neutral score
    }
    
    let expected_path = average_path_length(num_samples);
    if expected_path == 0.0 {
        return 0.5;
    }
    
    // Calculate score
    let exponent = -avg_path_length / expected_path;
    2.0_f32.powf(exponent)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_anomaly_score() {
        let score = AnomalyScore::new(0.7, 3.5, 100);
        
        assert!(score.is_anomaly(0.6));
        assert!(!score.is_anomaly(0.8));
        assert_eq!(score.confidence(), 1.0);
        
        let score2 = AnomalyScore::new(0.3, 5.2, 50);
        assert_eq!(score2.confidence(), 0.5);
    }
    
    #[test]
    fn test_score_history() {
        let mut history = ScoreHistory::<10>::new(0.2);
        
        // Add increasing scores
        for i in 0..5 {
            let score = AnomalyScore::new(0.3 + i as f32 * 0.1, 4.0, 100);
            history.add(score).unwrap();
        }
        
        assert!(history.is_increasing());
        assert!(history.trend() > 0.0);
        
        let stats = history.stats();
        assert_eq!(stats.count, 5);
        assert!(stats.mean > 0.4);
    }
    
    #[test]
    fn test_calculate_anomaly_score() {
        // Short path = anomaly (high score)
        let score1 = calculate_anomaly_score(2.0, 100);
        assert!(score1 > 0.6); // Very short path = high anomaly score
        
        // Normal path (expected path length)
        let expected = average_path_length(100);
        let score2 = calculate_anomaly_score(expected, 100);
        assert!((score2 - 0.5).abs() < 0.1); // Should be around 0.5
        
        // Slightly longer than expected = more normal
        let score3 = calculate_anomaly_score(expected * 1.2, 100);
        assert!(score3 < 0.5); // Longer path = lower score
        
        // Edge cases
        assert_eq!(calculate_anomaly_score(0.0, 0), 0.5);
        assert_eq!(calculate_anomaly_score(0.0, 1), 0.5);
    }
}