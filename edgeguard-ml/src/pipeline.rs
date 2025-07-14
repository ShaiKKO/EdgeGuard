//! Pipeline integration for ML-based anomaly detection
//!
//! This module provides pipeline stages that integrate Isolation Forest
//! anomaly detection into EdgeGuard's event processing pipeline.

use crate::{IsolationForest, ForestConfig, Sample};
use edgeguard_core::{
    buffer::CircularBuffer,
    events::{Event, SensorType, SystemEventType},
    pipeline::{PipelineStage, StageOutput, PipelineResult},
    time::Timestamp,
    traits::TimestampedReading,
};

#[cfg(not(feature = "std"))]
use heapless::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Feature complexity level for ML
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeatureComplexity {
    /// Basic features: mean, std, min, max, rate (5 features per sensor)
    Basic,
    /// Extended features: adds median, skewness, kurtosis, etc. (10 features per sensor)
    Extended,
    /// Full features: all statistical features (14 features per sensor)
    Full,
}

/// Configuration for ML anomaly detection stage
#[derive(Debug, Clone)]
pub struct MLConfig {
    /// Size of the window for feature extraction
    pub window_size: usize,
    /// Forest configuration
    pub forest_config: ForestConfig,
    /// Sensor types to monitor
    pub sensor_types: Vec<SensorType>,
    /// Time window for features in milliseconds
    pub feature_window_ms: u32,
    /// Retrain interval in seconds (None = no auto retrain)
    pub retrain_interval: Option<u32>,
    /// Anomaly score threshold
    pub anomaly_threshold: f32,
    /// Feature complexity level
    pub feature_complexity: FeatureComplexity,
    /// Enable cross-sensor correlation features
    pub enable_correlation: bool,
}

impl Default for MLConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            forest_config: ForestConfig::default(),
            sensor_types: vec![
                SensorType::Temperature,
                SensorType::Humidity,
                SensorType::Pressure,
            ],
            feature_window_ms: 5000,
            retrain_interval: Some(3600), // Retrain every hour
            anomaly_threshold: 0.7,
            feature_complexity: FeatureComplexity::Extended,
            enable_correlation: false,
        }
    }
}

/// Feature statistics for a sensor
#[derive(Debug, Clone, Copy)]
pub struct SensorStats {
    /// Mean value
    pub mean: f32,
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Rate of change per second
    pub rate_of_change: f32,
    /// Number of samples
    pub count: usize,
    /// Median value
    pub median: f32,
    /// Variance
    pub variance: f32,
    /// Skewness (third moment)
    pub skewness: f32,
    /// Kurtosis (fourth moment)
    pub kurtosis: f32,
    /// Root Mean Square
    pub rms: f32,
    /// Count of zero crossings
    pub zero_crossings: u32,
    /// Number of peaks
    pub peak_count: u32,
    /// Linear trend slope
    pub trend_slope: f32,
}

impl SensorStats {
    fn from_buffer(buffer: &CircularBuffer<100>) -> Self {
        if buffer.is_empty() {
            return Self {
                mean: 0.0,
                min: 0.0,
                max: 0.0,
                std_dev: 0.0,
                rate_of_change: 0.0,
                count: 0,
                median: 0.0,
                variance: 0.0,
                skewness: 0.0,
                kurtosis: 0.0,
                rms: 0.0,
                zero_crossings: 0,
                peak_count: 0,
                trend_slope: 0.0,
            };
        }

        let readings: Vec<f32> = buffer.iter().map(|r| r.value).collect();
        let timestamps: Vec<u64> = buffer.iter().map(|r| r.timestamp).collect();
        let count = readings.len();
        
        // Calculate mean
        let sum: f32 = readings.iter().sum();
        let mean = sum / count as f32;
        
        // Calculate min/max
        let min = readings.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = readings.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Calculate variance and standard deviation
        let variance: f32 = readings.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / count as f32;
        let std_dev = variance.sqrt();
        
        // Calculate median
        let mut sorted = readings.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if count % 2 == 0 {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            sorted[count / 2]
        };
        
        // Calculate skewness (third moment)
        let skewness = if std_dev > 0.0 {
            let third_moment: f32 = readings.iter()
                .map(|&x| ((x - mean) / std_dev).powi(3))
                .sum::<f32>() / count as f32;
            third_moment
        } else {
            0.0
        };
        
        // Calculate kurtosis (fourth moment)
        let kurtosis = if std_dev > 0.0 {
            let fourth_moment: f32 = readings.iter()
                .map(|&x| ((x - mean) / std_dev).powi(4))
                .sum::<f32>() / count as f32;
            fourth_moment - 3.0  // Excess kurtosis
        } else {
            0.0
        };
        
        // Calculate RMS (Root Mean Square)
        let sum_squares: f32 = readings.iter().map(|&x| x * x).sum();
        let rms = (sum_squares / count as f32).sqrt();
        
        // Count zero crossings
        let mut zero_crossings = 0u32;
        for i in 1..count {
            if (readings[i-1] >= 0.0) != (readings[i] >= 0.0) {
                zero_crossings += 1;
            }
        }
        
        // Count peaks (local maxima)
        let mut peak_count = 0u32;
        for i in 1..count-1 {
            if readings[i] > readings[i-1] && readings[i] > readings[i+1] {
                peak_count += 1;
            }
        }
        
        // Calculate linear trend slope using least squares
        let trend_slope = if count >= 2 {
            let t_mean = (timestamps.iter().sum::<u64>() as f32) / count as f32;
            let v_mean = mean;
            
            let numerator: f32 = readings.iter().zip(timestamps.iter())
                .map(|(&v, &t)| (t as f32 - t_mean) * (v - v_mean))
                .sum();
            let denominator: f32 = timestamps.iter()
                .map(|&t| (t as f32 - t_mean).powi(2))
                .sum();
            
            if denominator > 0.0 {
                numerator / denominator * 1000.0  // Convert to per second
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        // Calculate rate of change (simple difference between last and first)
        let rate_of_change = if count >= 2 {
            let first = buffer.iter().next().unwrap();
            let last = buffer.last().unwrap();
            let dt = (last.timestamp - first.timestamp) as f32 / 1000.0; // Convert to seconds
            if dt > 0.0 {
                (last.value - first.value) / dt
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        Self {
            mean,
            min,
            max,
            std_dev,
            rate_of_change,
            count,
            median,
            variance,
            skewness,
            kurtosis,
            rms,
            zero_crossings,
            peak_count,
            trend_slope,
        }
    }
    
    /// Convert to basic feature array for ML (5 features)
    fn to_basic_features(&self) -> [f32; 5] {
        [self.mean, self.std_dev, self.min, self.max, self.rate_of_change]
    }
    
    /// Convert to extended feature array for ML (10 features)
    fn to_extended_features(&self) -> [f32; 10] {
        [
            self.mean,
            self.std_dev,
            self.min,
            self.max,
            self.rate_of_change,
            self.median,
            self.skewness,
            self.kurtosis,
            self.zero_crossings as f32,
            self.trend_slope,
        ]
    }
    
    /// Convert to full feature array for ML (14 features)
    fn to_full_features(&self) -> [f32; 14] {
        [
            self.mean,
            self.std_dev,
            self.min,
            self.max,
            self.rate_of_change,
            self.median,
            self.variance,
            self.skewness,
            self.kurtosis,
            self.rms,
            self.zero_crossings as f32,
            self.peak_count as f32,
            self.trend_slope,
            self.count as f32,
        ]
    }
}

/// ML-based anomaly detection stage
pub struct MLAnomalyStage<const N: usize = 100> {
    /// The isolation forest
    forest: IsolationForest<N>,
    /// Sensor data windows
    temp_window: CircularBuffer<100>,
    humidity_window: CircularBuffer<100>,
    pressure_window: CircularBuffer<100>,
    /// Configuration
    config: MLConfig,
    /// Last retrain timestamp
    last_retrain: Timestamp,
    /// Training samples buffer
    #[cfg(not(feature = "std"))]
    training_buffer: Vec<Sample, 1000>,
    #[cfg(feature = "std")]
    training_buffer: Vec<Sample>,
    /// Is the model trained?
    is_trained: bool,
    /// Custom anomaly event type (using ValidatorError as closest match)
    anomaly_event_type: SystemEventType,
}

impl<const N: usize> MLAnomalyStage<N> {
    /// Create a new ML anomaly detection stage
    pub fn new(config: MLConfig) -> Self {
        let forest = IsolationForest::new(config.forest_config.clone());
        
        Self {
            forest,
            temp_window: CircularBuffer::new(),
            humidity_window: CircularBuffer::new(),
            pressure_window: CircularBuffer::new(),
            config,
            last_retrain: 0,
            training_buffer: Vec::new(),
            is_trained: false,
            // Use ValidatorError as the event type for anomalies
            anomaly_event_type: SystemEventType::ValidatorError,
        }
    }
    
    /// Extract features from all sensor windows
    fn extract_features(&self) -> Option<Sample> {
        // Get stats for each sensor type
        let temp_stats = SensorStats::from_buffer(&self.temp_window);
        let humidity_stats = SensorStats::from_buffer(&self.humidity_window);
        let pressure_stats = SensorStats::from_buffer(&self.pressure_window);
        
        // Need at least some data
        if temp_stats.count == 0 && humidity_stats.count == 0 && pressure_stats.count == 0 {
            return None;
        }
        
        // Combine features from all sensors
        let mut features = Vec::new();
        
        // Helper to add features based on complexity level
        let add_sensor_features = |features: &mut Vec<f32>, stats: &SensorStats| {
            let sensor_features = match self.config.feature_complexity {
                FeatureComplexity::Basic => stats.to_basic_features().to_vec(),
                FeatureComplexity::Extended => stats.to_extended_features().to_vec(),
                FeatureComplexity::Full => stats.to_full_features().to_vec(),
            };
            
            for f in sensor_features {
                #[cfg(not(feature = "std"))]
                let _ = features.push(f);
                #[cfg(feature = "std")]
                features.push(f);
            }
        };
        
        // Add features for each sensor if available
        if temp_stats.count > 0 {
            add_sensor_features(&mut features, &temp_stats);
        }
        
        if humidity_stats.count > 0 {
            add_sensor_features(&mut features, &humidity_stats);
        }
        
        if pressure_stats.count > 0 {
            add_sensor_features(&mut features, &pressure_stats);
        }
        
        // Add cross-sensor correlation features if enabled
        if self.config.enable_correlation {
            self.add_correlation_features(&mut features, &temp_stats, &humidity_stats, &pressure_stats);
        }
        
        // Create sample from features
        Sample::new(&features)
    }
    
    /// Train the forest with collected samples
    fn train(&mut self) -> Result<(), &'static str> {
        if self.training_buffer.len() < 10 {
            return Err("Insufficient training data");
        }
        
        // Train the forest
        self.forest.fit(&self.training_buffer)
            .map_err(|_| "Training failed")?;
        
        self.is_trained = true;
        Ok(())
    }
    
    /// Check if it's time to retrain
    fn should_retrain(&self, timestamp: Timestamp) -> bool {
        if let Some(interval) = self.config.retrain_interval {
            let elapsed = timestamp.saturating_sub(self.last_retrain);
            elapsed > (interval as u64 * 1000)
        } else {
            false
        }
    }
    
    /// Add correlation features using existing physics lookups
    fn add_correlation_features(
        &self, 
        features: &mut Vec<f32>,
        temp_stats: &SensorStats,
        humidity_stats: &SensorStats,
        pressure_stats: &SensorStats,
    ) {
        // Temperature-Humidity correlations
        if temp_stats.count > 0 && humidity_stats.count > 0 {
            // Use existing dew point lookup
            if let Some(dew_point) = edgeguard_core::lookup::dew_point_lookup(
                temp_stats.mean, 
                humidity_stats.mean
            ) {
                // Dew point margin - how far we are from condensation
                let dew_point_margin = temp_stats.mean - dew_point;
                #[cfg(not(feature = "std"))]
                let _ = features.push(dew_point_margin);
                #[cfg(feature = "std")]
                features.push(dew_point_margin);
                
                // Relative dew point depression
                let dew_point_depression = if temp_stats.mean != 0.0 {
                    dew_point_margin / temp_stats.mean.abs()
                } else {
                    0.0
                };
                #[cfg(not(feature = "std"))]
                let _ = features.push(dew_point_depression);
                #[cfg(feature = "std")]
                features.push(dew_point_depression);
            }
            
            // Variance ratio - indicates if sensors are responding similarly
            let variance_ratio = if humidity_stats.variance > 0.0 {
                temp_stats.variance / humidity_stats.variance
            } else {
                1.0
            };
            #[cfg(not(feature = "std"))]
            let _ = features.push(variance_ratio);
            #[cfg(feature = "std")]
            features.push(variance_ratio);
            
            // Trend alignment - are both sensors trending in same direction?
            let trend_alignment = if temp_stats.trend_slope.abs() > 0.01 && humidity_stats.trend_slope.abs() > 0.01 {
                if (temp_stats.trend_slope > 0.0) == (humidity_stats.trend_slope > 0.0) {
                    1.0
                } else {
                    -1.0
                }
            } else {
                0.0
            };
            #[cfg(not(feature = "std"))]
            let _ = features.push(trend_alignment);
            #[cfg(feature = "std")]
            features.push(trend_alignment);
        }
        
        // Pressure correlations
        if pressure_stats.count > 0 {
            // Normalized pressure (1013.25 hPa = 1.0)
            let normalized_pressure = pressure_stats.mean / 1013.25;
            #[cfg(not(feature = "std"))]
            let _ = features.push(normalized_pressure);
            #[cfg(feature = "std")]
            features.push(normalized_pressure);
            
            // Pressure deviation from sea level
            let pressure_deviation = (pressure_stats.mean - 1013.25).abs() / 1013.25;
            #[cfg(not(feature = "std"))]
            let _ = features.push(pressure_deviation);
            #[cfg(feature = "std")]
            features.push(pressure_deviation);
            
            // Pressure stability score (inverse of std dev)
            let pressure_stability = 1.0 / (pressure_stats.std_dev + 1.0);
            #[cfg(not(feature = "std"))]
            let _ = features.push(pressure_stability);
            #[cfg(feature = "std")]
            features.push(pressure_stability);
            
            // Temperature-Pressure correlation (simple, for HVAC detection)
            if temp_stats.count > 0 {
                let temp_pressure_ratio = temp_stats.mean / (pressure_stats.mean / 10.0);
                #[cfg(not(feature = "std"))]
                let _ = features.push(temp_pressure_ratio);
                #[cfg(feature = "std")]
                features.push(temp_pressure_ratio);
            }
        }
        
        // All-sensor stability metric
        if temp_stats.count > 0 && humidity_stats.count > 0 && pressure_stats.count > 0 {
            let combined_stability = 1.0 / (
                temp_stats.std_dev + humidity_stats.std_dev + pressure_stats.std_dev + 1.0
            );
            #[cfg(not(feature = "std"))]
            let _ = features.push(combined_stability);
            #[cfg(feature = "std")]
            features.push(combined_stability);
        }
    }
    
    /// Get current sensor statistics
    pub fn get_sensor_stats(&self) -> (SensorStats, SensorStats, SensorStats) {
        (
            SensorStats::from_buffer(&self.temp_window),
            SensorStats::from_buffer(&self.humidity_window),
            SensorStats::from_buffer(&self.pressure_window),
        )
    }
    
    /// Get feature names based on current configuration
    #[cfg(feature = "std")]
    pub fn get_feature_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        
        let base_features = match self.config.feature_complexity {
            FeatureComplexity::Basic => &["mean", "std_dev", "min", "max", "rate_of_change"][..],
            FeatureComplexity::Extended => &[
                "mean", "std_dev", "min", "max", "rate_of_change",
                "median", "skewness", "kurtosis", "zero_crossings", "trend_slope"
            ][..],
            FeatureComplexity::Full => &[
                "mean", "std_dev", "min", "max", "rate_of_change",
                "median", "variance", "skewness", "kurtosis", "rms",
                "zero_crossings", "peak_count", "trend_slope", "count"
            ][..],
        };
        
        // Add sensor-specific feature names
        for sensor in &["temp", "humidity", "pressure"] {
            for feature in base_features {
                names.push(format!("{}_{}", sensor, feature));
            }
        }
        
        // Add correlation features if enabled
        if self.config.enable_correlation {
            names.push("dew_point_margin".to_string());
            names.push("dew_point_depression".to_string());
            names.push("variance_ratio".to_string());
            names.push("trend_alignment".to_string());
            names.push("pressure_deviation".to_string());
            names.push("normalized_pressure".to_string());
            names.push("pressure_stability".to_string());
            names.push("temp_pressure_ratio".to_string());
            names.push("combined_stability".to_string());
        }
        
        names
    }
    
    /// Get feature names based on current configuration (no_std version)
    #[cfg(not(feature = "std"))]
    pub fn get_feature_count(&self) -> usize {
        let base_count = match self.config.feature_complexity {
            FeatureComplexity::Basic => 5,
            FeatureComplexity::Extended => 10,
            FeatureComplexity::Full => 14,
        };
        
        let sensor_count = 3; // temp, humidity, pressure
        let mut total = base_count * sensor_count;
        
        if self.config.enable_correlation {
            // Updated correlation features:
            // - dew_point_margin, dew_point_depression (2)
            // - variance_ratio, trend_alignment (2)
            // - normalized_pressure, pressure_deviation, pressure_stability (3)
            // - temp_pressure_ratio (1)
            // - combined_stability (1)
            total += 9;
        }
        
        total
    }
    
    /// Check if the model is trained
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }
    
    /// Get training buffer size
    pub fn training_buffer_size(&self) -> usize {
        self.training_buffer.len()
    }
}

impl<const N: usize> PipelineStage for MLAnomalyStage<N> {
    fn process(&mut self, event: Event, output: &mut StageOutput) -> PipelineResult<()> {
        match event {
            Event::SensorReading { sensor_id: _, sensor_type, value, timestamp, quality } => {
                // Skip low quality readings
                if quality < 0.5 {
                    output.push(event);
                    return Ok(());
                }
                
                // Create timestamped reading
                let reading = TimestampedReading { value, timestamp };
                
                // Add to appropriate window
                match sensor_type {
                    SensorType::Temperature => {
                        self.temp_window.push(reading);
                    }
                    SensorType::Humidity => {
                        self.humidity_window.push(reading);
                    }
                    SensorType::Pressure => {
                        self.pressure_window.push(reading);
                    }
                    _ => {
                        // Pass through sensors we don't monitor
                        output.push(event);
                        return Ok(());
                    }
                }
                
                // Extract features
                if let Some(sample) = self.extract_features() {
                    // During initial training phase
                    if !self.is_trained {
                        #[cfg(not(feature = "std"))]
                        let _ = self.training_buffer.push(sample);
                        #[cfg(feature = "std")]
                        self.training_buffer.push(sample);
                        
                        // Try to train once we have enough data
                        if self.training_buffer.len() >= 50 && !self.is_trained {
                            let _ = self.train();
                        }
                    } else {
                        // Score the sample
                        let anomaly_score = self.forest.anomaly_score(&sample);
                        
                        // Check for anomaly
                        if anomaly_score.is_anomaly(self.config.anomaly_threshold) {
                            // Emit anomaly event
                            // Details field encodes anomaly score as fixed point (0-1000)
                            let details = (anomaly_score.score * 1000.0) as u32;
                            
                            let anomaly_event = Event::SystemEvent {
                                event_type: self.anomaly_event_type,
                                timestamp,
                                details,
                            };
                            output.push(anomaly_event);
                        }
                        
                        // Check if retrain needed
                        if self.should_retrain(timestamp) {
                            // Add to training buffer for retrain
                            #[cfg(not(feature = "std"))]
                            {
                                if self.training_buffer.is_full() {
                                    self.training_buffer.clear();
                                }
                                let _ = self.training_buffer.push(sample);
                            }
                            #[cfg(feature = "std")]
                            {
                                if self.training_buffer.len() > 1000 {
                                    self.training_buffer.clear();
                                }
                                self.training_buffer.push(sample);
                            }
                            
                            // Retrain when we have enough new samples
                            if self.training_buffer.len() >= 50 {
                                let _ = self.train();
                                self.last_retrain = timestamp;
                                self.training_buffer.clear();
                            }
                        }
                    }
                }
                
                // Pass through original event
                output.push(event);
            }
            _ => {
                // Pass through other events
                output.push(event);
            }
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "MLAnomalyStage"
    }
    
    fn reset(&mut self) {
        self.temp_window.clear();
        self.humidity_window.clear();
        self.pressure_window.clear();
        self.training_buffer.clear();
        self.is_trained = false;
        self.last_retrain = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sensor_stats_basic() {
        let mut buffer = CircularBuffer::<100>::new();
        for i in 0..10 {
            buffer.push(TimestampedReading {
                value: 20.0 + i as f32,
                timestamp: i as u64 * 1000,
            });
        }
        
        let stats = SensorStats::from_buffer(&buffer);
        assert_eq!(stats.count, 10);
        assert!((stats.mean - 24.5).abs() < 0.1);
        assert_eq!(stats.min, 20.0);
        assert_eq!(stats.max, 29.0);
        assert!(stats.std_dev > 0.0);
        assert_eq!(stats.median, 24.5); // Average of 24 and 25
        assert!(stats.variance > 0.0);
        // The trend should be 1 unit per second (we're increasing by 1 every 1000ms)
        assert!((stats.trend_slope - 1.0).abs() < 0.1);
    }
    
    #[test]
    fn test_sensor_stats_advanced_features() {
        let mut buffer = CircularBuffer::<100>::new();
        
        // Create a sinusoidal pattern
        for i in 0..50 {
            let value = 25.0 + 5.0 * ((i as f32 * 0.2).sin());
            buffer.push(TimestampedReading {
                value,
                timestamp: i as u64 * 100, // 100ms intervals
            });
        }
        
        let stats = SensorStats::from_buffer(&buffer);
        
        // Check zero crossings (sine wave centered at 25 should have no zero crossings)
        assert_eq!(stats.zero_crossings, 0);
        
        // Check peaks (should have several peaks in a sine wave)
        assert!(stats.peak_count > 0);
        
        // RMS should be close to mean for a sine wave offset from zero
        assert!((stats.rms - stats.mean).abs() < 3.0);
        
        // Skewness should be near zero for symmetric sine wave
        assert!(stats.skewness.abs() < 0.5);
    }
    
    #[test]
    fn test_feature_extraction_basic() {
        let config = MLConfig {
            feature_complexity: FeatureComplexity::Basic,
            enable_correlation: false,
            ..Default::default()
        };
        let mut stage = MLAnomalyStage::<10>::new(config);
        
        // Add some data
        for i in 0..20 {
            stage.temp_window.push(TimestampedReading {
                value: 22.0 + (i as f32 * 0.1),
                timestamp: i as u64 * 1000,
            });
            stage.humidity_window.push(TimestampedReading {
                value: 60.0 - (i as f32 * 0.2),
                timestamp: i as u64 * 1000,
            });
            stage.pressure_window.push(TimestampedReading {
                value: 1013.0 + (i as f32 * 0.05),
                timestamp: i as u64 * 1000,
            });
        }
        
        let sample = stage.extract_features().unwrap();
        // Basic features: 5 per sensor * 3 sensors = 15 features
        assert_eq!(sample.num_features, 15);
    }
    
    #[test]
    fn test_feature_extraction_with_correlation() {
        let config = MLConfig {
            feature_complexity: FeatureComplexity::Extended,
            enable_correlation: true,
            ..Default::default()
        };
        let mut stage = MLAnomalyStage::<10>::new(config);
        
        // Add correlated data
        for i in 0..20 {
            let temp = 20.0 + (i as f32 * 0.5);
            let humidity = 80.0 - (i as f32 * 1.0); // Negative correlation
            let pressure = 1013.0 + (i as f32 * 0.1);
            
            stage.temp_window.push(TimestampedReading {
                value: temp,
                timestamp: i as u64 * 1000,
            });
            stage.humidity_window.push(TimestampedReading {
                value: humidity,
                timestamp: i as u64 * 1000,
            });
            stage.pressure_window.push(TimestampedReading {
                value: pressure,
                timestamp: i as u64 * 1000,
            });
        }
        
        let sample = stage.extract_features().unwrap();
        // Extended features: 10 per sensor * 3 sensors + 9 correlation = 39 features
        assert_eq!(sample.num_features, 39);
    }
    
    #[test]
    fn test_ml_stage_creation() {
        let config = MLConfig::default();
        let stage = MLAnomalyStage::<10>::new(config);
        assert_eq!(stage.name(), "MLAnomalyStage");
        assert!(!stage.is_trained());
    }
    
    #[cfg(feature = "std")]
    #[test]
    fn test_feature_names() {
        let config = MLConfig {
            feature_complexity: FeatureComplexity::Basic,
            enable_correlation: true,
            ..Default::default()
        };
        let stage = MLAnomalyStage::<10>::new(config);
        
        let names = stage.get_feature_names();
        assert_eq!(names.len(), 24); // 5 * 3 + 9 correlation features
        
        // Check some expected names
        assert!(names.contains(&"temp_mean".to_string()));
        assert!(names.contains(&"humidity_std_dev".to_string()));
        assert!(names.contains(&"pressure_rate_of_change".to_string()));
        assert!(names.contains(&"dew_point_margin".to_string()));
        assert!(names.contains(&"pressure_deviation".to_string()));
    }
}