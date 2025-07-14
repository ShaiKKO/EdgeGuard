# Machine Learning API

Anomaly detection and pattern recognition for sensor data processing.

## IsolationForest

Unsupervised anomaly detection using isolation forest algorithm optimized for edge devices.

### Constructor

```rust
impl IsolationForest {
    pub fn new(n_trees: usize, sample_size: usize) -> Self;
    pub fn with_contamination(mut self, contamination: f32) -> Self;
    pub fn with_random_seed(mut self, seed: u64) -> Self;
    pub fn with_feature_sampling(mut self, enabled: bool) -> Self;
    pub fn train(&mut self, data: &[&[f32]]) -> Result<(), MLError>;
    pub fn predict(&self, sample: &[f32]) -> Result<AnomalyScore, MLError>;
    pub fn predict_batch(&self, samples: &[&[f32]]) -> Result<Vec<AnomalyScore>, MLError>;
}
```

### Training

```rust
// Standard configuration for sensor data
let mut forest = IsolationForest::new(100, 256)
    .with_contamination(0.1)  // Expect 10% anomalies
    .with_random_seed(42);

// Prepare training data
let training_data: Vec<Vec<f32>> = collect_sensor_readings();
let training_refs: Vec<&[f32]> = training_data.iter().map(|v| v.as_slice()).collect();

// Train the model
forest.train(&training_refs)?;

// Model is now ready for inference
println!("Model trained with {} samples", training_data.len());
```

### Inference

```rust
// Single prediction
let sensor_reading = [23.5, 65.2, 1013.25];  // temp, humidity, pressure
let score = forest.predict(&sensor_reading)?;

if score.is_anomaly() {
    println!("Anomaly detected: score = {:.3}", score.value());
} else {
    println!("Normal reading: score = {:.3}", score.value());
}

// Batch prediction
let readings = vec![
    [23.5, 65.2, 1013.25],
    [25.0, 60.0, 1010.0],
    [100.0, 20.0, 950.0],  // Likely anomaly
];

let scores = forest.predict_batch(&readings.iter().map(|r| r.as_slice()).collect::<Vec<_>>())?;
for (reading, score) in readings.iter().zip(scores.iter()) {
    println!("Reading {:?}: {}", reading, score.value());
}
```

## AnomalyScore

Quantified anomaly score with configurable threshold.

### Constructor

```rust
impl AnomalyScore {
    pub fn new(value: f32) -> Self;
    pub fn with_threshold(mut self, threshold: f32) -> Self;
    pub fn value(&self) -> f32;
    pub fn is_anomaly(&self) -> bool;
    pub fn confidence(&self) -> f32;
    pub fn severity(&self) -> AnomalySeverity;
}
```

### Interpretation

```rust
let score = AnomalyScore::new(0.75)
    .with_threshold(0.6);

// Check if anomaly
if score.is_anomaly() {
    match score.severity() {
        AnomalySeverity::Low => println!("Minor anomaly detected"),
        AnomalySeverity::Medium => println!("Moderate anomaly detected"),
        AnomalySeverity::High => println!("Severe anomaly detected"),
        AnomalySeverity::Critical => println!("Critical anomaly detected"),
    }
}

// Get confidence in prediction
let confidence = score.confidence();
if confidence > 0.8 {
    println!("High confidence anomaly: {:.2}%", confidence * 100.0);
}
```

## MLAnomalyStage

Pipeline stage for real-time anomaly detection integration.

### Constructor

```rust
impl MLAnomalyStage {
    pub fn new(training_samples: usize) -> Self;
    pub fn with_model(model: IsolationForest) -> Self;
    pub fn with_feature_extractor(mut self, extractor: Box<dyn FeatureExtractor>) -> Self;
    pub fn with_retraining_interval(mut self, interval_ms: u64) -> Self;
    pub fn with_contamination_rate(mut self, rate: f32) -> Self;
}
```

### Pipeline Integration

```rust
use edgeguard::pipeline::{Pipeline, ValidationStage};
use edgeguard::ml::MLAnomalyStage;

// Create ML-enhanced pipeline
let ml_stage = MLAnomalyStage::new(500)
    .with_contamination_rate(0.05)  // 5% expected anomalies
    .with_retraining_interval(3600_000);  // Retrain every hour

let pipeline = Pipeline::<512>::builder()
    .add_stage(ValidationStage::new(
        TemperatureValidator::new(),
        SensorType::Temperature
    ))
    .add_stage(ml_stage)
    .build();

// Process sensor data through ML pipeline
let event = EventBuilder::new(timestamp)
    .sensor("sensor_001", SensorType::Temperature)
    .reading(23.5, 0.95)
    .unwrap();

pipeline.push_event(event);
pipeline.process_batch(10)?;

// Check for anomaly events
while let Some(result) = pipeline.pop_result() {
    if let Event::AnomalyDetected { sensor_id, score, .. } = result {
        println!("Anomaly in {}: score = {:.3}", sensor_id.as_str(), score.value());
    }
}
```

## FeatureExtractor

Extract relevant features from sensor data for ML processing.

### Trait Definition

```rust
pub trait FeatureExtractor: Send {
    fn extract(&self, event: &Event) -> Result<Vec<f32>, MLError>;
    fn feature_count(&self) -> usize;
    fn feature_names(&self) -> Vec<&'static str>;
}
```

### Built-in Extractors

```rust
// Basic statistical features
let extractor = StatisticalFeatureExtractor::new()
    .with_window_size(10)
    .with_features(vec![
        StatFeature::Mean,
        StatFeature::StdDev,
        StatFeature::Min,
        StatFeature::Max,
        StatFeature::RateOfChange,
    ]);

// Time-series features
let extractor = TimeSeriesFeatureExtractor::new()
    .with_window_size(20)
    .with_features(vec![
        TSFeature::Trend,
        TSFeature::Seasonality,
        TSFeature::Autocorrelation,
    ]);

// Physics-aware features
let extractor = PhysicsFeatureExtractor::new()
    .with_features(vec![
        PhysicsFeature::EnergyConsumption,
        PhysicsFeature::ThermalDynamics,
        PhysicsFeature::PressureGradient,
    ]);
```

### Custom Feature Extraction

```rust
struct CustomFeatureExtractor {
    window_size: usize,
    history: CircularBuffer<100>,
}

impl FeatureExtractor for CustomFeatureExtractor {
    fn extract(&self, event: &Event) -> Result<Vec<f32>, MLError> {
        if let Event::SensorReading { value, timestamp, .. } = event {
            let mut features = Vec::new();
            
            // Add current value
            features.push(*value);
            
            // Add moving average
            if self.history.len() >= self.window_size {
                let mean = self.history.as_slice().iter().sum::<f32>() / self.history.len() as f32;
                features.push(mean);
            }
            
            // Add rate of change
            if let Some(prev) = self.history.last() {
                let rate = (value - prev.value) / (timestamp - prev.timestamp) as f32;
                features.push(rate);
            }
            
            Ok(features)
        } else {
            Err(MLError::InvalidInput("Expected SensorReading event".to_string()))
        }
    }
    
    fn feature_count(&self) -> usize {
        3  // value, mean, rate
    }
    
    fn feature_names(&self) -> Vec<&'static str> {
        vec!["value", "moving_average", "rate_of_change"]
    }
}
```

## OnlineTraining

Continuous learning for model adaptation to changing conditions.

### Constructor

```rust
impl OnlineTraining {
    pub fn new(initial_model: IsolationForest) -> Self;
    pub fn with_update_frequency(mut self, frequency: UpdateFrequency) -> Self;
    pub fn with_adaptation_rate(mut self, rate: f32) -> Self;
    pub fn with_drift_detection(mut self, enabled: bool) -> Self;
    pub fn update(&mut self, sample: &[f32]) -> Result<(), MLError>;
    pub fn retrain(&mut self, data: &[&[f32]]) -> Result<(), MLError>;
}
```

### Adaptive Learning

```rust
let mut online_trainer = OnlineTraining::new(forest)
    .with_update_frequency(UpdateFrequency::Samples(100))
    .with_adaptation_rate(0.1)
    .with_drift_detection(true);

// Process streaming data with continuous learning
for sensor_reading in sensor_stream {
    let features = feature_extractor.extract(&sensor_reading)?;
    
    // Get prediction
    let score = online_trainer.predict(&features)?;
    
    // Update model with new data
    online_trainer.update(&features)?;
    
    // Handle concept drift
    if online_trainer.drift_detected() {
        println!("Concept drift detected, initiating retraining");
        let recent_data = collect_recent_training_data();
        online_trainer.retrain(&recent_data)?;
    }
}
```

## ModelPersistence

Save and load trained models for deployment.

### Serialization

```rust
use edgeguard::ml::ModelPersistence;

// Save trained model
let model_data = forest.serialize()?;
std::fs::write("anomaly_model.bin", model_data)?;

// Load model
let model_data = std::fs::read("anomaly_model.bin")?;
let forest = IsolationForest::deserialize(&model_data)?;
```

### Embedded Deployment

```rust
// For no_std environments
#[no_mangle]
static MODEL_DATA: &[u8] = include_bytes!("../models/anomaly_model.bin");

// Load model at runtime
let forest = IsolationForest::from_embedded(MODEL_DATA)?;
```

## Performance Characteristics

### Training Performance

- **Training time**: O(n log n) where n is sample count
- **Memory usage**: ~1KB per tree + sample storage
- **Typical training**: 10k samples in <1s on Cortex-M4
- **Model size**: 50-200KB for typical configurations

### Inference Performance

- **Prediction latency**: <1ms per sample
- **Throughput**: 1000+ predictions/sec on embedded devices
- **Memory footprint**: <100KB for deployed model
- **Batch processing**: 10k+ samples/sec with vectorization

### Optimization Techniques

```rust
// Use smaller tree count for faster inference
let fast_forest = IsolationForest::new(50, 128)  // Reduced from 100, 256
    .with_feature_sampling(true);

// Quantize model for embedded deployment
let quantized_forest = forest.quantize(8)?;  // 8-bit quantization

// Use feature selection
let selected_features = FeatureSelector::new()
    .select_k_best(5)  // Select top 5 features
    .fit(&training_data)?;
```

## Configuration Types

### FeatureComplexity

```rust
pub enum FeatureComplexity {
    Basic,      // Simple statistics (mean, std, min, max)
    Standard,   // + rate of change, trends
    Extended,   // + time series analysis, cross-correlations
    Full,       // + physics-based features, spectral analysis
}
```

### UpdateFrequency

```rust
pub enum UpdateFrequency {
    Samples(usize),     // Update every N samples
    Time(Duration),     // Update every time period
    Drift,              // Update when drift detected
    Manual,             // Manual updates only
}
```

### AnomalySeverity

```rust
pub enum AnomalySeverity {
    Low,        // Score 0.5-0.6
    Medium,     // Score 0.6-0.7
    High,       // Score 0.7-0.8
    Critical,   // Score 0.8-1.0
}
```

## Error Handling

### MLError

```rust
pub enum MLError {
    InsufficientData { required: usize, available: usize },
    InvalidInput(String),
    ModelNotTrained,
    NumericalInstability,
    FeatureExtractionFailed(String),
    SerializationError(String),
    IncompatibleModel { expected: String, actual: String },
}
```

### Error Recovery

```rust
match forest.predict(&features) {
    Ok(score) => {
        if score.is_anomaly() {
            handle_anomaly(&features, score);
        }
    }
    Err(MLError::ModelNotTrained) => {
        log::warn!("Model not trained, using fallback validation");
        use_fallback_validation(&features);
    }
    Err(MLError::InsufficientData { required, available }) => {
        log::info!("Need {} more samples for training", required - available);
    }
    Err(e) => {
        log::error!("ML prediction failed: {:?}", e);
        return Err(e);
    }
}
```

## Integration Examples

### MQTT Anomaly Detection

```rust
use edgeguard::connectors::mqtt::MqttConnector;
use edgeguard::ml::{IsolationForest, StatisticalFeatureExtractor};

// Set up MQTT client and ML model
let mut mqtt_client = MqttConnector::new(config)?;
let mut forest = IsolationForest::new(100, 256);
let extractor = StatisticalFeatureExtractor::new();

// Train model with historical data
let training_data = load_historical_data()?;
forest.train(&training_data)?;

// Process real-time MQTT messages
mqtt_client.subscribe("sensors/+/data", 1)?;

loop {
    let messages = mqtt_client.poll()?;
    for msg in messages {
        let sensor_data: SensorReading = serde_json::from_slice(&msg.payload)?;
        let event = Event::from_sensor_reading(sensor_data);
        
        // Extract features and predict
        let features = extractor.extract(&event)?;
        let score = forest.predict(&features)?;
        
        if score.is_anomaly() {
            // Publish anomaly alert
            let alert = AnomalyAlert {
                sensor_id: sensor_data.sensor_id,
                score: score.value(),
                timestamp: sensor_data.timestamp,
            };
            
            let alert_json = serde_json::to_vec(&alert)?;
            mqtt_client.publish("alerts/anomalies", &alert_json, 1)?;
        }
    }
}
```

### Edge ML Processing

```rust
// Configure for ESP32 deployment
let ml_stage = MLAnomalyStage::new(200)  // Reduced training samples
    .with_contamination_rate(0.1)
    .with_retraining_interval(7200_000);  // 2 hours

// Use basic feature extraction for performance
let extractor = StatisticalFeatureExtractor::new()
    .with_window_size(5)  // Small window for memory efficiency
    .with_features(vec![
        StatFeature::Mean,
        StatFeature::StdDev,
    ]);

let pipeline = Pipeline::<64>::builder()  // Small buffer for ESP32
    .add_stage(ValidationStage::new(
        TemperatureValidator::new(),
        SensorType::Temperature
    ))
    .add_stage(ml_stage.with_feature_extractor(Box::new(extractor)))
    .build();
```

## Best Practices

### Model Configuration

```rust
// Balance accuracy vs performance
let production_forest = IsolationForest::new(
    75,   // Moderate tree count
    256   // Sufficient sample size
)
.with_contamination(0.05)      // Conservative contamination rate
.with_feature_sampling(true);   // Enable for better generalization
```

### Feature Engineering

```rust
// Use domain knowledge for feature selection
let physics_extractor = PhysicsFeatureExtractor::new()
    .with_features(vec![
        PhysicsFeature::ThermalDynamics,    // For temperature sensors
        PhysicsFeature::PressureGradient,   // For pressure sensors
        PhysicsFeature::HumidityDewPoint,   // For humidity sensors
    ]);
```

### Deployment Strategy

```rust
// Implement A/B testing for model updates
let model_manager = ModelManager::new()
    .with_champion_model(current_forest)
    .with_challenger_model(new_forest)
    .with_traffic_split(0.9, 0.1);  // 90% champion, 10% challenger

// Gradual rollout based on performance metrics
if challenger_performance > champion_performance {
    model_manager.promote_challenger();
}
```

This machine learning API provides comprehensive anomaly detection capabilities optimized for edge deployment with real-time performance characteristics suitable for resource-constrained environments.