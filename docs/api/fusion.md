# Fusion API

Multi-sensor data fusion algorithms with confidence scoring.

## FusionAlgorithm Trait

Core trait for implementing fusion algorithms.

```rust
pub trait FusionAlgorithm: Send {
    fn update(&mut self, measurements: &[f32], timestamp: Timestamp) -> FusionResult<(f32, ConfidenceScore)>;
    fn reset(&mut self);
    fn name(&self) -> &'static str;
    fn is_converged(&self) -> bool { true }
    fn required_measurements(&self) -> usize { 1 }
}
```

## KalmanFilter

Optimal state estimation with process and measurement noise modeling.

### Constructor

```rust
impl<const N: usize, const M: usize> KalmanFilter<N, M> {
    pub fn new(config: KalmanConfig<N, M>) -> Self;
    pub fn predict(&mut self, dt_ms: u32) -> FusionResult<()>;
    pub fn update(&mut self, measurements: &[f32; M], timestamp: Timestamp, mask: Option<u32>) -> FusionResult<(f32, ConfidenceScore)>;
    pub fn state(&self) -> &Vector<N>;
    pub fn covariance(&self) -> &SquareMatrix<N>;
    pub fn reset(&mut self);
}
```

### Configuration

```rust
let config = KalmanConfig {
    initial_state: [25.0],              // Initial temperature
    initial_covariance: [[1.0]],        // Initial uncertainty
    process_noise: [[0.1]],             // Process noise
    measurement_noise: [[0.5]],         // Measurement noise
    transition: StateTransition {
        transition_matrix: [[1.0]],      // State transition
        control_matrix: None,
    },
    measurement_matrix: [[1.0]],        // Measurement model
    control_matrix: None,
    convergence_threshold: 0.01,
};

let mut filter = KalmanFilter::<1, 1>::new(config);
```

### Example Usage

```rust
// Multi-sensor temperature fusion
let config = KalmanConfig::<1, 3> {
    initial_state: [20.0],
    initial_covariance: [[1.0]],
    process_noise: [[0.1]],
    measurement_noise: [[0.5], [0.3], [0.4]],  // Three sensors
    transition: StateTransition {
        transition_matrix: [[1.0]],
        control_matrix: None,
    },
    measurement_matrix: [[1.0], [1.0], [1.0]],  // Direct measurements
    control_matrix: None,
    convergence_threshold: 0.01,
};

let mut filter = KalmanFilter::<1, 3>::new(config);

// Process measurements
let measurements = [20.1, 19.8, 20.3];
let (fused_value, confidence) = filter.update(
    &measurements,
    Timestamp::from_millis(1000),
    None
)?;

println!("Fused: {:.2}°C, Confidence: {:.2}", fused_value, confidence.as_f32());
```

## ExtendedKalmanFilter

Extended Kalman filter for non-linear systems with sensor models.

### Constructor

```rust
impl<const N: usize, const M: usize> ExtendedKalmanFilter<N, M> {
    pub fn new(config: KalmanConfig<N, M>) -> Self;
    pub fn with_model(config: KalmanConfig<N, M>, model: Box<dyn SensorModel>) -> Self;
    pub fn predict(&mut self, dt_ms: u32) -> FusionResult<()>;
    pub fn update(&mut self, measurements: &[f32; M], timestamp: Timestamp, mask: Option<u32>) -> FusionResult<(f32, ConfidenceScore)>;
}
```

### Example with Temperature Model

```rust
let config = KalmanConfig::<1, 1> {
    initial_state: [25.0],
    initial_covariance: [[1.0]],
    process_noise: [[0.1]],
    measurement_noise: [[0.5]],
    transition: StateTransition {
        transition_matrix: [[1.0]],
        control_matrix: None,
    },
    measurement_matrix: [[1.0]],
    control_matrix: None,
    convergence_threshold: 0.01,
};

let mut ekf = ExtendedKalmanFilter::with_model(
    config,
    Box::new(TemperatureModel::new(20.0, 0.5))  // Ambient temp, thermal mass
);
```

## WeightedAverageFusion

Simple weighted average fusion with configurable weights.

### Constructor

```rust
impl<const M: usize> WeightedAverageFusion<M> {
    pub fn new() -> Self;
    pub fn with_weights(weights: [f32; M]) -> Self;
    pub fn add_sensor(&mut self, model: Box<dyn SensorModel>) -> Result<(), FusionError>;
    pub fn set_weights(&mut self, weights: [f32; M]);
    pub fn get_weights(&self) -> &[f32; M];
}
```

### Example

```rust
// Equal weights for three sensors
let mut fusion = WeightedAverageFusion::<3>::new();

// Or custom weights
let fusion = WeightedAverageFusion::with_weights([0.5, 0.3, 0.2]);

// Process measurements
let measurements = [20.1, 19.8, 20.3];
let (fused_value, confidence) = fusion.update(
    &measurements,
    Timestamp::from_millis(1000)
)?;
```

## ComplementaryFilter

Complementary filter for combining high and low frequency components.

### Constructor

```rust
impl ComplementaryFilter {
    pub fn new(alpha: f32) -> Self;
    pub fn with_time_constant(time_constant_ms: u32) -> Self;
    pub fn set_alpha(&mut self, alpha: f32);
    pub fn get_alpha(&self) -> f32;
}
```

### Example

```rust
// 95% weight to first sensor, 5% to second
let mut filter = ComplementaryFilter::new(0.95);

// Or specify time constant
let filter = ComplementaryFilter::with_time_constant(1000);  // 1 second

let measurements = [gyro_reading, accelerometer_reading];
let (fused_value, confidence) = filter.update(
    &measurements,
    timestamp
)?;
```

## ConsensusVoting

Consensus-based fusion for redundant sensors.

### Constructor

```rust
impl<const M: usize> ConsensusVoting<M> {
    pub fn new(threshold: f32) -> Self;
    pub fn with_outlier_detection(mut self, enabled: bool) -> Self;
    pub fn with_agreement_window(mut self, window_ms: u32) -> Self;
}
```

### Example

```rust
// Require 70% agreement
let mut consensus = ConsensusVoting::<5>::new(0.7)
    .with_outlier_detection(true)
    .with_agreement_window(5000);

let measurements = [20.1, 20.0, 19.9, 25.0, 20.2];  // One outlier
let (fused_value, confidence) = consensus.update(
    &measurements,
    timestamp
)?;
```

## Sensor Models

### SensorModel Trait

```rust
pub trait SensorModel: Send {
    fn predict(&mut self, state: &mut Vector<1>, dt: f32);
    fn measurement_matrix(&self) -> &Matrix<1, 1>;
    fn process_noise(&self, dt: f32) -> f32;
    fn measurement_noise(&self) -> f32;
    fn name(&self) -> &'static str;
}
```

### TemperatureModel

```rust
impl TemperatureModel {
    pub fn new(ambient_temp: f32, thermal_mass: f32) -> Self;
    pub fn with_heat_capacity(mut self, capacity: f32) -> Self;
    pub fn with_thermal_conductivity(mut self, conductivity: f32) -> Self;
}
```

### PressureModel

```rust
impl PressureModel {
    pub fn new(reference_pressure: f32, altitude: f32) -> Self;
    pub fn with_weather_sensitivity(mut self, sensitivity: f32) -> Self;
    pub fn with_temperature_compensation(mut self, enabled: bool) -> Self;
}
```

### HumidityModel

```rust
impl HumidityModel {
    pub fn new(reference_humidity: f32, temperature: f32) -> Self;
    pub fn with_dew_point_tracking(mut self, enabled: bool) -> Self;
    pub fn with_vapor_pressure_model(mut self, model: VaporPressureModel) -> Self;
}
```

## ConfidenceScore

Fixed-point confidence scoring for memory efficiency.

### Constructor

```rust
impl ConfidenceScore {
    pub const MIN: Self = Self(0);
    pub const MAX: Self = Self(65535);
    
    pub fn new(value: f32) -> Self;
    pub fn from_u16(value: u16) -> Self;
    pub fn as_f32(&self) -> f32;
    pub fn as_u16(&self) -> u16;
    pub fn combine(&self, other: Self) -> Self;
    pub fn weighted_combine(&self, other: Self, weight: f32) -> Self;
}
```

### Example

```rust
let confidence1 = ConfidenceScore::new(0.8);
let confidence2 = ConfidenceScore::new(0.9);

// Combine confidences
let combined = confidence1.combine(confidence2);
println!("Combined confidence: {:.2}", combined.as_f32());

// Weighted combination
let weighted = confidence1.weighted_combine(confidence2, 0.7);
```

## Configuration Types

### KalmanConfig

```rust
pub struct KalmanConfig<const N: usize, const M: usize> {
    pub initial_state: Vector<N>,
    pub initial_covariance: SquareMatrix<N>,
    pub process_noise: SquareMatrix<N>,
    pub measurement_noise: SquareMatrix<M>,
    pub transition: StateTransition<N>,
    pub measurement_matrix: Matrix<M, N>,
    pub control_matrix: Option<Matrix<N, N>>,
    pub convergence_threshold: f32,
}
```

### StateTransition

```rust
pub struct StateTransition<const N: usize> {
    pub transition_matrix: SquareMatrix<N>,
    pub control_matrix: Option<SquareMatrix<N>>,
}
```

### Matrix Operations

```rust
// Type aliases for clarity
pub type Matrix<const R: usize, const C: usize> = [[f32; C]; R];
pub type SquareMatrix<const N: usize> = Matrix<N, N>;
pub type Vector<const N: usize> = [f32; N];

// Matrix operations
pub fn matrix_multiply<const R1: usize, const C1: usize, const C2: usize>(
    a: &Matrix<R1, C1>,
    b: &Matrix<C1, C2>,
) -> Matrix<R1, C2>;

pub fn matrix_transpose<const R: usize, const C: usize>(
    matrix: &Matrix<R, C>,
) -> Matrix<C, R>;

pub fn matrix_inverse<const N: usize>(
    matrix: &SquareMatrix<N>,
) -> Result<SquareMatrix<N>, FusionError>;
```

## Error Handling

### FusionError

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum FusionError {
    InsufficientData,
    NumericalInstability,
    InvalidConfiguration(&'static str),
    ModelError(&'static str),
    MatrixSingular,
    ConvergenceFailure,
    DimensionMismatch,
}
```

### Error Recovery

```rust
match fusion.update(&measurements, timestamp) {
    Ok((value, confidence)) => {
        process_fused_value(value, confidence);
    }
    Err(FusionError::NumericalInstability) => {
        // Reset and reinitialize
        fusion.reset();
        log::warn!("Fusion algorithm reset due to instability");
    }
    Err(FusionError::InsufficientData) => {
        // Wait for more measurements
        log::debug!("Insufficient data for fusion");
    }
    Err(e) => {
        log::error!("Fusion error: {:?}", e);
    }
}
```

## Advanced Usage

### Multi-Stage Fusion

```rust
// Stage 1: Sensor-level fusion
let mut temp_fusion = KalmanFilter::new(temp_config);
let mut pressure_fusion = KalmanFilter::new(pressure_config);

// Stage 2: Cross-sensor fusion
let mut meta_fusion = WeightedAverageFusion::with_weights([0.6, 0.4]);

// Process measurements
let temp_measurements = [20.1, 20.0, 19.9];
let (temp_fused, temp_confidence) = temp_fusion.update(&temp_measurements, timestamp)?;

let pressure_measurements = [1013.2, 1013.0];
let (pressure_fused, pressure_confidence) = pressure_fusion.update(&pressure_measurements, timestamp)?;

// Meta fusion
let meta_measurements = [temp_fused, pressure_fused];
let (final_value, final_confidence) = meta_fusion.update(&meta_measurements, timestamp)?;
```

### Adaptive Fusion

```rust
struct AdaptiveFusion {
    fusion: Box<dyn FusionAlgorithm>,
    performance_tracker: PerformanceTracker,
}

impl AdaptiveFusion {
    pub fn adapt_weights(&mut self, performance: &PerformanceMetrics) {
        if performance.accuracy < 0.8 {
            // Switch to more robust algorithm
            self.fusion = Box::new(ConsensusVoting::new(0.9));
        }
    }
}
```

## Performance Characteristics

### Memory Usage

| Algorithm | Memory | Complexity | Accuracy |
|-----------|---------|------------|----------|
| WeightedAverage | 128B | O(M) | Good |
| ComplementaryFilter | 64B | O(1) | Fair |
| KalmanFilter | 1-4KB | O(N³) | Excellent |
| ConsensusVoting | 512B | O(M log M) | Good |

### Latency

```rust
// Typical latencies on Cortex-M4 @168MHz
// WeightedAverage: ~10μs
// ComplementaryFilter: ~5μs
// KalmanFilter (1x1): ~500μs
// KalmanFilter (3x3): ~2ms
// ConsensusVoting: ~50μs
```

## Best Practices

### Algorithm Selection

```rust
// High-accuracy, low-latency requirements
let fusion = WeightedAverageFusion::with_weights([0.4, 0.4, 0.2]);

// Optimal performance with good models
let fusion = KalmanFilter::new(config);

// Robust against outliers
let fusion = ConsensusVoting::new(0.8);

// Simple frequency separation
let fusion = ComplementaryFilter::new(0.98);
```

### Sensor Quality Assessment

```rust
// Adjust weights based on sensor quality
let mut weights = [0.33, 0.33, 0.34];
if sensor_quality[0] < 0.8 {
    weights[0] *= 0.5;  // Reduce weight for poor quality sensor
}
fusion.set_weights(weights);
```

### Convergence Monitoring

```rust
if !fusion.is_converged() {
    log::warn!("Fusion algorithm not converged");
    // Allow more measurements before using result
}
```

This fusion API provides comprehensive multi-sensor data fusion capabilities with confidence scoring and error handling suitable for real-time edge applications.