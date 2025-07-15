# Python Fusion API

Multi-sensor data fusion algorithms with confidence scoring for improved accuracy and reliability.

## Overview

EdgeGuard's fusion system combines readings from multiple sensors to:
- **Improve Accuracy**: Reduce measurement uncertainty through statistical combination
- **Increase Reliability**: Detect and compensate for individual sensor failures
- **Provide Confidence**: Quantify certainty in fused measurements
- **Enable Redundancy**: Continue operation even with sensor failures

## Core Concepts

### Physics Background

Multi-sensor fusion leverages statistical principles and sensor physics:
- **Measurement Uncertainty**: Each sensor has inherent noise and bias
- **Correlation**: Related sensors provide complementary information
- **Optimal Estimation**: Mathematical techniques minimize estimation error
- **Confidence Scoring**: Quantifies reliability of fused measurements

## Base Classes

### FusionAlgorithm

Abstract base class for all fusion algorithms:

```python
from abc import ABC, abstractmethod

class FusionAlgorithm(ABC):
    """Base class for multi-sensor fusion algorithms."""
    
    @abstractmethod
    def update(self, measurements: List[float], timestamp: Timestamp) -> Tuple[float, float]:
        """Update fusion with new measurements.
        
        Args:
            measurements: List of sensor readings to fuse
            timestamp: Measurement timestamp
            
        Returns:
            Tuple of (fused_value, confidence_score)
            
        Raises:
            FusionError: If fusion fails
        """
        
    @abstractmethod
    def reset(self) -> None:
        """Reset fusion algorithm state."""
        
    @abstractmethod
    def name(self) -> str:
        """Get algorithm name."""
        
    def predict(self, dt_ms: int) -> Tuple[float, float]:
        """Predict next state (optional for some algorithms).
        
        Args:
            dt_ms: Time delta in milliseconds
            
        Returns:
            Tuple of (predicted_value, confidence_score)
        """
        raise NotImplementedError("Prediction not supported by this algorithm")
```

### ConfidenceScore

Confidence scoring for fusion results:

```python
class ConfidenceScore:
    """Confidence score for fusion results [0.0, 1.0]."""
    
    def __init__(self, value: float):
        """Create confidence score.
        
        Args:
            value: Confidence value [0.0, 1.0] where 1.0 is maximum confidence
        """
        
    @property
    def value(self) -> float:
        """Get confidence value [0.0, 1.0]."""
        
    @property
    def percentage(self) -> float:
        """Get confidence as percentage [0.0, 100.0]."""
        
    def combine(self, other: 'ConfidenceScore') -> 'ConfidenceScore':
        """Combine with another confidence score."""
        
    def __float__(self) -> float:
        """Convert to float value."""
        
    def __str__(self) -> str:
        """String representation."""
```

## Kalman Filter

Optimal state estimation with process and measurement noise modeling.

### Theory

Kalman filtering provides optimal estimates for linear systems with Gaussian noise:
- **State Prediction**: Predict next state based on process model
- **Measurement Update**: Incorporate new measurements with uncertainty
- **Covariance Tracking**: Track estimation uncertainty over time
- **Optimal Fusion**: Mathematically optimal for linear Gaussian systems

### KalmanFilter Class

```python
class KalmanFilter(FusionAlgorithm):
    """Kalman filter for optimal multi-sensor fusion."""
    
    def __init__(self, config: 'KalmanConfig'):
        """Initialize Kalman filter.
        
        Args:
            config: Kalman filter configuration
        """
        
    def update(self, measurements: List[float], timestamp: Timestamp) -> Tuple[float, float]:
        """Update filter with new measurements.
        
        Performs prediction and measurement update steps:
        1. Predict state forward in time
        2. Compute Kalman gain
        3. Update state estimate with measurements
        4. Update error covariance
        
        Args:
            measurements: Sensor measurements to incorporate
            timestamp: Measurement timestamp
            
        Returns:
            Tuple of (estimated_state, confidence_score)
        """
        
    def predict(self, dt_ms: int) -> Tuple[float, float]:
        """Predict state forward in time.
        
        Args:
            dt_ms: Time step in milliseconds
            
        Returns:
            Tuple of (predicted_state, confidence_score)
        """
        
    def reset(self) -> None:
        """Reset filter to initial conditions."""
        
    @property
    def state(self) -> float:
        """Current state estimate."""
        
    @property
    def covariance(self) -> float:
        """Current estimation uncertainty."""
        
    @property
    def innovation(self) -> float:
        """Last measurement innovation (residual)."""
```

### KalmanConfig

Configuration for Kalman filter parameters:

```python
class KalmanConfig:
    """Kalman filter configuration."""
    
    def __init__(self,
                 initial_state: float,
                 initial_covariance: float,
                 process_noise: float,
                 measurement_noise: float,
                 convergence_threshold: float = 0.01):
        """Create Kalman filter configuration.
        
        Args:
            initial_state: Initial state estimate
            initial_covariance: Initial estimation uncertainty
            process_noise: Process noise variance (state uncertainty growth)
            measurement_noise: Measurement noise variance (sensor uncertainty)
            convergence_threshold: Convergence detection threshold
        """
        
    @classmethod
    def for_temperature(cls, 
                       initial_temp: float = 20.0,
                       temp_stability: float = 0.1,
                       sensor_noise: float = 0.5) -> 'KalmanConfig':
        """Create configuration optimized for temperature sensing.
        
        Args:
            initial_temp: Expected initial temperature (°C)
            temp_stability: Temperature stability (°C/s std dev)
            sensor_noise: Sensor noise standard deviation (°C)
        """
        
    @classmethod
    def for_pressure(cls,
                    initial_pressure: float = 1013.25,
                    pressure_stability: float = 1.0,
                    sensor_noise: float = 0.1) -> 'KalmanConfig':
        """Create configuration optimized for pressure sensing."""
        
    @classmethod
    def for_humidity(cls,
                    initial_humidity: float = 50.0,
                    humidity_stability: float = 2.0,
                    sensor_noise: float = 1.0) -> 'KalmanConfig':
        """Create configuration optimized for humidity sensing."""
```

### Usage Examples

```python
import edgeguard as eg

# Basic Kalman filter for temperature fusion
config = eg.KalmanConfig.for_temperature(
    initial_temp=22.0,
    temp_stability=0.1,   # Stable environment
    sensor_noise=0.3      # Moderate sensor noise
)

kalman = eg.KalmanFilter(config)

# Fuse multiple temperature sensors
temp_readings = [22.1, 21.9, 22.3]  # Three temperature sensors
timestamp = eg.Timestamp.now()

fused_temp, confidence = kalman.update(temp_readings, timestamp)
print(f"Fused temperature: {fused_temp:.2f}°C (confidence: {confidence:.1%})")

# Continue with more measurements
for i in range(10):
    # Simulate sensor readings with noise
    readings = [22.0 + 0.1*i + random.gauss(0, 0.2) for _ in range(3)]
    timestamp = eg.Timestamp.now()
    
    fused_value, conf = kalman.update(readings, timestamp)
    print(f"Step {i}: {fused_value:.2f}°C (confidence: {conf:.1%})")
```

### Advanced Configuration

```python
# Custom Kalman configuration for specific application
config = eg.KalmanConfig(
    initial_state=25.0,        # 25°C initial estimate
    initial_covariance=2.0,    # High initial uncertainty
    process_noise=0.01,        # Very stable temperature
    measurement_noise=0.25,    # Quarter-degree sensor noise
    convergence_threshold=0.005 # Tight convergence criteria
)

kalman = eg.KalmanFilter(config)

# Monitor filter performance
for measurements in sensor_data_stream:
    fused, conf = kalman.update(measurements, timestamp)
    
    # Check filter health
    innovation = kalman.innovation
    if abs(innovation) > 3.0:  # 3-sigma test
        print(f"Warning: Large innovation {innovation:.2f} - possible sensor fault")
    
    if conf < 0.7:  # Low confidence
        print(f"Warning: Low confidence {conf:.1%} - check sensors")
```

## Weighted Average Fusion

Simple weighted combination of sensor readings.

### WeightedAverageFusion Class

```python
class WeightedAverageFusion(FusionAlgorithm):
    """Weighted average fusion of multiple sensors."""
    
    def __init__(self, weights: Optional[List[float]] = None):
        """Initialize weighted average fusion.
        
        Args:
            weights: Sensor weights (auto-normalized). Equal weights if None.
        """
        
    def set_weights(self, weights: List[float]) -> None:
        """Set sensor weights.
        
        Args:
            weights: Weights for each sensor (will be normalized)
        """
        
    def update(self, measurements: List[float], timestamp: Timestamp) -> Tuple[float, float]:
        """Compute weighted average of measurements.
        
        Confidence based on:
        - Agreement between sensors (lower spread = higher confidence)
        - Number of sensors (more sensors = higher confidence)
        - Individual sensor quality
        
        Args:
            measurements: Sensor readings
            timestamp: Measurement timestamp
            
        Returns:
            Tuple of (weighted_average, confidence_score)
        """
        
    def reset(self) -> None:
        """Reset fusion state."""
        
    @property
    def weights(self) -> List[float]:
        """Current sensor weights."""
```

### Usage Examples

```python
# Equal weight fusion
fusion = eg.WeightedAverageFusion()

readings = [22.1, 21.9, 22.3, 22.0]
fused, confidence = fusion.update(readings, eg.Timestamp.now())
print(f"Equal weight average: {fused:.2f}°C")

# Custom weights based on sensor quality
high_quality_weights = [0.4, 0.3, 0.2, 0.1]  # First sensor most trusted
fusion_weighted = eg.WeightedAverageFusion(high_quality_weights)

fused, confidence = fusion_weighted.update(readings, eg.Timestamp.now())
print(f"Weighted average: {fused:.2f}°C (confidence: {confidence:.1%})")

# Adaptive weights based on sensor performance
class AdaptiveWeightedFusion:
    def __init__(self):
        self.fusion = eg.WeightedAverageFusion()
        self.sensor_errors = [0.0] * 4  # Track error for 4 sensors
        self.measurement_count = 0
        
    def update(self, measurements: List[float], reference: float = None):
        """Update with adaptive weights based on sensor performance."""
        # Update weights based on historical performance
        if self.measurement_count > 10:  # Need history for adaptation
            # Lower error = higher weight
            max_error = max(self.sensor_errors) + 0.001  # Avoid division by zero
            weights = [(max_error - error) / max_error for error in self.sensor_errors]
            self.fusion.set_weights(weights)
        
        fused, conf = self.fusion.update(measurements, eg.Timestamp.now())
        
        # Update error tracking if reference available
        if reference is not None:
            for i, measurement in enumerate(measurements):
                error = abs(measurement - reference)
                # Exponential moving average of error
                alpha = 0.1
                self.sensor_errors[i] = alpha * error + (1 - alpha) * self.sensor_errors[i]
        
        self.measurement_count += 1
        return fused, conf

# Use adaptive fusion
adaptive = AdaptiveWeightedFusion()
for measurements in sensor_streams:
    fused, conf = adaptive.update(measurements)
    print(f"Adaptive fusion: {fused:.2f} (confidence: {conf:.1%})")
```

## Complementary Filter

Frequency-domain sensor fusion for combining sensors with different characteristics.

### ComplementaryFilter Class

```python
class ComplementaryFilter(FusionAlgorithm):
    """Complementary filter for frequency-domain sensor fusion."""
    
    def __init__(self, alpha: float = 0.98, time_constant_ms: Optional[int] = None):
        """Initialize complementary filter.
        
        Args:
            alpha: Filter coefficient [0.0, 1.0]
                  High values favor first sensor (low-pass)
                  Low values favor second sensor (high-pass)
            time_constant_ms: Time constant in milliseconds (alternative to alpha)
        """
        
    def update(self, measurements: List[float], timestamp: Timestamp) -> Tuple[float, float]:
        """Update complementary filter.
        
        Combines two sensors with complementary frequency characteristics:
        - First sensor: Low-pass filtered (long-term accuracy)
        - Second sensor: High-pass filtered (short-term dynamics)
        
        Args:
            measurements: Must contain exactly 2 measurements
            timestamp: Measurement timestamp
            
        Returns:
            Tuple of (filtered_value, confidence_score)
        """
        
    def reset(self) -> None:
        """Reset filter state."""
        
    @property
    def alpha(self) -> float:
        """Current filter coefficient."""
```

### Usage Examples

```python
# Temperature fusion: stable sensor + fast sensor
comp_filter = eg.ComplementaryFilter(alpha=0.95)

# Combine slow accurate sensor with fast responsive sensor
slow_accurate = 22.1    # High accuracy, slow response
fast_responsive = 22.3  # Lower accuracy, fast response

fused, conf = comp_filter.update([slow_accurate, fast_responsive], eg.Timestamp.now())
print(f"Complementary fusion: {fused:.2f}°C")

# Pressure fusion: barometer + altimeter
# Barometer: accurate long-term, affected by weather
# Altimeter: responsive to altitude changes, drifts over time
pressure_filter = eg.ComplementaryFilter(alpha=0.99)  # Favor barometer

barometer_reading = 1013.2
altimeter_reading = 1012.8

fused_pressure, conf = pressure_filter.update(
    [barometer_reading, altimeter_reading], 
    eg.Timestamp.now()
)
print(f"Fused pressure: {fused_pressure:.1f} hPa")
```

## Consensus Voting

Agreement-based fusion for redundant sensors.

### ConsensusVoting Class

```python
class ConsensusVoting(FusionAlgorithm):
    """Consensus voting fusion for redundant sensors."""
    
    def __init__(self, agreement_threshold: float = 0.8, outlier_rejection: bool = True):
        """Initialize consensus voting.
        
        Args:
            agreement_threshold: Minimum agreement ratio [0.0, 1.0]
            outlier_rejection: Enable automatic outlier rejection
        """
        
    def update(self, measurements: List[float], timestamp: Timestamp) -> Tuple[float, float]:
        """Perform consensus voting on measurements.
        
        Algorithm:
        1. Calculate pairwise agreements between sensors
        2. Reject outliers if enabled
        3. Compute consensus value from agreeing sensors
        4. Calculate confidence based on agreement level
        
        Args:
            measurements: Sensor readings (3 or more recommended)
            timestamp: Measurement timestamp
            
        Returns:
            Tuple of (consensus_value, confidence_score)
        """
        
    def reset(self) -> None:
        """Reset voting state."""
        
    @property
    def last_outliers(self) -> List[int]:
        """Indices of sensors rejected as outliers in last update."""
        
    @property
    def agreement_threshold(self) -> float:
        """Current agreement threshold."""
```

### Usage Examples

```python
# Redundant temperature sensors with fault detection
consensus = eg.ConsensusVoting(agreement_threshold=0.75)

# Multiple sensors measuring same phenomenon
temp_sensors = [22.1, 22.0, 22.2, 21.9, 25.5]  # Last one is faulty

consensus_temp, confidence = consensus.update(temp_sensors, eg.Timestamp.now())
outliers = consensus.last_outliers

print(f"Consensus temperature: {consensus_temp:.2f}°C")
print(f"Confidence: {confidence:.1%}")
print(f"Outlier sensors: {outliers}")  # Should identify sensor 4 (25.5°C)

# Fault-tolerant pressure measurement
pressure_consensus = eg.ConsensusVoting(agreement_threshold=0.8, outlier_rejection=True)

# Simulate sensor failure
pressure_readings = [1013.2, 1013.1, 1013.3, 999.9, 1013.0]  # Sensor 3 failed

consensus_pressure, conf = pressure_consensus.update(pressure_readings, eg.Timestamp.now())
failed_sensors = pressure_consensus.last_outliers

if failed_sensors:
    print(f"Sensor failure detected: sensors {failed_sensors}")
    print(f"Consensus pressure: {consensus_pressure:.1f} hPa (confidence: {conf:.1%})")
```

## Fusion Pipeline Integration

### FusionStage

Pipeline stage for multi-sensor fusion:

```python
class FusionStage:
    """Pipeline stage that performs multi-sensor fusion."""
    
    def __init__(self, algorithm: FusionAlgorithm, sensor_group: str = "default"):
        """Create fusion stage.
        
        Args:
            algorithm: Fusion algorithm to use
            sensor_group: Group name for sensors to fuse
        """
        
    def add_sensor(self, sensor_id: str, sensor_type: SensorType) -> None:
        """Add sensor to fusion group.
        
        Args:
            sensor_id: Unique sensor identifier
            sensor_type: Type of sensor
        """
        
    def process(self, event: Event) -> List[Event]:
        """Process incoming sensor events.
        
        Collects sensor readings and performs fusion when sufficient
        data is available.
        
        Args:
            event: Incoming sensor event
            
        Returns:
            List of fusion result events
        """
```

### Usage in Pipeline

```python
# Create fusion-enabled pipeline
pipeline = eg.Pipeline(buffer_size=512)

# Add validation stages
temp_validator = eg.TemperatureValidator().with_range(-20.0, 60.0)
pipeline.add_validator(temp_validator, eg.SensorType.Temperature)

# Add fusion stage
kalman_config = eg.KalmanConfig.for_temperature()
fusion_stage = eg.FusionStage(eg.KalmanFilter(kalman_config), "temperature_group")

# Add temperature sensors to fusion group
fusion_stage.add_sensor("temp_001", eg.SensorType.Temperature)
fusion_stage.add_sensor("temp_002", eg.SensorType.Temperature)
fusion_stage.add_sensor("temp_003", eg.SensorType.Temperature)

pipeline.add_stage(fusion_stage)

# Process sensor readings
for sensor_data in data_stream:
    reading = eg.SensorReading(
        sensor_data['id'], 
        eg.SensorType.Temperature, 
        sensor_data['value']
    )
    
    pipeline.push_event(reading)
    
    # Process and get fusion results
    results = pipeline.process_batch(10)
    for result in results:
        if hasattr(result, 'fused_value'):
            print(f"Fused result: {result.fused_value:.2f}°C (confidence: {result.confidence:.1%})")
```

## Advanced Topics

### Multi-Type Fusion

Fuse different sensor types for comprehensive environmental monitoring:

```python
class EnvironmentalFusion:
    """Multi-type environmental sensor fusion."""
    
    def __init__(self):
        self.temp_fusion = eg.KalmanFilter(eg.KalmanConfig.for_temperature())
        self.humidity_fusion = eg.WeightedAverageFusion()
        self.pressure_fusion = eg.ComplementaryFilter(alpha=0.95)
        
    def update(self, temp_readings: List[float], 
               humidity_readings: List[float],
               pressure_readings: List[float],
               timestamp: eg.Timestamp) -> dict:
        """Fuse multi-type environmental data."""
        
        # Individual fusion
        temp, temp_conf = self.temp_fusion.update(temp_readings, timestamp)
        humidity, humidity_conf = self.humidity_fusion.update(humidity_readings, timestamp)
        pressure, pressure_conf = self.pressure_fusion.update(pressure_readings, timestamp)
        
        # Cross-validation using physics
        dew_point = eg.calculate_dew_point(temp, humidity)
        altitude = eg.pressure_to_altitude(pressure)
        
        # Combined confidence
        overall_conf = min(temp_conf, humidity_conf, pressure_conf)
        
        return {
            'temperature': temp,
            'humidity': humidity,
            'pressure': pressure,
            'dew_point': dew_point,
            'altitude': altitude,
            'confidence': overall_conf,
            'physics_valid': dew_point <= temp  # Physics check
        }

# Use multi-type fusion
env_fusion = EnvironmentalFusion()

results = env_fusion.update(
    temp_readings=[22.1, 22.0, 22.2],
    humidity_readings=[65.0, 64.5],
    pressure_readings=[1013.2, 1013.0],
    timestamp=eg.Timestamp.now()
)

print(f"Environmental fusion results:")
print(f"  Temperature: {results['temperature']:.1f}°C")
print(f"  Humidity: {results['humidity']:.1f}% RH")
print(f"  Pressure: {results['pressure']:.1f} hPa")
print(f"  Dew Point: {results['dew_point']:.1f}°C")
print(f"  Confidence: {results['confidence']:.1%}")
```

### Adaptive Fusion

Dynamically adjust fusion parameters based on sensor performance:

```python
class AdaptiveFusion:
    """Adaptive fusion that adjusts to sensor conditions."""
    
    def __init__(self):
        self.algorithms = {
            'kalman': eg.KalmanFilter(eg.KalmanConfig.for_temperature()),
            'weighted': eg.WeightedAverageFusion(),
            'consensus': eg.ConsensusVoting()
        }
        self.current_algorithm = 'kalman'
        self.performance_history = []
        
    def update(self, measurements: List[float], timestamp: eg.Timestamp) -> Tuple[float, float]:
        """Adaptive fusion update."""
        
        # Try current algorithm
        algorithm = self.algorithms[self.current_algorithm]
        fused, confidence = algorithm.update(measurements, timestamp)
        
        # Track performance
        self.performance_history.append({
            'algorithm': self.current_algorithm,
            'confidence': confidence,
            'measurement_spread': max(measurements) - min(measurements)
        })
        
        # Adapt algorithm choice based on conditions
        if len(self.performance_history) > 10:
            recent_performance = self.performance_history[-10:]
            avg_confidence = sum(p['confidence'] for p in recent_performance) / 10
            avg_spread = sum(p['measurement_spread'] for p in recent_performance) / 10
            
            # Switch algorithms based on conditions
            if avg_confidence < 0.7 and avg_spread > 2.0:
                # High disagreement - use consensus voting
                self.current_algorithm = 'consensus'
            elif avg_spread < 0.5:
                # Good agreement - use weighted average
                self.current_algorithm = 'weighted'
            else:
                # Normal conditions - use Kalman
                self.current_algorithm = 'kalman'
        
        return fused, confidence

# Use adaptive fusion
adaptive = AdaptiveFusion()

for measurements in sensor_data_stream:
    fused, conf = adaptive.update(measurements, eg.Timestamp.now())
    print(f"Adaptive fusion ({adaptive.current_algorithm}): {fused:.2f} (conf: {conf:.1%})")
```

For integration with validation pipelines, see [Pipeline API](pipeline.md).
For sensor event handling, see [Events API](events.md).