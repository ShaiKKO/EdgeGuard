# Python API Reference

EdgeGuard Python bindings provide high-performance sensor validation with physics-aware constraints. Built on PyO3 for near-native performance while maintaining Python's ease of use.

## Installation

```bash
# Install from PyPI (when published)
pip install edgeguard

# Install from source
pip install maturin
maturin build --release
pip install target/wheels/edgeguard-*.whl
```

## Quick Start

```python
import edgeguard as eg

# Create temperature validator with physics constraints
validator = eg.TemperatureValidator() \
    .with_range(-20.0, 60.0) \
    .with_rate_limit(5.0)

# Validate sensor reading
try:
    valid_temp = validator.validate(23.5)
    print(f"Valid temperature: {valid_temp}°C")
except eg.ValidationError as e:
    print(f"Validation failed: {e}")
```

## Core Concepts

### Physics-Aware Validation

EdgeGuard applies real physics constraints to sensor validation:

```python
# Temperature validation with thermal mass
temp_validator = eg.TemperatureValidator() \
    .with_range(-40.0, 85.0) \
    .with_rate_limit(2.0) \
    .with_thermal_mass(0.5)  # 0.5kg thermal mass

# Humidity validation with dew point calculations
humidity_validator = eg.HumidityValidator() \
    .with_range(0.0, 100.0) \
    .with_rate_limit(10.0)

# Pressure validation with altitude compensation
pressure_validator = eg.PressureValidator() \
    .with_range(300.0, 1100.0) \
    .with_altitude(1000.0)  # 1000m elevation
```

### Event Processing Pipeline

Process sensor data through configurable validation stages:

```python
# Create pipeline with validation stages
pipeline = eg.Pipeline(buffer_size=256) \
    .add_validator(temp_validator, eg.SensorType.Temperature) \
    .add_cross_validation() \
    .add_fusion_stage()

# Process sensor readings
timestamp = eg.Timestamp.now()
reading = eg.SensorReading("temp_001", eg.SensorType.Temperature, 23.5, 0.95)
event = eg.Event.sensor_reading(reading, timestamp)

pipeline.push_event(event)
results = pipeline.process_batch(max_events=10)

for result in results:
    if isinstance(result, eg.ValidationResult):
        print(f"Validation: {result.status}")
```

### Multi-Sensor Fusion

Combine multiple sensors for improved accuracy:

```python
# Kalman filter configuration
config = eg.KalmanConfig(
    initial_state=[20.0],
    initial_covariance=[[1.0]],
    process_noise=[[0.1]],
    measurement_noise=[[0.5]]
)

# Create fusion algorithm
kalman = eg.KalmanFilter(config)

# Fuse sensor measurements
measurements = [20.1, 19.9, 20.2]
timestamp = eg.Timestamp.now()
fused_value, confidence = kalman.update(measurements, timestamp)

print(f"Fused: {fused_value} (confidence: {confidence})")
```

## API Components

### Validators
Physics-aware validation for sensor data.

- **[TemperatureValidator](validators.md#temperaturevalidator)** - Temperature validation with thermal mass modeling
- **[HumidityValidator](validators.md#humidityvalidator)** - Humidity validation with dew point calculations  
- **[PressureValidator](validators.md#pressurevalidator)** - Pressure validation with altitude compensation

### Event System
Unified event handling for sensor data processing.

- **[Event](events.md#event)** - Core event types for sensor readings and validation results
- **[SensorType](events.md#sensortype)** - Enumeration of supported sensor types
- **[ValidationStatus](events.md#validationstatus)** - Validation result status codes

### Pipeline
Event processing pipeline with composable stages.

- **[Pipeline](pipeline.md#pipeline)** - Main pipeline orchestrator
- **[ValidationStage](pipeline.md#validationstage)** - Validation pipeline stage
- **[FusionStage](pipeline.md#fusionstage)** - Multi-sensor fusion stage
- **[CrossValidationStage](pipeline.md#crossvalidationstage)** - Cross-sensor validation

### Fusion Algorithms
Multi-sensor data fusion with confidence scoring.

- **[KalmanFilter](fusion.md#kalmanfilter)** - Kalman filter implementation
- **[WeightedAverageFusion](fusion.md#weightedaveragefusion)** - Weighted average fusion
- **[ComplementaryFilter](fusion.md#complementaryfilter)** - Complementary filter

### Time Management
Time sources and timestamp handling.

- **[Timestamp](time.md#timestamp)** - High-precision timestamp with Python datetime integration
- **[SystemTime](time.md#systemtime)** - System wall clock time source
- **[MonotonicTime](time.md#monotonictime)** - Monotonic time source

### Helper Functions
Convenience functions for common operations.

- **[quick_validate()](helpers.md#quick-validate)** - Quick validation with automatic validator creation
- **[create_reading()](helpers.md#create-reading)** - Create sensor reading events
- **[get_sensor_ranges()](helpers.md#get-sensor-ranges)** - Get physics-based sensor ranges

### Type Conversion
Seamless conversion between Python and EdgeGuard types.

- **[ConversionUtils](conversion.md#conversionutils)** - Type conversion utilities
- **[parse_sensor_type()](conversion.md#parse-sensor-type)** - Parse sensor types from strings
- **[validate_sensor_config()](conversion.md#validate-sensor-config)** - Validate sensor configurations

## Error Handling

EdgeGuard provides comprehensive error handling with detailed error information:

```python
try:
    result = validator.validate(temperature)
except eg.ValidationError as e:
    print(f"Validation failed: {e.message}")
    print(f"Error code: {e.code}")
    if e.violation_type == "range":
        print(f"Value {e.actual_value} outside range {e.expected_range}")
    elif e.violation_type == "rate_of_change":
        print(f"Rate {e.actual_value} exceeds maximum {e.expected_range[1]}")

except eg.PhysicsViolationError as e:
    print(f"Physics violation: {e.message}")
    print(f"Constraint: {e.constraint}")

except eg.EdgeGuardError as e:
    print(f"General error: {e.message}")
```

## Performance

EdgeGuard Python bindings provide near-native performance:

| Operation | Performance | Memory |
|-----------|-------------|---------|
| Single validation | <10μs | Minimal |
| Pipeline processing | 100k+ events/sec | <10MB |
| Kalman filter update | <100μs | ~2KB |

### Performance Tips

```python
# Pre-create validators for better performance
validator = eg.TemperatureValidator().with_range(-20.0, 60.0)

# Process events in batches
results = pipeline.process_batch(max_events=100)

# Use appropriate buffer sizes
pipeline = eg.Pipeline(buffer_size=512)  # Larger for higher throughput

# Monitor pipeline metrics
metrics = pipeline.get_metrics()
if metrics.events_dropped > 0:
    print(f"Pipeline dropping events: {metrics.events_dropped}")
```

## Integration Examples

### MQTT Integration

```python
import paho.mqtt.client as mqtt

# Create MQTT client
client = mqtt.Client()

# Setup EdgeGuard pipeline
pipeline = eg.Pipeline(256) \
    .add_validator(eg.TemperatureValidator(), eg.SensorType.Temperature)

def on_message(client, userdata, msg):
    # Parse sensor data from MQTT
    data = json.loads(msg.payload.decode())
    
    # Create EdgeGuard event
    reading = eg.SensorReading(
        data['sensor_id'],
        eg.parse_sensor_type(data['type']),
        data['value'],
        data['quality']
    )
    
    # Validate through pipeline
    event = eg.Event.sensor_reading(reading, eg.Timestamp.now())
    pipeline.push_event(event)
    
    # Process validation results
    for result in pipeline.process_batch(10):
        if result.status == eg.ValidationStatus.Valid:
            print(f"Valid reading: {result.value}")

client.on_message = on_message
client.connect("localhost", 1883, 60)
client.subscribe("sensors/+/data")
client.loop_forever()
```

### Pandas Integration

```python
import pandas as pd

# Load sensor data
df = pd.read_csv('sensor_data.csv')

# Create validator
validator = eg.TemperatureValidator().with_range(-20.0, 60.0)

# Validate all readings
valid_mask = []
for temp in df['temperature']:
    try:
        validator.validate(temp)
        valid_mask.append(True)
    except eg.ValidationError:
        valid_mask.append(False)

# Filter to valid readings only
valid_df = df[valid_mask]
print(f"Valid readings: {len(valid_df)}/{len(df)}")
```

### Asyncio Integration

```python
import asyncio

async def process_sensor_stream():
    pipeline = eg.Pipeline(256)
    
    while True:
        # Simulate sensor reading
        reading = await get_sensor_reading()
        
        # Create event
        event = eg.Event.sensor_reading(
            eg.SensorReading("sensor_001", eg.SensorType.Temperature, reading, 0.95),
            eg.Timestamp.now()
        )
        
        # Process through pipeline
        pipeline.push_event(event)
        results = pipeline.process_batch(1)
        
        for result in results:
            print(f"Processed: {result}")
        
        await asyncio.sleep(0.1)  # 10Hz sampling

# Run sensor processing
asyncio.run(process_sensor_stream())
```

## Common Patterns

### Sensor Configuration

```python
# Configure validators based on sensor specifications
sensors = {
    'BME280': {
        'temperature': eg.TemperatureValidator().with_range(-40.0, 85.0).with_rate_limit(2.0),
        'humidity': eg.HumidityValidator().with_range(0.0, 100.0).with_rate_limit(5.0),
        'pressure': eg.PressureValidator().with_range(300.0, 1100.0).with_rate_limit(10.0)
    },
    'DHT22': {
        'temperature': eg.TemperatureValidator().with_range(-40.0, 80.0).with_rate_limit(1.0),
        'humidity': eg.HumidityValidator().with_range(0.0, 100.0).with_rate_limit(2.0)
    }
}
```

### Error Recovery

```python
def robust_validation(validator, value):
    try:
        return validator.validate(value)
    except eg.RateExceededError:
        # Rate exceeded - may be transient
        print("Rate limit exceeded, retrying...")
        time.sleep(0.1)
        return validator.validate(value)
    except eg.OutOfRangeError as e:
        # Out of range - log and skip
        print(f"Value {e.actual_value} out of range, skipping")
        return None
    except eg.PhysicsViolationError as e:
        # Physics violation - serious error
        print(f"Physics violation: {e.constraint}")
        raise
```

### Batch Processing

```python
def process_sensor_batch(readings):
    pipeline = eg.Pipeline(len(readings))
    validator = eg.TemperatureValidator().with_range(-20.0, 60.0)
    pipeline.add_validator(validator, eg.SensorType.Temperature)
    
    # Add all readings to pipeline
    for i, reading in enumerate(readings):
        event = eg.Event.sensor_reading(
            eg.SensorReading(f"sensor_{i}", eg.SensorType.Temperature, reading, 0.95),
            eg.Timestamp.now()
        )
        pipeline.push_event(event)
    
    # Process all at once
    results = pipeline.process_batch(len(readings))
    
    # Extract valid readings
    valid_readings = []
    for result in results:
        if result.status == eg.ValidationStatus.Valid:
            valid_readings.append(result.value)
    
    return valid_readings
```

For detailed API documentation, see the individual component pages:
- [Validators API](validators.md) - Physics-aware validation with Python interfaces
- [Events API](events.md) - Event system, sensor types, and validation status
- [Pipeline API](pipeline.md) - Event processing pipeline with composable stages
- [Fusion API](fusion.md) - Multi-sensor data fusion algorithms
- [Time API](time.md) - High-precision time management and datetime integration
- [Helpers API](helpers.md) - Convenience functions and utilities
- [Conversion API](conversion.md) - Type conversion and data format utilities