# Python Helpers API

Convenience functions and utilities for common EdgeGuard operations.

## Overview

EdgeGuard's helper functions provide:
- **Quick Access**: Simplified APIs for common operations
- **Smart Defaults**: Sensible default configurations for typical use cases
- **Type Safety**: Automatic type conversion and validation
- **Performance**: Optimized implementations for frequent operations
- **Integration**: Seamless integration with external data sources

## Module-Level Functions

### quick_validate()

Validate sensor readings with automatic validator creation:

```python
def quick_validate(sensor_type: Union[str, SensorType], 
                  value: float,
                  quality: Optional[float] = None,
                  **kwargs) -> ValidationResult:
    """Quick validation with automatic validator creation.
    
    Args:
        sensor_type: Sensor type (string or SensorType enum)
        value: Sensor reading to validate
        quality: Reading quality [0.0, 1.0] (optional)
        **kwargs: Additional validator configuration
        
    Returns:
        ValidationResult with validation outcome
        
    Raises:
        ValidationError: If validation fails
        
    Example:
        result = eg.quick_validate("temperature", 23.5, quality=0.95)
        if result.is_valid:
            print(f"Valid temperature: {result.value}°C")
    """
```

### Usage Examples

```python
import edgeguard as eg

# Basic validation with string sensor types
temp_result = eg.quick_validate("temperature", 23.5)
humidity_result = eg.quick_validate("humidity", 65.0)
pressure_result = eg.quick_validate("pressure", 1013.2)

print(f"Temperature valid: {temp_result.is_valid}")
print(f"Humidity valid: {humidity_result.is_valid}")
print(f"Pressure valid: {pressure_result.is_valid}")

# Validation with custom parameters
industrial_temp = eg.quick_validate(
    "temperature", 
    150.0,
    min_range=-200.0,
    max_range=500.0,
    rate_limit=1.0
)

# Validation with quality scoring
noisy_reading = eg.quick_validate(
    eg.SensorType.Temperature,
    23.5,
    quality=0.7  # Lower quality reading
)

# Batch validation
readings = [
    ("temperature", 23.1),
    ("temperature", 23.5),
    ("humidity", 65.0),
    ("pressure", 1013.2)
]

results = []
for sensor_type, value in readings:
    try:
        result = eg.quick_validate(sensor_type, value)
        results.append(result)
        print(f"✓ {sensor_type}: {value}")
    except eg.ValidationError as e:
        print(f"✗ {sensor_type}: {value} - {e}")
```

### create_reading()

Create sensor reading events with automatic type conversion:

```python
def create_reading(sensor_id: str,
                  sensor_type: Union[str, SensorType],
                  value: float,
                  quality: float = 1.0,
                  timestamp: Optional[Timestamp] = None) -> SensorReading:
    """Create sensor reading event with type conversion.
    
    Args:
        sensor_id: Unique sensor identifier
        sensor_type: Sensor type (string or enum)
        value: Sensor reading value
        quality: Reading quality [0.0, 1.0]
        timestamp: Reading timestamp (auto-generated if None)
        
    Returns:
        SensorReading event ready for pipeline processing
        
    Example:
        reading = eg.create_reading("temp_001", "temperature", 23.5, 0.95)
        pipeline.push_event(reading)
    """
```

### Usage Examples

```python
# Simple reading creation
temp_reading = eg.create_reading("temp_001", "temperature", 23.5)
humidity_reading = eg.create_reading("humidity_01", "humidity", 65.0, 0.98)

# With explicit timestamp
timestamp = eg.Timestamp.now()
pressure_reading = eg.create_reading(
    "pressure_01", 
    eg.SensorType.Pressure, 
    1013.2, 
    0.95,
    timestamp
)

# Batch reading creation
sensor_data = [
    {"id": "temp_001", "type": "temperature", "value": 23.1, "quality": 0.95},
    {"id": "temp_002", "type": "temperature", "value": 23.3, "quality": 0.97},
    {"id": "humidity_01", "type": "humidity", "value": 65.0, "quality": 0.92}
]

readings = []
for data in sensor_data:
    reading = eg.create_reading(
        data["id"],
        data["type"],
        data["value"],
        data["quality"]
    )
    readings.append(reading)

print(f"Created {len(readings)} sensor readings")
```

### get_sensor_ranges()

Get physics-based sensor ranges and constraints:

```python
def get_sensor_ranges(sensor_type: Union[str, SensorType]) -> Dict[str, Any]:
    """Get physics-based ranges and constraints for sensor type.
    
    Args:
        sensor_type: Sensor type to query
        
    Returns:
        Dictionary with sensor specifications:
        - physical_range: Theoretical physical limits
        - typical_range: Common operating range
        - default_rate_limit: Default rate of change limit
        - units: Standard units
        - description: Sensor type description
        
    Example:
        specs = eg.get_sensor_ranges("temperature")
        print(f"Temperature range: {specs['typical_range']} {specs['units']}")
    """
```

### Usage Examples

```python
# Get temperature sensor specifications
temp_specs = eg.get_sensor_ranges("temperature")
print(f"Temperature specifications:")
print(f"  Physical range: {temp_specs['physical_range']} {temp_specs['units']}")
print(f"  Typical range: {temp_specs['typical_range']} {temp_specs['units']}")
print(f"  Rate limit: {temp_specs['default_rate_limit']} {temp_specs['units']}/s")
print(f"  Description: {temp_specs['description']}")

# Configure validator using sensor specs
def create_optimized_validator(sensor_type: str):
    """Create validator with optimized settings."""
    specs = eg.get_sensor_ranges(sensor_type)
    
    if sensor_type == "temperature":
        return eg.TemperatureValidator() \
            .with_range(*specs['typical_range']) \
            .with_rate_limit(specs['default_rate_limit'])
    elif sensor_type == "humidity":
        return eg.HumidityValidator() \
            .with_range(*specs['typical_range']) \
            .with_rate_limit(specs['default_rate_limit'])
    elif sensor_type == "pressure":
        return eg.PressureValidator() \
            .with_range(*specs['typical_range']) \
            .with_rate_limit(specs['default_rate_limit'])

# Create validators for multiple sensor types
sensor_types = ["temperature", "humidity", "pressure"]
validators = {}

for sensor_type in sensor_types:
    validators[sensor_type] = create_optimized_validator(sensor_type)
    specs = eg.get_sensor_ranges(sensor_type)
    print(f"Created {sensor_type} validator: {specs['typical_range']} {specs['units']}")
```

## EdgeGuardHelpers Class

Stateful helper class for complex operations:

### Class Definition

```python
class EdgeGuardHelpers:
    """Helper class for common EdgeGuard operations with state management."""
    
    def __init__(self):
        """Initialize helpers with default configurations."""
        self._validator_cache = {}
        self._fusion_cache = {}
        self._default_configs = self._load_default_configs()
        
    def validate_reading(self, sensor_type: Union[str, SensorType], 
                        value: float, **kwargs) -> ValidationResult:
        """Validate reading with cached validator."""
        
    def create_pipeline(self, sensor_types: List[Union[str, SensorType]], 
                       buffer_size: int = 256,
                       enable_fusion: bool = False,
                       enable_cross_validation: bool = False) -> Pipeline:
        """Create pre-configured pipeline for sensor types."""
        
    def batch_validate(self, readings: List[Tuple[str, str, float]]) -> List[ValidationResult]:
        """Validate batch of readings efficiently."""
        
    def process_csv_file(self, filepath: str, 
                        sensor_id_col: str = "sensor_id",
                        sensor_type_col: str = "sensor_type", 
                        value_col: str = "value",
                        timestamp_col: Optional[str] = None) -> Dict[str, Any]:
        """Process CSV file through EdgeGuard validation."""
        
    def calculate_statistics(self, readings: List[SensorReading]) -> Dict[str, float]:
        """Calculate comprehensive statistics for sensor readings."""
```

### Usage Examples

```python
# Create helper instance
helpers = eg.EdgeGuardHelpers()

# Validate readings with caching (faster for repeated validations)
result1 = helpers.validate_reading("temperature", 23.5)
result2 = helpers.validate_reading("temperature", 24.1)  # Uses cached validator

# Create pre-configured pipeline
environmental_pipeline = helpers.create_pipeline(
    sensor_types=["temperature", "humidity", "pressure"],
    buffer_size=512,
    enable_fusion=True,
    enable_cross_validation=True
)

print(f"Created pipeline with {environmental_pipeline.stage_count} stages")

# Batch validation (efficient for large datasets)
readings_data = [
    ("temp_001", "temperature", 23.1),
    ("temp_002", "temperature", 23.5),
    ("humidity_01", "humidity", 65.0),
    ("pressure_01", "pressure", 1013.2),
    ("temp_003", "temperature", 99.9)  # Invalid reading
]

validation_results = helpers.batch_validate(readings_data)
valid_count = sum(1 for result in validation_results if result.is_valid)
print(f"Validated {len(validation_results)} readings, {valid_count} valid")

# Process CSV file
csv_results = helpers.process_csv_file(
    "sensor_data.csv",
    sensor_id_col="id",
    sensor_type_col="type",
    value_col="reading",
    timestamp_col="timestamp"
)

print(f"CSV Processing Results:")
print(f"  Total readings: {csv_results['total_readings']}")
print(f"  Valid readings: {csv_results['valid_readings']}")
print(f"  Validation rate: {csv_results['validation_rate']:.1%}")
print(f"  Processing time: {csv_results['processing_time_ms']:.1f}ms")
```

## Physics Calculation Functions

### calculate_dew_point()

Calculate dew point from temperature and humidity:

```python
def calculate_dew_point(temperature: float, humidity: float) -> float:
    """Calculate dew point from temperature and relative humidity.
    
    Uses Magnus formula for accurate dew point calculation:
    Td = (b * γ) / (a - γ)
    where γ = ln(RH/100) + (a * T)/(b + T)
    
    Args:
        temperature: Air temperature (°C)
        humidity: Relative humidity (% RH)
        
    Returns:
        Dew point temperature (°C)
        
    Raises:
        ValueError: If humidity not in range [0, 100] or invalid temperature
        
    Example:
        dp = eg.calculate_dew_point(25.0, 60.0)  # ~16.7°C
    """
```

### pressure_to_altitude()

Convert pressure to altitude using barometric formula:

```python
def pressure_to_altitude(pressure_hpa: float, 
                        sea_level_pressure: float = 1013.25) -> float:
    """Convert atmospheric pressure to altitude.
    
    Uses international barometric formula:
    h = (T₀/L) * [(P₀/P)^(R*L/g*M) - 1]
    
    Args:
        pressure_hpa: Atmospheric pressure (hPa)
        sea_level_pressure: Reference sea level pressure (hPa)
        
    Returns:
        Altitude above sea level (meters)
        
    Example:
        altitude = eg.pressure_to_altitude(900.0)  # ~1000m
    """
```

### altitude_to_pressure()

Convert altitude to expected pressure:

```python
def altitude_to_pressure(altitude_m: float,
                        sea_level_pressure: float = 1013.25) -> float:
    """Convert altitude to expected atmospheric pressure.
    
    Args:
        altitude_m: Altitude above sea level (meters)
        sea_level_pressure: Reference sea level pressure (hPa)
        
    Returns:
        Expected atmospheric pressure (hPa)
        
    Example:
        pressure = eg.altitude_to_pressure(1500.0)  # ~845 hPa
    """
```

### Usage Examples

```python
# Dew point calculations for different conditions
conditions = [
    (25.0, 60.0),  # Comfortable room
    (30.0, 80.0),  # Hot humid day
    (15.0, 40.0),  # Cool dry day
    (5.0, 90.0)    # Cold humid morning
]

for temp, humidity in conditions:
    dew_point = eg.calculate_dew_point(temp, humidity)
    print(f"T={temp}°C, RH={humidity}% → Dew point: {dew_point:.1f}°C")

# Pressure-altitude conversions
altitudes = [0, 500, 1000, 1500, 3000]  # meters
for alt in altitudes:
    pressure = eg.altitude_to_pressure(alt)
    calculated_alt = eg.pressure_to_altitude(pressure)
    print(f"Altitude: {alt}m → Pressure: {pressure:.1f} hPa → Back: {calculated_alt:.0f}m")

# Cross-validation using physics
def validate_environmental_reading(temp: float, humidity: float, pressure: float, altitude: float = 0):
    """Validate environmental reading using physics relationships."""
    issues = []
    
    # Check dew point physics
    dew_point = eg.calculate_dew_point(temp, humidity)
    if dew_point > temp:
        issues.append(f"Dew point {dew_point:.1f}°C exceeds air temperature {temp}°C")
    
    # Check pressure-altitude relationship
    expected_pressure = eg.altitude_to_pressure(altitude)
    pressure_diff = abs(pressure - expected_pressure)
    if pressure_diff > 20.0:  # Allow 20 hPa tolerance for weather variation
        issues.append(f"Pressure {pressure} hPa inconsistent with altitude {altitude}m (expected ~{expected_pressure:.1f} hPa)")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'dew_point': dew_point,
        'expected_pressure': expected_pressure
    }

# Test environmental validation
result = validate_environmental_reading(
    temp=22.0,
    humidity=65.0,
    pressure=1013.2,
    altitude=0  # Sea level
)

if result['valid']:
    print("✓ Environmental reading is physically consistent")
    print(f"  Dew point: {result['dew_point']:.1f}°C")
else:
    print("✗ Environmental reading has physics violations:")
    for issue in result['issues']:
        print(f"  - {issue}")
```

## Data Processing Helpers

### process_pandas_dataframe()

Process pandas DataFrame through EdgeGuard validation:

```python
def process_pandas_dataframe(df: 'pd.DataFrame',
                           sensor_id_col: str = 'sensor_id',
                           sensor_type_col: str = 'sensor_type',
                           value_col: str = 'value',
                           timestamp_col: Optional[str] = None,
                           quality_col: Optional[str] = None) -> 'pd.DataFrame':
    """Process pandas DataFrame through EdgeGuard validation.
    
    Args:
        df: Input DataFrame with sensor data
        sensor_id_col: Column name for sensor IDs
        sensor_type_col: Column name for sensor types
        value_col: Column name for sensor values
        timestamp_col: Column name for timestamps (optional)
        quality_col: Column name for quality scores (optional)
        
    Returns:
        DataFrame with additional columns:
        - validation_status: Validation outcome
        - validation_error: Error details (if validation failed)
        - is_valid: Boolean validation result
        
    Example:
        df_validated = eg.process_pandas_dataframe(
            df, 
            sensor_id_col='id',
            sensor_type_col='type', 
            value_col='reading'
        )
    """
```

### batch_process_files()

Process multiple files through EdgeGuard validation:

```python
def batch_process_files(file_paths: List[str],
                       output_dir: str,
                       file_format: str = 'csv',
                       **kwargs) -> Dict[str, Any]:
    """Process multiple sensor data files.
    
    Args:
        file_paths: List of input file paths
        output_dir: Directory for output files
        file_format: Input file format ('csv', 'json', 'parquet')
        **kwargs: Additional processing options
        
    Returns:
        Processing summary with statistics
        
    Example:
        results = eg.batch_process_files(
            ['data/day1.csv', 'data/day2.csv'],
            'validated_data/',
            file_format='csv'
        )
    """
```

### Usage Examples

```python
import pandas as pd

# Create sample DataFrame
data = {
    'sensor_id': ['temp_001', 'temp_002', 'humidity_01', 'pressure_01'] * 100,
    'sensor_type': ['temperature', 'temperature', 'humidity', 'pressure'] * 100,
    'value': [23.5, 23.1, 65.0, 1013.2] * 100,
    'timestamp': pd.date_range('2024-01-01', periods=400, freq='1min'),
    'quality': [0.95, 0.97, 0.92, 0.98] * 100
}

df = pd.DataFrame(data)

# Add some invalid readings
df.loc[50, 'value'] = 150.0  # Invalid temperature
df.loc[150, 'value'] = -10.0  # Invalid humidity

# Process through EdgeGuard
df_validated = eg.process_pandas_dataframe(df)

# Analyze results
print(f"Validation Results:")
print(f"  Total readings: {len(df_validated)}")
print(f"  Valid readings: {df_validated['is_valid'].sum()}")
print(f"  Invalid readings: {(~df_validated['is_valid']).sum()}")
print(f"  Validation rate: {df_validated['is_valid'].mean():.1%}")

# Filter to valid readings only
df_valid = df_validated[df_validated['is_valid']]

# Group by sensor type for analysis
for sensor_type in df_valid['sensor_type'].unique():
    type_data = df_valid[df_valid['sensor_type'] == sensor_type]
    mean_value = type_data['value'].mean()
    std_value = type_data['value'].std()
    print(f"{sensor_type}: μ={mean_value:.2f}, σ={std_value:.2f}")

# Export validated data
df_valid.to_csv('validated_sensor_data.csv', index=False)
print("Exported validated data to CSV")
```

## Performance Utilities

### benchmark_validation()

Benchmark validation performance:

```python
def benchmark_validation(sensor_type: Union[str, SensorType],
                        num_readings: int = 10000,
                        value_range: Tuple[float, float] = None) -> Dict[str, float]:
    """Benchmark validation performance.
    
    Args:
        sensor_type: Type of sensor to benchmark
        num_readings: Number of readings to validate
        value_range: Range of values to test (uses sensor defaults if None)
        
    Returns:
        Performance metrics:
        - total_time_ms: Total validation time
        - avg_time_per_reading_us: Average time per reading (microseconds)
        - throughput_hz: Validations per second
        - memory_usage_mb: Peak memory usage
        
    Example:
        perf = eg.benchmark_validation("temperature", 100000)
        print(f"Throughput: {perf['throughput_hz']:.0f} validations/sec")
    """
```

### profile_pipeline()

Profile pipeline performance:

```python
def profile_pipeline(pipeline: Pipeline,
                    events: List[Event],
                    iterations: int = 10) -> Dict[str, Any]:
    """Profile pipeline performance with detailed metrics.
    
    Args:
        pipeline: Pipeline to profile
        events: Test events to process
        iterations: Number of profiling iterations
        
    Returns:
        Detailed performance profile:
        - stage_timings: Per-stage processing times
        - memory_usage: Memory usage statistics
        - throughput_metrics: Overall throughput statistics
        - bottlenecks: Identified performance bottlenecks
        
    Example:
        profile = eg.profile_pipeline(pipeline, test_events, iterations=5)
        for stage, timing in profile['stage_timings'].items():
            print(f"{stage}: {timing:.2f}ms avg")
    """
```

### Usage Examples

```python
# Benchmark different validator types
sensor_types = ["temperature", "humidity", "pressure"]
benchmark_results = {}

for sensor_type in sensor_types:
    results = eg.benchmark_validation(sensor_type, num_readings=50000)
    benchmark_results[sensor_type] = results
    
    print(f"{sensor_type.capitalize()} Validation Performance:")
    print(f"  Throughput: {results['throughput_hz']:,.0f} validations/sec")
    print(f"  Avg time: {results['avg_time_per_reading_us']:.1f} μs/reading")
    print(f"  Memory: {results['memory_usage_mb']:.1f} MB")

# Find best performing validator
best_sensor = max(benchmark_results.keys(), 
                 key=lambda x: benchmark_results[x]['throughput_hz'])
print(f"\nBest performance: {best_sensor} ({benchmark_results[best_sensor]['throughput_hz']:,.0f} Hz)")

# Profile pipeline performance
pipeline = eg.Pipeline(1024)
# Add stages to pipeline...

test_events = [
    eg.SensorReading(f"temp_{i:03d}", eg.SensorType.Temperature, 20.0 + i * 0.1)
    for i in range(1000)
]

profile = eg.profile_pipeline(pipeline, test_events, iterations=10)

print("Pipeline Performance Profile:")
for stage_name, timing_ms in profile['stage_timings'].items():
    percentage = (timing_ms / sum(profile['stage_timings'].values())) * 100
    print(f"  {stage_name}: {timing_ms:.2f}ms ({percentage:.1f}%)")

if profile['bottlenecks']:
    print("\nIdentified bottlenecks:")
    for bottleneck in profile['bottlenecks']:
        print(f"  - {bottleneck}")
```

For detailed API documentation of individual components, see:
- [Validators API](validators.md) - Physics-aware validation
- [Events API](events.md) - Event system and types
- [Pipeline API](pipeline.md) - Pipeline processing
- [Fusion API](fusion.md) - Multi-sensor fusion