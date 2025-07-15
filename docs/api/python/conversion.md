# Python Conversion API

Type conversion utilities for seamless integration between Python and EdgeGuard types.

## Overview

EdgeGuard's conversion system provides:
- **Seamless Integration**: Convert between Python and EdgeGuard types
- **Fuzzy Matching**: Intelligent parsing of sensor type strings
- **Validation**: Ensure type safety during conversion
- **Performance**: Optimized for high-frequency operations
- **Flexibility**: Support multiple input formats and conventions

## Core Conversion Classes

### ConversionUtils

Main utility class for type conversions:

```python
class ConversionUtils:
    """Utilities for converting between Python and EdgeGuard types."""
    
    @staticmethod
    def parse_sensor_type(type_str: str) -> SensorType:
        """Parse sensor type from string with fuzzy matching.
        
        Supports various string formats:
        - Standard names: "temperature", "humidity", "pressure"
        - Abbreviations: "temp", "hum", "press"
        - Single letters: "t", "h", "p"
        - Case insensitive: "TEMPERATURE", "Temperature"
        
        Args:
            type_str: String representation of sensor type
            
        Returns:
            Corresponding SensorType enum value
            
        Raises:
            ValueError: If sensor type cannot be parsed
            
        Example:
            sensor_type = ConversionUtils.parse_sensor_type("temp")
            # Returns SensorType.Temperature
        """
        
    @staticmethod
    def sensor_type_to_string(sensor_type: SensorType, format: str = "full") -> str:
        """Convert SensorType to string representation.
        
        Args:
            sensor_type: SensorType enum value
            format: Output format ("full", "short", "abbrev", "letter")
            
        Returns:
            String representation of sensor type
            
        Example:
            name = ConversionUtils.sensor_type_to_string(
                SensorType.Temperature, "short"
            )  # Returns "temp"
        """
        
    @staticmethod
    def parse_validation_status(status_str: str) -> ValidationStatus:
        """Parse validation status from string.
        
        Args:
            status_str: String representation ("valid", "error", "out_of_range", etc.)
            
        Returns:
            Corresponding ValidationStatus enum value
        """
        
    @staticmethod
    def timestamp_from_various(timestamp_input: Any) -> Timestamp:
        """Convert various timestamp formats to EdgeGuard Timestamp.
        
        Supports:
        - Python datetime objects
        - Unix timestamps (int/float)
        - ISO 8601 strings
        - Pandas Timestamp objects
        
        Args:
            timestamp_input: Various timestamp formats
            
        Returns:
            EdgeGuard Timestamp object
        """
        
    @staticmethod
    def validate_sensor_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize sensor configuration.
        
        Args:
            config: Sensor configuration dictionary
            
        Returns:
            Validated and normalized configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
```

## Sensor Type Parsing

### parse_sensor_type()

Module-level function for convenient sensor type parsing:

```python
def parse_sensor_type(type_input: Union[str, int, SensorType]) -> SensorType:
    """Parse sensor type from various input formats.
    
    Args:
        type_input: Sensor type in various formats:
                   - String: "temperature", "temp", "t"
                   - Integer: 0 for temperature, 1 for humidity, etc.
                   - SensorType: Pass through unchanged
                   
    Returns:
        SensorType enum value
        
    Raises:
        ValueError: If type cannot be parsed
        
    Example:
        temp1 = parse_sensor_type("temperature")
        temp2 = parse_sensor_type("temp")
        temp3 = parse_sensor_type("t")
        temp4 = parse_sensor_type(0)
        # All return SensorType.Temperature
    """
```

### Usage Examples

```python
import edgeguard as eg

# Various ways to specify temperature sensor
temperature_inputs = [
    "temperature",
    "TEMPERATURE", 
    "temp",
    "Temp",
    "t",
    "T",
    0,
    eg.SensorType.Temperature
]

for input_val in temperature_inputs:
    sensor_type = eg.parse_sensor_type(input_val)
    print(f"'{input_val}' → {sensor_type}")

# Sensor type string mappings
sensor_mappings = {
    # Temperature variants
    "temperature": eg.SensorType.Temperature,
    "temp": eg.SensorType.Temperature,
    "t": eg.SensorType.Temperature,
    
    # Humidity variants
    "humidity": eg.SensorType.Humidity,
    "hum": eg.SensorType.Humidity,
    "rh": eg.SensorType.Humidity,
    "h": eg.SensorType.Humidity,
    
    # Pressure variants
    "pressure": eg.SensorType.Pressure,
    "press": eg.SensorType.Pressure,
    "p": eg.SensorType.Pressure,
    "bar": eg.SensorType.Pressure,
    "atm": eg.SensorType.Pressure,
    
    # Other sensor types
    "voc": eg.SensorType.Voc,
    "pm": eg.SensorType.Particulate,
    "pm2.5": eg.SensorType.Particulate,
    "sound": eg.SensorType.Acoustic,
    "noise": eg.SensorType.Acoustic,
    "vibration": eg.SensorType.Vibration,
    "vib": eg.SensorType.Vibration,
    "emf": eg.SensorType.Emf,
    "magnetic": eg.SensorType.Emf
}

# Test all mappings
for input_str, expected_type in sensor_mappings.items():
    parsed_type = eg.parse_sensor_type(input_str)
    assert parsed_type == expected_type
    print(f"✓ {input_str} → {parsed_type}")
```

## Data Format Conversion

### from_dict()

Convert dictionary data to EdgeGuard objects:

```python
def from_dict(data: Dict[str, Any], target_type: str) -> Any:
    """Convert dictionary to EdgeGuard object.
    
    Args:
        data: Dictionary with object data
        target_type: Target type ("SensorReading", "ValidationResult", etc.)
        
    Returns:
        EdgeGuard object of specified type
        
    Example:
        reading_data = {
            "sensor_id": "temp_001",
            "sensor_type": "temperature", 
            "value": 23.5,
            "quality": 0.95,
            "timestamp": "2024-01-15T10:30:00Z"
        }
        reading = from_dict(reading_data, "SensorReading")
    """
```

### to_dict()

Convert EdgeGuard objects to dictionaries:

```python
def to_dict(obj: Any, include_metadata: bool = True) -> Dict[str, Any]:
    """Convert EdgeGuard object to dictionary.
    
    Args:
        obj: EdgeGuard object (SensorReading, ValidationResult, etc.)
        include_metadata: Include metadata like timestamps, IDs
        
    Returns:
        Dictionary representation of object
        
    Example:
        reading = eg.SensorReading("temp_001", "temperature", 23.5)
        data = to_dict(reading)
        # Returns: {"sensor_id": "temp_001", "sensor_type": "temperature", ...}
    """
```

### Usage Examples

```python
# Convert from various data formats
json_data = {
    "sensor_id": "temp_001",
    "type": "temperature",
    "reading": 23.5,
    "timestamp": "2024-01-15T10:30:00Z",
    "quality": 0.95
}

# Convert to SensorReading
reading = eg.from_dict(json_data, "SensorReading")
print(f"Created reading: {reading.sensor_id} = {reading.value}")

# Convert back to dictionary
reading_dict = eg.to_dict(reading)
print(f"Dictionary: {reading_dict}")

# Batch conversion from JSON array
json_array = [
    {"sensor_id": "temp_001", "type": "temperature", "value": 23.5},
    {"sensor_id": "humidity_01", "type": "humidity", "value": 65.0},
    {"sensor_id": "pressure_01", "type": "pressure", "value": 1013.2}
]

readings = []
for item in json_array:
    reading = eg.from_dict(item, "SensorReading")
    readings.append(reading)

print(f"Converted {len(readings)} readings from JSON")

# Convert validation results
validation_data = {
    "sensor_id": "temp_001",
    "status": "valid",
    "constraints_applied": ["range", "rate_limit"],
    "original_value": 23.5,
    "validated_value": 23.5
}

validation_result = eg.from_dict(validation_data, "ValidationResult")
print(f"Validation: {validation_result.sensor_id} → {validation_result.status}")
```

## File Format Conversion

### read_csv()

Read sensor data from CSV files:

```python
def read_csv(filepath: str, 
            sensor_id_col: str = "sensor_id",
            sensor_type_col: str = "sensor_type",
            value_col: str = "value",
            timestamp_col: Optional[str] = None,
            quality_col: Optional[str] = None) -> List[SensorReading]:
    """Read sensor data from CSV file.
    
    Args:
        filepath: Path to CSV file
        sensor_id_col: Column name for sensor IDs
        sensor_type_col: Column name for sensor types
        value_col: Column name for sensor values
        timestamp_col: Column name for timestamps (optional)
        quality_col: Column name for quality scores (optional)
        
    Returns:
        List of SensorReading objects
        
    Example:
        readings = read_csv("data.csv", 
                           sensor_id_col="id",
                           sensor_type_col="type",
                           value_col="reading")
    """
```

### write_csv()

Write EdgeGuard objects to CSV files:

```python
def write_csv(objects: List[Any], 
             filepath: str,
             include_metadata: bool = True) -> None:
    """Write EdgeGuard objects to CSV file.
    
    Args:
        objects: List of EdgeGuard objects
        filepath: Output CSV file path
        include_metadata: Include timestamps and other metadata
        
    Example:
        results = [validation_result1, validation_result2, ...]
        write_csv(results, "validation_results.csv")
    """
```

### read_json()

Read sensor data from JSON files:

```python
def read_json(filepath: str) -> List[Any]:
    """Read EdgeGuard objects from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        List of EdgeGuard objects
        
    Example:
        events = read_json("sensor_events.json")
    """
```

### Usage Examples

```python
# Read CSV data with automatic type conversion
readings = eg.read_csv(
    "sensor_data.csv",
    sensor_id_col="device_id",
    sensor_type_col="measurement_type", 
    value_col="measured_value",
    timestamp_col="datetime",
    quality_col="signal_strength"
)

print(f"Loaded {len(readings)} readings from CSV")

# Process and validate readings
pipeline = eg.Pipeline(1024)
temp_validator = eg.TemperatureValidator().with_range(-40.0, 85.0)
pipeline.add_validator(temp_validator, eg.SensorType.Temperature)

for reading in readings:
    pipeline.push_event(reading)

results = pipeline.process_all()

# Write validation results to CSV
validation_results = [r for r in results if isinstance(r, eg.ValidationResult)]
eg.write_csv(validation_results, "validation_output.csv")

# JSON format conversion
sensor_events = [
    {"type": "SensorReading", "sensor_id": "temp_001", "value": 23.5},
    {"type": "ValidationResult", "sensor_id": "temp_001", "status": "valid"}
]

# Convert from JSON objects
events = []
for event_data in sensor_events:
    event = eg.from_dict(event_data, event_data["type"])
    events.append(event)

# Write back to JSON
eg.write_json(events, "processed_events.json")
```

## Configuration Conversion

### validate_sensor_config()

Validate and normalize sensor configurations:

```python
def validate_sensor_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize sensor configuration.
    
    Args:
        config: Raw configuration dictionary
        
    Returns:
        Validated configuration with normalized values
        
    Raises:
        ValueError: If configuration is invalid
        
    Example:
        raw_config = {
            "type": "temp",
            "range": [-20, 60],
            "rate_limit": "5.0",
            "id": "temp_001"
        }
        validated = validate_sensor_config(raw_config)
    """
```

### create_validator_from_config()

Create validators from configuration dictionaries:

```python
def create_validator_from_config(config: Dict[str, Any]) -> Validator:
    """Create validator from configuration dictionary.
    
    Args:
        config: Validator configuration
        
    Returns:
        Configured validator instance
        
    Example:
        config = {
            "type": "temperature",
            "min_range": -40.0,
            "max_range": 85.0,
            "rate_limit": 2.0,
            "thermal_mass": 0.1
        }
        validator = create_validator_from_config(config)
    """
```

### Usage Examples

```python
# Configuration validation and normalization
sensor_configs = [
    {
        "id": "temp_001",
        "type": "temp",  # Will be normalized to "temperature"
        "range": "(-20, 60)",  # Will be parsed to tuple
        "rate_limit": "5.0",  # Will be converted to float
        "location": "room_a"
    },
    {
        "id": "humidity_01", 
        "type": "RH",  # Will be normalized to "humidity"
        "range": [0, 100],  # Already a list
        "rate_limit": 10.0,  # Already a float
        "calibration_date": "2024-01-01"
    }
]

validators = {}
for config in sensor_configs:
    try:
        # Validate and normalize configuration
        validated_config = eg.validate_sensor_config(config)
        print(f"Validated config for {validated_config['id']}")
        
        # Create validator from config
        validator = eg.create_validator_from_config(validated_config)
        validators[validated_config['id']] = validator
        
    except ValueError as e:
        print(f"Invalid config for {config.get('id', 'unknown')}: {e}")

print(f"Created {len(validators)} validators from configuration")

# Configuration from YAML/JSON files
import yaml

# Load configuration from YAML
with open('sensor_config.yaml', 'r') as f:
    yaml_config = yaml.safe_load(f)

# Process sensor definitions
for sensor_def in yaml_config['sensors']:
    validated_config = eg.validate_sensor_config(sensor_def)
    validator = eg.create_validator_from_config(validated_config)
    print(f"Configured sensor: {validated_config['id']} ({validated_config['type']})")

# Create pipeline from configuration
pipeline_config = yaml_config.get('pipeline', {})
pipeline = eg.Pipeline(
    buffer_size=pipeline_config.get('buffer_size', 256)
)

# Add validators to pipeline
for sensor_id, validator in validators.items():
    sensor_type = eg.parse_sensor_type(sensor_configs[0]['type'])  # Get type from original config
    pipeline.add_validator(validator, sensor_type)

print(f"Pipeline configured with {pipeline.stage_count} stages")
```

## Pandas Integration

### from_pandas()

Convert pandas DataFrames to EdgeGuard objects:

```python
def from_pandas(df: 'pd.DataFrame',
               sensor_id_col: str = 'sensor_id',
               sensor_type_col: str = 'sensor_type', 
               value_col: str = 'value',
               timestamp_col: Optional[str] = None,
               quality_col: Optional[str] = None) -> List[SensorReading]:
    """Convert pandas DataFrame to SensorReading objects.
    
    Args:
        df: Input DataFrame
        sensor_id_col: Column containing sensor IDs
        sensor_type_col: Column containing sensor types
        value_col: Column containing sensor values
        timestamp_col: Column containing timestamps (optional)
        quality_col: Column containing quality scores (optional)
        
    Returns:
        List of SensorReading objects
        
    Example:
        import pandas as pd
        df = pd.read_csv('sensor_data.csv')
        readings = from_pandas(df)
    """
```

### to_pandas()

Convert EdgeGuard objects to pandas DataFrame:

```python
def to_pandas(objects: List[Any]) -> 'pd.DataFrame':
    """Convert EdgeGuard objects to pandas DataFrame.
    
    Args:
        objects: List of EdgeGuard objects
        
    Returns:
        DataFrame with object data
        
    Example:
        df = to_pandas(sensor_readings)
        df.to_csv('output.csv', index=False)
    """
```

### Usage Examples

```python
import pandas as pd

# Create sample DataFrame
data = {
    'device_id': ['temp_001', 'temp_002', 'hum_001'] * 100,
    'measurement_type': ['temperature', 'temperature', 'humidity'] * 100,
    'reading_value': [23.5, 23.1, 65.0] * 100,
    'data_quality': [0.95, 0.97, 0.92] * 100,
    'timestamp': pd.date_range('2024-01-01', periods=300, freq='1min')
}

df = pd.DataFrame(data)

# Convert to EdgeGuard objects
readings = eg.from_pandas(
    df,
    sensor_id_col='device_id',
    sensor_type_col='measurement_type',
    value_col='reading_value',
    quality_col='data_quality',
    timestamp_col='timestamp'
)

print(f"Converted {len(readings)} pandas rows to EdgeGuard readings")

# Process through pipeline
pipeline = eg.Pipeline(1024)
temp_validator = eg.TemperatureValidator().with_range(-20.0, 60.0)
humidity_validator = eg.HumidityValidator().with_range(0.0, 100.0)

pipeline.add_validator(temp_validator, eg.SensorType.Temperature)
pipeline.add_validator(humidity_validator, eg.SensorType.Humidity)

# Process all readings
for reading in readings:
    pipeline.push_event(reading)

results = pipeline.process_all()

# Convert results back to DataFrame
results_df = eg.to_pandas(results)

# Analyze validation results
validation_df = results_df[results_df['event_type'] == 'ValidationResult']
success_rate = (validation_df['status'] == 'Valid').mean()

print(f"Validation success rate: {success_rate:.1%}")

# Export results
validation_df.to_csv('validation_results.csv', index=False)
print("Results exported to CSV")
```

For detailed type documentation, see:
- [Events API](events.md) - Event types and structures
- [Validators API](validators.md) - Validator configuration options
- [Time API](time.md) - Timestamp handling and conversion