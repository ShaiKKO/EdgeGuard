# Python Events API

Event system for sensor data processing with unified event handling and type-safe sensor classification.

## Event System Overview

EdgeGuard processes all sensor data through a unified event system that provides:
- **Type Safety**: Strongly typed sensor classifications and validation statuses
- **Timestamp Precision**: Millisecond-precision timestamps for accurate ordering
- **Memory Efficiency**: Fixed-size data structures for predictable memory usage
- **Cross-Platform**: Consistent behavior across embedded and desktop platforms

## Core Event Types

### Event

The base event class representing all sensor data and validation results:

```python
class Event:
    """Base class for all EdgeGuard events."""
    
    @property
    def timestamp(self) -> Timestamp:
        """Event timestamp with millisecond precision."""
        
    @property
    def event_type(self) -> str:
        """String identifier for the event type."""
        
    def to_dict(self) -> dict:
        """Convert event to dictionary representation."""
        
    @classmethod
    def from_dict(cls, data: dict) -> 'Event':
        """Create event from dictionary representation."""
```

### SensorReading

Represents raw sensor data with quality information:

```python
class SensorReading(Event):
    """Raw sensor reading event."""
    
    def __init__(self, 
                 sensor_id: str,
                 sensor_type: SensorType,
                 value: float,
                 quality: float = 1.0,
                 timestamp: Optional[Timestamp] = None):
        """Create sensor reading event.
        
        Args:
            sensor_id: Unique sensor identifier (max 15 characters)
            sensor_type: Type of sensor (Temperature, Humidity, etc.)
            value: Sensor reading value
            quality: Reading quality [0.0, 1.0] where 1.0 is perfect
            timestamp: Reading timestamp (auto-generated if None)
        """
        
    @property
    def sensor_id(self) -> str:
        """Unique identifier for the sensor."""
        
    @property
    def sensor_type(self) -> SensorType:
        """Type classification of the sensor."""
        
    @property
    def value(self) -> float:
        """The sensor reading value."""
        
    @property
    def quality(self) -> float:
        """Quality score [0.0, 1.0] indicating reading reliability."""
```

### ValidationResult

Represents the result of physics-aware validation:

```python
class ValidationResult(Event):
    """Sensor validation result event."""
    
    def __init__(self,
                 sensor_id: str,
                 status: ValidationStatus,
                 constraints_applied: List[str],
                 original_value: Optional[float] = None,
                 validated_value: Optional[float] = None,
                 error_details: Optional[str] = None,
                 timestamp: Optional[Timestamp] = None):
        """Create validation result event.
        
        Args:
            sensor_id: Sensor identifier that was validated
            status: Validation outcome status
            constraints_applied: List of constraint types applied
            original_value: Original sensor reading
            validated_value: Validated value (if successful)
            error_details: Error description (if validation failed)
            timestamp: Validation timestamp
        """
        
    @property
    def sensor_id(self) -> str:
        """Identifier of the validated sensor."""
        
    @property
    def status(self) -> ValidationStatus:
        """Validation outcome status."""
        
    @property
    def is_valid(self) -> bool:
        """True if validation passed."""
        
    @property
    def constraints_applied(self) -> List[str]:
        """List of constraint types that were checked."""
        
    @property
    def error_details(self) -> Optional[str]:
        """Detailed error description if validation failed."""
```

### CrossValidationResult

Represents cross-sensor validation results:

```python
class CrossValidationResult(Event):
    """Cross-sensor validation result event."""
    
    def __init__(self,
                 primary_sensor: str,
                 related_sensor: str,
                 validation_type: str,
                 status: ValidationStatus,
                 details: Optional[dict] = None,
                 timestamp: Optional[Timestamp] = None):
        """Create cross-validation result.
        
        Args:
            primary_sensor: Primary sensor being validated
            related_sensor: Related sensor used for validation
            validation_type: Type of cross-validation performed
            status: Validation outcome
            details: Additional validation details
            timestamp: Validation timestamp
        """
        
    @property
    def primary_sensor(self) -> str:
        """Primary sensor identifier."""
        
    @property
    def related_sensor(self) -> str:
        """Related sensor identifier."""
        
    @property
    def validation_type(self) -> str:
        """Type of cross-validation performed."""
```

## Sensor Types

### SensorType Enumeration

Physics-aware sensor classification with built-in constraints:

```python
class SensorType(Enum):
    """Enumeration of supported sensor types with physics constraints."""
    
    Temperature = 0    # Temperature sensors (-273.15°C to 2000°C theoretical)
    Humidity = 1       # Relative humidity sensors (0-100% RH)
    Pressure = 2       # Atmospheric pressure sensors (0-2000 hPa typical)
    Voc = 3           # Volatile Organic Compound sensors (ppb/ppm)
    Particulate = 4    # Particulate matter sensors (μg/m³)
    Acoustic = 5       # Sound level sensors (dB)
    Vibration = 6      # Vibration/acceleration sensors (g-force)
    Emf = 7           # Electromagnetic field sensors (μT/mT)
    Custom = 8         # Custom sensor types
    
    def physical_range(self) -> Tuple[float, float]:
        """Get theoretical physical range for sensor type."""
        
    def typical_range(self) -> Tuple[float, float]:
        """Get typical operating range for sensor type."""
        
    def default_rate_limit(self) -> float:
        """Get default rate of change limit for sensor type."""
        
    def units(self) -> str:
        """Get standard units for sensor type."""
```

### Usage Examples

```python
import edgeguard as eg

# Temperature sensor
temp_type = eg.SensorType.Temperature
print(f"Temperature range: {temp_type.typical_range()}")  # (-40.0, 125.0)
print(f"Temperature units: {temp_type.units()}")          # "°C"
print(f"Rate limit: {temp_type.default_rate_limit()}")    # 10.0 °C/s

# Humidity sensor
humidity_type = eg.SensorType.Humidity  
print(f"Humidity range: {humidity_type.typical_range()}")  # (0.0, 100.0)
print(f"Humidity units: {humidity_type.units()}")          # "% RH"

# Custom sensor type
custom_type = eg.SensorType.Custom
print(f"Custom sensor: {custom_type}")
```

## Validation Status

### ValidationStatus Enumeration

Detailed validation outcome classification:

```python
class ValidationStatus(Enum):
    """Enumeration of validation outcomes."""
    
    Valid = 0                    # Reading passed all validation checks
    OutOfRange = 1              # Reading outside valid range
    RateExceeded = 2            # Rate of change exceeded limit
    CrossValidationFailed = 3    # Failed cross-sensor validation
    SensorQualityBad = 4        # Sensor quality below threshold
    InvalidValue = 5            # NaN, infinite, or malformed value
    PhysicsViolation = 6        # Violated physics constraints
    CalibrationNeeded = 7       # Sensor appears to need calibration
    SensorTimeout = 8           # Sensor not responding
    
    def is_error(self) -> bool:
        """True if status represents an error condition."""
        
    def is_warning(self) -> bool:
        """True if status represents a warning condition."""
        
    def severity(self) -> str:
        """Get severity level: 'info', 'warning', 'error', 'critical'."""
```

### Status Usage

```python
# Check validation outcomes
for result in validation_results:
    if result.status == eg.ValidationStatus.Valid:
        print(f"✓ {result.sensor_id}: {result.validated_value}")
    elif result.status.is_warning():
        print(f"⚠ {result.sensor_id}: {result.error_details}")
    elif result.status.is_error():
        print(f"✗ {result.sensor_id}: {result.error_details}")
```

## Event Creation

### Builder Pattern

Create events using builder pattern for clarity:

```python
# Create sensor reading event
reading = eg.SensorReading("temp_001", eg.SensorType.Temperature, 23.5, 0.95)

# Create with explicit timestamp
timestamp = eg.Timestamp.now()
reading = eg.SensorReading("humidity_01", eg.SensorType.Humidity, 65.0, 0.98, timestamp)

# Create validation result
result = eg.ValidationResult(
    sensor_id="temp_001",
    status=eg.ValidationStatus.Valid,
    constraints_applied=["range", "rate_limit"],
    original_value=23.5,
    validated_value=23.5
)
```

### Factory Functions

Convenience functions for common event creation:

```python
# Quick sensor reading creation
reading = eg.create_reading("temp_001", "temperature", 23.5)
reading = eg.create_reading("humidity_01", eg.SensorType.Humidity, 65.0, quality=0.95)

# Batch reading creation
readings = eg.create_readings([
    ("temp_001", "temperature", 23.5),
    ("humidity_01", "humidity", 65.0),
    ("pressure_01", "pressure", 1013.2)
])

# Create validation result from exception
try:
    validator.validate(invalid_value)
except eg.ValidationError as e:
    result = eg.create_validation_result_from_error("sensor_01", e)
```

## Event Serialization

### JSON Serialization

Events support JSON serialization for storage and transmission:

```python
# Convert event to JSON
reading = eg.SensorReading("temp_001", eg.SensorType.Temperature, 23.5)
json_data = reading.to_json()
print(json_data)
# {"event_type": "SensorReading", "sensor_id": "temp_001", ...}

# Create event from JSON
event = eg.Event.from_json(json_data)
assert isinstance(event, eg.SensorReading)
assert event.sensor_id == "temp_001"

# Batch serialization
events = [reading1, reading2, result1]
json_array = eg.events_to_json(events)
restored_events = eg.events_from_json(json_array)
```

### Binary Serialization

Efficient binary serialization for embedded systems:

```python
# Convert to binary format (Avro/MessagePack)
reading = eg.SensorReading("temp_001", eg.SensorType.Temperature, 23.5)
binary_data = reading.to_bytes()

# Restore from binary
event = eg.Event.from_bytes(binary_data)

# Batch binary serialization
events = [reading1, reading2, result1]
binary_array = eg.events_to_bytes(events)
restored_events = eg.events_from_bytes(binary_array)
```

## Event Filtering

### Filter Functions

Built-in filters for common event processing:

```python
# Filter by sensor type
temp_events = eg.filter_by_sensor_type(events, eg.SensorType.Temperature)

# Filter by validation status  
valid_events = eg.filter_by_status(events, eg.ValidationStatus.Valid)
error_events = eg.filter_by_status(events, lambda s: s.is_error())

# Filter by time range
recent_events = eg.filter_by_time_range(
    events, 
    start=eg.Timestamp.now() - 3600000,  # Last hour
    end=eg.Timestamp.now()
)

# Filter by sensor ID pattern
temp_sensor_events = eg.filter_by_sensor_pattern(events, r"temp_\d+")

# Complex filtering
high_quality_temp_events = eg.filter_events(
    events,
    sensor_type=eg.SensorType.Temperature,
    min_quality=0.9,
    status=eg.ValidationStatus.Valid
)
```

### Custom Filters

Create custom event filters:

```python
def filter_by_value_range(events: List[eg.Event], min_val: float, max_val: float):
    """Filter events by sensor value range."""
    return [
        event for event in events
        if isinstance(event, eg.SensorReading) and min_val <= event.value <= max_val
    ]

def filter_recent_errors(events: List[eg.Event], minutes: int = 5):
    """Filter recent validation errors."""
    cutoff = eg.Timestamp.now() - (minutes * 60 * 1000)  # Convert to milliseconds
    return [
        event for event in events
        if isinstance(event, eg.ValidationResult) 
        and event.timestamp >= cutoff
        and event.status.is_error()
    ]

# Use custom filters
recent_temp_readings = filter_by_value_range(events, 20.0, 30.0)
recent_errors = filter_recent_errors(events, minutes=10)
```

## Event Statistics

### Statistical Analysis

Built-in statistical functions for event analysis:

```python
# Calculate statistics for sensor readings
temp_events = eg.filter_by_sensor_type(events, eg.SensorType.Temperature)
stats = eg.calculate_statistics(temp_events)

print(f"Temperature statistics:")
print(f"  Count: {stats.count}")
print(f"  Mean: {stats.mean:.2f}°C")
print(f"  Std Dev: {stats.std_dev:.2f}°C")
print(f"  Min: {stats.min:.2f}°C")
print(f"  Max: {stats.max:.2f}°C")
print(f"  Range: {stats.range:.2f}°C")

# Validation statistics
validation_events = eg.filter_by_event_type(events, eg.ValidationResult)
validation_stats = eg.calculate_validation_statistics(validation_events)

print(f"Validation statistics:")
print(f"  Total validations: {validation_stats.total}")
print(f"  Success rate: {validation_stats.success_rate:.1%}")
print(f"  Error rate: {validation_stats.error_rate:.1%}")
print(f"  Most common error: {validation_stats.most_common_error}")
```

### Time Series Analysis

Analyze temporal patterns in sensor data:

```python
# Calculate sampling rate
sampling_rate = eg.calculate_sampling_rate(temp_events)
print(f"Average sampling rate: {sampling_rate:.1f} Hz")

# Detect gaps in data
gaps = eg.detect_time_gaps(temp_events, expected_interval_ms=1000)
for gap in gaps:
    print(f"Data gap: {gap.duration_ms}ms from {gap.start_time}")

# Calculate trend
trend = eg.calculate_trend(temp_events, window_minutes=10)
if trend.direction == "increasing":
    print(f"Temperature increasing at {trend.rate:.2f}°C/hour")
elif trend.direction == "decreasing":
    print(f"Temperature decreasing at {trend.rate:.2f}°C/hour")
else:
    print("Temperature stable")
```

## Integration Examples

### MQTT Integration

Process events from MQTT messages:

```python
import json
import paho.mqtt.client as mqtt

def on_message(client, userdata, msg):
    """Handle MQTT sensor data message."""
    try:
        # Parse JSON message
        data = json.loads(msg.payload.decode())
        
        # Create sensor reading event
        reading = eg.SensorReading(
            sensor_id=data['sensor_id'],
            sensor_type=eg.parse_sensor_type(data['type']),
            value=data['value'],
            quality=data.get('quality', 1.0)
        )
        
        # Process through pipeline
        pipeline.push_event(reading)
        
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error processing MQTT message: {e}")

# Setup MQTT client
client = mqtt.Client()
client.on_message = on_message
client.connect("localhost", 1883, 60)
client.subscribe("sensors/+/data")
```

### Database Storage

Store events in database with proper indexing:

```python
import sqlite3

def create_events_table(conn):
    """Create events table with proper indexing."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            sensor_id TEXT,
            sensor_type INTEGER,
            value REAL,
            quality REAL,
            status INTEGER,
            data TEXT
        )
    """)
    
    # Create indexes for efficient querying
    conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sensor_id ON events(sensor_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sensor_type ON events(sensor_type)")

def store_event(conn, event: eg.Event):
    """Store event in database."""
    if isinstance(event, eg.SensorReading):
        conn.execute("""
            INSERT INTO events (timestamp, event_type, sensor_id, sensor_type, value, quality)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            event.timestamp.to_millis(),
            "SensorReading",
            event.sensor_id,
            event.sensor_type.value,
            event.value,
            event.quality
        ))
    elif isinstance(event, eg.ValidationResult):
        conn.execute("""
            INSERT INTO events (timestamp, event_type, sensor_id, status, data)
            VALUES (?, ?, ?, ?, ?)
        """, (
            event.timestamp.to_millis(),
            "ValidationResult", 
            event.sensor_id,
            event.status.value,
            json.dumps({
                'constraints_applied': event.constraints_applied,
                'error_details': event.error_details
            })
        ))

# Usage
conn = sqlite3.connect('sensor_data.db')
create_events_table(conn)

for event in events:
    store_event(conn, event)
    
conn.commit()
```

For event processing through validation pipelines, see [Pipeline API](pipeline.md).
For time management and timestamp handling, see [Time API](time.md).