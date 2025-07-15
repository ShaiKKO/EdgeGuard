# Python Time API

High-precision time management with millisecond accuracy and Python datetime integration.

## Overview

EdgeGuard's time system provides:
- **Millisecond Precision**: High-resolution timestamps for accurate event ordering
- **Python Integration**: Seamless conversion to/from Python datetime objects
- **Timezone Support**: UTC-based timestamps with timezone conversion
- **Performance**: Optimized for high-frequency sensor data processing
- **Embedded Compatibility**: Works across platforms from ESP32 to desktop

## Core Types

### Timestamp

High-precision timestamp with millisecond resolution:

```python
class Timestamp:
    """High-precision timestamp with millisecond resolution."""
    
    def __init__(self, millis: int):
        """Create timestamp from milliseconds since Unix epoch.
        
        Args:
            millis: Milliseconds since January 1, 1970 UTC
        """
        
    @classmethod
    def now(cls) -> 'Timestamp':
        """Get current timestamp with system time."""
        
    @classmethod
    def from_seconds(cls, seconds: float) -> 'Timestamp':
        """Create timestamp from seconds since Unix epoch.
        
        Args:
            seconds: Seconds since epoch (fractional seconds supported)
        """
        
    @classmethod
    def from_datetime(cls, dt: datetime.datetime) -> 'Timestamp':
        """Create timestamp from Python datetime object.
        
        Args:
            dt: Python datetime (timezone-aware recommended)
        """
        
    def to_millis(self) -> int:
        """Get milliseconds since Unix epoch."""
        
    def to_seconds(self) -> float:
        """Get seconds since Unix epoch (with fractional part)."""
        
    def to_datetime(self) -> datetime.datetime:
        """Convert to Python datetime object (UTC timezone)."""
        
    def to_local_datetime(self) -> datetime.datetime:
        """Convert to local timezone datetime object."""
        
    def __add__(self, delta_ms: int) -> 'Timestamp':
        """Add milliseconds to timestamp."""
        
    def __sub__(self, other) -> int:
        """Subtract timestamps to get millisecond difference."""
        
    def __eq__(self, other: 'Timestamp') -> bool:
        """Compare timestamps for equality."""
        
    def __lt__(self, other: 'Timestamp') -> bool:
        """Compare timestamps for ordering."""
        
    def format(self, fmt: str = "%Y-%m-%d %H:%M:%S.%f") -> str:
        """Format timestamp as string."""
```

### Usage Examples

```python
import edgeguard as eg
from datetime import datetime, timezone

# Create timestamps
now = eg.Timestamp.now()
epoch = eg.Timestamp(0)  # Unix epoch
future = eg.Timestamp.from_seconds(time.time() + 3600)  # One hour from now

# From Python datetime
dt = datetime.now(timezone.utc)
timestamp = eg.Timestamp.from_datetime(dt)

# Conversions
millis = now.to_millis()
seconds = now.to_seconds()
dt_utc = now.to_datetime()
dt_local = now.to_local_datetime()

print(f"Current time: {now.format()}")
print(f"Milliseconds: {millis}")
print(f"UTC datetime: {dt_utc}")
print(f"Local datetime: {dt_local}")

# Arithmetic
one_second_later = now + 1000  # Add 1000ms
time_diff = one_second_later - now  # 1000ms difference

# Comparisons
if timestamp < now:
    print("Timestamp is in the past")
```

## Time Sources

### SystemTime

System wall clock time source:

```python
class SystemTime:
    """System wall clock time source."""
    
    @staticmethod
    def now() -> Timestamp:
        """Get current system time as timestamp."""
        
    @staticmethod
    def resolution_ms() -> int:
        """Get system clock resolution in milliseconds."""
        
    @staticmethod
    def is_monotonic() -> bool:
        """Check if system clock is monotonic (always False)."""
```

### MonotonicTime

Monotonic time source that doesn't jump backwards:

```python
class MonotonicTime:
    """Monotonic time source that never goes backwards."""
    
    @staticmethod
    def now() -> Timestamp:
        """Get current monotonic time as timestamp."""
        
    @staticmethod
    def resolution_ms() -> int:
        """Get monotonic clock resolution in milliseconds."""
        
    @staticmethod
    def is_monotonic() -> bool:
        """Check if time source is monotonic (always True)."""
```

### Time Source Selection

Choose appropriate time source for your application:

```python
# For timestamping events with real wall-clock time
system_time = eg.SystemTime()
timestamp = system_time.now()

# For measuring durations and intervals (preferred for embedded)
monotonic_time = eg.MonotonicTime()
start_time = monotonic_time.now()
# ... do work ...
end_time = monotonic_time.now()
duration_ms = end_time - start_time

# Time source comparison
print(f"System time resolution: {eg.SystemTime.resolution_ms()}ms")
print(f"Monotonic time resolution: {eg.MonotonicTime.resolution_ms()}ms")
print(f"System time monotonic: {eg.SystemTime.is_monotonic()}")
print(f"Monotonic time monotonic: {eg.MonotonicTime.is_monotonic()}")
```

## Time Utilities

### Duration and Intervals

Helper functions for working with time durations:

```python
class Duration:
    """Time duration utilities."""
    
    @staticmethod
    def milliseconds(ms: int) -> int:
        """Create duration from milliseconds."""
        return ms
        
    @staticmethod 
    def seconds(s: float) -> int:
        """Create duration from seconds."""
        return int(s * 1000)
        
    @staticmethod
    def minutes(m: float) -> int:
        """Create duration from minutes."""
        return int(m * 60 * 1000)
        
    @staticmethod
    def hours(h: float) -> int:
        """Create duration from hours."""
        return int(h * 60 * 60 * 1000)
        
    @staticmethod
    def days(d: float) -> int:
        """Create duration from days."""
        return int(d * 24 * 60 * 60 * 1000)

# Usage
one_second = eg.Duration.seconds(1)
five_minutes = eg.Duration.minutes(5)
one_hour = eg.Duration.hours(1)

# Use with timestamps
now = eg.Timestamp.now()
later = now + one_hour
earlier = now - five_minutes
```

### Time Range Operations

Work with time ranges and intervals:

```python
class TimeRange:
    """Time range operations."""
    
    def __init__(self, start: Timestamp, end: Timestamp):
        """Create time range.
        
        Args:
            start: Range start time
            end: Range end time
        """
        
    def contains(self, timestamp: Timestamp) -> bool:
        """Check if timestamp is within range."""
        
    def duration_ms(self) -> int:
        """Get range duration in milliseconds."""
        
    def split(self, interval_ms: int) -> List['TimeRange']:
        """Split range into intervals."""
        
    def overlap(self, other: 'TimeRange') -> Optional['TimeRange']:
        """Find overlap with another range."""

# Usage
start = eg.Timestamp.now()
end = start + eg.Duration.hours(1)
time_range = eg.TimeRange(start, end)

# Check if timestamp is in range
test_time = start + eg.Duration.minutes(30)
if time_range.contains(test_time):
    print("Timestamp is within range")

# Split into 10-minute intervals
intervals = time_range.split(eg.Duration.minutes(10))
print(f"Split into {len(intervals)} intervals")
```

## Timezone Handling

### Timezone Conversion

Convert between timezones while maintaining precision:

```python
import pytz

# Create timezone-aware timestamps
utc_now = eg.Timestamp.now()  # Always UTC internally
eastern = pytz.timezone('US/Eastern')
pacific = pytz.timezone('US/Pacific')

# Convert to different timezones for display
dt_utc = utc_now.to_datetime()
dt_eastern = dt_utc.replace(tzinfo=pytz.UTC).astimezone(eastern)
dt_pacific = dt_utc.replace(tzinfo=pytz.UTC).astimezone(pacific)

print(f"UTC: {dt_utc}")
print(f"Eastern: {dt_eastern}")
print(f"Pacific: {dt_pacific}")

# Create timestamp from timezone-aware datetime
eastern_dt = datetime(2024, 1, 15, 14, 30, 0, tzinfo=eastern)
timestamp = eg.Timestamp.from_datetime(eastern_dt)
print(f"Converted to UTC timestamp: {timestamp.to_datetime()}")
```

### Daylight Saving Time

Handle DST transitions correctly:

```python
# DST transition handling
eastern = pytz.timezone('US/Eastern')

# Before DST (EST)
before_dst = datetime(2024, 3, 10, 1, 30, 0)
before_dst_aware = eastern.localize(before_dst)
timestamp_before = eg.Timestamp.from_datetime(before_dst_aware)

# After DST (EDT) 
after_dst = datetime(2024, 3, 10, 3, 30, 0)
after_dst_aware = eastern.localize(after_dst)
timestamp_after = eg.Timestamp.from_datetime(after_dst_aware)

# Time difference accounts for DST transition
diff_ms = timestamp_after - timestamp_before
print(f"Time difference: {diff_ms}ms")  # Should be 1 hour despite 2-hour clock difference
```

## Performance Optimization

### High-Frequency Timestamping

Optimize for high-frequency sensor data:

```python
# Pre-allocate timestamp objects for performance
class TimestampCache:
    def __init__(self, size: int = 1000):
        self.cache = [None] * size
        self.index = 0
        
    def get_timestamp(self) -> eg.Timestamp:
        """Get cached timestamp object."""
        if self.cache[self.index] is None:
            self.cache[self.index] = eg.Timestamp.now()
        else:
            # Reuse object, update time
            self.cache[self.index] = eg.Timestamp.now()
            
        timestamp = self.cache[self.index]
        self.index = (self.index + 1) % len(self.cache)
        return timestamp

# Use cached timestamps for high-frequency data
cache = TimestampCache(100)

for i in range(1000):
    timestamp = cache.get_timestamp()
    reading = eg.SensorReading("sensor_001", eg.SensorType.Temperature, 23.5, timestamp=timestamp)
    # Process reading...
```

### Batch Timestamp Operations

Process multiple timestamps efficiently:

```python
# Batch create timestamps
start_time = eg.Timestamp.now()
timestamps = []
for i in range(100):
    timestamps.append(start_time + (i * 100))  # 100ms intervals

# Batch conversion to datetime
datetimes = [ts.to_datetime() for ts in timestamps]

# Batch filtering by time range
recent_timestamps = [
    ts for ts in timestamps 
    if ts >= (eg.Timestamp.now() - eg.Duration.minutes(5))
]
```

## Integration Examples

### Event Timestamping

Timestamp sensor events with high precision:

```python
# Create timestamped sensor readings
def create_timestamped_reading(sensor_id: str, sensor_type: eg.SensorType, value: float) -> eg.SensorReading:
    """Create sensor reading with precise timestamp."""
    timestamp = eg.Timestamp.now()
    return eg.SensorReading(sensor_id, sensor_type, value, timestamp=timestamp)

# Batch timestamping
readings = []
base_time = eg.Timestamp.now()
for i, value in enumerate([23.1, 23.2, 23.0, 22.9]):
    timestamp = base_time + (i * 250)  # 250ms intervals
    reading = eg.SensorReading("temp_001", eg.SensorType.Temperature, value, timestamp=timestamp)
    readings.append(reading)
```

### Time Series Analysis

Analyze temporal patterns in sensor data:

```python
def analyze_sampling_rate(events: List[eg.Event]) -> dict:
    """Analyze sampling rate of timestamped events."""
    if len(events) < 2:
        return {"error": "Need at least 2 events"}
        
    # Sort by timestamp
    sorted_events = sorted(events, key=lambda e: e.timestamp)
    
    # Calculate intervals
    intervals = []
    for i in range(1, len(sorted_events)):
        interval_ms = sorted_events[i].timestamp - sorted_events[i-1].timestamp
        intervals.append(interval_ms)
    
    # Statistics
    mean_interval = sum(intervals) / len(intervals)
    min_interval = min(intervals)
    max_interval = max(intervals)
    
    return {
        "mean_interval_ms": mean_interval,
        "mean_rate_hz": 1000.0 / mean_interval if mean_interval > 0 else 0,
        "min_interval_ms": min_interval,
        "max_interval_ms": max_interval,
        "jitter_ms": max_interval - min_interval,
        "total_duration_ms": sorted_events[-1].timestamp - sorted_events[0].timestamp
    }

# Analyze sensor data timing
analysis = analyze_sampling_rate(sensor_readings)
print(f"Average sampling rate: {analysis['mean_rate_hz']:.1f} Hz")
print(f"Timing jitter: {analysis['jitter_ms']}ms")
```

### Database Time Indexing

Efficient time-based database queries:

```python
import sqlite3

def create_time_indexed_table(conn):
    """Create table with efficient time indexing."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_ms INTEGER NOT NULL,
            sensor_id TEXT NOT NULL,
            value REAL NOT NULL,
            FOREIGN KEY (sensor_id) REFERENCES sensors(id)
        )
    """)
    
    # Create time-based indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON sensor_data(timestamp_ms)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sensor_time ON sensor_data(sensor_id, timestamp_ms)")

def query_time_range(conn, start: eg.Timestamp, end: eg.Timestamp, sensor_id: str = None):
    """Query data within time range."""
    if sensor_id:
        cursor = conn.execute("""
            SELECT timestamp_ms, value FROM sensor_data 
            WHERE sensor_id = ? AND timestamp_ms BETWEEN ? AND ?
            ORDER BY timestamp_ms
        """, (sensor_id, start.to_millis(), end.to_millis()))
    else:
        cursor = conn.execute("""
            SELECT timestamp_ms, sensor_id, value FROM sensor_data 
            WHERE timestamp_ms BETWEEN ? AND ?
            ORDER BY timestamp_ms
        """, (start.to_millis(), end.to_millis()))
    
    results = []
    for row in cursor:
        timestamp = eg.Timestamp(row[0])
        if sensor_id:
            results.append((timestamp, row[1]))
        else:
            results.append((timestamp, row[1], row[2]))
    
    return results

# Usage
conn = sqlite3.connect('sensor_data.db')
create_time_indexed_table(conn)

# Query last hour of data
end_time = eg.Timestamp.now()
start_time = end_time - eg.Duration.hours(1)
recent_data = query_time_range(conn, start_time, end_time, "temp_001")

print(f"Found {len(recent_data)} readings in the last hour")
```

### Real-time Processing

Handle real-time data streams with precise timing:

```python
import asyncio

class RealTimeProcessor:
    """Real-time sensor data processor with timing constraints."""
    
    def __init__(self, max_latency_ms: int = 100):
        self.max_latency_ms = max_latency_ms
        self.processed_count = 0
        self.late_count = 0
        
    async def process_reading(self, reading: eg.SensorReading):
        """Process reading with latency monitoring."""
        processing_start = eg.Timestamp.now()
        
        # Check if reading is too old
        latency_ms = processing_start - reading.timestamp
        if latency_ms > self.max_latency_ms:
            self.late_count += 1
            print(f"Warning: Late processing {latency_ms}ms for {reading.sensor_id}")
        
        # Simulate processing
        await asyncio.sleep(0.01)  # 10ms processing time
        
        # Update statistics
        self.processed_count += 1
        
        if self.processed_count % 100 == 0:
            success_rate = (self.processed_count - self.late_count) / self.processed_count
            print(f"Processed {self.processed_count} readings, {success_rate:.1%} on time")

# Usage
processor = RealTimeProcessor(max_latency_ms=50)

async def process_stream():
    """Process stream of sensor readings."""
    for i in range(1000):
        # Simulate sensor reading with current timestamp
        reading = eg.SensorReading(
            "temp_001", 
            eg.SensorType.Temperature, 
            23.0 + 0.1 * i,
            timestamp=eg.Timestamp.now()
        )
        
        await processor.process_reading(reading)
        await asyncio.sleep(0.01)  # 100Hz data rate

# Run real-time processing
asyncio.run(process_stream())
```

For event processing with timestamps, see [Events API](events.md).
For pipeline timing and performance, see [Pipeline API](pipeline.md).