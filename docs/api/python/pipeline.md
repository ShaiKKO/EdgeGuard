# Python Pipeline API

Event processing pipeline with composable stages for real-time sensor data validation and fusion.

## Overview

EdgeGuard's pipeline system provides:
- **Composable Architecture**: Chain processing stages for complex data flows
- **Real-time Processing**: Handle high-frequency sensor data with minimal latency
- **Backpressure Handling**: Graceful degradation under load
- **Built-in Metrics**: Monitor pipeline performance and health
- **Type Safety**: Strongly typed event processing with validation

## Core Concepts

### Pipeline Architecture

The pipeline processes events through sequential stages:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Sensor    │───▶│ Validation  │───▶│   Fusion    │───▶│ Aggregation │
│   Events    │    │   Stage     │    │   Stage     │    │   Stage     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

Each stage can:
- Transform events (validation results, fused values)
- Filter events (remove invalid readings)
- Generate new events (cross-validation results)
- Route events (split by sensor type)

## Core Classes

### Pipeline

Main pipeline orchestrator for event processing:

```python
class Pipeline:
    """Event processing pipeline with composable stages."""
    
    def __init__(self, buffer_size: int = 256):
        """Create pipeline with specified buffer size.
        
        Args:
            buffer_size: Maximum events in internal queue
        """
        
    def add_stage(self, stage: 'PipelineStage') -> 'Pipeline':
        """Add processing stage to pipeline.
        
        Args:
            stage: Stage to add (processed in order added)
            
        Returns:
            Self for method chaining
        """
        
    def add_validator(self, validator: 'Validator', sensor_type: 'SensorType') -> 'Pipeline':
        """Add validation stage for specific sensor type.
        
        Args:
            validator: Physics-aware validator
            sensor_type: Type of sensor to validate
            
        Returns:
            Self for method chaining
        """
        
    def push_event(self, event: 'Event') -> bool:
        """Push event into pipeline for processing.
        
        Args:
            event: Event to process
            
        Returns:
            True if event was queued, False if pipeline is full
        """
        
    def process_batch(self, max_events: int = 100) -> List['Event']:
        """Process batch of events through pipeline.
        
        Args:
            max_events: Maximum events to process in this batch
            
        Returns:
            List of output events from pipeline
        """
        
    def process_all(self) -> List['Event']:
        """Process all queued events through pipeline.
        
        Returns:
            List of all output events
        """
        
    def get_metrics(self) -> 'PipelineMetrics':
        """Get pipeline performance metrics."""
        
    def clear(self) -> None:
        """Clear all queued events and reset pipeline state."""
        
    @property
    def queue_size(self) -> int:
        """Current number of queued events."""
        
    @property
    def is_full(self) -> bool:
        """True if pipeline queue is full."""
        
    @property
    def stage_count(self) -> int:
        """Number of processing stages in pipeline."""
```

### PipelineStage

Base class for all pipeline processing stages:

```python
from abc import ABC, abstractmethod

class PipelineStage(ABC):
    """Base class for pipeline processing stages."""
    
    @abstractmethod
    def process(self, event: 'Event') -> List['Event']:
        """Process incoming event.
        
        Args:
            event: Event to process
            
        Returns:
            List of output events (can be empty, single, or multiple)
        """
        
    @abstractmethod
    def name(self) -> str:
        """Get stage name for debugging and metrics."""
        
    def reset(self) -> None:
        """Reset stage state (optional override)."""
        pass
        
    def get_metrics(self) -> dict:
        """Get stage-specific metrics (optional override)."""
        return {}
```

## Built-in Stages

### ValidationStage

Validates sensor readings using physics-aware constraints:

```python
class ValidationStage(PipelineStage):
    """Pipeline stage for sensor validation."""
    
    def __init__(self, validator: 'Validator', sensor_type: 'SensorType', 
                 emit_valid_only: bool = False):
        """Create validation stage.
        
        Args:
            validator: Physics-aware validator to use
            sensor_type: Type of sensor this stage validates
            emit_valid_only: If True, only emit valid readings (filter errors)
        """
        
    def process(self, event: 'Event') -> List['Event']:
        """Validate sensor reading events.
        
        Process flow:
        1. Check if event is SensorReading of matching type
        2. Apply validation using configured validator
        3. Generate ValidationResult event
        4. Optionally filter based on emit_valid_only
        
        Args:
            event: Incoming event (processes SensorReading events)
            
        Returns:
            List containing ValidationResult and optionally original reading
        """
        
    @property
    def validation_count(self) -> int:
        """Total number of validations performed."""
        
    @property
    def success_rate(self) -> float:
        """Validation success rate [0.0, 1.0]."""
```

### FusionStage

Combines multiple sensor readings using fusion algorithms:

```python
class FusionStage(PipelineStage):
    """Pipeline stage for multi-sensor fusion."""
    
    def __init__(self, algorithm: 'FusionAlgorithm', 
                 sensor_group: str = "default",
                 fusion_window_ms: int = 1000):
        """Create fusion stage.
        
        Args:
            algorithm: Fusion algorithm (Kalman, WeightedAverage, etc.)
            sensor_group: Name of sensor group to fuse
            fusion_window_ms: Time window for collecting measurements
        """
        
    def add_sensor(self, sensor_id: str, sensor_type: 'SensorType') -> None:
        """Add sensor to fusion group.
        
        Args:
            sensor_id: Unique sensor identifier
            sensor_type: Type of sensor
        """
        
    def process(self, event: 'Event') -> List['Event']:
        """Process sensor readings for fusion.
        
        Collects readings from registered sensors and performs fusion
        when sufficient data is available within the time window.
        
        Args:
            event: Incoming sensor reading event
            
        Returns:
            List containing FusionResult event when fusion is performed
        """
        
    @property
    def fusion_count(self) -> int:
        """Number of fusion operations performed."""
        
    @property
    def average_confidence(self) -> float:
        """Average confidence of fusion results."""
```

### CrossValidationStage

Performs cross-sensor validation using physics relationships:

```python
class CrossValidationStage(PipelineStage):
    """Pipeline stage for cross-sensor validation."""
    
    def __init__(self, validation_window_ms: int = 5000):
        """Create cross-validation stage.
        
        Args:
            validation_window_ms: Time window for collecting related readings
        """
        
    def add_validation_rule(self, primary_type: 'SensorType', 
                           secondary_type: 'SensorType',
                           validation_func: callable) -> None:
        """Add cross-validation rule.
        
        Args:
            primary_type: Primary sensor type being validated
            secondary_type: Secondary sensor type used for validation
            validation_func: Function(primary_val, secondary_val) -> bool
        """
        
    def process(self, event: 'Event') -> List['Event']:
        """Perform cross-sensor validation.
        
        Checks physics relationships between different sensor types:
        - Temperature vs Humidity (dew point consistency)
        - Pressure vs Altitude (barometric relationship)
        - Custom validation rules
        
        Args:
            event: Incoming sensor reading event
            
        Returns:
            List containing CrossValidationResult events
        """
```

### FilterStage

Filters events based on custom criteria:

```python
class FilterStage(PipelineStage):
    """Pipeline stage for event filtering."""
    
    def __init__(self, filter_func: callable, name: str = "filter"):
        """Create filter stage.
        
        Args:
            filter_func: Function(event) -> bool, True to keep event
            name: Name for this filter stage
        """
        
    @classmethod
    def by_sensor_type(cls, sensor_type: 'SensorType') -> 'FilterStage':
        """Create filter for specific sensor type."""
        
    @classmethod
    def by_quality_threshold(cls, min_quality: float) -> 'FilterStage':
        """Create filter for minimum quality threshold."""
        
    @classmethod
    def by_validation_status(cls, status: 'ValidationStatus') -> 'FilterStage':
        """Create filter for specific validation status."""
        
    def process(self, event: 'Event') -> List['Event']:
        """Filter events based on criteria."""
```

### AggregationStage

Aggregates sensor data over time windows:

```python
class AggregationStage(PipelineStage):
    """Pipeline stage for data aggregation."""
    
    def __init__(self, window_size_ms: int, 
                 aggregation_func: str = "mean",
                 sensor_type: Optional['SensorType'] = None):
        """Create aggregation stage.
        
        Args:
            window_size_ms: Aggregation window size in milliseconds
            aggregation_func: "mean", "median", "min", "max", "std", "count"
            sensor_type: Specific sensor type to aggregate (None for all)
        """
        
    def process(self, event: 'Event') -> List['Event']:
        """Aggregate sensor readings over time windows.
        
        Collects readings within time windows and emits aggregated values:
        - Mean, median, min, max, standard deviation
        - Count of readings in window
        - Data quality metrics
        
        Args:
            event: Incoming sensor reading event
            
        Returns:
            List containing AggregatedReading events when windows complete
        """
```

### RouterStage

Routes events to different processing paths:

```python
class RouterStage(PipelineStage):
    """Pipeline stage for event routing."""
    
    def __init__(self):
        """Create router stage."""
        
    def add_route(self, condition: callable, stage: 'PipelineStage') -> None:
        """Add routing rule.
        
        Args:
            condition: Function(event) -> bool for routing decision
            stage: Stage to route matching events to
        """
        
    def add_sensor_type_route(self, sensor_type: 'SensorType', 
                             stage: 'PipelineStage') -> None:
        """Add route based on sensor type."""
        
    def process(self, event: 'Event') -> List['Event']:
        """Route events based on configured rules."""
```

## Usage Examples

### Basic Pipeline

```python
import edgeguard as eg

# Create pipeline with validation
pipeline = eg.Pipeline(buffer_size=512)

# Add temperature validation
temp_validator = eg.TemperatureValidator().with_range(-20.0, 60.0)
pipeline.add_validator(temp_validator, eg.SensorType.Temperature)

# Add humidity validation  
humidity_validator = eg.HumidityValidator().with_range(0.0, 100.0)
pipeline.add_validator(humidity_validator, eg.SensorType.Humidity)

# Process sensor readings
readings = [
    eg.SensorReading("temp_001", eg.SensorType.Temperature, 23.5),
    eg.SensorReading("humidity_01", eg.SensorType.Humidity, 65.0),
    eg.SensorReading("temp_002", eg.SensorType.Temperature, 24.1)
]

# Push events and process
for reading in readings:
    pipeline.push_event(reading)

results = pipeline.process_all()
for result in results:
    if isinstance(result, eg.ValidationResult):
        print(f"Validation: {result.sensor_id} -> {result.status}")
```

### Advanced Pipeline with Fusion

```python
# Create sophisticated processing pipeline
pipeline = eg.Pipeline(buffer_size=1024)

# Step 1: Validation stages
temp_validator = eg.TemperatureValidator().with_range(-40.0, 85.0)
humidity_validator = eg.HumidityValidator().with_range(0.0, 100.0)
pressure_validator = eg.PressureValidator().with_range(300.0, 1100.0)

pipeline.add_validator(temp_validator, eg.SensorType.Temperature)
pipeline.add_validator(humidity_validator, eg.SensorType.Humidity)
pipeline.add_validator(pressure_validator, eg.SensorType.Pressure)

# Step 2: Filter to valid readings only
valid_filter = eg.FilterStage.by_validation_status(eg.ValidationStatus.Valid)
pipeline.add_stage(valid_filter)

# Step 3: Cross-validation stage
cross_val = eg.CrossValidationStage(validation_window_ms=5000)

# Add physics validation rules
def temp_humidity_check(temp: float, humidity: float) -> bool:
    """Check temperature-humidity physics consistency."""
    dew_point = eg.calculate_dew_point(temp, humidity)
    return dew_point <= temp  # Dew point must be <= air temperature

cross_val.add_validation_rule(
    eg.SensorType.Temperature,
    eg.SensorType.Humidity,
    temp_humidity_check
)

pipeline.add_stage(cross_val)

# Step 4: Fusion stages for redundant sensors
kalman_config = eg.KalmanConfig.for_temperature()
temp_fusion = eg.FusionStage(eg.KalmanFilter(kalman_config), "temperature_group")
temp_fusion.add_sensor("temp_001", eg.SensorType.Temperature)
temp_fusion.add_sensor("temp_002", eg.SensorType.Temperature)
temp_fusion.add_sensor("temp_003", eg.SensorType.Temperature)

pipeline.add_stage(temp_fusion)

# Step 5: Aggregation for data reduction
aggregation = eg.AggregationStage(
    window_size_ms=60000,  # 1 minute windows
    aggregation_func="mean"
)
pipeline.add_stage(aggregation)

print(f"Pipeline created with {pipeline.stage_count} stages")
```

### Real-time Processing

```python
import asyncio
import time

class RealTimePipeline:
    """Real-time sensor processing pipeline."""
    
    def __init__(self):
        self.pipeline = self._create_pipeline()
        self.processing_stats = {
            'events_processed': 0,
            'processing_time_ms': 0,
            'last_batch_size': 0
        }
        
    def _create_pipeline(self) -> eg.Pipeline:
        """Create optimized real-time pipeline."""
        pipeline = eg.Pipeline(buffer_size=2048)  # Large buffer for real-time
        
        # Fast validation stages
        temp_validator = eg.TemperatureValidator().with_range(-20.0, 60.0)
        pipeline.add_validator(temp_validator, eg.SensorType.Temperature)
        
        # High-quality filter
        quality_filter = eg.FilterStage.by_quality_threshold(0.8)
        pipeline.add_stage(quality_filter)
        
        # Fast fusion with weighted average
        fusion = eg.FusionStage(eg.WeightedAverageFusion(), "temp_group")
        fusion.add_sensor("temp_001", eg.SensorType.Temperature)
        fusion.add_sensor("temp_002", eg.SensorType.Temperature)
        pipeline.add_stage(fusion)
        
        return pipeline
    
    async def process_stream(self, sensor_stream):
        """Process continuous sensor data stream."""
        batch_size = 50  # Process in batches for efficiency
        
        async for sensor_data in sensor_stream:
            # Create sensor reading
            reading = eg.SensorReading(
                sensor_data['id'],
                eg.parse_sensor_type(sensor_data['type']),
                sensor_data['value'],
                sensor_data.get('quality', 1.0)
            )
            
            # Queue for processing
            if not self.pipeline.push_event(reading):
                print("Warning: Pipeline queue full, dropping event")
                continue
            
            # Process batch when ready
            if self.pipeline.queue_size >= batch_size:
                await self._process_batch()
            
            # Yield control for other tasks
            await asyncio.sleep(0.001)
    
    async def _process_batch(self):
        """Process batch of events with timing."""
        start_time = time.perf_counter()
        
        results = self.pipeline.process_batch(100)
        
        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000
        
        # Update statistics
        self.processing_stats['events_processed'] += len(results)
        self.processing_stats['processing_time_ms'] += processing_time_ms
        self.processing_stats['last_batch_size'] = len(results)
        
        # Log performance every 1000 events
        if self.processing_stats['events_processed'] % 1000 == 0:
            avg_time_per_event = (
                self.processing_stats['processing_time_ms'] / 
                self.processing_stats['events_processed']
            )
            print(f"Processed {self.processing_stats['events_processed']} events")
            print(f"Average time per event: {avg_time_per_event:.3f}ms")
        
        return results

# Usage
async def main():
    processor = RealTimePipeline()
    
    # Simulate sensor data stream
    async def sensor_stream():
        for i in range(10000):
            yield {
                'id': f'temp_{i % 3 + 1:03d}',
                'type': 'temperature',
                'value': 20.0 + 5.0 * math.sin(i * 0.1) + random.gauss(0, 0.2),
                'quality': 0.95
            }
            await asyncio.sleep(0.01)  # 100Hz data rate
    
    await processor.process_stream(sensor_stream())

# Run real-time processing
asyncio.run(main())
```

## Pipeline Metrics

### PipelineMetrics

Monitor pipeline performance and health:

```python
class PipelineMetrics:
    """Pipeline performance metrics."""
    
    @property
    def events_processed(self) -> int:
        """Total events processed through pipeline."""
        
    @property
    def events_dropped(self) -> int:
        """Events dropped due to queue overflow."""
        
    @property
    def processing_time_ms(self) -> float:
        """Total processing time in milliseconds."""
        
    @property
    def average_latency_ms(self) -> float:
        """Average event processing latency."""
        
    @property
    def throughput_hz(self) -> float:
        """Events processed per second."""
        
    @property
    def queue_utilization(self) -> float:
        """Queue utilization [0.0, 1.0]."""
        
    @property
    def stage_metrics(self) -> Dict[str, dict]:
        """Per-stage performance metrics."""
        
    def reset(self) -> None:
        """Reset all metrics counters."""
```

### Performance Monitoring

```python
def monitor_pipeline_performance(pipeline: eg.Pipeline, duration_seconds: int = 60):
    """Monitor pipeline performance over time."""
    start_time = time.time()
    initial_metrics = pipeline.get_metrics()
    
    while time.time() - start_time < duration_seconds:
        time.sleep(5)  # Check every 5 seconds
        
        current_metrics = pipeline.get_metrics()
        
        # Calculate rates
        events_delta = current_metrics.events_processed - initial_metrics.events_processed
        time_delta = time.time() - start_time
        throughput = events_delta / time_delta if time_delta > 0 else 0
        
        print(f"Pipeline Performance:")
        print(f"  Throughput: {throughput:.1f} events/sec")
        print(f"  Queue utilization: {current_metrics.queue_utilization:.1%}")
        print(f"  Average latency: {current_metrics.average_latency_ms:.2f}ms")
        print(f"  Events dropped: {current_metrics.events_dropped}")
        
        # Check for performance issues
        if current_metrics.queue_utilization > 0.8:
            print("⚠️  Warning: High queue utilization - consider increasing buffer size")
        
        if current_metrics.events_dropped > 0:
            print("⚠️  Warning: Events being dropped - pipeline overloaded")
        
        if current_metrics.average_latency_ms > 100:
            print("⚠️  Warning: High latency - check stage performance")

# Usage
monitor_pipeline_performance(pipeline, duration_seconds=300)  # Monitor for 5 minutes
```

## Error Handling

### Pipeline Errors

Handle pipeline errors gracefully:

```python
try:
    results = pipeline.process_batch(100)
except eg.PipelineError as e:
    if e.error_type == "queue_full":
        print("Pipeline queue full - reduce input rate or increase buffer size")
    elif e.error_type == "stage_error":
        print(f"Stage '{e.stage_name}' failed: {e.details}")
        # Reset problematic stage
        pipeline.reset_stage(e.stage_name)
    elif e.error_type == "timeout":
        print("Pipeline processing timeout - check for blocking stages")
except eg.ValidationError as e:
    print(f"Validation error in pipeline: {e}")
except Exception as e:
    print(f"Unexpected pipeline error: {e}")
    # Log error and continue processing
    pipeline.clear()  # Clear queue and reset
```

### Stage Error Recovery

```python
class RobustPipeline(eg.Pipeline):
    """Pipeline with automatic error recovery."""
    
    def __init__(self, buffer_size: int = 256):
        super().__init__(buffer_size)
        self.error_counts = {}
        self.max_errors_per_stage = 10
        
    def process_batch(self, max_events: int = 100) -> List['Event']:
        """Process batch with error recovery."""
        try:
            return super().process_batch(max_events)
        except eg.PipelineError as e:
            # Track errors per stage
            stage_name = getattr(e, 'stage_name', 'unknown')
            self.error_counts[stage_name] = self.error_counts.get(stage_name, 0) + 1
            
            # Disable problematic stages
            if self.error_counts[stage_name] > self.max_errors_per_stage:
                print(f"Disabling stage '{stage_name}' due to repeated errors")
                self.disable_stage(stage_name)
                self.error_counts[stage_name] = 0
            
            # Reset and continue
            self.reset_stage(stage_name)
            return []  # Return empty results for this batch

# Usage
robust_pipeline = RobustPipeline(buffer_size=1024)
# Add stages...

# Process with automatic recovery
while True:
    try:
        results = robust_pipeline.process_batch(50)
        for result in results:
            print(f"Processed: {result}")
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Unexpected error: {e}")
        time.sleep(1)  # Brief pause before retrying
```

## Integration Patterns

### MQTT Pipeline Integration

```python
import paho.mqtt.client as mqtt
import json

class MqttPipelineIntegration:
    """Integrate MQTT data stream with EdgeGuard pipeline."""
    
    def __init__(self, pipeline: eg.Pipeline):
        self.pipeline = pipeline
        self.client = mqtt.Client()
        self.client.on_message = self._on_mqtt_message
        self.client.on_connect = self._on_connect
        
    def _on_connect(self, client, userdata, flags, rc):
        """Handle MQTT connection."""
        print(f"Connected to MQTT broker with result code {rc}")
        client.subscribe("sensors/+/data")
        
    def _on_mqtt_message(self, client, userdata, msg):
        """Process MQTT sensor data through pipeline."""
        try:
            # Parse sensor data
            data = json.loads(msg.payload.decode())
            
            # Create EdgeGuard event
            reading = eg.SensorReading(
                data['sensor_id'],
                eg.parse_sensor_type(data['type']),
                data['value'],
                data.get('quality', 1.0)
            )
            
            # Process through pipeline
            if self.pipeline.push_event(reading):
                # Process batch if queue is getting full
                if self.pipeline.queue_size > 100:
                    results = self.pipeline.process_batch(50)
                    self._handle_results(results)
            else:
                print("Pipeline queue full - dropping MQTT message")
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error processing MQTT message: {e}")
    
    def _handle_results(self, results: List[eg.Event]):
        """Handle pipeline results."""
        for result in results:
            if isinstance(result, eg.ValidationResult):
                if result.status != eg.ValidationStatus.Valid:
                    print(f"Validation failed: {result.sensor_id} - {result.error_details}")
            elif isinstance(result, eg.FusionResult):
                print(f"Fused data: {result.sensor_group} = {result.value} (conf: {result.confidence:.1%})")
    
    def start(self, broker_host: str, broker_port: int = 1883):
        """Start MQTT integration."""
        self.client.connect(broker_host, broker_port, 60)
        self.client.loop_start()
        
        # Process pipeline periodically
        import threading
        def process_pipeline():
            while True:
                results = self.pipeline.process_batch(100)
                if results:
                    self._handle_results(results)
                time.sleep(0.1)  # 10Hz processing
        
        processing_thread = threading.Thread(target=process_pipeline, daemon=True)
        processing_thread.start()

# Usage
pipeline = eg.Pipeline(1024)
# Configure pipeline...

mqtt_integration = MqttPipelineIntegration(pipeline)
mqtt_integration.start("localhost", 1883)

# Keep running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down MQTT integration")
```

For detailed validator documentation, see [Validators API](validators.md).
For fusion algorithm details, see [Fusion API](fusion.md).
For event types and handling, see [Events API](events.md).