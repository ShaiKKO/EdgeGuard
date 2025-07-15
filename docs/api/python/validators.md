# Python Validators API

Physics-aware validation components for sensor data with Python-native interfaces.

## Core Validator Interface

All validators implement a common interface for consistent usage:

```python
from abc import ABC, abstractmethod

class Validator(ABC):
    @abstractmethod
    def validate(self, value: float) -> float:
        """Validate a sensor reading.
        
        Args:
            value: The sensor reading to validate
            
        Returns:
            The validated value (may be the same as input)
            
        Raises:
            ValidationError: If validation fails
        """
        pass
```

## TemperatureValidator

Validates temperature readings with physics-aware constraints including thermal mass modeling and rate limiting.

### Physics Background

Temperature sensors are subject to several physical constraints:
- **Thermal Mass**: Objects cannot change temperature instantaneously due to heat capacity
- **Rate Limits**: Maximum possible temperature change rates based on thermal dynamics
- **Operating Ranges**: Physical sensor limitations and environmental conditions
- **Thermal Equilibrium**: Consistency with surrounding environmental conditions

### Constructor and Configuration

```python
class TemperatureValidator:
    def __init__(self) -> None:
        """Create temperature validator with default physics constraints.
        
        Default configuration:
        - Range: -80°C to 125°C (absolute physical limits)
        - Rate limit: 10°C/s (conservative thermal dynamics)
        - Thermal mass: 0.1kg (small sensor assumption)
        """
        
    def with_range(self, min_temp: float, max_temp: float) -> 'TemperatureValidator':
        """Set temperature range constraints.
        
        Args:
            min_temp: Minimum valid temperature (°C)
            max_temp: Maximum valid temperature (°C)
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If min_temp >= max_temp or outside absolute limits
        """
        
    def with_rate_limit(self, max_rate: float) -> 'TemperatureValidator':
        """Set maximum rate of change constraint.
        
        Rate limiting prevents detection of impossible temperature changes
        that could indicate sensor malfunction or electromagnetic interference.
        
        Args:
            max_rate: Maximum temperature change rate (°C/s)
                     Typical values: 0.1-10.0°C/s depending on thermal mass
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If max_rate <= 0
        """
        
    def with_thermal_mass(self, mass_kg: float) -> 'TemperatureValidator':
        """Set thermal mass for rate calculations.
        
        Thermal mass affects how quickly temperature can change:
        - Small sensors (0.01-0.1kg): Fast response, higher rate limits
        - Medium objects (0.1-1.0kg): Moderate response
        - Large thermal masses (>1.0kg): Slow response, low rate limits
        
        Args:
            mass_kg: Thermal mass in kilograms
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If mass_kg <= 0
        """
        
    def with_ambient_context(self, ambient_temp: float) -> 'TemperatureValidator':
        """Set ambient temperature for thermal equilibrium validation.
        
        Args:
            ambient_temp: Expected ambient temperature (°C)
            
        Returns:
            Self for method chaining
        """
```

### Usage Examples

```python
import edgeguard as eg

# Basic temperature validation
validator = eg.TemperatureValidator()
try:
    temp = validator.validate(23.5)
    print(f"Valid temperature: {temp}°C")
except eg.ValidationError as e:
    print(f"Validation failed: {e}")

# Sensor-specific configuration
bme280_validator = eg.TemperatureValidator() \
    .with_range(-40.0, 85.0) \
    .with_rate_limit(2.0) \
    .with_thermal_mass(0.05)  # Small sensor

# Industrial sensor configuration  
industrial_validator = eg.TemperatureValidator() \
    .with_range(-200.0, 500.0) \
    .with_rate_limit(0.5) \
    .with_thermal_mass(2.0)  # Large thermal mass

# Validate readings
readings = [23.1, 23.2, 23.0, 22.9, 23.3]
for reading in readings:
    try:
        valid_temp = bme280_validator.validate(reading)
        print(f"✓ {valid_temp}°C")
    except eg.ValidationError as e:
        print(f"✗ {reading}°C - {e}")
```

### Error Handling

```python
try:
    temp = validator.validate(reading)
except eg.OutOfRangeError as e:
    print(f"Temperature {e.actual_value}°C outside range [{e.expected_range[0]}, {e.expected_range[1]}]°C")
except eg.RateExceededError as e:
    print(f"Temperature change rate {e.actual_value}°C/s exceeds maximum {e.expected_range[1]}°C/s")
except eg.PhysicsViolationError as e:
    print(f"Physics constraint violated: {e.constraint}")
except eg.InvalidValueError as e:
    print(f"Invalid temperature value: {e.actual_value} (NaN or infinite)")
```

## HumidityValidator

Validates humidity readings with dew point calculations and physical moisture dynamics.

### Physics Background

Humidity sensors measure relative humidity (RH) as percentage of moisture saturation:
- **Physical Range**: 0-100% RH (cannot exceed saturation)
- **Dew Point Relationship**: RH and temperature determine dew point
- **Vapor Dynamics**: Humidity changes limited by air volume and diffusion rates
- **Condensation Physics**: RH approaching 100% indicates possible condensation

### Constructor and Configuration

```python
class HumidityValidator:
    def __init__(self) -> None:
        """Create humidity validator with default physics constraints.
        
        Default configuration:
        - Range: 0-100% RH (physical limits)
        - Rate limit: 20%/s (moderate air circulation)
        - Dew point validation: enabled
        """
        
    def with_range(self, min_humidity: float, max_humidity: float) -> 'HumidityValidator':
        """Set humidity range constraints.
        
        Args:
            min_humidity: Minimum valid humidity (% RH)
            max_humidity: Maximum valid humidity (% RH)
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If not in range [0, 100] or min >= max
        """
        
    def with_rate_limit(self, max_rate: float) -> 'HumidityValidator':
        """Set maximum humidity change rate.
        
        Rate limiting based on physical vapor diffusion:
        - Open air: 10-50%/s possible with rapid air changes
        - Enclosed spaces: 1-10%/s typical for room volumes
        - Large volumes: <1%/s for warehouses, outdoors
        
        Args:
            max_rate: Maximum humidity change rate (%/s)
            
        Returns:
            Self for method chaining
        """
        
    def with_temperature_context(self, temperature: float) -> 'HumidityValidator':
        """Set temperature for dew point validation.
        
        Enables cross-sensor validation using dew point physics:
        - Dew point must be ≤ air temperature
        - Higher accuracy when temperature is known
        
        Args:
            temperature: Air temperature (°C) for dew point calculation
            
        Returns:
            Self for method chaining
        """
        
    def with_air_volume(self, volume_m3: float) -> 'HumidityValidator':
        """Set air volume for rate limit calculations.
        
        Args:
            volume_m3: Air volume in cubic meters
            
        Returns:
            Self for method chaining
        """
```

### Dew Point Validation

```python
# Temperature-aware humidity validation
temp_validator = eg.TemperatureValidator()
humidity_validator = eg.HumidityValidator() \
    .with_temperature_context(25.0)  # 25°C air temperature

# Validate humidity with dew point check
try:
    humidity = humidity_validator.validate(80.0)  # 80% RH at 25°C
    dew_point = eg.calculate_dew_point(25.0, humidity)
    print(f"Humidity: {humidity}% RH, Dew point: {dew_point:.1f}°C")
except eg.PhysicsViolationError as e:
    print(f"Dew point violation: {e.constraint}")
```

### Usage Examples

```python
# Basic humidity validation
humidity_validator = eg.HumidityValidator()

# DHT22 sensor configuration
dht22_validator = eg.HumidityValidator() \
    .with_range(0.0, 100.0) \
    .with_rate_limit(5.0)

# SHT30 precision sensor
sht30_validator = eg.HumidityValidator() \
    .with_range(0.0, 100.0) \
    .with_rate_limit(10.0) \
    .with_temperature_context(22.0)

# Validate sensor readings
readings = [45.2, 46.1, 44.8, 45.5, 47.0]
for reading in readings:
    try:
        valid_humidity = dht22_validator.validate(reading)
        print(f"✓ {valid_humidity}% RH")
    except eg.ValidationError as e:
        print(f"✗ {reading}% RH - {e}")
```

## PressureValidator

Validates atmospheric pressure readings with altitude compensation and barometric physics.

### Physics Background

Atmospheric pressure varies with altitude following barometric formula:
- **Sea Level**: ~1013.25 hPa standard atmosphere
- **Altitude Effect**: ~12 hPa decrease per 100m elevation
- **Weather Variation**: ±30 hPa typical weather-driven changes
- **Rate Limits**: Pressure changes limited by atmospheric dynamics

### Constructor and Configuration

```python
class PressureValidator:
    def __init__(self) -> None:
        """Create pressure validator with default atmospheric constraints.
        
        Default configuration:
        - Range: 540-1080 hPa (extreme weather + altitude range)
        - Rate limit: 50 hPa/s (rapid weather changes)
        - Sea level pressure: 1013.25 hPa (standard atmosphere)
        """
        
    def with_range(self, min_pressure: float, max_pressure: float) -> 'PressureValidator':
        """Set pressure range constraints.
        
        Typical ranges:
        - Sea level: 980-1050 hPa (normal weather)
        - 1000m elevation: 880-950 hPa
        - 3000m elevation: 680-750 hPa
        
        Args:
            min_pressure: Minimum valid pressure (hPa)
            max_pressure: Maximum valid pressure (hPa)
            
        Returns:
            Self for method chaining
        """
        
    def with_altitude(self, altitude_m: float) -> 'PressureValidator':
        """Set altitude for pressure compensation.
        
        Enables altitude-corrected validation using barometric formula:
        P(h) = P₀ * (1 - 0.0065h/T₀)^(gM/R*0.0065)
        
        Where:
        - h: altitude (m)
        - P₀: sea level pressure (hPa)  
        - T₀: standard temperature (288.15K)
        
        Args:
            altitude_m: Altitude above sea level (meters)
            
        Returns:
            Self for method chaining
        """
        
    def with_sea_level_pressure(self, pressure_hpa: float) -> 'PressureValidator':
        """Set reference sea level pressure.
        
        Args:
            pressure_hpa: Reference sea level pressure (hPa)
                         Default: 1013.25 hPa (standard atmosphere)
            
        Returns:
            Self for method chaining
        """
        
    def with_rate_limit(self, max_rate: float) -> 'PressureValidator':
        """Set maximum pressure change rate.
        
        Atmospheric pressure changes limited by:
        - Weather systems: 1-20 hPa/hour typical
        - Extreme weather: up to 50 hPa/hour (hurricanes)
        - Elevation changes: depends on movement speed
        
        Args:
            max_rate: Maximum pressure change rate (hPa/s)
            
        Returns:
            Self for method chaining
        """
```

### Altitude Compensation

```python
# Pressure validation with altitude compensation
sea_level_validator = eg.PressureValidator() \
    .with_range(980.0, 1050.0)  # Sea level range

mountain_validator = eg.PressureValidator() \
    .with_range(680.0, 750.0) \
    .with_altitude(3000.0)  # 3000m elevation

# Calculate expected pressure at altitude
expected_pressure = eg.calculate_pressure_at_altitude(1013.25, 1500.0)
print(f"Expected pressure at 1500m: {expected_pressure:.1f} hPa")

# Validate with altitude context
try:
    pressure = mountain_validator.validate(720.5)
    corrected = eg.pressure_to_sea_level(pressure, 3000.0)
    print(f"Pressure: {pressure} hPa (sea level: {corrected:.1f} hPa)")
except eg.ValidationError as e:
    print(f"Pressure validation failed: {e}")
```

### Usage Examples

```python
# Basic pressure validation
pressure_validator = eg.PressureValidator()

# BMP280 sensor at sea level
bmp280_validator = eg.PressureValidator() \
    .with_range(300.0, 1100.0) \
    .with_rate_limit(10.0)

# MS5611 barometer with altitude compensation
ms5611_validator = eg.PressureValidator() \
    .with_range(450.0, 1100.0) \
    .with_altitude(500.0) \
    .with_rate_limit(20.0)

# Validate barometer readings
readings = [1013.2, 1012.8, 1013.5, 1012.1, 1014.0]
for reading in readings:
    try:
        valid_pressure = bmp280_validator.validate(reading)
        print(f"✓ {valid_pressure} hPa")
    except eg.ValidationError as e:
        print(f"✗ {reading} hPa - {e}")
```

## Custom Validators

Create custom validators for specialized sensors:

```python
class VibrationValidator(eg.Validator):
    """Validate vibration sensor readings."""
    
    def __init__(self, max_amplitude: float = 10.0, frequency_range: tuple = (0.1, 1000.0)):
        """Initialize vibration validator.
        
        Args:
            max_amplitude: Maximum vibration amplitude (g-force)
            frequency_range: Valid frequency range (Hz)
        """
        self.max_amplitude = max_amplitude
        self.frequency_range = frequency_range
        
    def validate(self, value: float) -> float:
        """Validate vibration reading."""
        if math.isnan(value) or math.isinf(value):
            raise eg.InvalidValueError(value, "Vibration reading must be finite")
            
        if not (0.0 <= value <= self.max_amplitude):
            raise eg.OutOfRangeError(
                value, 
                (0.0, self.max_amplitude),
                "vibration_amplitude"
            )
            
        return value

# Use custom validator
vibration_validator = VibrationValidator(max_amplitude=5.0)
try:
    vibration = vibration_validator.validate(2.3)
    print(f"Vibration: {vibration}g")
except eg.ValidationError as e:
    print(f"Vibration validation failed: {e}")
```

## Validator Composition

Combine multiple validators for comprehensive sensor validation:

```python
class EnvironmentalValidator:
    """Combined environmental sensor validation."""
    
    def __init__(self):
        self.temp_validator = eg.TemperatureValidator() \
            .with_range(-20.0, 60.0) \
            .with_rate_limit(2.0)
            
        self.humidity_validator = eg.HumidityValidator() \
            .with_range(0.0, 100.0) \
            .with_rate_limit(5.0)
            
        self.pressure_validator = eg.PressureValidator() \
            .with_range(980.0, 1050.0) \
            .with_rate_limit(10.0)
    
    def validate_reading(self, temp: float, humidity: float, pressure: float) -> dict:
        """Validate complete environmental reading."""
        results = {}
        
        # Validate individual sensors
        try:
            results['temperature'] = self.temp_validator.validate(temp)
        except eg.ValidationError as e:
            results['temperature_error'] = str(e)
            
        try:
            results['humidity'] = self.humidity_validator.validate(humidity)
        except eg.ValidationError as e:
            results['humidity_error'] = str(e)
            
        try:
            results['pressure'] = self.pressure_validator.validate(pressure)
        except eg.ValidationError as e:
            results['pressure_error'] = str(e)
        
        # Cross-sensor validation
        if 'temperature' in results and 'humidity' in results:
            dew_point = eg.calculate_dew_point(results['temperature'], results['humidity'])
            if dew_point > results['temperature']:
                results['physics_error'] = f"Dew point {dew_point:.1f}°C exceeds air temperature"
        
        return results

# Use combined validator
env_validator = EnvironmentalValidator()
results = env_validator.validate_reading(23.5, 65.0, 1013.2)
print("Validation results:", results)
```

## Performance Considerations

### Optimization Tips

```python
# Pre-create validators for better performance
temp_validator = eg.TemperatureValidator().with_range(-20.0, 60.0)

# Batch validation when possible
readings = [23.1, 23.2, 22.9, 23.5]
valid_readings = []
for reading in readings:
    try:
        valid_readings.append(temp_validator.validate(reading))
    except eg.ValidationError:
        continue  # Skip invalid readings

# Use appropriate rate limits for your application
fast_validator = eg.TemperatureValidator().with_rate_limit(10.0)  # Fast changes OK
slow_validator = eg.TemperatureValidator().with_rate_limit(0.5)   # Slow thermal mass
```

### Memory Usage

Validators use minimal memory (typically <1KB each) and can be safely cached:

```python
# Cache validators for reuse
validator_cache = {
    'BME280_temp': eg.TemperatureValidator().with_range(-40.0, 85.0),
    'BME280_humidity': eg.HumidityValidator().with_range(0.0, 100.0),
    'BME280_pressure': eg.PressureValidator().with_range(300.0, 1100.0)
}

def validate_bme280_reading(sensor_type: str, value: float) -> float:
    """Validate BME280 sensor reading using cached validator."""
    validator = validator_cache.get(f'BME280_{sensor_type}')
    if validator is None:
        raise ValueError(f"Unknown BME280 sensor type: {sensor_type}")
    return validator.validate(value)
```

For integration with pipeline processing, see [Pipeline API](pipeline.md).
For multi-sensor fusion after validation, see [Fusion API](fusion.md).