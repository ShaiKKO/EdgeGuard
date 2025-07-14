# Validators API

Physics-aware validation components for sensor data.

## Validator Trait

Core trait for implementing custom validators:

```rust
pub trait Validator: Send {
    type Value;
    type Error;
    
    fn validate(&self, value: Self::Value) -> Result<Self::Value, Self::Error>;
    fn validate_with_context(&self, value: Self::Value, context: &ValidationContext) -> Result<Self::Value, Self::Error>;
}
```

### ValidationContext

Context information for validation:

```rust
pub struct ValidationContext {
    pub timestamp: Timestamp,
    pub quality: f32,
    pub sensor_id: InlineString,
}
```

## TemperatureValidator

Validates temperature readings with physics-aware constraints.

### Constructor

```rust
impl TemperatureValidator {
    pub fn new() -> Self;
    pub fn with_range(mut self, min: f32, max: f32) -> Self;
    pub fn with_rate_limit(mut self, max_rate: f32) -> Self;
    pub fn with_thermal_mass(mut self, mass_kg: f32) -> Self;
}
```

### Configuration

```rust
// Standard configuration
let validator = TemperatureValidator::new()
    .with_range(-20.0, 60.0)    // Operating range
    .with_rate_limit(5.0)       // 5°C/s maximum rate
    .with_thermal_mass(0.5);    // 0.5kg thermal mass
```

### Physics Constraints

- **Range validation**: -80°C to 125°C absolute limits
- **Rate limiting**: Based on thermal mass and heat capacity
- **Thermal dynamics**: Prevents instantaneous temperature changes

### Error Types

```rust
pub enum ValidationError {
    InvalidValue(f32),
    OutOfRange { value: f32, min: f32, max: f32 },
    RateOfChangeExceeded { rate: f32, max_rate: f32 },
    QualityTooLow { quality: f32, min_quality: f32 },
    PhysicsViolation(&'static str),
}
```

### Example Usage

```rust
let validator = TemperatureValidator::new()
    .with_range(-40.0, 85.0)
    .with_rate_limit(2.0);

match validator.validate(23.5) {
    Ok(value) => println!("Valid: {}°C", value),
    Err(ValidationError::OutOfRange { value, min, max }) => {
        println!("Temperature {} outside range [{}, {}]", value, min, max);
    }
    Err(ValidationError::RateOfChangeExceeded { rate, max_rate }) => {
        println!("Rate {} exceeds maximum {}", rate, max_rate);
    }
    Err(e) => println!("Validation error: {:?}", e),
}
```

## HumidityValidator

Validates humidity readings with dew point calculations.

### Constructor

```rust
impl HumidityValidator {
    pub fn new() -> Self;
    pub fn with_range(mut self, min: f32, max: f32) -> Self;
    pub fn with_rate_limit(mut self, max_rate: f32) -> Self;
    pub fn with_temperature_context(mut self, temperature: f32) -> Self;
}
```

### Configuration

```rust
let validator = HumidityValidator::new()
    .with_range(0.0, 100.0)     // 0-100% range
    .with_rate_limit(10.0)      // 10%/s maximum rate
    .with_temperature_context(25.0); // For dew point calculation
```

### Physics Constraints

- **Range validation**: 0-100% relative humidity
- **Rate limiting**: Based on air volume and vapor diffusion
- **Dew point validation**: Ensures physical consistency with temperature

### Cross-Sensor Validation

```rust
// Validate humidity with temperature context
let context = ValidationContext {
    timestamp: SystemTime.now(),
    quality: 0.95,
    sensor_id: InlineString::new("humidity_01").unwrap(),
};

let result = validator.validate_with_context(80.0, &context);
```

## PressureValidator

Validates pressure readings with altitude compensation.

### Constructor

```rust
impl PressureValidator {
    pub fn new() -> Self;
    pub fn with_range(mut self, min: f32, max: f32) -> Self;
    pub fn with_rate_limit(mut self, max_rate: f32) -> Self;
    pub fn with_altitude(mut self, altitude_m: f32) -> Self;
    pub fn with_sea_level_pressure(mut self, pressure_hpa: f32) -> Self;
}
```

### Configuration

```rust
let validator = PressureValidator::new()
    .with_range(300.0, 1100.0)  // hPa range
    .with_rate_limit(50.0)      // 50 hPa/s maximum rate
    .with_altitude(1000.0)      // 1000m altitude
    .with_sea_level_pressure(1013.25); // Standard sea level
```

### Physics Constraints

- **Range validation**: 300-1100 hPa typical range
- **Rate limiting**: Based on atmospheric dynamics
- **Altitude compensation**: Adjusts expected pressure for elevation

### Altitude Calculation

```rust
// Pressure decreases ~12 hPa per 100m elevation
let expected_pressure = validator.altitude_corrected_pressure(1013.25, 500.0);
```

## Custom Validators

Implement custom validators for specialized sensors:

```rust
struct VibrationValidator {
    max_amplitude: f32,
    frequency_range: (f32, f32),
}

impl Validator for VibrationValidator {
    type Value = f32;
    type Error = ValidationError;
    
    fn validate(&self, value: f32) -> Result<f32, ValidationError> {
        if value.is_nan() || value.is_infinite() {
            return Err(ValidationError::InvalidValue(value));
        }
        
        if value > self.max_amplitude {
            return Err(ValidationError::OutOfRange {
                value,
                min: 0.0,
                max: self.max_amplitude,
            });
        }
        
        Ok(value)
    }
}
```

## ValidationError

Complete error enumeration for all validation failures:

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    InvalidValue(f32),
    OutOfRange { value: f32, min: f32, max: f32 },
    RateOfChangeExceeded { rate: f32, max_rate: f32 },
    QualityTooLow { quality: f32, min_quality: f32 },
    PhysicsViolation(&'static str),
    CrossSensorInconsistency {
        primary_value: f32,
        secondary_value: f32,
        constraint: &'static str,
    },
    SensorTimeout { last_reading: Timestamp },
    CalibrationRequired { drift_detected: bool },
}
```

## Best Practices

### Sensor-Specific Configuration

```rust
// BME280 sensor
let temp_validator = TemperatureValidator::new()
    .with_range(-40.0, 85.0)    // Sensor operating range
    .with_rate_limit(2.0);      // Conservative rate limit

// DHT22 sensor
let humidity_validator = HumidityValidator::new()
    .with_range(0.0, 100.0)
    .with_rate_limit(5.0);
```

### Error Handling

```rust
match validator.validate(reading) {
    Ok(value) => process_valid_reading(value),
    Err(ValidationError::OutOfRange { .. }) => {
        // Log error, continue processing
        log::warn!("Reading out of range");
    }
    Err(ValidationError::RateOfChangeExceeded { .. }) => {
        // Possible sensor malfunction
        log::error!("Sensor changing too rapidly");
    }
    Err(ValidationError::PhysicsViolation(msg)) => {
        // Physics constraint violated
        log::error!("Physics violation: {}", msg);
    }
}
```

### Performance Optimization

```rust
// Pre-allocate validators
let temp_validator = TemperatureValidator::new()
    .with_range(-20.0, 60.0);

// Use consistent validator instances
let mut pipeline = Pipeline::<64>::builder()
    .add_stage(ValidationStage::new(temp_validator, SensorType::Temperature))
    .build();
```

## Testing

Mock validators for testing:

```rust
use edgeguard::validators::MockValidator;

let mut validator = MockValidator::new();
validator.expect_validate()
    .with(eq(23.5))
    .return_once(|v| Ok(v));

assert_eq!(validator.validate(23.5).unwrap(), 23.5);
```

## Integration

Validators integrate seamlessly with the pipeline system:

```rust
let mut pipeline = Pipeline::<256>::builder()
    .add_stage(ValidationStage::new(
        TemperatureValidator::new().with_range(-20.0, 60.0),
        SensorType::Temperature
    ))
    .add_stage(ValidationStage::new(
        HumidityValidator::new().with_range(0.0, 100.0),
        SensorType::Humidity
    ))
    .build();
```

This provides comprehensive validation with physics-aware constraints for reliable sensor data processing.