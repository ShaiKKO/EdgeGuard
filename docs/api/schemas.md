# Schemas API

Avro schema validation and physics-aware constraints for sensor data.

## SchemaRegistry

Central registry for managing sensor data schemas with version control.

### Constructor

```rust
impl SchemaRegistry {
    pub fn new() -> Self;
    pub fn with_capacity(capacity: usize) -> Self;
    pub fn register_schema(&mut self, name: &str, schema: Schema) -> Result<SchemaId, SchemaError>;
    pub fn get_schema(&self, id: SchemaId) -> Option<&Schema>;
    pub fn get_schema_by_name(&self, name: &str) -> Option<&Schema>;
    pub fn list_schemas(&self) -> Vec<&str>;
}
```

### Configuration

```rust
// Standard registry setup
let mut registry = SchemaRegistry::new();

// Register sensor schemas
let temp_schema_id = registry.register_schema(
    "temperature_reading",
    Schema::from_str(TEMPERATURE_SCHEMA_JSON)?
)?;

let humidity_schema_id = registry.register_schema(
    "humidity_reading", 
    Schema::from_str(HUMIDITY_SCHEMA_JSON)?
)?;
```

### Schema Management

```rust
// List all registered schemas
let schema_names = registry.list_schemas();
println!("Available schemas: {:?}", schema_names);

// Retrieve schema by ID
if let Some(schema) = registry.get_schema(temp_schema_id) {
    println!("Temperature schema: {}", schema.canonical_form());
}

// Retrieve schema by name
if let Some(schema) = registry.get_schema_by_name("temperature_reading") {
    // Use schema for validation
}
```

## Schema

Avro schema with embedded physics constraints for sensor validation.

### Constructor

```rust
impl Schema {
    pub fn from_str(schema_json: &str) -> Result<Self, SchemaError>;
    pub fn from_value(schema_value: serde_json::Value) -> Result<Self, SchemaError>;
    pub fn with_physics_constraints(mut self, constraints: PhysicsConstraints) -> Self;
    pub fn canonical_form(&self) -> String;
    pub fn fingerprint(&self) -> u64;
}
```

### Physics Constraints

```rust
// Define physics-aware constraints
let constraints = PhysicsConstraints::new()
    .with_range(-40.0, 85.0)
    .with_rate_limit(5.0)
    .with_units("celsius")
    .with_precision(0.1);

let schema = Schema::from_str(TEMPERATURE_SCHEMA_JSON)?
    .with_physics_constraints(constraints);
```

### Schema Validation

```rust
// Validate data against schema
let data = serde_json::json!({
    "sensor_id": "temp_001",
    "temperature": 23.5,
    "timestamp": 1640995200000,
    "quality": 0.95
});

match schema.validate(&data) {
    Ok(()) => println!("Data valid"),
    Err(SchemaError::ValidationFailed { field, reason }) => {
        eprintln!("Validation failed for {}: {}", field, reason);
    }
}
```

## SchemaValidator

High-performance validator that combines Avro schema validation with physics constraints.

### Constructor

```rust
impl SchemaValidator {
    pub fn new(schema: &Schema) -> Self;
    pub fn with_registry(registry: &SchemaRegistry, schema_name: &str) -> Result<Self, SchemaError>;
    pub fn validate(&self, data: &serde_json::Value) -> Result<(), SchemaError>;
    pub fn validate_bytes(&self, data: &[u8]) -> Result<(), SchemaError>;
    pub fn validate_with_context(&self, data: &serde_json::Value, context: &ValidationContext) -> Result<(), SchemaError>;
}
```

### Validation Workflow

```rust
// Create validator from schema
let validator = SchemaValidator::new(&temperature_schema);

// Validate sensor readings
let sensor_data = serde_json::json!({
    "sensor_id": "temp_001",
    "temperature": 23.5,
    "timestamp": 1640995200000,
    "quality": 0.95
});

// Basic validation
match validator.validate(&sensor_data) {
    Ok(()) => process_valid_data(&sensor_data),
    Err(e) => handle_validation_error(e),
}

// Context-aware validation
let context = ValidationContext {
    timestamp: Timestamp::from_millis(1640995200000),
    quality: 0.95,
    sensor_id: InlineString::new("temp_001").unwrap(),
};

validator.validate_with_context(&sensor_data, &context)?;
```

### Batch Validation

```rust
// Validate multiple readings efficiently
let readings = vec![reading1, reading2, reading3];
let mut valid_count = 0;
let mut error_count = 0;

for reading in readings {
    match validator.validate(&reading) {
        Ok(()) => valid_count += 1,
        Err(_) => error_count += 1,
    }
}

println!("Validated {} readings, {} errors", valid_count, error_count);
```

## Built-in Schemas

EdgeGuard provides pre-defined schemas for common sensor types.

### Temperature Schema

```rust
pub const TEMPERATURE_SCHEMA: &str = r#"
{
    "type": "record",
    "name": "TemperatureReading",
    "fields": [
        {"name": "sensor_id", "type": "string"},
        {"name": "temperature", "type": "float"},
        {"name": "timestamp", "type": "long"},
        {"name": "quality", "type": "float", "default": 1.0}
    ],
    "physics": {
        "temperature": {
            "range": [-40.0, 85.0],
            "rate_limit": 5.0,
            "units": "celsius"
        }
    }
}
"#;
```

### Humidity Schema

```rust
pub const HUMIDITY_SCHEMA: &str = r#"
{
    "type": "record",
    "name": "HumidityReading",
    "fields": [
        {"name": "sensor_id", "type": "string"},
        {"name": "humidity", "type": "float"},
        {"name": "temperature", "type": ["null", "float"], "default": null},
        {"name": "timestamp", "type": "long"},
        {"name": "quality", "type": "float", "default": 1.0}
    ],
    "physics": {
        "humidity": {
            "range": [0.0, 100.0],
            "rate_limit": 10.0,
            "units": "percent"
        }
    }
}
"#;
```

### Pressure Schema

```rust
pub const PRESSURE_SCHEMA: &str = r#"
{
    "type": "record",
    "name": "PressureReading",
    "fields": [
        {"name": "sensor_id", "type": "string"},
        {"name": "pressure", "type": "float"},
        {"name": "altitude", "type": ["null", "float"], "default": null},
        {"name": "timestamp", "type": "long"},
        {"name": "quality", "type": "float", "default": 1.0}
    ],
    "physics": {
        "pressure": {
            "range": [300.0, 1100.0],
            "rate_limit": 10.0,
            "units": "hPa"
        }
    }
}
"#;
```

## PhysicsConstraints

Physics-aware constraints that extend Avro schema validation.

### Constructor

```rust
impl PhysicsConstraints {
    pub fn new() -> Self;
    pub fn with_range(mut self, min: f32, max: f32) -> Self;
    pub fn with_rate_limit(mut self, max_rate: f32) -> Self;
    pub fn with_units(mut self, units: &str) -> Self;
    pub fn with_precision(mut self, precision: f32) -> Self;
    pub fn with_dependencies(mut self, dependencies: Vec<String>) -> Self;
}
```

### Constraint Types

```rust
// Range constraints
let temp_constraints = PhysicsConstraints::new()
    .with_range(-40.0, 85.0)
    .with_units("celsius");

// Rate limiting
let pressure_constraints = PhysicsConstraints::new()
    .with_range(300.0, 1100.0)
    .with_rate_limit(50.0)  // 50 hPa/s maximum
    .with_units("hPa");

// Cross-sensor dependencies
let humidity_constraints = PhysicsConstraints::new()
    .with_range(0.0, 100.0)
    .with_dependencies(vec!["temperature".to_string()])
    .with_units("percent");
```

### Validation Integration

```rust
// Apply constraints during validation
let validator = SchemaValidator::new(&schema);
let constraints = PhysicsConstraints::new()
    .with_range(0.0, 100.0)
    .with_rate_limit(20.0);

// Constraints are automatically applied during validation
match validator.validate(&data) {
    Ok(()) => println!("Data passes schema and physics validation"),
    Err(SchemaError::PhysicsViolation { field, constraint, value }) => {
        eprintln!("Physics constraint violated for {}: {} = {}", field, constraint, value);
    }
}
```

## Schema Evolution

Support for schema versioning and backward compatibility.

### Version Management

```rust
// Register multiple versions of a schema
let v1_schema = Schema::from_str(TEMPERATURE_SCHEMA_V1)?;
let v2_schema = Schema::from_str(TEMPERATURE_SCHEMA_V2)?;

let mut registry = SchemaRegistry::new();
let v1_id = registry.register_schema("temperature_v1", v1_schema)?;
let v2_id = registry.register_schema("temperature_v2", v2_schema)?;

// Set default version
registry.set_default_version("temperature", v2_id)?;
```

### Compatibility Checking

```rust
// Check schema compatibility
let compatibility = registry.check_compatibility("temperature_v1", "temperature_v2")?;

match compatibility {
    Compatibility::Full => println!("Schemas are fully compatible"),
    Compatibility::Forward => println!("Forward compatible only"),
    Compatibility::Backward => println!("Backward compatible only"),
    Compatibility::None => println!("Schemas are incompatible"),
}
```

### Migration Support

```rust
// Migrate data between schema versions
let migrator = SchemaMigrator::new(&v1_schema, &v2_schema);

let old_data = serde_json::json!({
    "sensor_id": "temp_001",
    "temp": 23.5,  // Old field name
    "timestamp": 1640995200000
});

let new_data = migrator.migrate(&old_data)?;
// Results in: {"sensor_id": "temp_001", "temperature": 23.5, "timestamp": 1640995200000}
```

## Performance Characteristics

### Validation Performance

- **Schema validation**: <50μs per record
- **Physics constraint checking**: <20μs per constraint
- **Batch validation**: 10k+ records/sec on Cortex-M4
- **Memory usage**: <1KB per validator instance

### Optimization Techniques

```rust
// Pre-compile schemas for better performance
let compiled_schema = CompiledSchema::from_schema(&schema)?;
let validator = SchemaValidator::from_compiled(&compiled_schema);

// Use schema fingerprints for caching
let fingerprint = schema.fingerprint();
if let Some(cached_validator) = validator_cache.get(&fingerprint) {
    // Use cached validator
} else {
    // Create new validator and cache it
}
```

## Error Handling

### SchemaError

```rust
pub enum SchemaError {
    ParseError(String),
    ValidationFailed { field: String, reason: String },
    PhysicsViolation { field: String, constraint: String, value: f32 },
    SchemaNotFound(String),
    IncompatibleTypes { expected: String, actual: String },
    RegistryError(String),
}
```

### Error Recovery

```rust
match validator.validate(&sensor_data) {
    Ok(()) => process_valid_data(&sensor_data),
    Err(SchemaError::ValidationFailed { field, reason }) => {
        log::warn!("Schema validation failed for {}: {}", field, reason);
        // Continue with next record
    }
    Err(SchemaError::PhysicsViolation { field, constraint, value }) => {
        log::error!("Physics constraint {} violated for {}: {}", constraint, field, value);
        // May indicate sensor malfunction
    }
    Err(e) => {
        log::error!("Schema validation error: {:?}", e);
        return Err(e);
    }
}
```

## Integration Patterns

### Pipeline Integration

```rust
use edgeguard::pipeline::{Pipeline, ValidationStage};

// Create schema-aware validation stage
let schema_validator = SchemaValidator::new(&temperature_schema);
let validation_stage = ValidationStage::new(schema_validator, SensorType::Temperature);

let pipeline = Pipeline::<256>::builder()
    .add_stage(validation_stage)
    .build();
```

### Stream Processing

```rust
use edgeguard::stream::{Stream, StreamProcessor};

// Validate streaming data with schemas
let mut stream_processor = StreamProcessor::new(sensor_stream, pipeline);

while let Ok(Some(result)) = stream_processor.process_next() {
    match result {
        Event::ValidationResult { status: ValidationStatus::Valid, .. } => {
            // Process valid data
        }
        Event::ValidationResult { status, .. } => {
            // Handle validation failures
        }
    }
}
```

### Network Integration

```rust
use edgeguard::connectors::mqtt::MqttConnector;

// Validate incoming MQTT messages
let mut mqtt_client = MqttConnector::new(config)?;
mqtt_client.subscribe("sensors/+/data", 1)?;

loop {
    let messages = mqtt_client.poll()?;
    for msg in messages {
        let data: serde_json::Value = serde_json::from_slice(&msg.payload)?;
        
        match schema_validator.validate(&data) {
            Ok(()) => {
                // Process valid sensor data
                let event = create_event_from_data(&data);
                pipeline.push_event(event);
            }
            Err(e) => {
                log::warn!("Invalid sensor data from {}: {:?}", msg.topic, e);
            }
        }
    }
}
```

## Best Practices

### Schema Design

```rust
// Use appropriate data types
{
    "name": "temperature", 
    "type": "float"  // Not "double" for embedded efficiency
}

// Include quality indicators
{
    "name": "quality",
    "type": "float",
    "default": 1.0,
    "doc": "Quality indicator from 0.0 to 1.0"
}

// Add physics constraints
"physics": {
    "temperature": {
        "range": [-40.0, 85.0],
        "rate_limit": 5.0,
        "units": "celsius"
    }
}
```

### Performance Optimization

```rust
// Pre-compile frequently used schemas
let compiled_schemas: HashMap<String, CompiledSchema> = schemas
    .iter()
    .map(|(name, schema)| (name.clone(), CompiledSchema::from_schema(schema)))
    .collect();

// Use schema registry for caching
let registry = SchemaRegistry::with_capacity(100);
let validator = SchemaValidator::with_registry(&registry, "temperature")?;
```

### Error Handling

```rust
// Implement graceful degradation
match validator.validate(&data) {
    Ok(()) => process_data(&data),
    Err(SchemaError::PhysicsViolation { .. }) => {
        // Log but continue processing
        log::warn!("Physics constraint violated, continuing");
        process_data_with_warning(&data);
    }
    Err(SchemaError::ValidationFailed { .. }) => {
        // Skip invalid data
        log::error!("Invalid data format, skipping");
    }
}
```

This schemas API provides comprehensive data validation with physics-aware constraints, enabling robust sensor data processing with strong type safety and performance characteristics suitable for edge deployment.