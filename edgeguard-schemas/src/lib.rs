//! Avro Schema Definitions with Embedded Physics Constraints
//!
//! ## Overview
//!
//! This module bridges the gap between data serialization and physics-based validation
//! by embedding physical constraints directly into Avro schemas. This approach ensures
//! that data validation rules travel with the data, enabling consistent validation
//! across different systems and languages.
//!
//! ## Why Avro?
//!
//! Apache Avro was chosen for EdgeGuard because:
//!
//! 1. **Compact Binary Format**: Critical for bandwidth-constrained IoT devices
//! 2. **Schema Evolution**: Sensors and requirements change over time
//! 3. **Cross-Language Support**: C, Python, Java, and more can parse our data
//! 4. **Self-Describing**: Schema travels with data for long-term storage
//! 5. **Fast Serialization**: Minimal CPU overhead on edge devices
//!
//! ## Schema Design Philosophy
//!
//! ### Physics Constraints in Schemas
//!
//! Traditional Avro schemas only define data types. EdgeGuard extends this with
//! physics constraints embedded as schema properties:
//!
//! ```json
//! {
//!   "name": "temperature",
//!   "type": "double",
//!   "edgeguard.constraints": {
//!     "min": -273.15,
//!     "max": 1000.0,
//!     "maxRateChange": 10.0,
//!     "unit": "celsius"
//!   }
//! }
//! ```
//!
//! This ensures any system reading the data understands its physical limits.
//!
//! ### Schema Evolution Strategy
//!
//! Sensors evolve, but data lives forever. Our evolution strategy:
//!
//! 1. **Always Append**: New fields are added, never removed
//! 2. **Default Values**: Old consumers ignore new fields gracefully
//! 3. **Version in Name**: `sensor_reading_v1`, `sensor_reading_v2`
//! 4. **Compatibility Rules**: v2 can read v1, but not vice versa
//!
//! ### Compression Considerations
//!
//! Avro's binary encoding is already compact, but for time series:
//! - Use arrays for batch readings
//! - Delta encoding for timestamps
//! - Quantization for redundant precision
//! - Compression algorithms (Snappy, ZSTD) for transport
//!
//! ## Integration with Validators
//!
//! Schemas can generate validators automatically:
//!
//! ```rust
//! // Future functionality
//! let schema = SchemaRegistry::new().get("temperature_v1")?;
//! let validator = TemperatureValidator::from_schema(schema)?;
//! ```
//!
//! This ensures validation rules stay synchronized with data definitions.
//!
//! ## Schema Registry Pattern
//!
//! The registry pattern provides:
//! - **Central Management**: All schemas in one place
//! - **Version Control**: Track schema evolution
//! - **Runtime Loading**: Update schemas without recompiling
//! - **Multi-Tenant**: Different schemas for different deployments
//!
//! ## Common Schema Patterns
//!
//! ### Sensor Reading Schema
//! ```json
//! {
//!   "type": "record",
//!   "name": "SensorReading",
//!   "fields": [
//!     {"name": "timestamp", "type": "long"},
//!     {"name": "value", "type": "double"},
//!     {"name": "quality", "type": "float"},
//!     {"name": "sensor_id", "type": "string"}
//!   ]
//! }
//! ```
//!
//! ### Batch Reading Schema (Efficient)
//! ```json
//! {
//!   "type": "record",
//!   "name": "BatchReading",
//!   "fields": [
//!     {"name": "base_timestamp", "type": "long"},
//!     {"name": "interval_ms", "type": "int"},
//!     {"name": "values", "type": {"type": "array", "items": "double"}},
//!     {"name": "sensor_id", "type": "string"}
//!   ]
//! }
//! ```
//!
//! ## Usage Example
//!
//! ```rust
//! use edgeguard_schemas::{SchemaRegistry, PhysicsConstraints};
//!
//! // Initialize registry with default schemas
//! let mut registry = SchemaRegistry::new();
//! registry.load_defaults()?;
//!
//! // Get a schema for serialization
//! let schema = registry.get("sensor_reading_v1").unwrap();
//!
//! // Future: Extract constraints for validation
//! let constraints = PhysicsConstraints::from_schema(schema)?;
//! assert_eq!(constraints.unit, "celsius");
//! assert_eq!(constraints.min, Some(-80.0));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use apache_avro::Schema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod schemas;

/// Schema registry for managing multiple schema versions
pub struct SchemaRegistry {
    schemas: HashMap<String, Schema>,
}

impl SchemaRegistry {
    /// Create new empty registry
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
        }
    }
    
    /// Register a schema
    pub fn register(&mut self, name: &str, schema: Schema) {
        self.schemas.insert(name.to_string(), schema);
    }
    
    /// Get schema by name
    pub fn get(&self, name: &str) -> Option<&Schema> {
        self.schemas.get(name)
    }
    
    /// Load default EdgeGuard schemas
    pub fn load_defaults(&mut self) -> Result<(), SchemaError> {
        // Load sensor reading schema
        let sensor_reading = schemas::sensor_reading_v1()?;
        self.register("sensor_reading_v1", sensor_reading);
        
        // Load device status schema
        let device_status = schemas::device_status_v1()?;
        self.register("device_status_v1", device_status);
        
        Ok(())
    }
}

/// Schema-related errors
#[derive(Debug, thiserror_no_std::Error)]
pub enum SchemaError {
    #[error("Failed to parse schema: {0}")]
    ParseError(String),
    
    #[error("Schema not found: {0}")]
    NotFound(String),
    
    #[error("Validation failed: {0}")]
    ValidationError(String),
}

/// Physics constraints that can be embedded in Avro schemas
/// 
/// These constraints extend standard Avro schemas with physical validation rules
/// that travel with the data. This ensures consistent validation across different
/// systems, languages, and time periods.
/// 
/// ## Design Rationale
/// 
/// Why embed constraints in schemas rather than code?
/// 
/// 1. **Data Longevity**: Data outlives code. A sensor reading from 2020 should
///    still be validatable in 2030 using its original constraints.
/// 
/// 2. **Cross-Language**: A Python analytics system should apply the same validation
///    as the Rust edge device that generated the data.
/// 
/// 3. **Configuration as Code**: Constraints can be updated without recompiling
///    by updating schema definitions.
/// 
/// ## Constraint Types
/// 
/// - **Range**: Physical limits (min/max)
/// - **Rate**: Maximum change per time unit  
/// - **Cross-Sensor**: Relationships between measurements
/// - **Environmental**: Context-dependent rules
/// 
/// ## Example
/// 
/// ```json
/// {
///   "edgeguard.constraints": {
///     "min": -40.0,
///     "max": 85.0,
///     "maxRateChange": 5.0,
///     "unit": "celsius",
///     "rules": [
///       {
///         "ruleType": "cross_sensor",
///         "params": {
///           "requires": "humidity",
///           "constraint": "dew_point_valid"
///         }
///       }
///     ]
///   }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsConstraints {
    /// Minimum valid value based on physics
    /// 
    /// Example: -273.15 for temperature (absolute zero)
    pub min: Option<f64>,
    
    /// Maximum valid value based on physics or sensor limits
    /// 
    /// Example: 125.0 for consumer temperature sensors
    pub max: Option<f64>,
    
    /// Maximum rate of change per second
    /// 
    /// Based on physical properties like thermal mass
    pub max_rate_change: Option<f64>,
    
    /// Unit of measurement (SI preferred)
    /// 
    /// Examples: "celsius", "pascal", "percent"
    pub unit: String,
    
    /// Additional validation rules beyond simple ranges
    /// 
    /// These handle complex physics relationships
    pub rules: Vec<ValidationRule>,
}

/// Custom validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule type (e.g., "cross_sensor", "environmental")
    pub rule_type: String,
    
    /// Parameters for the rule
    pub params: HashMap<String, serde_json::Value>,
}

impl Default for SchemaRegistry {
    fn default() -> Self {
        Self::new()
    }
}