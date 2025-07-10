//! Avro schemas for EdgeGuard sensor data
//!
//! Provides schemas with embedded physics constraints for validation.
//! Schemas are versioned and backward-compatible by default.

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

/// Physics constraints that can be embedded in schemas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsConstraints {
    /// Minimum valid value
    pub min: Option<f64>,
    
    /// Maximum valid value
    pub max: Option<f64>,
    
    /// Maximum rate of change per second
    pub max_rate_change: Option<f64>,
    
    /// Unit of measurement
    pub unit: String,
    
    /// Additional validation rules
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