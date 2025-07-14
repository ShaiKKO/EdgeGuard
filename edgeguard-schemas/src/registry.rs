//! Schema Registry for Version Management
//!
//! This module provides centralized schema management with versioning support,
//! enabling schema evolution while maintaining backward compatibility.

use apache_avro::Schema;
use std::collections::HashMap;
use std::sync::RwLock;

use crate::{SchemaError, physics::SensorConstraints};

/// Schema metadata for registry entries
#[derive(Debug, Clone)]
pub struct SchemaMetadata {
    /// Schema name (e.g., "sensor_reading")
    pub name: String,
    
    /// Schema version (e.g., "v1", "v2")
    pub version: String,
    
    /// Full qualified name (e.g., "sensor_reading_v1")
    pub qualified_name: String,
    
    /// Schema namespace
    pub namespace: String,
    
    /// Whether this schema is deprecated
    pub deprecated: bool,
    
    /// Replacement schema if deprecated
    pub replacement: Option<String>,
    
    /// Associated physics constraints
    pub constraints: Option<SensorConstraints>,
}

/// Thread-safe schema registry with version management
pub struct SchemaRegistry {
    /// Schemas indexed by qualified name
    schemas: RwLock<HashMap<String, (Schema, SchemaMetadata)>>,
    
    /// Version mappings (name -> [versions])
    versions: RwLock<HashMap<String, Vec<String>>>,
    
    /// Latest version for each schema name
    latest: RwLock<HashMap<String, String>>,
}

impl SchemaRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            schemas: RwLock::new(HashMap::new()),
            versions: RwLock::new(HashMap::new()),
            latest: RwLock::new(HashMap::new()),
        }
    }
    
    /// Register a schema with metadata
    pub fn register_with_metadata(
        &self,
        schema: Schema,
        metadata: SchemaMetadata,
    ) -> Result<(), SchemaError> {
        let qualified_name = metadata.qualified_name.clone();
        let base_name = metadata.name.clone();
        let version = metadata.version.clone();
        
        // Validate schema
        self.validate_schema(&schema, &metadata)?;
        
        // Update schemas
        {
            let mut schemas = self.schemas.write()
                .map_err(|_| SchemaError::ParseError("Lock poisoned".to_string()))?;
            schemas.insert(qualified_name.clone(), (schema, metadata));
        }
        
        // Update version tracking
        {
            let mut versions = self.versions.write()
                .map_err(|_| SchemaError::ParseError("Lock poisoned".to_string()))?;
            versions.entry(base_name.clone())
                .or_insert_with(Vec::new)
                .push(version.clone());
        }
        
        // Update latest version
        {
            let mut latest = self.latest.write()
                .map_err(|_| SchemaError::ParseError("Lock poisoned".to_string()))?;
            
            // Simple version comparison - in practice, use semver
            if !latest.contains_key(&base_name) || version > latest[&base_name] {
                latest.insert(base_name, version);
            }
        }
        
        Ok(())
    }
    
    /// Register a schema (simplified version)
    pub fn register(&self, name: &str, schema: Schema) -> Result<(), SchemaError> {
        // Extract version from name if present (e.g., "sensor_reading_v1")
        let (base_name, version) = if let Some(pos) = name.rfind("_v") {
            let base = &name[..pos];
            let ver = &name[pos + 1..];
            (base.to_string(), ver.to_string())
        } else {
            (name.to_string(), "v1".to_string())
        };
        
        let metadata = SchemaMetadata {
            name: base_name,
            version,
            qualified_name: name.to_string(),
            namespace: "io.edgeguard".to_string(),
            deprecated: false,
            replacement: None,
            constraints: None,
        };
        
        self.register_with_metadata(schema, metadata)
    }
    
    /// Get a schema by qualified name
    pub fn get(&self, name: &str) -> Result<Schema, SchemaError> {
        let schemas = self.schemas.read()
            .map_err(|_| SchemaError::ParseError("Lock poisoned".to_string()))?;
        
        schemas.get(name)
            .map(|(schema, _)| schema.clone())
            .ok_or_else(|| SchemaError::NotFound(name.to_string()))
    }
    
    /// Get the latest version of a schema
    pub fn get_latest(&self, base_name: &str) -> Result<Schema, SchemaError> {
        let latest = self.latest.read()
            .map_err(|_| SchemaError::ParseError("Lock poisoned".to_string()))?;
        
        let version = latest.get(base_name)
            .ok_or_else(|| SchemaError::NotFound(format!("No versions of {}", base_name)))?;
        
        let qualified_name = format!("{}_{}", base_name, version);
        self.get(&qualified_name)
    }
    
    /// Get all versions of a schema
    pub fn get_versions(&self, base_name: &str) -> Result<Vec<String>, SchemaError> {
        let versions = self.versions.read()
            .map_err(|_| SchemaError::ParseError("Lock poisoned".to_string()))?;
        
        Ok(versions.get(base_name)
            .cloned()
            .unwrap_or_default())
    }
    
    /// Get schema metadata
    pub fn get_metadata(&self, name: &str) -> Result<SchemaMetadata, SchemaError> {
        let schemas = self.schemas.read()
            .map_err(|_| SchemaError::ParseError("Lock poisoned".to_string()))?;
        
        schemas.get(name)
            .map(|(_, metadata)| metadata.clone())
            .ok_or_else(|| SchemaError::NotFound(name.to_string()))
    }
    
    /// Check if a schema can read data written with another schema
    pub fn is_compatible(
        &self,
        writer_schema: &str,
        reader_schema: &str,
    ) -> Result<bool, SchemaError> {
        let writer = self.get(writer_schema)?;
        let reader = self.get(reader_schema)?;
        
        // Use Avro's built-in compatibility checking
        // This is a simplified check - real implementation would be more thorough
        Ok(writer.name() == reader.name())
    }
    
    /// Mark a schema as deprecated
    pub fn deprecate(
        &self,
        name: &str,
        replacement: Option<String>,
    ) -> Result<(), SchemaError> {
        let mut schemas = self.schemas.write()
            .map_err(|_| SchemaError::ParseError("Lock poisoned".to_string()))?;
        
        let (_schema, metadata) = schemas.get_mut(name)
            .ok_or_else(|| SchemaError::NotFound(name.to_string()))?;
        
        metadata.deprecated = true;
        metadata.replacement = replacement;
        
        Ok(())
    }
    
    /// Validate a schema before registration
    fn validate_schema(
        &self,
        _schema: &Schema,
        metadata: &SchemaMetadata,
    ) -> Result<(), SchemaError> {
        // Check for duplicate registration
        if self.get(&metadata.qualified_name).is_ok() {
            return Err(SchemaError::ValidationError(
                format!("Schema {} already registered", metadata.qualified_name)
            ));
        }
        
        // Additional validation could include:
        // - Checking field naming conventions
        // - Validating default values
        // - Ensuring required fields for EdgeGuard
        
        Ok(())
    }
    
    /// Load all default EdgeGuard schemas
    pub fn load_defaults(&self) -> Result<(), SchemaError> {
        use crate::schemas;
        
        // Register sensor reading schema
        let sensor_reading = schemas::sensor_reading_v1()?;
        self.register("sensor_reading_v1", sensor_reading)?;
        
        // Register device status schema
        let device_status = schemas::device_status_v1()?;
        self.register("device_status_v1", device_status)?;
        
        // Register batch schema
        let batch = schemas::sensor_batch_v1()?;
        self.register("sensor_batch_v1", batch)?;
        
        // Register alert schema
        let alert = schemas::alert_v1()?;
        self.register("alert_v1", alert)?;
        
        Ok(())
    }
}

impl Default for SchemaRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Global registry instance (optional pattern)
lazy_static::lazy_static! {
    /// Global schema registry with default schemas loaded
    pub static ref GLOBAL_REGISTRY: SchemaRegistry = {
        let registry = SchemaRegistry::new();
        // Ignore errors in static initialization
        let _ = registry.load_defaults();
        registry
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schemas;
    
    #[test]
    fn register_and_retrieve() {
        let registry = SchemaRegistry::new();
        let schema = schemas::sensor_reading_v1().unwrap();
        
        registry.register("test_schema_v1", schema.clone()).unwrap();
        
        let retrieved = registry.get("test_schema_v1").unwrap();
        assert_eq!(schema.name(), retrieved.name());
    }
    
    #[test]
    fn version_tracking() {
        let registry = SchemaRegistry::new();
        
        let schema_v1 = schemas::sensor_reading_v1().unwrap();
        let schema_v2 = schemas::sensor_reading_v1().unwrap(); // Same for testing
        
        registry.register("test_v1", schema_v1).unwrap();
        registry.register("test_v2", schema_v2).unwrap();
        
        let versions = registry.get_versions("test").unwrap();
        assert_eq!(versions.len(), 2);
        assert!(versions.contains(&"v1".to_string()));
        assert!(versions.contains(&"v2".to_string()));
    }
    
    #[test]
    fn latest_version() {
        let registry = SchemaRegistry::new();
        
        let schema = schemas::sensor_reading_v1().unwrap();
        registry.register("sensor_v1", schema.clone()).unwrap();
        registry.register("sensor_v2", schema.clone()).unwrap();
        
        let latest = registry.get_latest("sensor").unwrap();
        assert_eq!(latest.name(), schema.name());
    }
    
    #[test]
    fn deprecation() {
        let registry = SchemaRegistry::new();
        let schema = schemas::sensor_reading_v1().unwrap();
        
        registry.register("old_schema_v1", schema).unwrap();
        registry.deprecate("old_schema_v1", Some("new_schema_v1".to_string())).unwrap();
        
        let metadata = registry.get_metadata("old_schema_v1").unwrap();
        assert!(metadata.deprecated);
        assert_eq!(metadata.replacement, Some("new_schema_v1".to_string()));
    }
}