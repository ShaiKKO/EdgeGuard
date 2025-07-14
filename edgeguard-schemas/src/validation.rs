//! Schema-Based Validation
//!
//! This module provides validation functionality that uses Avro schemas
//! with embedded physics constraints to validate sensor data.

use apache_avro::{Schema, types::Value};

use crate::{SchemaError, physics::SensorConstraints};

/// Schema validator that combines Avro validation with physics constraints
pub struct SchemaValidator {
    /// The Avro schema for structural validation
    _schema: Schema,
    
    /// Physics constraints for value validation
    constraints: Option<SensorConstraints>,
    
    /// Whether to enforce cross-sensor validation
    enable_cross_validation: bool,
}

impl SchemaValidator {
    /// Create a new validator from schema
    pub fn new(schema: Schema) -> Self {
        // Try to extract constraints from schema
        let constraints = crate::physics::extract_from_schema(&schema);
        
        Self {
            _schema: schema,
            constraints,
            enable_cross_validation: true,
        }
    }
    
    /// Create validator with explicit constraints
    pub fn with_constraints(schema: Schema, constraints: SensorConstraints) -> Self {
        Self {
            _schema: schema,
            constraints: Some(constraints),
            enable_cross_validation: true,
        }
    }
    
    /// Disable cross-sensor validation
    pub fn disable_cross_validation(mut self) -> Self {
        self.enable_cross_validation = false;
        self
    }
    
    /// Validate a value against schema and physics constraints
    pub fn validate(&self, value: &Value) -> Result<ValidationReport, SchemaError> {
        let mut report = ValidationReport::new();
        
        // First, validate against Avro schema structure
        self.validate_structure(value, &mut report)?;
        
        // Then, apply physics constraints if available
        if let Some(constraints) = &self.constraints {
            self.validate_physics(value, constraints, &mut report)?;
        }
        
        // Finally, check cross-sensor rules if enabled
        if self.enable_cross_validation && self.constraints.is_some() {
            self.validate_cross_sensor(value, &mut report)?;
        }
        
        Ok(report)
    }
    
    /// Validate structural compliance with Avro schema
    fn validate_structure(
        &self,
        value: &Value,
        report: &mut ValidationReport,
    ) -> Result<(), SchemaError> {
        // For now, we do basic validation for records
        // Full Avro schema validation would require traversing the schema tree
        match value {
            Value::Record(fields) => {
                // Basic check that we have a record
                if fields.is_empty() {
                    report.add_error(ValidationIssue {
                        issue_type: IssueType::MissingField,
                        field: None,
                        message: "Record has no fields".to_string(),
                        severity: Severity::Error,
                    });
                }
                
                // Check for required fields based on common sensor reading fields
                let field_map: std::collections::HashMap<_, _> = fields
                    .iter()
                    .map(|(k, v)| (k.as_str(), v))
                    .collect();
                
                // Check required fields for sensor readings
                let required_fields = ["sensor_id", "timestamp", "value", "unit"];
                for field_name in &required_fields {
                    if !field_map.contains_key(field_name) {
                        report.add_error(ValidationIssue {
                            issue_type: IssueType::MissingField,
                            field: Some(field_name.to_string()),
                            message: format!("Required field '{}' is missing", field_name),
                            severity: Severity::Error,
                        });
                    }
                }
            }
            _ => {
                report.add_error(ValidationIssue {
                    issue_type: IssueType::TypeMismatch,
                    field: None,
                    message: "Expected record type".to_string(),
                    severity: Severity::Error,
                });
            }
        }
        
        Ok(())
    }
    
    /// Validate physics constraints
    fn validate_physics(
        &self,
        value: &Value,
        constraints: &SensorConstraints,
        report: &mut ValidationReport,
    ) -> Result<(), SchemaError> {
        // Extract numeric value from Avro record
        let numeric_value = self.extract_numeric_value(value)?;
        
        // Check absolute limits
        if let Some(min) = constraints.absolute_min {
            if numeric_value < min {
                report.add_error(ValidationIssue {
                    issue_type: IssueType::PhysicsViolation,
                    field: Some("value".to_string()),
                    message: format!(
                        "Value {} is below absolute minimum {} {}",
                        numeric_value, min, constraints.display_unit
                    ),
                    severity: Severity::Error,
                });
            }
        }
        
        if let Some(max) = constraints.absolute_max {
            if numeric_value > max {
                report.add_error(ValidationIssue {
                    issue_type: IssueType::PhysicsViolation,
                    field: Some("value".to_string()),
                    message: format!(
                        "Value {} exceeds absolute maximum {} {}",
                        numeric_value, max, constraints.display_unit
                    ),
                    severity: Severity::Error,
                });
            }
        }
        
        // Check typical ranges (warnings only)
        if let Some(min) = constraints.typical_min {
            if numeric_value < min {
                report.add_warning(ValidationIssue {
                    issue_type: IssueType::UnusualValue,
                    field: Some("value".to_string()),
                    message: format!(
                        "Value {} is below typical minimum {} {}",
                        numeric_value, min, constraints.display_unit
                    ),
                    severity: Severity::Warning,
                });
            }
        }
        
        Ok(())
    }
    
    /// Validate cross-sensor constraints
    fn validate_cross_sensor(
        &self,
        _value: &Value,
        report: &mut ValidationReport,
    ) -> Result<(), SchemaError> {
        // This would require access to other sensor values
        // For now, we just note which validations are required
        if let Some(constraints) = &self.constraints {
            for rule in &constraints.cross_validation {
                report.add_info(ValidationIssue {
                    issue_type: IssueType::CrossValidationRequired,
                    field: None,
                    message: format!(
                        "Cross-validation required with {} sensor",
                        rule.requires_sensor
                    ),
                    severity: Severity::Info,
                });
            }
        }
        
        Ok(())
    }
    
    /// Extract numeric value from Avro record
    fn extract_numeric_value(&self, value: &Value) -> Result<f64, SchemaError> {
        match value {
            Value::Record(fields) => {
                // Look for "value" field
                fields.iter()
                    .find(|(name, _)| name == "value")
                    .and_then(|(_, val)| match val {
                        Value::Double(d) => Some(*d),
                        Value::Float(f) => Some(*f as f64),
                        Value::Int(i) => Some(*i as f64),
                        Value::Long(l) => Some(*l as f64),
                        _ => None,
                    })
                    .ok_or_else(|| SchemaError::ValidationError(
                        "No numeric 'value' field found".to_string()
                    ))
            }
            Value::Double(d) => Ok(*d),
            Value::Float(f) => Ok(*f as f64),
            Value::Int(i) => Ok(*i as f64),
            Value::Long(l) => Ok(*l as f64),
            _ => Err(SchemaError::ValidationError(
                "Cannot extract numeric value".to_string()
            )),
        }
    }
}

/// Validation report containing all issues found
#[derive(Debug, Default)]
pub struct ValidationReport {
    /// Validation errors (must be fixed)
    pub errors: Vec<ValidationIssue>,
    
    /// Validation warnings (should be reviewed)
    pub warnings: Vec<ValidationIssue>,
    
    /// Informational messages
    pub info: Vec<ValidationIssue>,
}

impl ValidationReport {
    /// Create new empty report
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Check if validation passed (no errors)
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
    
    /// Add an error
    pub fn add_error(&mut self, issue: ValidationIssue) {
        self.errors.push(issue);
    }
    
    /// Add a warning
    pub fn add_warning(&mut self, issue: ValidationIssue) {
        self.warnings.push(issue);
    }
    
    /// Add info
    pub fn add_info(&mut self, issue: ValidationIssue) {
        self.info.push(issue);
    }
    
    /// Get total issue count
    pub fn total_issues(&self) -> usize {
        self.errors.len() + self.warnings.len() + self.info.len()
    }
}

/// Individual validation issue
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Type of issue
    pub issue_type: IssueType,
    
    /// Field that caused the issue (if applicable)
    pub field: Option<String>,
    
    /// Human-readable message
    pub message: String,
    
    /// Issue severity
    pub severity: Severity,
}

/// Types of validation issues
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueType {
    /// Required field is missing
    MissingField,
    
    /// Type doesn't match schema
    TypeMismatch,
    
    /// Value violates physics laws
    PhysicsViolation,
    
    /// Value is unusual but not impossible
    UnusualValue,
    
    /// Rate of change too high
    RateViolation,
    
    /// Cross-sensor validation needed
    CrossValidationRequired,
    
    /// Cross-sensor validation failed
    CrossValidationFailed,
}

/// Issue severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational only
    Info,
    
    /// Should be reviewed
    Warning,
    
    /// Must be fixed
    Error,
}

/// Convert EdgeGuard validation error to schema validation report
pub fn from_validation_error(
    error: &edgeguard_core::errors::ValidationError,
) -> ValidationReport {
    let mut report = ValidationReport::new();
    
    let issue = match error {
        edgeguard_core::errors::ValidationError::OutOfRange { value, min, max } => {
            ValidationIssue {
                issue_type: IssueType::PhysicsViolation,
                field: Some("value".to_string()),
                message: format!("Value {} outside range [{}, {}]", value, min, max),
                severity: Severity::Error,
            }
        }
        edgeguard_core::errors::ValidationError::RateExceeded { rate, max_rate } => {
            ValidationIssue {
                issue_type: IssueType::RateViolation,
                field: Some("value".to_string()),
                message: format!("Rate {} exceeds maximum {}", rate, max_rate),
                severity: Severity::Error,
            }
        }
        edgeguard_core::errors::ValidationError::CrossValidationFailed { reason } => {
            ValidationIssue {
                issue_type: IssueType::CrossValidationFailed,
                field: None,
                message: reason.to_string(),
                severity: Severity::Error,
            }
        }
        _ => {
            ValidationIssue {
                issue_type: IssueType::PhysicsViolation,
                field: None,
                message: format!("Validation error: {:?}", error),
                severity: Severity::Error,
            }
        }
    };
    
    report.add_error(issue);
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schemas;
    
    #[test]
    fn validate_sensor_reading() {
        let schema = schemas::sensor_reading_v1().unwrap();
        let validator = SchemaValidator::new(schema);
        
        // Create a test value
        let value = Value::Record(vec![
            ("sensor_id".to_string(), Value::String("test".to_string())),
            ("timestamp".to_string(), Value::Long(1234567890)),
            ("value".to_string(), Value::Double(25.0)),
            ("unit".to_string(), Value::String("celsius".to_string())),
        ]);
        
        let report = validator.validate(&value).unwrap();
        assert!(report.is_valid());
    }
    
    #[test]
    fn detect_missing_field() {
        let schema = schemas::sensor_reading_v1().unwrap();
        let validator = SchemaValidator::new(schema);
        
        // Missing required fields
        let value = Value::Record(vec![
            ("sensor_id".to_string(), Value::String("test".to_string())),
        ]);
        
        let report = validator.validate(&value).unwrap();
        assert!(!report.is_valid());
        assert!(report.errors.iter().any(|e| e.issue_type == IssueType::MissingField));
    }
    
    #[test]
    fn physics_validation() {
        let schema = schemas::sensor_reading_v1().unwrap();
        let constraints = crate::physics::SensorConstraints::temperature_celsius();
        let validator = SchemaValidator::with_constraints(schema, constraints);
        
        // Temperature below absolute zero
        let value = Value::Record(vec![
            ("sensor_id".to_string(), Value::String("test".to_string())),
            ("timestamp".to_string(), Value::Long(1234567890)),
            ("value".to_string(), Value::Double(-300.0)),
            ("unit".to_string(), Value::String("celsius".to_string())),
        ]);
        
        let report = validator.validate(&value).unwrap();
        assert!(!report.is_valid());
        assert!(report.errors.iter().any(|e| e.issue_type == IssueType::PhysicsViolation));
    }
}