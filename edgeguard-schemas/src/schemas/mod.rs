//! EdgeGuard Avro schemas with physics constraints
//!
//! Each schema includes metadata for physics-based validation.
//! Schemas follow semantic versioning for compatibility.

use apache_avro::Schema;
use serde_json::json;

use crate::SchemaError;

/// Sensor reading schema v1.0.0
pub fn sensor_reading_v1() -> Result<Schema, SchemaError> {
    let schema_json = json!({
        "namespace": "io.edgeguard.sensors.v1",
        "type": "record",
        "name": "SensorReading",
        "doc": "Sensor reading with physics constraints",
        "fields": [
            {
                "name": "sensor_id",
                "type": "string",
                "doc": "Unique sensor identifier"
            },
            {
                "name": "timestamp",
                "type": "long",
                "logicalType": "timestamp-millis",
                "doc": "Reading timestamp in milliseconds since epoch"
            },
            {
                "name": "sensor_type",
                "type": {
                    "type": "enum",
                    "name": "SensorType",
                    "symbols": [
                        "TEMPERATURE",
                        "HUMIDITY", 
                        "PRESSURE",
                        "VOC",
                        "PARTICULATE",
                        "ACOUSTIC",
                        "VIBRATION",
                        "EMF"
                    ]
                }
            },
            {
                "name": "value",
                "type": "double",
                "doc": "Primary sensor reading value"
            },
            {
                "name": "unit",
                "type": "string",
                "doc": "Unit of measurement"
            },
            {
                "name": "quality",
                "type": ["null", "float"],
                "default": null,
                "doc": "Data quality score (0.0-1.0)"
            },
            {
                "name": "physics_constraints",
                "type": ["null", {
                    "type": "record",
                    "name": "PhysicsConstraints",
                    "fields": [
                        {
                            "name": "min_value",
                            "type": ["null", "double"],
                            "default": null
                        },
                        {
                            "name": "max_value", 
                            "type": ["null", "double"],
                            "default": null
                        },
                        {
                            "name": "max_rate_change",
                            "type": ["null", "double"],
                            "default": null,
                            "doc": "Maximum change per second"
                        }
                    ]
                }],
                "default": null
            },
            {
                "name": "additional_values",
                "type": ["null", {
                    "type": "map",
                    "values": "double"
                }],
                "default": null,
                "doc": "Additional sensor readings (e.g., for multi-channel sensors)"
            },
            {
                "name": "metadata",
                "type": {
                    "type": "map",
                    "values": "string"
                },
                "default": {},
                "doc": "Additional metadata"
            }
        ]
    });
    
    Schema::parse(&schema_json)
        .map_err(|e| SchemaError::ParseError(e.to_string()))
}

/// Device status schema v1.0.0
pub fn device_status_v1() -> Result<Schema, SchemaError> {
    let schema_json = json!({
        "namespace": "io.edgeguard.device.v1",
        "type": "record",
        "name": "DeviceStatus",
        "doc": "Edge device status information",
        "fields": [
            {
                "name": "device_id",
                "type": "string"
            },
            {
                "name": "timestamp",
                "type": "long",
                "logicalType": "timestamp-millis"
            },
            {
                "name": "battery_percent",
                "type": ["null", "float"],
                "default": null,
                "doc": "Battery level 0-100%"
            },
            {
                "name": "memory_free_kb",
                "type": ["null", "int"],
                "default": null,
                "doc": "Free memory in KB"
            },
            {
                "name": "cpu_usage_percent",
                "type": ["null", "float"],
                "default": null,
                "doc": "CPU usage 0-100%"
            },
            {
                "name": "temperature_celsius",
                "type": ["null", "float"],
                "default": null,
                "doc": "Device temperature"
            },
            {
                "name": "uptime_seconds",
                "type": "long",
                "doc": "Seconds since boot"
            },
            {
                "name": "error_count",
                "type": "int",
                "default": 0,
                "doc": "Errors since boot"
            },
            {
                "name": "sensors",
                "type": {
                    "type": "array",
                    "items": {
                        "type": "record",
                        "name": "SensorStatus",
                        "fields": [
                            {"name": "sensor_id", "type": "string"},
                            {"name": "enabled", "type": "boolean"},
                            {"name": "quality", "type": "float"},
                            {"name": "last_reading", "type": ["null", "long"], "default": null}
                        ]
                    }
                },
                "default": []
            }
        ]
    });
    
    Schema::parse(&schema_json)
        .map_err(|e| SchemaError::ParseError(e.to_string()))
}

/// Batch message schema for high-frequency data
pub fn sensor_batch_v1() -> Result<Schema, SchemaError> {
    let schema_json = json!({
        "namespace": "io.edgeguard.batch.v1",
        "type": "record",
        "name": "SensorBatch",
        "doc": "Batched sensor readings for efficient transmission",
        "fields": [
            {
                "name": "batch_id",
                "type": "string",
                "doc": "Unique batch identifier"
            },
            {
                "name": "device_id",
                "type": "string"
            },
            {
                "name": "timestamp_start",
                "type": "long",
                "logicalType": "timestamp-millis"
            },
            {
                "name": "timestamp_end",
                "type": "long",
                "logicalType": "timestamp-millis"
            },
            {
                "name": "count",
                "type": "int",
                "doc": "Number of readings in batch"
            },
            {
                "name": "compression",
                "type": {
                    "type": "enum",
                    "name": "CompressionType",
                    "symbols": ["NONE", "LZ4", "GZIP", "DELTA"]
                },
                "default": "NONE"
            },
            {
                "name": "readings",
                "type": "bytes",
                "doc": "Compressed sensor readings"
            },
            {
                "name": "statistics",
                "type": ["null", {
                    "type": "record",
                    "name": "BatchStatistics",
                    "fields": [
                        {"name": "min_value", "type": "double"},
                        {"name": "max_value", "type": "double"},
                        {"name": "mean_value", "type": "double"},
                        {"name": "std_dev", "type": ["null", "double"], "default": null}
                    ]
                }],
                "default": null
            }
        ]
    });
    
    Schema::parse(&schema_json)
        .map_err(|e| SchemaError::ParseError(e.to_string()))
}

/// Alert schema for critical events
pub fn alert_v1() -> Result<Schema, SchemaError> {
    let schema_json = json!({
        "namespace": "io.edgeguard.alert.v1",
        "type": "record",
        "name": "Alert",
        "doc": "Alert for validation failures or anomalies",
        "fields": [
            {
                "name": "alert_id",
                "type": "string"
            },
            {
                "name": "timestamp",
                "type": "long",
                "logicalType": "timestamp-millis"
            },
            {
                "name": "severity",
                "type": {
                    "type": "enum",
                    "name": "Severity",
                    "symbols": ["INFO", "WARNING", "ERROR", "CRITICAL"]
                }
            },
            {
                "name": "alert_type",
                "type": {
                    "type": "enum",
                    "name": "AlertType",
                    "symbols": [
                        "OUT_OF_RANGE",
                        "RATE_EXCEEDED",
                        "CROSS_VALIDATION_FAILED",
                        "SENSOR_QUALITY_BAD",
                        "ANOMALY_DETECTED",
                        "DEVICE_ERROR"
                    ]
                }
            },
            {
                "name": "sensor_id",
                "type": ["null", "string"],
                "default": null
            },
            {
                "name": "message",
                "type": "string"
            },
            {
                "name": "details",
                "type": {
                    "type": "map",
                    "values": "string"
                },
                "default": {}
            }
        ]
    });
    
    Schema::parse(&schema_json)
        .map_err(|e| SchemaError::ParseError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn parse_sensor_reading_schema() {
        let schema = sensor_reading_v1();
        assert!(schema.is_ok());
    }
    
    #[test]
    fn parse_device_status_schema() {
        let schema = device_status_v1();
        assert!(schema.is_ok());
    }
    
    #[test]
    fn parse_batch_schema() {
        let schema = sensor_batch_v1();
        assert!(schema.is_ok());
    }
    
    #[test]
    fn parse_alert_schema() {
        let schema = alert_v1();
        assert!(schema.is_ok());
    }
}