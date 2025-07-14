//! IoT Protocol Connectors for Edge-to-Cloud Communication
//!
//! ## Overview
//!
//! This module provides protocol adapters optimized for different IoT deployment
//! scenarios. Each protocol has distinct advantages depending on network conditions,
//! device capabilities, and use cases.
//!
//! ## Protocol Selection Guide
//!
//! ### MQTT (Message Queuing Telemetry Transport)
//!
//! **When to use:**
//! - Reliable networks (WiFi, Ethernet)
//! - Need for pub/sub patterns
//! - Multiple consumers for sensor data
//! - QoS guarantees required
//!
//! **Characteristics:**
//! - Header overhead: 2-5 bytes minimum
//! - Persistent connections reduce handshakes
//! - Built-in QoS levels (0, 1, 2)
//! - Topic-based routing
//!
//! **EdgeGuard optimizations:**
//! - Batching for high-frequency sensors
//! - Topic hierarchy for efficient filtering
//! - Retained messages for last-known-good
//! - LWT (Last Will and Testament) for failures
//!
//! ### CoAP (Constrained Application Protocol)
//!
//! **When to use:**
//! - Constrained networks (6LoWPAN, NB-IoT)
//! - Battery-powered devices
//! - UDP is acceptable
//! - RESTful paradigm preferred
//!
//! **Characteristics:**
//! - Header overhead: 4 bytes minimum
//! - Stateless (no connection overhead)
//! - Optional reliability (Confirmable messages)
//! - Multicast support
//!
//! **EdgeGuard optimizations:**
//! - Block-wise transfers for large payloads
//! - Observe pattern for subscriptions
//! - DTLS for security without TLS overhead
//! - Resource discovery for auto-configuration
//!
//! ### HTTP/HTTPS
//!
//! **When to use:**
//! - Integration with existing web services
//! - Firewall-friendly environments
//! - Complex authentication required
//! - Development/debugging
//!
//! **Characteristics:**
//! - Header overhead: 200+ bytes typical
//! - Stateless requests
//! - Wide compatibility
//! - TLS security standard
//!
//! **EdgeGuard optimizations:**
//! - HTTP/2 for multiplexing
//! - Keep-alive for connection reuse
//! - Compression for headers/body
//! - Batch endpoints for multiple readings
//!
//! ## Connector Design Patterns
//!
//! ### 1. Buffering Strategy
//!
//! All connectors implement smart buffering:
//! ```rust
//! // Pseudo-code for buffering logic
//! if network_available() {
//!     send_immediate(data)
//! } else {
//!     buffer.push(data);
//!     if buffer.full() {
//!         buffer.drop_oldest();  // Prioritize recent data
//!     }
//! }
//! ```
//!
//! ### 2. Retry Logic
//!
//! Exponential backoff with jitter:
//! ```text
//! retry_delay = min(base * 2^attempt + random_jitter, max_delay)
//! ```
//!
//! ### 3. Compression
//!
//! Automatic compression for efficiency:
//! - Small payloads (<100 bytes): No compression
//! - Medium payloads (100-1000 bytes): LZ4
//! - Large payloads (>1000 bytes): Zstandard
//!
//! ## Security Considerations
//!
//! ### Authentication
//! - **MQTT**: Username/password, client certificates, OAuth
//! - **CoAP**: Pre-shared keys, raw public keys, certificates
//! - **HTTP**: Basic auth, bearer tokens, mutual TLS
//!
//! ### Encryption
//! - **MQTT**: TLS 1.2+ (configurable cipher suites)
//! - **CoAP**: DTLS 1.2+ (lighter than TLS)
//! - **HTTP**: HTTPS mandatory for production
//!
//! ### Best Practices
//! 1. Never hard-code credentials
//! 2. Use certificate pinning for critical deployments
//! 3. Implement perfect forward secrecy
//! 4. Regular key rotation
//!
//! ## Resource Usage
//!
//! Typical memory footprint per connector:
//!
//! | Protocol | ROM  | RAM (idle) | RAM (active) |
//! |----------|------|------------|--------------|
//! | MQTT     | 50KB | 4KB        | 8-16KB       |
//! | CoAP     | 30KB | 2KB        | 4-8KB        |
//! | HTTP     | 80KB | 8KB        | 16-32KB      |
//!
//! ## Example Usage
//!
//! ```rust
//! use edgeguard_connectors::{Connector, mqtt::MqttConnector};
//!
//! // Configure MQTT for sensor data
//! let mut mqtt = MqttConnector::new("broker.local:1883")?;
//! mqtt.set_client_id("sensor_001");
//! mqtt.set_keep_alive(60);  // seconds
//!
//! // Send validated data
//! let topic = "sensors/temperature/sensor_001";
//! let data = serialize_reading(&validated_reading)?;
//! mqtt.send(topic, &data)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "mqtt")]
pub mod mqtt;

#[cfg(feature = "coap")]
pub mod coap;

#[cfg(feature = "http")]
pub mod http;

// Re-export common types
#[cfg(feature = "mqtt")]
pub use mqtt::{MqttConnector, MqttConfig, MqttError, QoS};

use thiserror::Error;

/// Common connector errors
#[derive(Debug, Error)]
pub enum ConnectorError {
    #[error("Not connected")]
    NotConnected,
    
    #[error("Buffer full")]
    BufferFull,
    
    #[error("Timeout")]
    Timeout,
    
    #[error("Protocol error: {0}")]
    ProtocolError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Trait for all protocol connectors
pub trait Connector {
    type Error;
    
    /// Send sensor data
    fn send(&mut self, topic: &str, data: &[u8]) -> Result<(), Self::Error>;
    
    /// Check if connected
    fn is_connected(&self) -> bool;
}

/// Async version of the Connector trait
/// 
/// This is the preferred trait for new implementations
#[cfg(feature = "std")]
#[async_trait::async_trait]
pub trait AsyncConnector: Send {
    type Error;
    
    /// Send sensor data asynchronously
    async fn send(&mut self, topic: &str, data: &[u8]) -> Result<(), Self::Error>;
    
    /// Check if connected
    fn is_connected(&self) -> bool;
    
    /// Get connection statistics
    fn stats(&self) -> ConnectionStats;
}

/// Connection statistics common to all connectors
#[derive(Debug, Default, Clone)]
pub struct ConnectionStats {
    /// Total messages sent successfully
    pub messages_sent: u64,
    /// Total messages failed to send
    pub messages_failed: u64,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Number of reconnections
    pub reconnections: u32,
    /// Last error message
    pub last_error: Option<String>,
}