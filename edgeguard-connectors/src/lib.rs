//! IoT protocol connectors for EdgeGuard
//!
//! Provides adapters for common IoT protocols:
//! - MQTT for standard IoT deployments
//! - CoAP for constrained devices
//! - HTTP for cloud integration

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "mqtt")]
pub mod mqtt;

#[cfg(feature = "coap")]
pub mod coap;

/// Trait for all protocol connectors
pub trait Connector {
    type Error;
    
    /// Send sensor data
    fn send(&mut self, topic: &str, data: &[u8]) -> Result<(), Self::Error>;
    
    /// Check if connected
    fn is_connected(&self) -> bool;
}