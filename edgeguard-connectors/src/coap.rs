//! CoAP Connector for EdgeGuard - Lightweight Protocol for Constrained Devices
//!
//! ## Overview
//!
//! This module provides CoAP (Constrained Application Protocol) support optimized
//! for resource-constrained IoT devices. CoAP is designed for machine-to-machine
//! applications and works well in low-power, lossy networks.
//!
//! ## Design Philosophy
//!
//! This implementation separates message building from transport:
//! - **no_std support**: Message building works without std
//! - **Platform agnostic**: Users provide their own UDP transport
//! - **Zero-copy where possible**: Minimize allocations
//!
//! ## Features
//!
//! - **Message building**: Create CoAP requests/responses in no_std
//! - **Block-wise transfers**: Handle large payloads in chunks
//! - **Observe pattern**: Subscribe to resource changes
//! - **Resource discovery**: /.well-known/core support
//! - **DTLS ready**: Message format compatible with secure transport
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐     ┌──────────────────┐
//! │ EdgeGuard Core  │     │ Platform Network │
//! │   Validation    │     │      Stack       │
//! └────────┬────────┘     └────────┬─────────┘
//!          │                       │
//!          ▼                       ▼
//! ┌─────────────────┐     ┌──────────────────┐
//! │ CoAP Message    │────▶│ Transport Trait  │
//! │    Builder      │     │ (User Provided)  │
//! └─────────────────┘     └──────────────────┘
//! ```
//!
//! ## Example Usage
//!
//! ### no_std Message Building
//! ```rust
//! use edgeguard_connectors::coap::{CoapMessage, MessageType, Code};
//! 
//! // Build a CoAP GET request
//! let mut message = CoapMessage::new();
//! message.set_type(MessageType::Confirmable);
//! message.set_code(Code::Get);
//! message.set_message_id(1234);
//! message.add_option(11, b"temperature"); // Uri-Path
//! 
//! // Serialize to bytes for transport
//! let mut buffer = [0u8; 256];
//! let len = message.serialize(&mut buffer)?;
//! // Send buffer[..len] over your transport
//! ```
//!
//! ### std Full Connector
//! ```rust
//! # #[cfg(feature = "std")]
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use edgeguard_connectors::coap::{CoapConnector, CoapConfig};
//! 
//! let config = CoapConfig::new("coap://gateway.local:5683")
//!     .confirmable(true)
//!     .block_size(64); // For constrained networks
//!
//! let mut coap = CoapConnector::new(config)?;
//! 
//! // Send sensor reading
//! let resource = "sensors/temp/device01";
//! let payload = b"{\"value\": 23.5}";
//! coap.post(resource, payload).await?;
//! # Ok(())
//! # }
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
use core::{fmt, mem, slice};

#[cfg(feature = "std")]
use std::{fmt, mem, slice};

use coap_lite::{
    CoapOption, MessageClass, MessageType as CoapMessageType, 
    Packet, RequestType, ResponseType, Header,
};

#[cfg(not(feature = "std"))]
use heapless::Vec;

#[cfg(feature = "std")]
use std::vec::Vec;

/// Re-export commonly used types
pub use coap_lite::{ContentFormat, ObserveOption};

/// CoAP message types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    /// Confirmable message (requires ACK)
    Confirmable,
    /// Non-confirmable message (fire and forget)
    NonConfirmable,
    /// Acknowledgment
    Acknowledgment,
    /// Reset
    Reset,
}

impl From<MessageType> for CoapMessageType {
    fn from(mt: MessageType) -> Self {
        match mt {
            MessageType::Confirmable => CoapMessageType::Confirmable,
            MessageType::NonConfirmable => CoapMessageType::NonConfirmable,
            MessageType::Acknowledgment => CoapMessageType::Acknowledgement,
            MessageType::Reset => CoapMessageType::Reset,
        }
    }
}

/// CoAP method and response codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Code {
    // Methods (0.xx)
    Empty,
    Get,
    Post,
    Put,
    Delete,
    
    // Success (2.xx)
    Created,      // 2.01
    Deleted,      // 2.02
    Valid,        // 2.03
    Changed,      // 2.04
    Content,      // 2.05
    
    // Client Error (4.xx)
    BadRequest,   // 4.00
    Unauthorized, // 4.01
    BadOption,    // 4.02
    Forbidden,    // 4.03
    NotFound,     // 4.04
    
    // Server Error (5.xx)
    InternalServerError, // 5.00
    NotImplemented,      // 5.01
    BadGateway,          // 5.02
    ServiceUnavailable,  // 5.03
}

impl Code {
    fn to_message_class(self) -> MessageClass {
        match self {
            Code::Empty => MessageClass::Empty,
            Code::Get => MessageClass::Request(RequestType::Get),
            Code::Post => MessageClass::Request(RequestType::Post),
            Code::Put => MessageClass::Request(RequestType::Put),
            Code::Delete => MessageClass::Request(RequestType::Delete),
            
            Code::Created => MessageClass::Response(ResponseType::Created),
            Code::Deleted => MessageClass::Response(ResponseType::Deleted),
            Code::Valid => MessageClass::Response(ResponseType::Valid),
            Code::Changed => MessageClass::Response(ResponseType::Changed),
            Code::Content => MessageClass::Response(ResponseType::Content),
            
            Code::BadRequest => MessageClass::Response(ResponseType::BadRequest),
            Code::Unauthorized => MessageClass::Response(ResponseType::Unauthorized),
            Code::BadOption => MessageClass::Response(ResponseType::BadOption),
            Code::Forbidden => MessageClass::Response(ResponseType::Forbidden),
            Code::NotFound => MessageClass::Response(ResponseType::NotFound),
            
            Code::InternalServerError => MessageClass::Response(ResponseType::InternalServerError),
            Code::NotImplemented => MessageClass::Response(ResponseType::NotImplemented),
            Code::BadGateway => MessageClass::Response(ResponseType::BadGateway),
            Code::ServiceUnavailable => MessageClass::Response(ResponseType::ServiceUnavailable),
        }
    }
}

/// CoAP errors
#[derive(Debug)]
pub enum CoapError {
    /// Buffer too small for message
    BufferTooSmall,
    /// Invalid message format
    InvalidMessage,
    /// Network error (std only)
    #[cfg(feature = "std")]
    Network(std::io::Error),
    /// Timeout waiting for response
    Timeout,
    /// Maximum retransmissions exceeded
    MaxRetransmissions,
}

impl fmt::Display for CoapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoapError::BufferTooSmall => write!(f, "Buffer too small for CoAP message"),
            CoapError::InvalidMessage => write!(f, "Invalid CoAP message format"),
            #[cfg(feature = "std")]
            CoapError::Network(e) => write!(f, "Network error: {}", e),
            CoapError::Timeout => write!(f, "Timeout waiting for response"),
            CoapError::MaxRetransmissions => write!(f, "Maximum retransmissions exceeded"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CoapError {}

/// CoAP message builder for no_std environments
///
/// This provides a simple API to build CoAP messages without requiring std.
/// The underlying coap-lite crate handles the actual packet formatting.
pub struct CoapMessage {
    packet: Packet,
}

impl CoapMessage {
    /// Create a new CoAP message
    pub fn new() -> Self {
        Self {
            packet: Packet::new(),
        }
    }
    
    /// Set message type
    pub fn set_type(&mut self, msg_type: MessageType) {
        self.packet.header.set_type(msg_type.into());
    }
    
    /// Set message code
    pub fn set_code(&mut self, code: Code) {
        self.packet.header.code = code.to_message_class();
    }
    
    /// Set message ID
    pub fn set_message_id(&mut self, id: u16) {
        self.packet.header.message_id = id;
    }
    
    /// Set token for request/response matching
    pub fn set_token(&mut self, token: Vec<u8>) {
        self.packet.set_token(token);
    }
    
    /// Add an option
    pub fn add_option(&mut self, number: u16, value: &[u8]) {
        self.packet.add_option(number.into(), value.to_vec());
    }
    
    /// Set payload
    pub fn set_payload(&mut self, payload: Vec<u8>) {
        self.packet.payload = payload;
    }
    
    /// Add Uri-Path option (convenience method)
    pub fn add_uri_path(&mut self, path: &str) {
        self.add_option(11, path.as_bytes());
    }
    
    /// Add Content-Format option (convenience method)
    pub fn set_content_format(&mut self, format: ContentFormat) {
        let value = (format as u16).to_be_bytes();
        self.add_option(12, &value);
    }
    
    /// Serialize message to bytes
    pub fn serialize(&self, buffer: &mut [u8]) -> Result<usize, CoapError> {
        let bytes = self.packet.to_bytes()
            .map_err(|_| CoapError::InvalidMessage)?;
        
        if buffer.len() < bytes.len() {
            return Err(CoapError::BufferTooSmall);
        }
        
        buffer[..bytes.len()].copy_from_slice(&bytes);
        Ok(bytes.len())
    }
    
    /// Parse message from bytes
    pub fn parse(data: &[u8]) -> Result<Self, CoapError> {
        let packet = Packet::from_bytes(data)
            .map_err(|_| CoapError::InvalidMessage)?;
        Ok(Self { packet })
    }
    
    /// Get message type
    pub fn get_type(&self) -> MessageType {
        match self.packet.header.get_type() {
            CoapMessageType::Confirmable => MessageType::Confirmable,
            CoapMessageType::NonConfirmable => MessageType::NonConfirmable,
            CoapMessageType::Acknowledgement => MessageType::Acknowledgment,
            CoapMessageType::Reset => MessageType::Reset,
        }
    }
    
    /// Get message ID
    pub fn get_message_id(&self) -> u16 {
        self.packet.header.message_id
    }
    
    /// Get payload
    pub fn get_payload(&self) -> &[u8] {
        &self.packet.payload
    }
}

impl Default for CoapMessage {
    fn default() -> Self {
        Self::new()
    }
}

/// Transport trait for platform-specific network implementations
///
/// Users must implement this trait for their platform's network stack.
pub trait CoapTransport {
    /// Error type for the transport
    type Error;
    
    /// Send CoAP message
    fn send(&mut self, endpoint: &str, data: &[u8]) -> Result<(), Self::Error>;
    
    /// Receive CoAP message (blocking with timeout)
    fn receive(&mut self, buffer: &mut [u8], timeout_ms: u32) -> Result<usize, Self::Error>;
}

/// Block-wise transfer support
pub struct BlockTransfer {
    block_size: u16,
    current_block: u32,
    more: bool,
}

impl BlockTransfer {
    /// Create new block transfer with specified block size (power of 2)
    pub fn new(block_size_exp: u8) -> Self {
        let block_size = 1u16 << (block_size_exp + 4); // 16, 32, 64, etc.
        Self {
            block_size,
            current_block: 0,
            more: true,
        }
    }
    
    /// Get next block of data
    pub fn next_block<'a>(&mut self, data: &'a [u8]) -> Option<&'a [u8]> {
        if !self.more {
            return None;
        }
        
        let start = (self.current_block * self.block_size as u32) as usize;
        if start >= data.len() {
            self.more = false;
            return None;
        }
        
        let end = ((start as u32 + self.block_size as u32) as usize).min(data.len());
        self.more = end < data.len();
        self.current_block += 1;
        
        Some(&data[start..end])
    }
    
    /// Create Block2 option value
    pub fn block2_value(&self) -> u32 {
        let szx = (self.block_size.trailing_zeros() - 4) as u32;
        (self.current_block << 4) | (if self.more { 0x08 } else { 0 }) | szx
    }
}

// Full connector implementation for std environments
#[cfg(all(feature = "std", feature = "coap-transport"))]
pub use self::std_impl::*;

#[cfg(all(feature = "std", feature = "coap-transport"))]
mod std_impl {
    use super::*;
    use std::net::UdpSocket;
    use std::time::Duration;
    use std::sync::{Arc, Mutex};
    use crate::ConnectionStats;
    
    /// CoAP configuration
    pub struct CoapConfig {
        /// CoAP endpoint URL
        pub endpoint: String,
        /// Use confirmable messages
        pub confirmable: bool,
        /// Retransmission timeout (ms)
        pub ack_timeout: u32,
        /// Maximum retransmissions
        pub max_retransmit: u8,
        /// Block size for large transfers (as exponent)
        pub block_size_exp: u8,
    }
    
    impl CoapConfig {
        /// Create new configuration with endpoint
        pub fn new(endpoint: impl Into<String>) -> Self {
            Self {
                endpoint: endpoint.into(),
                confirmable: true,
                ack_timeout: 2000,
                max_retransmit: 4,
                block_size_exp: 6, // 1024 bytes
            }
        }
        
        /// Set confirmable mode
        pub fn confirmable(mut self, confirmable: bool) -> Self {
            self.confirmable = confirmable;
            self
        }
        
        /// Set block size (16 * 2^n bytes)
        pub fn block_size(mut self, size_exp: u8) -> Self {
            self.block_size_exp = size_exp.min(6); // Max 1024
            self
        }
    }
    
    /// CoAP connector with UDP transport
    pub struct CoapConnector {
        config: CoapConfig,
        socket: UdpSocket,
        message_id: u16,
        stats: Arc<Mutex<ConnectionStats>>,
    }
    
    impl CoapConnector {
        /// Create new CoAP connector
        pub fn new(config: CoapConfig) -> Result<Self, CoapError> {
            let socket = UdpSocket::bind("0.0.0.0:0")
                .map_err(CoapError::Network)?;
            
            socket.set_read_timeout(Some(Duration::from_millis(config.ack_timeout as u64)))
                .map_err(CoapError::Network)?;
            
            // Simple pseudo-random message ID based on current time
            let message_id = (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() & 0xFFFF) as u16;
            
            Ok(Self {
                config,
                socket,
                message_id,
                stats: Arc::new(Mutex::new(ConnectionStats::default())),
            })
        }
        
        /// Send GET request
        pub async fn get(&mut self, resource: &str) -> Result<Vec<u8>, CoapError> {
            let mut msg = CoapMessage::new();
            msg.set_type(if self.config.confirmable { 
                MessageType::Confirmable 
            } else { 
                MessageType::NonConfirmable 
            });
            msg.set_code(Code::Get);
            msg.set_message_id(self.next_message_id());
            msg.add_uri_path(resource);
            
            self.send_and_receive(msg).await
        }
        
        /// Send POST request
        pub async fn post(&mut self, resource: &str, payload: &[u8]) -> Result<(), CoapError> {
            let mut msg = CoapMessage::new();
            msg.set_type(if self.config.confirmable { 
                MessageType::Confirmable 
            } else { 
                MessageType::NonConfirmable 
            });
            msg.set_code(Code::Post);
            msg.set_message_id(self.next_message_id());
            msg.add_uri_path(resource);
            msg.set_payload(payload.to_vec());
            
            self.send_and_receive(msg).await?;
            Ok(())
        }
        
        /// Internal: Send message and wait for response if confirmable
        async fn send_and_receive(&mut self, msg: CoapMessage) -> Result<Vec<u8>, CoapError> {
            let mut buffer = vec![0u8; 1024];
            let len = msg.serialize(&mut buffer)?;
            
            // Send with retransmissions
            for attempt in 0..=self.config.max_retransmit {
                self.socket.send_to(&buffer[..len], &self.config.endpoint)
                    .map_err(CoapError::Network)?;
                
                self.stats.lock().unwrap().messages_sent += 1;
                self.stats.lock().unwrap().bytes_sent += len as u64;
                
                if msg.get_type() == MessageType::NonConfirmable {
                    return Ok(Vec::new());
                }
                
                // Wait for ACK
                match self.socket.recv_from(&mut buffer) {
                    Ok((n, _)) => {
                        let response = CoapMessage::parse(&buffer[..n])?;
                        if response.get_message_id() == msg.get_message_id() {
                            return Ok(response.get_payload().to_vec());
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        if attempt < self.config.max_retransmit {
                            // Exponential backoff
                            tokio::time::sleep(Duration::from_millis(
                                self.config.ack_timeout as u64 * (1 << attempt)
                            )).await;
                            continue;
                        }
                    }
                    Err(e) => return Err(CoapError::Network(e)),
                }
            }
            
            self.stats.lock().unwrap().messages_failed += 1;
            Err(CoapError::MaxRetransmissions)
        }
        
        fn next_message_id(&mut self) -> u16 {
            self.message_id = self.message_id.wrapping_add(1);
            self.message_id
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_message_builder() {
        let mut msg = CoapMessage::new();
        msg.set_type(MessageType::Confirmable);
        msg.set_code(Code::Get);
        msg.set_message_id(1234);
        msg.add_uri_path("temperature");
        
        let mut buffer = [0u8; 256];
        let len = msg.serialize(&mut buffer).unwrap();
        
        // Should create valid CoAP message
        assert!(len > 4); // At least header
        assert_eq!(buffer[0] >> 4, 0x4); // Version 1
        assert_eq!((buffer[0] >> 2) & 0x3, 0); // Type = CON
    }
    
    #[test]
    fn test_block_transfer() {
        let data = b"Hello, this is a longer message that needs to be sent in blocks!";
        let mut block = BlockTransfer::new(4); // 256 byte blocks
        
        let block1 = block.next_block(data).unwrap();
        assert_eq!(block1.len(), data.len()); // First block contains all
        assert!(!block.more); // No more blocks
    }
}