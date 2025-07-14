# Connectors API

Network connectivity for edge devices with cloud integration.

## Connector Trait

Core trait for implementing network connectors.

```rust
pub trait Connector: Send {
    type Config;
    type Error;
    
    fn connect(&mut self, config: Self::Config) -> Result<(), Self::Error>;
    fn send(&mut self, data: &[u8]) -> Result<(), Self::Error>;
    fn receive(&mut self) -> Result<Vec<u8>, Self::Error>;
    fn disconnect(&mut self) -> Result<(), Self::Error>;
    fn is_connected(&self) -> bool;
    fn statistics(&self) -> ConnectionStatistics;
}
```

## MqttConnector

MQTT client with auto-reconnection and offline buffering.

### Constructor

```rust
impl MqttConnector {
    pub fn new(config: MqttConfig) -> Result<Self, MqttError>;
    pub fn subscribe(&mut self, topic: &str, qos: u8) -> Result<(), MqttError>;
    pub fn unsubscribe(&mut self, topic: &str) -> Result<(), MqttError>;
    pub fn publish(&mut self, topic: &str, payload: &[u8], qos: u8) -> Result<(), MqttError>;
    pub fn poll(&mut self) -> Result<Vec<MqttMessage>, MqttError>;
    pub fn disconnect(&mut self) -> Result<(), MqttError>;
}
```

### Configuration

```rust
let config = MqttConfig::new("sensor_gateway", "mqtt://localhost:1883")
    .with_credentials("username", "password")
    .with_clean_session(false)
    .with_keep_alive(Duration::from_secs(60))
    .with_max_reconnect_attempts(10)
    .with_offline_buffer_size(1000)
    .with_tls_config(TlsConfig::default());

let mut client = MqttConnector::new(config)?;
```

### Example Usage

```rust
use edgeguard::connectors::mqtt::{MqttConnector, MqttConfig};
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = MqttConfig::new("temperature_sensor", "mqtt://broker:1883")
        .with_credentials("sensor_user", "sensor_pass")
        .with_keep_alive(Duration::from_secs(30));
    
    let mut client = MqttConnector::new(config)?;
    
    // Subscribe to command topic
    client.subscribe("sensors/commands", 1)?;
    
    // Publish sensor data
    let sensor_data = r#"{"temperature": 23.5, "timestamp": 1640995200}"#;
    client.publish("sensors/temperature", sensor_data.as_bytes(), 1)?;
    
    // Poll for messages
    loop {
        match client.poll() {
            Ok(messages) => {
                for msg in messages {
                    println!("Received: {} on {}", 
                        String::from_utf8_lossy(&msg.payload), 
                        msg.topic);
                }
            }
            Err(e) => {
                eprintln!("MQTT error: {}", e);
                sleep(Duration::from_secs(1)).await;
            }
        }
    }
}
```

### Advanced Features

```rust
// Offline message buffering
let config = MqttConfig::new("device", "mqtt://broker:1883")
    .with_offline_buffer_size(1000)
    .with_offline_buffer_strategy(OfflineBufferStrategy::DropOldest);

// TLS configuration
let tls_config = TlsConfig::new()
    .with_ca_certificate("ca.pem")
    .with_client_certificate("client.pem", "client.key")
    .with_verify_hostname(true);

let config = MqttConfig::new("secure_device", "mqtts://broker:8883")
    .with_tls_config(tls_config);
```

## CoapConnector

CoAP client for constrained devices with UDP transport.

### Constructor

```rust
impl CoapConnector {
    pub fn new(config: CoapConfig) -> Result<Self, CoapError>;
    pub fn get(&mut self, path: &str) -> Result<CoapResponse, CoapError>;
    pub fn post(&mut self, path: &str, payload: &[u8]) -> Result<CoapResponse, CoapError>;
    pub fn put(&mut self, path: &str, payload: &[u8]) -> Result<CoapResponse, CoapError>;
    pub fn delete(&mut self, path: &str) -> Result<CoapResponse, CoapError>;
    pub fn observe(&mut self, path: &str) -> Result<CoapObserver, CoapError>;
}
```

### Configuration

```rust
let config = CoapConfig::new("coap://server:5683")
    .with_timeout(Duration::from_secs(5))
    .with_retries(3)
    .with_confirmable(true)
    .with_block_size(BlockSize::Block1024);

let mut client = CoapConnector::new(config)?;
```

### Example Usage

```rust
use edgeguard::connectors::coap::{CoapConnector, CoapConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = CoapConfig::new("coap://sensor-server:5683")
        .with_timeout(Duration::from_secs(10))
        .with_retries(3);
    
    let mut client = CoapConnector::new(config)?;
    
    // POST sensor data
    let sensor_data = r#"{"temperature": 23.5}"#;
    let response = client.post("sensors/temperature", sensor_data.as_bytes())?;
    
    if response.is_success() {
        println!("Data sent successfully");
    }
    
    // GET configuration
    let config_response = client.get("config/intervals")?;
    if let Some(payload) = config_response.payload() {
        println!("Config: {}", String::from_utf8_lossy(payload));
    }
    
    // Observe temperature changes
    let observer = client.observe("sensors/temperature")?;
    for notification in observer {
        println!("Temperature update: {:?}", notification);
    }
    
    Ok(())
}
```

### Blockwise Transfer

```rust
// Large payload transfer
let large_data = vec![0u8; 2048];
let response = client.post("sensors/bulk", &large_data)?;

// Configure block size
let config = CoapConfig::new("coap://server:5683")
    .with_block_size(BlockSize::Block512);
```

## HttpConnector

HTTP/HTTPS client with REST API integration.

### Constructor

```rust
impl HttpConnector {
    pub fn new(config: HttpConfig) -> Result<Self, HttpError>;
    pub fn get(&mut self, path: &str) -> Result<HttpResponse, HttpError>;
    pub fn post(&mut self, path: &str, payload: &[u8]) -> Result<HttpResponse, HttpError>;
    pub fn put(&mut self, path: &str, payload: &[u8]) -> Result<HttpResponse, HttpError>;
    pub fn delete(&mut self, path: &str) -> Result<HttpResponse, HttpError>;
    pub fn patch(&mut self, path: &str, payload: &[u8]) -> Result<HttpResponse, HttpError>;
}
```

### Configuration

```rust
let config = HttpConfig::new("https://api.example.com")
    .with_timeout(Duration::from_secs(30))
    .with_auth_bearer("api_token_here")
    .with_header("Content-Type", "application/json")
    .with_header("User-Agent", "EdgeGuard/1.0")
    .with_retry_policy(RetryPolicy::exponential(3))
    .with_tls_verification(true);

let mut client = HttpConnector::new(config)?;
```

### Example Usage

```rust
use edgeguard::connectors::http::{HttpConnector, HttpConfig};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = HttpConfig::new("https://api.iot-platform.com")
        .with_auth_bearer("your-api-token")
        .with_header("Content-Type", "application/json")
        .with_timeout(Duration::from_secs(30));
    
    let mut client = HttpConnector::new(config)?;
    
    // POST sensor data
    let sensor_data = json!({
        "device_id": "sensor_001",
        "temperature": 23.5,
        "humidity": 60.2,
        "timestamp": 1640995200
    });
    
    let response = client.post(
        "/v1/sensors/data",
        sensor_data.to_string().as_bytes()
    )?;
    
    if response.is_success() {
        println!("Data uploaded successfully");
    } else {
        println!("Upload failed: {}", response.status());
    }
    
    // GET device configuration
    let config_response = client.get("/v1/devices/sensor_001/config")?;
    if let Some(body) = config_response.body() {
        println!("Config: {}", String::from_utf8_lossy(body));
    }
    
    Ok(())
}
```

### Authentication

```rust
// Bearer token authentication
let config = HttpConfig::new("https://api.example.com")
    .with_auth_bearer("your_token_here");

// Basic authentication
let config = HttpConfig::new("https://api.example.com")
    .with_auth_basic("username", "password");

// Custom authentication header
let config = HttpConfig::new("https://api.example.com")
    .with_header("X-API-Key", "your_api_key");
```

## Configuration Types

### MqttConfig

```rust
pub struct MqttConfig {
    pub fn new(client_id: &str, broker_url: &str) -> Self;
    pub fn with_credentials(mut self, username: &str, password: &str) -> Self;
    pub fn with_clean_session(mut self, clean: bool) -> Self;
    pub fn with_keep_alive(mut self, duration: Duration) -> Self;
    pub fn with_max_reconnect_attempts(mut self, attempts: u32) -> Self;
    pub fn with_offline_buffer_size(mut self, size: usize) -> Self;
    pub fn with_tls_config(mut self, config: TlsConfig) -> Self;
}
```

### CoapConfig

```rust
pub struct CoapConfig {
    pub fn new(server_url: &str) -> Self;
    pub fn with_timeout(mut self, timeout: Duration) -> Self;
    pub fn with_retries(mut self, retries: u32) -> Self;
    pub fn with_confirmable(mut self, confirmable: bool) -> Self;
    pub fn with_block_size(mut self, size: BlockSize) -> Self;
    pub fn with_dtls_config(mut self, config: DtlsConfig) -> Self;
}
```

### HttpConfig

```rust
pub struct HttpConfig {
    pub fn new(base_url: &str) -> Self;
    pub fn with_timeout(mut self, timeout: Duration) -> Self;
    pub fn with_auth_bearer(mut self, token: &str) -> Self;
    pub fn with_auth_basic(mut self, username: &str, password: &str) -> Self;
    pub fn with_header(mut self, key: &str, value: &str) -> Self;
    pub fn with_retry_policy(mut self, policy: RetryPolicy) -> Self;
    pub fn with_tls_verification(mut self, verify: bool) -> Self;
}
```

## Response Types

### MqttMessage

```rust
pub struct MqttMessage {
    pub topic: String,
    pub payload: Vec<u8>,
    pub qos: u8,
    pub retain: bool,
    pub timestamp: Timestamp,
}
```

### CoapResponse

```rust
pub struct CoapResponse {
    pub fn status(&self) -> CoapStatus;
    pub fn payload(&self) -> Option<&[u8]>;
    pub fn is_success(&self) -> bool;
    pub fn content_format(&self) -> Option<ContentFormat>;
    pub fn max_age(&self) -> Option<Duration>;
}
```

### HttpResponse

```rust
pub struct HttpResponse {
    pub fn status(&self) -> u16;
    pub fn headers(&self) -> &HeaderMap;
    pub fn body(&self) -> Option<&[u8]>;
    pub fn is_success(&self) -> bool;
    pub fn content_type(&self) -> Option<&str>;
    pub fn content_length(&self) -> Option<usize>;
}
```

## Error Handling

### ConnectorError

```rust
#[derive(Debug)]
pub enum ConnectorError {
    ConnectionFailed(String),
    Timeout,
    AuthenticationFailed,
    NetworkError(String),
    ProtocolError(String),
    ConfigurationError(String),
    BufferFull,
    Disconnected,
}
```

### Error Recovery

```rust
match client.publish("sensors/data", payload, 1) {
    Ok(()) => println!("Message sent"),
    Err(ConnectorError::ConnectionFailed(_)) => {
        // Attempt reconnection
        client.connect(config)?;
    }
    Err(ConnectorError::Timeout) => {
        // Retry with exponential backoff
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
    Err(ConnectorError::BufferFull) => {
        // Wait for buffer space
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Err(e) => {
        eprintln!("Connector error: {:?}", e);
    }
}
```

## Statistics and Monitoring

### ConnectionStatistics

```rust
pub struct ConnectionStatistics {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub connection_uptime: Duration,
    pub reconnection_count: u32,
    pub last_error: Option<String>,
}
```

### Monitoring

```rust
let stats = client.statistics();
println!("Messages sent: {}", stats.messages_sent);
println!("Uptime: {:?}", stats.connection_uptime);
println!("Reconnections: {}", stats.reconnection_count);

if let Some(error) = &stats.last_error {
    println!("Last error: {}", error);
}
```

## Integration with Pipeline

### MQTT Stream Integration

```rust
use edgeguard::connectors::mqtt::MqttConnector;
use edgeguard::pipeline::Pipeline;

async fn mqtt_pipeline_integration() -> Result<(), Box<dyn std::error::Error>> {
    let mqtt_config = MqttConfig::new("pipeline_client", "mqtt://localhost:1883");
    let mut client = MqttConnector::new(mqtt_config)?;
    
    let mut pipeline = Pipeline::<256>::builder()
        .add_stage(ValidationStage::new(
            TemperatureValidator::new(),
            SensorType::Temperature
        ))
        .build();
    
    client.subscribe("sensors/+/data", 1)?;
    
    loop {
        // Poll for MQTT messages
        let messages = client.poll()?;
        
        for msg in messages {
            // Parse sensor data
            if let Ok(sensor_data) = parse_sensor_data(&msg.payload) {
                // Create event and process
                let event = create_event_from_sensor_data(sensor_data);
                pipeline.push_event(event);
            }
        }
        
        // Process pipeline
        pipeline.process_batch(10)?;
        
        // Send results back via MQTT
        while let Some(result) = pipeline.pop_result() {
            let result_json = serialize_result(&result);
            client.publish("results/validated", result_json.as_bytes(), 1)?;
        }
    }
}
```

## Best Practices

### Connection Management

```rust
// Implement connection health checking
async fn maintain_connection(client: &mut MqttConnector) {
    if !client.is_connected() {
        match client.connect(config) {
            Ok(()) => log::info!("Reconnected successfully"),
            Err(e) => log::error!("Reconnection failed: {}", e),
        }
    }
}

// Use connection pooling for HTTP
let mut pool = HttpConnectionPool::new(5);
let client = pool.get_connection()?;
```

### Data Serialization

```rust
// Use efficient serialization formats
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct SensorReading {
    device_id: String,
    sensor_type: String,
    value: f32,
    timestamp: u64,
    quality: f32,
}

// Serialize for transmission
let reading = SensorReading {
    device_id: "sensor_001".to_string(),
    sensor_type: "temperature".to_string(),
    value: 23.5,
    timestamp: 1640995200,
    quality: 0.95,
};

let json_data = serde_json::to_vec(&reading)?;
client.publish("sensors/data", &json_data, 1)?;
```

### Security

```rust
// Use secure connections
let tls_config = TlsConfig::new()
    .with_ca_certificate("ca.pem")
    .with_client_certificate("client.pem", "client.key")
    .with_verify_hostname(true);

let config = MqttConfig::new("secure_device", "mqtts://broker:8883")
    .with_tls_config(tls_config);

// Validate certificates
let http_config = HttpConfig::new("https://api.example.com")
    .with_tls_verification(true);
```

### Performance Optimization

```rust
// Use connection pooling
let pool = HttpConnectionPool::new(10);

// Batch messages for efficiency
let mut batch = MessageBatch::new();
batch.add_message("topic1", payload1);
batch.add_message("topic2", payload2);
client.publish_batch(batch)?;

// Configure appropriate timeouts
let config = HttpConfig::new("https://api.example.com")
    .with_timeout(Duration::from_secs(5))  // Short timeout for responsiveness
    .with_retry_policy(RetryPolicy::exponential(3));
```

This connectors API provides comprehensive network integration capabilities for edge devices with robust error handling and performance optimization.