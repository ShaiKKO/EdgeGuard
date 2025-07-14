# Raspberry Pi Deployment Guide

Complete guide for deploying EdgeGuard on Raspberry Pi with GPIO integration, performance optimization, and production configurations.

## Hardware Requirements

### Supported Models
- **Raspberry Pi 4**: 2GB+ RAM recommended for full features
- **Raspberry Pi 3B+**: 1GB RAM, suitable for basic validation
- **Raspberry Pi Zero 2W**: 512MB RAM, minimal configurations only
- **Raspberry Pi CM4**: Industrial applications

### Storage Requirements
- **MicroSD Card**: 16GB minimum, 32GB recommended
- **SSD**: Optional, improves I/O performance significantly
- **Network**: Ethernet or WiFi connectivity

### Sensor Connectivity
- **GPIO sensors**: DHT22, DS18B20, PIR sensors
- **I2C sensors**: BME280, SHT30, BMP280
- **SPI sensors**: High-speed sensor modules
- **USB sensors**: Serial or HID sensor devices
- **1-Wire sensors**: Temperature sensor chains

## Operating System Setup

### Raspberry Pi OS Installation

```bash
# Download Raspberry Pi Imager
wget -O rpi-imager.deb https://downloads.raspberrypi.org/imager/imager_latest_amd64.deb
sudo dpkg -i rpi-imager.deb

# Flash OS to SD card using imager
# Enable SSH and configure WiFi during imaging
```

### System Configuration

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y \
    build-essential \
    curl \
    git \
    i2c-tools \
    python3-pip \
    wiringpi

# Enable I2C and SPI
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_spi 0

# Verify I2C
sudo i2cdetect -y 1
```

### Rust Installation

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Add ARM target for cross-compilation
rustup target add armv7-unknown-linux-gnueabihf

# Verify installation
rustc --version
cargo --version
```

## Basic Project Setup

### Project Structure

```
raspberry-pi-edgeguard/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── sensors/
│   │   ├── mod.rs
│   │   ├── gpio.rs
│   │   ├── i2c.rs
│   │   └── spi.rs
│   ├── network/
│   │   ├── mod.rs
│   │   ├── mqtt.rs
│   │   └── http.rs
│   └── config/
│       ├── mod.rs
│       └── device.rs
├── config/
│   └── device.toml
└── systemd/
    └── edgeguard.service
```

### Cargo.toml Configuration

```toml
[package]
name = "raspberry-pi-edgeguard"
version = "0.1.0"
edition = "2021"

[dependencies]
edgeguard = { version = "0.1.0", features = [
    "std",
    "validation-core",
    "pipeline-core",
    "fusion-core",
    "ml",
    "mqtt",
    "http"
] }

# Raspberry Pi GPIO
rppal = "0.14"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
toml = "0.8"
tokio = { version = "1.0", features = ["full"] }
tracing = "0.1"
tracing-subscriber = "0.3"
clap = { version = "4.0", features = ["derive"] }

# Sensor drivers
bme280 = "0.4"
dht22_pi = "0.1"
ds18b20 = "0.1"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

## GPIO Sensor Integration

### DHT22 Temperature/Humidity Sensor

```rust
use rppal::gpio::{Gpio, OutputPin, InputPin, Level, Trigger};
use edgeguard::{
    validators::{TemperatureValidator, HumidityValidator},
    events::{EventBuilder, SensorType},
    time::SystemTime,
};
use std::time::{Duration, Instant};

pub struct DHT22Sensor {
    pin: u8,
    gpio: Gpio,
    temp_validator: TemperatureValidator,
    humidity_validator: HumidityValidator,
    time_source: SystemTime,
}

impl DHT22Sensor {
    pub fn new(pin: u8) -> Result<Self, Box<dyn std::error::Error>> {
        let gpio = Gpio::new()?;
        
        Ok(Self {
            pin,
            gpio,
            temp_validator: TemperatureValidator::new()
                .with_range(-40.0, 80.0)
                .with_rate_limit(2.0),
            humidity_validator: HumidityValidator::new()
                .with_range(0.0, 100.0)
                .with_rate_limit(5.0),
            time_source: SystemTime::new(),
        })
    }
    
    pub fn read_sensor(&mut self) -> Result<(f32, f32), SensorError> {
        let mut pin = self.gpio.get(self.pin)?.into_output();
        
        // Send start signal
        pin.set_low();
        std::thread::sleep(Duration::from_millis(18));
        pin.set_high();
        std::thread::sleep(Duration::from_micros(30));
        
        // Switch to input mode
        let mut pin = pin.into_input();
        pin.set_interrupt(Trigger::Both)?;
        
        // Read data bits
        let mut data = [0u8; 5];
        let mut bit_count = 0;
        let mut byte_count = 0;
        
        let start_time = Instant::now();
        
        while byte_count < 5 && start_time.elapsed() < Duration::from_millis(100) {
            if let Ok(level) = pin.read() {
                if level == Level::High {
                    // High pulse duration determines bit value
                    let pulse_start = Instant::now();
                    while pin.read()? == Level::High {
                        if pulse_start.elapsed() > Duration::from_millis(1) {
                            break;
                        }
                    }
                    
                    let pulse_duration = pulse_start.elapsed();
                    if pulse_duration > Duration::from_micros(50) {
                        data[byte_count] |= 1 << (7 - bit_count);
                    }
                    
                    bit_count += 1;
                    if bit_count == 8 {
                        bit_count = 0;
                        byte_count += 1;
                    }
                }
            }
        }
        
        // Verify checksum
        let checksum = data[0].wrapping_add(data[1]).wrapping_add(data[2]).wrapping_add(data[3]);
        if checksum != data[4] {
            return Err(SensorError::ChecksumError);
        }
        
        // Convert to temperature and humidity
        let humidity = ((data[0] as u16) << 8 | data[1] as u16) as f32 / 10.0;
        let temperature = (((data[2] & 0x7F) as u16) << 8 | data[3] as u16) as f32 / 10.0;
        let temperature = if data[2] & 0x80 != 0 { -temperature } else { temperature };
        
        Ok((temperature, humidity))
    }
    
    pub fn create_events(&mut self) -> Result<Vec<Event>, SensorError> {
        let (temperature, humidity) = self.read_sensor()?;
        let timestamp = self.time_source.now();
        
        let mut events = Vec::new();
        
        // Validate and create temperature event
        match self.temp_validator.validate(temperature) {
            Ok(valid_temp) => {
                if let Some(event) = EventBuilder::new(timestamp)
                    .sensor("dht22_temp", SensorType::Temperature)
                    .reading(valid_temp, 0.85)
                {
                    events.push(event);
                }
            }
            Err(e) => {
                tracing::warn!("Temperature validation failed: {:?}", e);
            }
        }
        
        // Validate and create humidity event
        match self.humidity_validator.validate(humidity) {
            Ok(valid_humidity) => {
                if let Some(event) = EventBuilder::new(timestamp)
                    .sensor("dht22_humidity", SensorType::Humidity)
                    .reading(valid_humidity, 0.80)
                {
                    events.push(event);
                }
            }
            Err(e) => {
                tracing::warn!("Humidity validation failed: {:?}", e);
            }
        }
        
        Ok(events)
    }
}

#[derive(Debug)]
pub enum SensorError {
    GpioError(rppal::gpio::Error),
    TimeoutError,
    ChecksumError,
    ValidationError,
}

impl From<rppal::gpio::Error> for SensorError {
    fn from(err: rppal::gpio::Error) -> Self {
        SensorError::GpioError(err)
    }
}
```

### DS18B20 Temperature Sensor Chain

```rust
use std::fs;
use std::path::Path;
use edgeguard::validators::TemperatureValidator;

pub struct DS18B20Chain {
    base_path: String,
    sensors: Vec<DS18B20Sensor>,
    validator: TemperatureValidator,
}

struct DS18B20Sensor {
    id: String,
    path: String,
    last_reading: f32,
}

impl DS18B20Chain {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let base_path = "/sys/bus/w1/devices".to_string();
        let mut sensors = Vec::new();
        
        // Scan for DS18B20 sensors
        if let Ok(entries) = fs::read_dir(&base_path) {
            for entry in entries {
                if let Ok(entry) = entry {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.starts_with("28-") {  // DS18B20 family code
                        let sensor_path = format!("{}/{}/w1_slave", base_path, name);
                        sensors.push(DS18B20Sensor {
                            id: name,
                            path: sensor_path,
                            last_reading: 0.0,
                        });
                    }
                }
            }
        }
        
        tracing::info!("Found {} DS18B20 sensors", sensors.len());
        
        Ok(Self {
            base_path,
            sensors,
            validator: TemperatureValidator::new()
                .with_range(-55.0, 125.0)  // DS18B20 operating range
                .with_rate_limit(1.0),
        })
    }
    
    pub fn read_all_sensors(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        let timestamp = SystemTime::new().now();
        
        for sensor in &mut self.sensors {
            match self.read_sensor(sensor) {
                Ok(temperature) => {
                    if let Ok(valid_temp) = self.validator.validate(temperature) {
                        if let Some(event) = EventBuilder::new(timestamp)
                            .sensor(&sensor.id, SensorType::Temperature)
                            .reading(valid_temp, 0.95)
                        {
                            events.push(event);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to read sensor {}: {:?}", sensor.id, e);
                }
            }
        }
        
        events
    }
    
    fn read_sensor(&self, sensor: &mut DS18B20Sensor) -> Result<f32, SensorError> {
        let content = fs::read_to_string(&sensor.path)?;
        let lines: Vec<&str> = content.trim().split('\n').collect();
        
        if lines.len() < 2 {
            return Err(SensorError::InvalidData);
        }
        
        // Check CRC
        if !lines[0].ends_with("YES") {
            return Err(SensorError::CrcError);
        }
        
        // Extract temperature
        if let Some(temp_pos) = lines[1].find("t=") {
            let temp_str = &lines[1][temp_pos + 2..];
            let temp_millis: i32 = temp_str.parse()?;
            let temperature = temp_millis as f32 / 1000.0;
            
            sensor.last_reading = temperature;
            Ok(temperature)
        } else {
            Err(SensorError::ParseError)
        }
    }
}
```

## I2C Sensor Integration

### BME280 Environmental Sensor

```rust
use rppal::i2c::I2c;
use edgeguard::{
    validators::{TemperatureValidator, HumidityValidator, PressureValidator},
    events::{EventBuilder, SensorType},
    time::SystemTime,
};

pub struct BME280Sensor {
    i2c: I2c,
    address: u16,
    temp_validator: TemperatureValidator,
    humidity_validator: HumidityValidator,
    pressure_validator: PressureValidator,
    time_source: SystemTime,
}

impl BME280Sensor {
    pub fn new(address: u16) -> Result<Self, Box<dyn std::error::Error>> {
        let mut i2c = I2c::new()?;
        i2c.set_slave_address(address)?;
        
        Ok(Self {
            i2c,
            address,
            temp_validator: TemperatureValidator::new()
                .with_range(-40.0, 85.0)
                .with_rate_limit(2.0),
            humidity_validator: HumidityValidator::new()
                .with_range(0.0, 100.0)
                .with_rate_limit(5.0),
            pressure_validator: PressureValidator::new()
                .with_range(300.0, 1100.0)
                .with_rate_limit(10.0),
            time_source: SystemTime::new(),
        })
    }
    
    pub fn initialize(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Read chip ID
        let chip_id = self.read_register(0xD0)?;
        if chip_id != 0x60 {
            return Err("Invalid BME280 chip ID".into());
        }
        
        // Reset sensor
        self.write_register(0xE0, 0xB6)?;
        std::thread::sleep(Duration::from_millis(10));
        
        // Configure sensor
        self.write_register(0xF2, 0x01)?;  // Humidity oversampling x1
        self.write_register(0xF4, 0x25)?;  // Temp/pressure oversampling x1, normal mode
        self.write_register(0xF5, 0x00)?;  // Config register
        
        tracing::info!("BME280 initialized at address 0x{:02X}", self.address);
        Ok(())
    }
    
    pub fn read_measurements(&mut self) -> Result<Vec<Event>, Box<dyn std::error::Error>> {
        let mut buffer = [0u8; 8];
        self.i2c.block_read(0xF7, &mut buffer)?;
        
        // Parse raw data
        let pressure_raw = ((buffer[0] as u32) << 12) | ((buffer[1] as u32) << 4) | ((buffer[2] as u32) >> 4);
        let temperature_raw = ((buffer[3] as u32) << 12) | ((buffer[4] as u32) << 4) | ((buffer[5] as u32) >> 4);
        let humidity_raw = ((buffer[6] as u32) << 8) | (buffer[7] as u32);
        
        // Convert to physical values (simplified - use proper BME280 library)
        let temperature = (temperature_raw as f32) / 100.0 - 273.15;
        let humidity = (humidity_raw as f32) / 1024.0;
        let pressure = (pressure_raw as f32) / 256.0;
        
        let timestamp = self.time_source.now();
        let mut events = Vec::new();
        
        // Validate and create events
        if let Ok(valid_temp) = self.temp_validator.validate(temperature) {
            if let Some(event) = EventBuilder::new(timestamp)
                .sensor("bme280_temp", SensorType::Temperature)
                .reading(valid_temp, 0.95)
            {
                events.push(event);
            }
        }
        
        if let Ok(valid_humidity) = self.humidity_validator.validate(humidity) {
            if let Some(event) = EventBuilder::new(timestamp)
                .sensor("bme280_humidity", SensorType::Humidity)
                .reading(valid_humidity, 0.90)
            {
                events.push(event);
            }
        }
        
        if let Ok(valid_pressure) = self.pressure_validator.validate(pressure) {
            if let Some(event) = EventBuilder::new(timestamp)
                .sensor("bme280_pressure", SensorType::Pressure)
                .reading(valid_pressure, 0.95)
            {
                events.push(event);
            }
        }
        
        Ok(events)
    }
    
    fn read_register(&mut self, register: u8) -> Result<u8, rppal::i2c::Error> {
        let mut buffer = [0u8; 1];
        self.i2c.block_read(register, &mut buffer)?;
        Ok(buffer[0])
    }
    
    fn write_register(&mut self, register: u8, value: u8) -> Result<(), rppal::i2c::Error> {
        self.i2c.block_write(register, &[value])
    }
}
```

## Advanced Pipeline Configuration

### Multi-Sensor Processing Pipeline

```rust
use edgeguard::{
    pipeline::{Pipeline, ValidationStage, CrossValidationStage, FusionStage, AggregationStage},
    fusion::{KalmanFilter, KalmanConfig},
    events::{CrossValidationType, WindowSpec, AggregationMethod},
};

pub struct SensorHub {
    pipeline: Pipeline<1024>,
    dht22: DHT22Sensor,
    bme280: BME280Sensor,
    ds18b20_chain: DS18B20Chain,
}

impl SensorHub {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let pipeline = Self::create_pipeline()?;
        
        Ok(Self {
            pipeline,
            dht22: DHT22Sensor::new(4)?,
            bme280: BME280Sensor::new(0x76)?,
            ds18b20_chain: DS18B20Chain::new()?,
        })
    }
    
    fn create_pipeline() -> Result<Pipeline<1024>, Box<dyn std::error::Error>> {
        // Configure Kalman filter for temperature fusion
        let kalman_config = KalmanConfig {
            initial_state: [20.0],
            initial_covariance: [[1.0]],
            process_noise: [[0.1]],
            measurement_noise: [[0.5]],
            transition: StateTransition {
                transition_matrix: [[1.0]],
                control_matrix: None,
            },
            measurement_matrix: [[1.0]],
            control_matrix: None,
            convergence_threshold: 0.01,
        };
        
        let kalman_filter = KalmanFilter::new(kalman_config);
        
        // Create cross-validation stage
        let mut cross_validation = CrossValidationStage::new();
        cross_validation.add_pair(
            SensorType::Temperature,
            SensorType::Humidity,
            CrossValidationType::DewPoint
        )?;
        
        // Create aggregation stage
        let aggregation = AggregationStage::new(
            WindowSpec::TimeWindow(60_000),  // 1 minute window
            AggregationMethod::Statistics,
            SensorType::Temperature
        );
        
        let pipeline = Pipeline::<1024>::builder()
            .add_stage(ValidationStage::new(
                TemperatureValidator::new()
                    .with_range(-40.0, 85.0)
                    .with_rate_limit(5.0),
                SensorType::Temperature
            ))
            .add_stage(ValidationStage::new(
                HumidityValidator::new()
                    .with_range(0.0, 100.0)
                    .with_rate_limit(10.0),
                SensorType::Humidity
            ))
            .add_stage(ValidationStage::new(
                PressureValidator::new()
                    .with_range(300.0, 1100.0)
                    .with_rate_limit(20.0),
                SensorType::Pressure
            ))
            .add_stage(cross_validation)
            .add_stage(FusionStage::new(Box::new(kalman_filter)))
            .add_stage(aggregation)
            .build();
        
        Ok(pipeline)
    }
    
    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut interval = tokio::time::interval(Duration::from_secs(5));
        
        loop {
            interval.tick().await;
            
            // Collect sensor readings
            let mut all_events = Vec::new();
            
            // DHT22 readings
            match self.dht22.create_events() {
                Ok(mut events) => all_events.append(&mut events),
                Err(e) => tracing::warn!("DHT22 error: {:?}", e),
            }
            
            // BME280 readings
            match self.bme280.read_measurements() {
                Ok(mut events) => all_events.append(&mut events),
                Err(e) => tracing::warn!("BME280 error: {:?}", e),
            }
            
            // DS18B20 readings
            let mut ds18b20_events = self.ds18b20_chain.read_all_sensors();
            all_events.append(&mut ds18b20_events);
            
            // Process through pipeline
            for event in all_events {
                self.pipeline.push_event(event);
            }
            
            if let Ok(processed) = self.pipeline.process_batch(100) {
                tracing::info!("Processed {} events", processed);
            }
            
            // Handle pipeline results
            while let Some(result) = self.pipeline.pop_result() {
                self.handle_pipeline_result(result).await;
            }
        }
    }
    
    async fn handle_pipeline_result(&self, result: Event) {
        match result {
            Event::ValidationResult { sensor_id, status, .. } => {
                tracing::info!("Sensor {}: {:?}", sensor_id.as_str(), status);
            }
            Event::FusionResult { value, confidence, .. } => {
                tracing::info!("Fused value: {:.2} (confidence: {:.2})", 
                    value, confidence.as_f32());
            }
            Event::AggregationResult { statistics, .. } => {
                tracing::info!("Aggregated stats: mean={:.2}, std={:.2}", 
                    statistics.mean, statistics.std_dev);
            }
            _ => {}
        }
    }
}
```

## Network Integration

### MQTT Client Integration

```rust
use edgeguard::connectors::mqtt::{MqttConnector, MqttConfig};
use tokio::time::{Duration, interval};

pub struct MqttGateway {
    sensor_hub: SensorHub,
    mqtt_client: MqttConnector,
}

impl MqttGateway {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let sensor_hub = SensorHub::new()?;
        
        let mqtt_config = MqttConfig::new("raspberry-pi-gateway", "mqtt://localhost:1883")
            .with_credentials("sensor_user", "sensor_password")
            .with_clean_session(false)
            .with_keep_alive(Duration::from_secs(60))
            .with_max_reconnect_attempts(10);
        
        let mqtt_client = MqttConnector::new(mqtt_config)?;
        
        Ok(Self {
            sensor_hub,
            mqtt_client,
        })
    }
    
    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Subscribe to command topics
        self.mqtt_client.subscribe("sensors/raspberry-pi/commands", 1)?;
        
        let mut sensor_interval = interval(Duration::from_secs(10));
        let mut mqtt_interval = interval(Duration::from_secs(1));
        
        loop {
            tokio::select! {
                _ = sensor_interval.tick() => {
                    self.process_sensors().await?;
                }
                _ = mqtt_interval.tick() => {
                    self.process_mqtt().await?;
                }
            }
        }
    }
    
    async fn process_sensors(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Process sensor data and send to MQTT
        let events = self.collect_sensor_events().await?;
        
        for event in events {
            let topic = format!("sensors/raspberry-pi/{}", 
                self.get_sensor_type(&event));
            
            let payload = self.serialize_event(&event)?;
            self.mqtt_client.publish(&topic, &payload, 1)?;
        }
        
        Ok(())
    }
    
    async fn process_mqtt(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let messages = self.mqtt_client.poll()?;
        
        for message in messages {
            if message.topic.ends_with("/commands") {
                self.handle_command(&message.payload).await?;
            }
        }
        
        Ok(())
    }
    
    async fn handle_command(&mut self, payload: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        let command: serde_json::Value = serde_json::from_slice(payload)?;
        
        match command.get("action").and_then(|v| v.as_str()) {
            Some("restart") => {
                tracing::info!("Restart command received");
                // Implement restart logic
            }
            Some("configure") => {
                tracing::info!("Configure command received");
                // Implement configuration update
            }
            Some("status") => {
                let status = self.get_system_status().await;
                let status_json = serde_json::to_vec(&status)?;
                self.mqtt_client.publish("sensors/raspberry-pi/status", &status_json, 1)?;
            }
            _ => {
                tracing::warn!("Unknown command: {:?}", command);
            }
        }
        
        Ok(())
    }
    
    fn get_sensor_type(&self, event: &Event) -> &str {
        match event {
            Event::SensorReading { sensor_type, .. } => {
                match sensor_type {
                    SensorType::Temperature => "temperature",
                    SensorType::Humidity => "humidity",
                    SensorType::Pressure => "pressure",
                    _ => "unknown",
                }
            }
            _ => "event",
        }
    }
    
    fn serialize_event(&self, event: &Event) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(event)
    }
    
    async fn get_system_status(&self) -> serde_json::Value {
        serde_json::json!({
            "timestamp": SystemTime::new().now(),
            "uptime": self.get_uptime(),
            "memory_usage": self.get_memory_usage(),
            "cpu_temperature": self.get_cpu_temperature(),
            "pipeline_metrics": self.sensor_hub.pipeline.metrics()
        })
    }
    
    fn get_uptime(&self) -> f64 {
        // Read from /proc/uptime
        if let Ok(content) = std::fs::read_to_string("/proc/uptime") {
            if let Some(uptime_str) = content.split_whitespace().next() {
                return uptime_str.parse().unwrap_or(0.0);
            }
        }
        0.0
    }
    
    fn get_memory_usage(&self) -> f64 {
        // Read from /proc/meminfo
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            let mut total = 0;
            let mut available = 0;
            
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    total = line.split_whitespace().nth(1)
                        .and_then(|s| s.parse().ok()).unwrap_or(0);
                } else if line.starts_with("MemAvailable:") {
                    available = line.split_whitespace().nth(1)
                        .and_then(|s| s.parse().ok()).unwrap_or(0);
                }
            }
            
            if total > 0 {
                return ((total - available) as f64 / total as f64) * 100.0;
            }
        }
        0.0
    }
    
    fn get_cpu_temperature(&self) -> f64 {
        // Read from /sys/class/thermal/thermal_zone0/temp
        if let Ok(content) = std::fs::read_to_string("/sys/class/thermal/thermal_zone0/temp") {
            if let Ok(temp_millis) = content.trim().parse::<i32>() {
                return temp_millis as f64 / 1000.0;
            }
        }
        0.0
    }
}
```

## Performance Optimization

### Multi-Threading Configuration

```rust
use tokio::runtime::Runtime;
use std::sync::Arc;
use parking_lot::Mutex;

pub struct OptimizedSensorHub {
    sensor_tasks: Vec<tokio::task::JoinHandle<()>>,
    pipeline: Arc<Mutex<Pipeline<2048>>>,
    config: Arc<HubConfig>,
}

impl OptimizedSensorHub {
    pub fn new(config: HubConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let pipeline = Arc::new(Mutex::new(Self::create_optimized_pipeline()?));
        let config = Arc::new(config);
        
        Ok(Self {
            sensor_tasks: Vec::new(),
            pipeline,
            config,
        })
    }
    
    pub async fn start(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Start sensor reading tasks
        self.start_sensor_tasks().await;
        
        // Start processing task
        self.start_processing_task().await;
        
        // Start network tasks
        self.start_network_tasks().await;
        
        Ok(())
    }
    
    async fn start_sensor_tasks(&mut self) {
        let pipeline = Arc::clone(&self.pipeline);
        let config = Arc::clone(&self.config);
        
        // I2C sensor task
        let i2c_task = tokio::spawn(async move {
            let mut bme280 = BME280Sensor::new(0x76).unwrap();
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            
            loop {
                interval.tick().await;
                
                if let Ok(events) = bme280.read_measurements() {
                    let mut pipeline_guard = pipeline.lock();
                    for event in events {
                        pipeline_guard.push_event(event);
                    }
                }
            }
        });
        
        // GPIO sensor task
        let gpio_task = tokio::spawn(async move {
            let mut dht22 = DHT22Sensor::new(4).unwrap();
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                
                if let Ok(events) = dht22.create_events() {
                    let mut pipeline_guard = pipeline.lock();
                    for event in events {
                        pipeline_guard.push_event(event);
                    }
                }
            }
        });
        
        self.sensor_tasks.push(i2c_task);
        self.sensor_tasks.push(gpio_task);
    }
    
    async fn start_processing_task(&mut self) {
        let pipeline = Arc::clone(&self.pipeline);
        
        let processing_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(100));
            
            loop {
                interval.tick().await;
                
                let mut pipeline_guard = pipeline.lock();
                if let Ok(processed) = pipeline_guard.process_batch(50) {
                    if processed > 0 {
                        tracing::debug!("Processed {} events", processed);
                    }
                }
            }
        });
        
        self.sensor_tasks.push(processing_task);
    }
    
    fn create_optimized_pipeline() -> Result<Pipeline<2048>, Box<dyn std::error::Error>> {
        // Create pipeline optimized for Raspberry Pi performance
        let pipeline = Pipeline::<2048>::builder()
            .add_stage(ValidationStage::new(
                TemperatureValidator::new()
                    .with_range(-40.0, 85.0)
                    .with_rate_limit(5.0),
                SensorType::Temperature
            ))
            .add_stage(ValidationStage::new(
                HumidityValidator::new()
                    .with_range(0.0, 100.0)
                    .with_rate_limit(10.0),
                SensorType::Humidity
            ))
            .add_stage(ValidationStage::new(
                PressureValidator::new()
                    .with_range(300.0, 1100.0)
                    .with_rate_limit(20.0),
                SensorType::Pressure
            ))
            .build();
        
        Ok(pipeline)
    }
}

pub struct HubConfig {
    pub sensor_interval: Duration,
    pub processing_interval: Duration,
    pub network_interval: Duration,
    pub buffer_size: usize,
}

impl Default for HubConfig {
    fn default() -> Self {
        Self {
            sensor_interval: Duration::from_secs(5),
            processing_interval: Duration::from_millis(100),
            network_interval: Duration::from_secs(1),
            buffer_size: 2048,
        }
    }
}
```

## System Integration

### Systemd Service Configuration

```ini
# /etc/systemd/system/edgeguard.service
[Unit]
Description=EdgeGuard Sensor Gateway
After=network.target
Wants=network.target

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=/home/pi/edgeguard
ExecStart=/home/pi/edgeguard/target/release/raspberry-pi-edgeguard
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

# Environment variables
Environment=RUST_LOG=info
Environment=RUST_BACKTRACE=1

# Resource limits
LimitNOFILE=65536
MemoryMax=512M

[Install]
WantedBy=multi-user.target
```

### Installation Script

```bash
#!/bin/bash
# install.sh

set -e

# Build application
echo "Building EdgeGuard..."
cargo build --release

# Copy binary
sudo cp target/release/raspberry-pi-edgeguard /usr/local/bin/
sudo chmod +x /usr/local/bin/raspberry-pi-edgeguard

# Copy systemd service
sudo cp systemd/edgeguard.service /etc/systemd/system/
sudo systemctl daemon-reload

# Enable and start service
sudo systemctl enable edgeguard
sudo systemctl start edgeguard

# Check status
sudo systemctl status edgeguard

echo "EdgeGuard installed and started successfully!"
```

## Configuration Management

### Device Configuration

```toml
# config/device.toml
[device]
name = "raspberry-pi-001"
location = "Living Room"
timezone = "UTC"

[sensors]
[sensors.dht22]
enabled = true
pin = 4
interval = 10

[sensors.bme280]
enabled = true
address = 0x76
interval = 5

[sensors.ds18b20]
enabled = true
scan_interval = 30

[network]
[network.mqtt]
enabled = true
broker = "mqtt://localhost:1883"
username = "sensor_user"
password = "sensor_password"
topics = ["sensors/raspberry-pi"]

[network.http]
enabled = false
endpoint = "https://api.example.com/sensors"
api_key = "your-api-key"

[pipeline]
buffer_size = 1024
batch_size = 50
processing_interval = 100

[validation]
[validation.temperature]
min_range = -40.0
max_range = 85.0
rate_limit = 5.0

[validation.humidity]
min_range = 0.0
max_range = 100.0
rate_limit = 10.0

[validation.pressure]
min_range = 300.0
max_range = 1100.0
rate_limit = 20.0
```

### Configuration Loading

```rust
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Deserialize, Serialize)]
pub struct DeviceConfig {
    pub device: DeviceInfo,
    pub sensors: SensorConfig,
    pub network: NetworkConfig,
    pub pipeline: PipelineConfig,
    pub validation: ValidationConfig,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DeviceInfo {
    pub name: String,
    pub location: String,
    pub timezone: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SensorConfig {
    pub dht22: Option<DHT22Config>,
    pub bme280: Option<BME280Config>,
    pub ds18b20: Option<DS18B20Config>,
}

impl DeviceConfig {
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: DeviceConfig = toml::from_str(&content)?;
        Ok(config)
    }
    
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let content = toml::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }
}
```

## Production Deployment

### Monitoring and Logging

```rust
use tracing::{info, warn, error};
use tracing_subscriber::{EnvFilter, fmt::format::FmtSpan};

pub fn setup_logging() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_span_events(FmtSpan::CLOSE)
        .with_target(false)
        .with_thread_ids(true)
        .init();
}

pub struct SystemMonitor {
    start_time: std::time::Instant,
    metrics: SystemMetrics,
}

#[derive(Debug, Default)]
pub struct SystemMetrics {
    pub uptime: Duration,
    pub memory_usage: f64,
    pub cpu_temperature: f64,
    pub sensor_errors: u32,
    pub network_errors: u32,
}

impl SystemMonitor {
    pub fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
            metrics: SystemMetrics::default(),
        }
    }
    
    pub fn update_metrics(&mut self) {
        self.metrics.uptime = self.start_time.elapsed();
        self.metrics.memory_usage = self.get_memory_usage();
        self.metrics.cpu_temperature = self.get_cpu_temperature();
    }
    
    pub fn log_metrics(&self) {
        info!(
            "System metrics: uptime={:.1}s, memory={:.1}%, cpu_temp={:.1}°C",
            self.metrics.uptime.as_secs_f64(),
            self.metrics.memory_usage,
            self.metrics.cpu_temperature
        );
    }
    
    fn get_memory_usage(&self) -> f64 {
        // Implementation from previous example
        0.0
    }
    
    fn get_cpu_temperature(&self) -> f64 {
        // Implementation from previous example
        0.0
    }
}
```

### Health Checks

```rust
use serde_json::json;

pub struct HealthChecker {
    last_check: std::time::Instant,
    check_interval: Duration,
}

impl HealthChecker {
    pub fn new() -> Self {
        Self {
            last_check: std::time::Instant::now(),
            check_interval: Duration::from_secs(60),
        }
    }
    
    pub fn check_system_health(&mut self) -> HealthStatus {
        if self.last_check.elapsed() < self.check_interval {
            return HealthStatus::Ok;
        }
        
        self.last_check = std::time::Instant::now();
        
        // Check memory usage
        let memory_usage = self.get_memory_usage();
        if memory_usage > 90.0 {
            return HealthStatus::Critical("High memory usage".to_string());
        }
        
        // Check CPU temperature
        let cpu_temp = self.get_cpu_temperature();
        if cpu_temp > 80.0 {
            return HealthStatus::Warning("High CPU temperature".to_string());
        }
        
        // Check disk space
        let disk_usage = self.get_disk_usage();
        if disk_usage > 95.0 {
            return HealthStatus::Critical("Low disk space".to_string());
        }
        
        HealthStatus::Ok
    }
    
    fn get_memory_usage(&self) -> f64 {
        // Implementation from previous example
        0.0
    }
    
    fn get_cpu_temperature(&self) -> f64 {
        // Implementation from previous example
        0.0
    }
    
    fn get_disk_usage(&self) -> f64 {
        // Check root filesystem usage
        if let Ok(output) = std::process::Command::new("df")
            .args(&["-h", "/"])
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = output_str.lines().nth(1) {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 5 {
                    let usage_str = parts[4].trim_end_matches('%');
                    return usage_str.parse().unwrap_or(0.0);
                }
            }
        }
        0.0
    }
}

#[derive(Debug)]
pub enum HealthStatus {
    Ok,
    Warning(String),
    Critical(String),
}
```

This comprehensive Raspberry Pi deployment guide provides complete coverage of EdgeGuard implementation with GPIO integration, performance optimization, and production-ready configurations.