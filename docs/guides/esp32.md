# ESP32 Deployment Guide

Complete guide for deploying EdgeGuard on ESP32 microcontrollers with optimized configurations and examples.

## Hardware Requirements

### Minimum Requirements
- **ESP32 DevKit**: 4MB flash, 520KB RAM
- **Power supply**: 3.3V, 500mA minimum
- **Sensors**: BME280, DHT22, DS18B20, or compatible
- **Development board**: ESP32-WROOM-32 or ESP32-S3

### Recommended Hardware
- **ESP32-S3**: 8MB flash, 512KB RAM for advanced features
- **External sensors**: I2C/SPI sensors for better performance
- **External antenna**: For improved WiFi range
- **Level shifters**: For 5V sensor compatibility

### Supported Sensor Modules
- **BME280**: Temperature, humidity, pressure (I2C/SPI)
- **DHT22**: Temperature, humidity (GPIO)
- **DS18B20**: Temperature (OneWire)
- **BMP280**: Pressure, temperature (I2C/SPI)
- **SHT30**: Temperature, humidity (I2C)

## Development Environment Setup

### Install ESP32 Toolchain

```bash
# Install espup
cargo install espup

# Install ESP32 toolchain
espup install

# Source environment variables
source ~/export-esp.sh

# Add to shell profile for persistence
echo 'source ~/export-esp.sh' >> ~/.bashrc
```

### Install Development Tools

```bash
# Install flashing and monitoring tools
cargo install espflash espmonitor

# Install additional tools
cargo install cargo-espflash
```

### Verify Installation

```bash
# Check toolchain
rustc --version --verbose

# Check targets
rustup target list | grep esp32

# Test compilation
cargo check --target xtensa-esp32-none-elf
```

## Project Configuration

### Cargo.toml Setup

```toml
[package]
name = "esp32-edgeguard"
version = "0.1.0"
edition = "2021"

[dependencies]
edgeguard = { version = "0.1.0", default-features = false, features = [
    "embedded",
    "validation-core",
    "pipeline-core",
    "fusion-core"
] }

# ESP32 specific dependencies
esp-hal = "0.10"
esp-println = "0.6"
esp-backtrace = { version = "0.8", features = ["esp32", "panic-handler", "print-uart"] }
nb = "1.0"
heapless = "0.8"

[profile.release]
opt-level = "s"        # Optimize for size
lto = true             # Link-time optimization
codegen-units = 1      # Single codegen unit
strip = true           # Strip debug symbols
panic = "abort"        # Smaller panic handler

[profile.dev]
debug = true           # Enable debug info
opt-level = "z"        # Optimize for size even in debug
```

### Memory Configuration

```rust
// src/main.rs
#![no_std]
#![no_main]

use esp_backtrace as _;
use esp_hal::{
    clock::ClockControl,
    peripherals::Peripherals,
    prelude::*,
    system::SystemControl,
    timer::TimerGroup,
    Delay,
};

// Configure heap size
const HEAP_SIZE: usize = 32 * 1024;  // 32KB heap

#[global_allocator]
static ALLOCATOR: esp_alloc::EspHeap = esp_alloc::EspHeap::empty();

fn init_heap() {
    use core::mem::MaybeUninit;
    static mut HEAP: MaybeUninit<[u8; HEAP_SIZE]> = MaybeUninit::uninit();
    unsafe { ALLOCATOR.init(HEAP.as_mut_ptr() as *mut u8, HEAP_SIZE) }
}
```

## Basic Implementation

### Simple Temperature Validation

```rust
use edgeguard::{
    validators::TemperatureValidator,
    events::{EventBuilder, SensorType},
    time::MonotonicTime,
};
use esp_hal::{
    gpio::{GpioPin, Input, PullUp},
    peripherals::GPIO,
};

#[entry]
fn main() -> ! {
    init_heap();
    
    let peripherals = Peripherals::take();
    let system = SystemControl::new(peripherals.SYSTEM);
    let clocks = ClockControl::boot_defaults(system.clock_control).freeze();
    
    // Initialize time source
    let time_source = MonotonicTime::new();
    
    // Create temperature validator
    let validator = TemperatureValidator::new()
        .with_range(-20.0, 60.0)    // Indoor temperature range
        .with_rate_limit(2.0);      // Conservative rate limit
    
    // Configure sensor pin (DHT22)
    let gpio = GPIO::new(peripherals.GPIO);
    let sensor_pin = gpio.gpio4.into_push_pull_output();
    
    let mut delay = Delay::new(&clocks);
    
    loop {
        // Read sensor (simplified)
        let temperature = read_dht22_temperature(&sensor_pin, &mut delay);
        
        // Validate reading
        match validator.validate(temperature) {
            Ok(valid_temp) => {
                esp_println::println!("Valid temperature: {:.1}°C", valid_temp);
            }
            Err(e) => {
                esp_println::println!("Validation error: {:?}", e);
            }
        }
        
        delay.delay_ms(5000u32);  // 5 second intervals
    }
}

fn read_dht22_temperature(pin: &GpioPin, delay: &mut Delay) -> f32 {
    // DHT22 reading implementation
    // This is a simplified version - use proper DHT22 library
    23.5  // Placeholder value
}
```

### Multi-Sensor Pipeline

```rust
use edgeguard::{
    pipeline::{Pipeline, ValidationStage},
    validators::{TemperatureValidator, HumidityValidator},
    events::{EventBuilder, SensorType},
    time::MonotonicTime,
};
use esp_hal::{
    i2c::I2C,
    peripherals::{I2C0, GPIO},
    gpio::{GpioPin, InputOutput, PullUp},
};

struct SensorHub {
    pipeline: Pipeline<64>,
    i2c: I2C<'static, I2C0>,
    time_source: MonotonicTime,
}

impl SensorHub {
    fn new(i2c: I2C<'static, I2C0>) -> Self {
        let pipeline = Pipeline::<64>::builder()
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
            .build();
        
        Self {
            pipeline,
            i2c,
            time_source: MonotonicTime::new(),
        }
    }
    
    fn process_sensors(&mut self) {
        // Read BME280 sensor
        let (temperature, humidity) = self.read_bme280();
        
        // Create events
        let temp_event = EventBuilder::new(self.time_source.now())
            .sensor("bme280", SensorType::Temperature)
            .reading(temperature, 0.95)
            .unwrap();
        
        let humidity_event = EventBuilder::new(self.time_source.now())
            .sensor("bme280", SensorType::Humidity)
            .reading(humidity, 0.90)
            .unwrap();
        
        // Process through pipeline
        self.pipeline.push_event(temp_event);
        self.pipeline.push_event(humidity_event);
        
        if let Ok(processed) = self.pipeline.process_batch(10) {
            esp_println::println!("Processed {} events", processed);
        }
        
        // Handle results
        while let Some(result) = self.pipeline.pop_result() {
            match result {
                Event::ValidationResult { sensor_id, status, .. } => {
                    esp_println::println!("Sensor {}: {:?}", sensor_id.as_str(), status);
                }
                _ => {}
            }
        }
    }
    
    fn read_bme280(&mut self) -> (f32, f32) {
        // BME280 I2C reading implementation
        // Use proper BME280 library in production
        (23.5, 65.0)  // Placeholder values
    }
}
```

## Advanced Features

### Kalman Filter Fusion

```rust
use edgeguard::{
    fusion::{KalmanFilter, KalmanConfig, StateTransition},
    pipeline::FusionStage,
};

fn create_fusion_pipeline() -> Pipeline<128> {
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
    let fusion_stage = FusionStage::new(Box::new(kalman_filter));
    
    Pipeline::<128>::builder()
        .add_stage(ValidationStage::new(
            TemperatureValidator::new(),
            SensorType::Temperature
        ))
        .add_stage(fusion_stage)
        .build()
}
```

### WiFi Integration

```rust
use esp_hal::{
    peripherals::WIFI,
    wifi::{WifiController, WifiDevice, WifiEvent, WifiStaDevice, WifiState},
};

struct WiFiSensorGateway {
    pipeline: Pipeline<256>,
    wifi: WifiController<'static>,
    connected: bool,
}

impl WiFiSensorGateway {
    fn new(wifi: WifiController<'static>) -> Self {
        let pipeline = Pipeline::<256>::builder()
            .add_stage(ValidationStage::new(
                TemperatureValidator::new(),
                SensorType::Temperature
            ))
            .build();
        
        Self {
            pipeline,
            wifi,
            connected: false,
        }
    }
    
    fn connect_wifi(&mut self, ssid: &str, password: &str) -> Result<(), WifiError> {
        // WiFi connection implementation
        self.wifi.set_configuration(&wifi::Configuration::Client(
            wifi::ClientConfiguration {
                ssid: ssid.into(),
                password: password.into(),
                ..Default::default()
            }
        ))?;
        
        self.wifi.start()?;
        self.wifi.connect()?;
        
        // Wait for connection
        loop {
            if self.wifi.is_connected()? {
                self.connected = true;
                break;
            }
            // Small delay
        }
        
        Ok(())
    }
    
    fn send_sensor_data(&mut self, data: &[u8]) -> Result<(), NetworkError> {
        if !self.connected {
            return Err(NetworkError::NotConnected);
        }
        
        // Send data over WiFi
        // Implementation depends on chosen protocol (HTTP, MQTT, etc.)
        Ok(())
    }
}
```

## Memory Optimization

### Buffer Size Configuration

```rust
// Configure for different ESP32 variants
#[cfg(feature = "esp32-c3")]
const PIPELINE_SIZE: usize = 32;

#[cfg(feature = "esp32-s3")]
const PIPELINE_SIZE: usize = 128;

#[cfg(not(any(feature = "esp32-c3", feature = "esp32-s3")))]
const PIPELINE_SIZE: usize = 64;

// Create pipeline with appropriate size
let pipeline = Pipeline::<PIPELINE_SIZE>::builder()
    .add_stage(ValidationStage::new(
        TemperatureValidator::new(),
        SensorType::Temperature
    ))
    .build();
```

### Memory Usage Monitoring

```rust
use esp_hal::system::SystemControl;

fn check_memory_usage() {
    let free_heap = unsafe { esp_alloc::HEAP.used() };
    let total_heap = HEAP_SIZE;
    let used_percentage = (free_heap * 100) / total_heap;
    
    esp_println::println!("Heap usage: {}/{} bytes ({}%)", 
        free_heap, total_heap, used_percentage);
    
    if used_percentage > 80 {
        esp_println::println!("WARNING: High memory usage!");
    }
}
```

## Performance Optimization

### Sensor Reading Optimization

```rust
use esp_hal::timer::TimerGroup;

struct OptimizedSensorReader {
    last_reading: u64,
    reading_interval: u64,
    cached_value: f32,
}

impl OptimizedSensorReader {
    fn new(interval_ms: u64) -> Self {
        Self {
            last_reading: 0,
            reading_interval: interval_ms,
            cached_value: 0.0,
        }
    }
    
    fn read_temperature(&mut self, current_time: u64) -> Option<f32> {
        if current_time - self.last_reading >= self.reading_interval {
            // Only read sensor when interval elapsed
            self.cached_value = read_actual_sensor();
            self.last_reading = current_time;
            Some(self.cached_value)
        } else {
            // Return cached value
            None
        }
    }
}

fn read_actual_sensor() -> f32 {
    // Actual sensor reading implementation
    23.5
}
```

### Pipeline Optimization

```rust
// Batch processing for better performance
fn process_sensor_batch(pipeline: &mut Pipeline<64>, sensors: &[f32]) {
    let time_source = MonotonicTime::new();
    
    // Create events in batch
    for (i, &temperature) in sensors.iter().enumerate() {
        let event = EventBuilder::new(time_source.now())
            .sensor(&format!("sensor_{}", i), SensorType::Temperature)
            .reading(temperature, 0.95)
            .unwrap();
        
        pipeline.push_event(event);
    }
    
    // Process all events at once
    if let Ok(processed) = pipeline.process_batch(sensors.len()) {
        esp_println::println!("Processed {} sensor readings", processed);
    }
}
```

## Error Handling

### Robust Error Recovery

```rust
use edgeguard::validators::ValidationError;

fn handle_sensor_errors(result: Result<f32, ValidationError>) {
    match result {
        Ok(value) => {
            esp_println::println!("Valid reading: {:.1}°C", value);
        }
        Err(ValidationError::OutOfRange { value, min, max }) => {
            esp_println::println!("Temperature {} outside range [{}, {}]", value, min, max);
            // Continue processing - might be temporary spike
        }
        Err(ValidationError::RateOfChangeExceeded { rate, max_rate }) => {
            esp_println::println!("Rate {} exceeds maximum {}", rate, max_rate);
            // Possible sensor malfunction - increase monitoring
        }
        Err(ValidationError::PhysicsViolation(msg)) => {
            esp_println::println!("Physics violation: {}", msg);
            // Serious issue - may need sensor recalibration
        }
        Err(e) => {
            esp_println::println!("Validation error: {:?}", e);
        }
    }
}
```

### Watchdog Timer

```rust
use esp_hal::timer::TimerGroup;

struct WatchdogManager {
    timer: TimerGroup,
    last_feed: u64,
    timeout_ms: u64,
}

impl WatchdogManager {
    fn new(timer: TimerGroup, timeout_ms: u64) -> Self {
        Self {
            timer,
            last_feed: 0,
            timeout_ms,
        }
    }
    
    fn feed(&mut self) {
        self.timer.wdt.feed();
        self.last_feed = MonotonicTime::new().now();
    }
    
    fn check_timeout(&self) -> bool {
        let current_time = MonotonicTime::new().now();
        current_time - self.last_feed > self.timeout_ms
    }
}
```

## Deployment Strategies

### Over-the-Air Updates

```rust
use esp_hal::flash::Flash;

struct OTAManager {
    flash: Flash<'static>,
    update_available: bool,
}

impl OTAManager {
    fn new(flash: Flash<'static>) -> Self {
        Self {
            flash,
            update_available: false,
        }
    }
    
    fn check_for_updates(&mut self) -> Result<bool, OTAError> {
        // Check for firmware updates
        // Implementation depends on update mechanism
        Ok(false)
    }
    
    fn perform_update(&mut self) -> Result<(), OTAError> {
        if self.update_available {
            // Download and flash new firmware
            esp_println::println!("Starting OTA update...");
            // Implementation depends on update source
        }
        Ok(())
    }
}
```

### Power Management

```rust
use esp_hal::power::PowerMode;

fn configure_power_management() {
    // Configure for low power operation
    let power_config = PowerConfiguration {
        cpu_frequency: CpuFrequency::MHz160,  // Reduce from 240MHz
        wifi_power_save: true,
        light_sleep_enabled: true,
    };
    
    // Apply power configuration
    apply_power_config(power_config);
}

fn enter_deep_sleep(duration_ms: u64) {
    esp_println::println!("Entering deep sleep for {} ms", duration_ms);
    
    // Configure wake-up source
    configure_wakeup_timer(duration_ms);
    
    // Enter deep sleep
    esp_hal::power::deep_sleep();
}
```

## Testing and Debugging

### Serial Debugging

```rust
use esp_println::println;

fn debug_sensor_pipeline(pipeline: &Pipeline<64>) {
    let metrics = pipeline.metrics();
    
    println!("Pipeline metrics:");
    println!("  Events processed: {}", metrics.events_processed);
    println!("  Events dropped: {}", metrics.events_dropped);
    println!("  Processing errors: {}", metrics.processing_errors);
    println!("  Current depth: {}", metrics.current_depth);
}
```

### Hardware Testing

```rust
fn test_sensor_connectivity() {
    esp_println::println!("Testing sensor connectivity...");
    
    // Test I2C sensors
    test_i2c_sensors();
    
    // Test GPIO sensors
    test_gpio_sensors();
    
    // Test OneWire sensors
    test_onewire_sensors();
}

fn test_i2c_sensors() {
    // I2C sensor detection
    let i2c_addresses = [0x76, 0x77, 0x44, 0x45];  // Common sensor addresses
    
    for addr in i2c_addresses {
        if probe_i2c_device(addr) {
            esp_println::println!("Found I2C device at 0x{:02X}", addr);
        }
    }
}

fn probe_i2c_device(address: u8) -> bool {
    // I2C device probing implementation
    false  // Placeholder
}
```

## Production Considerations

### Monitoring and Logging

```rust
use esp_hal::rtc::Rtc;

struct ProductionMonitor {
    start_time: u64,
    error_count: u32,
    last_error: Option<&'static str>,
}

impl ProductionMonitor {
    fn new() -> Self {
        Self {
            start_time: MonotonicTime::new().now(),
            error_count: 0,
            last_error: None,
        }
    }
    
    fn report_error(&mut self, error: &'static str) {
        self.error_count += 1;
        self.last_error = Some(error);
        
        esp_println::println!("ERROR #{}: {}", self.error_count, error);
        
        // Send to monitoring system if connected
        if self.error_count > 100 {
            esp_println::println!("High error count - consider restart");
        }
    }
    
    fn uptime(&self) -> u64 {
        MonotonicTime::new().now() - self.start_time
    }
}
```

### Configuration Management

```rust
use esp_hal::storage::FlashStorage;

struct DeviceConfig {
    sensor_interval: u32,
    wifi_ssid: heapless::String<32>,
    wifi_password: heapless::String<64>,
    server_url: heapless::String<128>,
}

impl DeviceConfig {
    fn load_from_flash() -> Result<Self, ConfigError> {
        // Load configuration from flash storage
        Ok(Self::default())
    }
    
    fn save_to_flash(&self) -> Result<(), ConfigError> {
        // Save configuration to flash storage
        Ok(())
    }
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            sensor_interval: 5000,  // 5 seconds
            wifi_ssid: heapless::String::new(),
            wifi_password: heapless::String::new(),
            server_url: heapless::String::new(),
        }
    }
}
```

## Best Practices

### Code Organization

```rust
// src/sensors/mod.rs
pub mod bme280;
pub mod dht22;
pub mod ds18b20;

// src/network/mod.rs
pub mod wifi;
pub mod mqtt;

// src/config/mod.rs
pub mod device_config;
pub mod sensor_config;

// src/main.rs
mod sensors;
mod network;
mod config;

use edgeguard::prelude::*;
```

### Error Handling Strategy

```rust
// Define application-specific error types
#[derive(Debug)]
enum AppError {
    Sensor(SensorError),
    Network(NetworkError),
    Validation(ValidationError),
    Configuration(ConfigError),
}

// Implement error conversion
impl From<ValidationError> for AppError {
    fn from(err: ValidationError) -> Self {
        AppError::Validation(err)
    }
}

// Centralized error handling
fn handle_app_error(error: AppError) {
    match error {
        AppError::Sensor(e) => {
            esp_println::println!("Sensor error: {:?}", e);
            // Maybe restart sensor
        }
        AppError::Network(e) => {
            esp_println::println!("Network error: {:?}", e);
            // Maybe reconnect
        }
        AppError::Validation(e) => {
            esp_println::println!("Validation error: {:?}", e);
            // Continue processing
        }
        AppError::Configuration(e) => {
            esp_println::println!("Config error: {:?}", e);
            // Use defaults
        }
    }
}
```

### Performance Monitoring

```rust
use esp_hal::timer::TimerGroup;

struct PerformanceMonitor {
    cycle_start: u64,
    max_cycle_time: u64,
    avg_cycle_time: f32,
    cycle_count: u32,
}

impl PerformanceMonitor {
    fn new() -> Self {
        Self {
            cycle_start: 0,
            max_cycle_time: 0,
            avg_cycle_time: 0.0,
            cycle_count: 0,
        }
    }
    
    fn start_cycle(&mut self) {
        self.cycle_start = MonotonicTime::new().now();
    }
    
    fn end_cycle(&mut self) {
        let cycle_time = MonotonicTime::new().now() - self.cycle_start;
        
        if cycle_time > self.max_cycle_time {
            self.max_cycle_time = cycle_time;
        }
        
        self.avg_cycle_time = (self.avg_cycle_time * self.cycle_count as f32 + cycle_time as f32) / (self.cycle_count + 1) as f32;
        self.cycle_count += 1;
        
        if self.cycle_count % 1000 == 0 {
            esp_println::println!("Cycle stats: avg={:.1}ms, max={}ms", 
                self.avg_cycle_time, self.max_cycle_time);
        }
    }
}
```

This ESP32 deployment guide provides comprehensive coverage of EdgeGuard implementation on ESP32 microcontrollers with optimized configurations for resource-constrained environments.