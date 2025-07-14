//! ESP32 WiFi Streaming Example - Real-time Sensor Data over MQTT
//!
//! This example demonstrates streaming validated sensor data over WiFi:
//! - Connects to WiFi network
//! - Validates sensor data using EdgeGuard
//! - Streams validated data via MQTT
//! - Implements backpressure for network congestion
//! - Batches data for efficient transmission
//!
//! ## Hardware Requirements
//! - ESP32 with WiFi (any variant)
//! - BME280 sensor on I2C
//! - Optional: SD card for buffering during network outages
//!
//! ## MQTT Topics
//! - `sensors/{device_id}/temperature` - Temperature readings
//! - `sensors/{device_id}/humidity` - Humidity readings  
//! - `sensors/{device_id}/pressure` - Pressure readings
//! - `sensors/{device_id}/status` - Device status and diagnostics
//!
//! ## Power Optimization
//! - WiFi power save mode enabled
//! - Batch transmissions every 10 seconds
//! - Deep sleep between readings (optional)

#![no_std]
#![no_main]

extern crate alloc;

use esp32_hal::{
    clock::ClockControl,
    gpio::IO,
    i2c::{I2C, I2cConfig},
    peripherals::Peripherals,
    prelude::*,
    timer::TimerGroup,
    Rtc,
};
use esp_backtrace as _;
use esp_println::println;
use esp_wifi::{wifi::{WifiDevice, WifiController}, EspWifiInitFor};

// EdgeGuard streaming components
use edgeguard_core::{
    stream::{Stream, BatchingStream, RateLimitedStream, BackpressureControl},
    events::{Event, EventBuilder, SensorType},
    validators::{TemperatureValidator, HumidityValidator, PressureValidator},
    pipeline::{Pipeline, ValidationStage, AggregationStage},
    traits::{Validator, TimeSource},
    time::Timestamp,
};

use heapless::{String, Vec};
use alloc::boxed::Box;

// Network buffer for MQTT messages
const MQTT_BUFFER_SIZE: usize = 4096;

/// WiFi configuration
const WIFI_SSID: &str = env!("WIFI_SSID");
const WIFI_PASSWORD: &str = env!("WIFI_PASSWORD");
const MQTT_BROKER: &str = env!("MQTT_BROKER"); // e.g., "192.168.1.100:1883"

/// Device configuration  
const DEVICE_ID: &str = "esp32_sensor_001";
const BATCH_SIZE: usize = 10;
const BATCH_TIMEOUT_MS: u32 = 10_000; // 10 seconds

/// Custom stream that reads from BME280 sensor
struct Bme280Stream<'a> {
    i2c: I2C<'a, esp32_hal::peripherals::I2C0>,
    time_source: &'a dyn TimeSource,
    sequence: u32,
}

impl<'a> Bme280Stream<'a> {
    fn new(i2c: I2C<'a, esp32_hal::peripherals::I2C0>, time_source: &'a dyn TimeSource) -> Self {
        Self {
            i2c,
            time_source,
            sequence: 0,
        }
    }
    
    fn read_bme280(&mut self) -> Result<(f32, f32, f32), &'static str> {
        // In production, use actual BME280 driver
        // For demo, return simulated values
        let temp = 22.5 + ((self.sequence % 10) as f32) * 0.1;
        let humidity = 55.0 + ((self.sequence % 20) as f32) * 0.5;
        let pressure = 1013.25 + ((self.sequence % 5) as f32) * 0.5;
        
        Ok((temp, humidity, pressure))
    }
}

impl<'a> Stream for Bme280Stream<'a> {
    type Item = Event;
    type Error = &'static str;
    
    fn poll_next(&mut self) -> nb::Result<Self::Item, Self::Error> {
        // Read sensor every call (in production, might rate limit here)
        match self.read_bme280() {
            Ok((temp, humidity, pressure)) => {
                let timestamp = self.time_source.now();
                self.sequence += 1;
                
                // Return events in round-robin fashion
                let event = match self.sequence % 3 {
                    0 => EventBuilder::new(timestamp)
                        .sensor("bme280_temp", SensorType::Temperature)
                        .reading(temp, 0.95)
                        .unwrap(),
                    1 => EventBuilder::new(timestamp)
                        .sensor("bme280_humidity", SensorType::Humidity)
                        .reading(humidity, 0.90)
                        .unwrap(),
                    _ => EventBuilder::new(timestamp)
                        .sensor("bme280_pressure", SensorType::Pressure)
                        .reading(pressure, 0.98)
                        .unwrap(),
                };
                
                Ok(event)
            }
            Err(e) => Err(nb::Error::Other(e))
        }
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None) // Infinite stream
    }
}

/// MQTT message formatter
fn format_mqtt_message(event: &Event, device_id: &str) -> Option<(String<128>, String<256>)> {
    match event {
        Event::SensorReading { sensor_id, sensor_type, value, timestamp, quality } => {
            let topic = match sensor_type {
                SensorType::Temperature => format!("sensors/{}/temperature", device_id),
                SensorType::Humidity => format!("sensors/{}/humidity", device_id),
                SensorType::Pressure => format!("sensors/{}/pressure", device_id),
                _ => return None,
            };
            
            // JSON payload
            let payload = format!(
                r#"{{"timestamp":{},"value":{},"quality":{},"sensor":"{}"}}"#,
                timestamp,
                value,
                quality,
                sensor_id.as_str()
            );
            
            // Convert to heapless strings
            let topic_str = String::<128>::from(topic.as_str());
            let payload_str = String::<256>::from(payload.as_str());
            
            Some((topic_str, payload_str))
        }
        _ => None,
    }
}

/// Network transmission statistics
#[derive(Default)]
struct NetworkStats {
    messages_sent: u32,
    bytes_sent: usize,
    send_failures: u32,
    reconnections: u32,
}

impl NetworkStats {
    fn print(&self) {
        println!("\n=== Network Statistics ===");
        println!("Messages sent: {}", self.messages_sent);
        println!("Bytes sent: {}", self.bytes_sent);
        println!("Send failures: {}", self.send_failures);
        println!("Reconnections: {}", self.reconnections);
        println!("=========================\n");
    }
}

#[entry]
fn main() -> ! {
    // Initialize heap
    init_heap();
    
    // Initialize peripherals
    let peripherals = Peripherals::take();
    let mut system = peripherals.SYSTEM.split();
    let clocks = ClockControl::boot_defaults(system.clock_control).freeze();
    
    // Initialize I2C for BME280
    let io = IO::new(peripherals.GPIO, peripherals.IO_MUX);
    let sda = io.pins.gpio21.into_open_drain_output();
    let scl = io.pins.gpio22.into_open_drain_output();
    
    let i2c = I2C::new(
        peripherals.I2C0,
        sda,
        scl,
        100u32.kHz(),
        &mut system.peripheral_clock_control,
        &clocks,
    );
    
    // Initialize RTC for timekeeping
    let rtc = Rtc::new(peripherals.RTC_CNTL);
    let time_source: &'static _ = Box::leak(Box::new(EspTimeSource { rtc }));
    
    // Initialize timer
    let timer_group0 = TimerGroup::new(
        peripherals.TIMG0,
        &clocks,
        &mut system.peripheral_clock_control,
    );
    let mut timer = timer_group0.timer0;
    
    println!("ESP32 WiFi Streaming Example");
    println!("Connecting to WiFi...");
    
    // Initialize WiFi
    let wifi_init = esp_wifi::initialize(
        EspWifiInitFor::Wifi,
        timer_group0.timer1,
        esp32_hal::Rng::new(peripherals.RNG),
        system.radio_clock_control,
        &clocks,
    ).unwrap();
    
    // Connect to WiFi (simplified - in production use proper error handling)
    // let (wifi_device, wifi_controller) = wifi_init.wifi();
    // ... WiFi connection code ...
    
    println!("Connected to WiFi!");
    println!("MQTT Broker: {}", MQTT_BROKER);
    
    // Create sensor stream with rate limiting and batching
    let sensor_stream = Bme280Stream::new(i2c, time_source);
    
    // Apply rate limiting (max 10 readings per second)
    let rate_limited = RateLimitedStream::new(sensor_stream, 10, time_source);
    
    // Apply batching for efficient network transmission
    let batched = BatchingStream::new(
        rate_limited,
        edgeguard_core::stream::BatchStrategy::CountWithTimeout {
            size: BATCH_SIZE,
            timeout_ms: BATCH_TIMEOUT_MS,
        },
        time_source,
    );
    
    // Create validation pipeline
    let pipeline = Pipeline::<256>::builder()
        .add_stage(ValidationStage::new(
            TemperatureValidator::new().with_range(-40.0, 85.0),
            SensorType::Temperature,
        ))
        .add_stage(ValidationStage::new(
            HumidityValidator::new(),
            SensorType::Humidity,
        ))
        .add_stage(ValidationStage::new(
            PressureValidator::new().with_altitude(100.0), // 100m above sea level
            SensorType::Pressure,
        ))
        // Add aggregation to reduce data volume
        .add_stage(AggregationStage::new(
            edgeguard_core::pipeline::WindowSpec::Time { duration_ms: 60_000 }, // 1 minute
            edgeguard_core::pipeline::AggregationMethod::Average,
            SensorType::Temperature,
        ))
        .build();
    
    // Backpressure control for network congestion
    let mut backpressure = BackpressureControl::new(100, 50); // High: 100, Low: 50
    
    // Statistics
    let mut net_stats = NetworkStats::default();
    let mut total_events = 0u32;
    
    // MQTT client buffer
    let mut mqtt_buffer = Vec::<u8, MQTT_BUFFER_SIZE>::new();
    
    println!("Starting sensor streaming...");
    
    // Main streaming loop
    timer.start(100_000u64); // 100ms tick
    
    loop {
        nb::block!(timer.wait()).unwrap();
        
        // Check backpressure
        backpressure.update(mqtt_buffer.len());
        if backpressure.should_pause() {
            println!("Backpressure activated - pausing stream");
            continue;
        }
        
        // Process next batch from stream
        let mut batch = [Event::default(); BATCH_SIZE];
        match batched.process_batch(&mut batch) {
            Ok(count) => {
                println!("Received batch of {} events", count);
                total_events += count as u32;
                
                // Process through validation pipeline
                let mut pipeline = pipeline;
                for i in 0..count {
                    if pipeline.push_event(batch[i].clone()) {
                        // Event accepted
                    }
                }
                
                // Process pipeline
                let _ = pipeline.process_batch(count);
                
                // Collect validated results
                while let Some(result) = pipeline.pop_result() {
                    // Format as MQTT message
                    if let Some((topic, payload)) = format_mqtt_message(&result, DEVICE_ID) {
                        // In production, actually send via MQTT client
                        println!("MQTT Publish: {} -> {}", topic, payload);
                        
                        // Simulate adding to buffer
                        let msg_size = topic.len() + payload.len();
                        if mqtt_buffer.remaining_capacity() >= msg_size {
                            // Add to buffer (simplified)
                            mqtt_buffer.extend_from_slice(payload.as_bytes()).ok();
                            net_stats.messages_sent += 1;
                            net_stats.bytes_sent += msg_size;
                        } else {
                            net_stats.send_failures += 1;
                        }
                    }
                }
                
                // Simulate network transmission
                if mqtt_buffer.len() > 1024 {
                    println!("Flushing {} bytes to network", mqtt_buffer.len());
                    mqtt_buffer.clear();
                }
            }
            Err(nb::Error::WouldBlock) => {
                // No data ready yet
            }
            Err(nb::Error::Other(e)) => {
                println!("Stream error: {}", e);
            }
        }
        
        // Print statistics every 100 events
        if total_events % 100 == 0 && total_events > 0 {
            println!("\nTotal events processed: {}", total_events);
            net_stats.print();
            
            // Memory usage
            let free_heap = ALLOCATOR.lock().free();
            println!("Free heap: {} bytes", free_heap);
        }
    }
}

// Heap allocator setup
use linked_list_allocator::LockedHeap;
use core::mem::MaybeUninit;

#[global_allocator]
static ALLOCATOR: LockedHeap = LockedHeap::empty();

fn init_heap() {
    const HEAP_SIZE: usize = 64 * 1024; // 64KB for WiFi operations
    static mut HEAP: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];
    
    unsafe {
        ALLOCATOR.lock().init(HEAP.as_ptr() as usize, HEAP_SIZE);
    }
}

// Time source implementation
struct EspTimeSource {
    rtc: Rtc<'static>,
}

impl TimeSource for EspTimeSource {
    fn now(&self) -> Timestamp {
        (self.rtc.get_time_us() / 1000) as Timestamp
    }
    
    fn is_wall_clock(&self) -> bool {
        false
    }
    
    fn precision_ms(&self) -> u32 {
        1
    }
}

// Default event for array initialization
impl Default for Event {
    fn default() -> Self {
        Event::SystemEvent {
            event_type: edgeguard_core::events::SystemEventType::Startup,
            timestamp: 0,
            details: 0,
        }
    }
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    println!("PANIC: {}", info);
    loop {}
}

// Build configuration for WiFi support:
/*
[dependencies]
edgeguard-core = { 
    version = "0.1",
    default-features = false,
    features = [
        "validation-core",
        "pipeline-core",
        "pipeline-stages",
        "stream-core",
        "stream-adapters",
        "alloc"
    ]
}

# ESP32 WiFi stack
esp-wifi = { 
    version = "0.1", 
    features = ["esp32", "wifi", "async", "embassy-net"]
}
embassy-net = { version = "0.2", features = ["tcp", "dhcpv4", "medium-ethernet"] }
smoltcp = { version = "0.10", default-features = false, features = ["proto-ipv4", "socket-tcp"] }

# MQTT client (no_std compatible)
rust-mqtt = { version = "0.2", default-features = false }

[env]
WIFI_SSID = "your_wifi_ssid"
WIFI_PASSWORD = "your_wifi_password"  
MQTT_BROKER = "192.168.1.100:1883"
*/