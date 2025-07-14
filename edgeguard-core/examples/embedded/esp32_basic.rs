//! ESP32 Basic Example - Temperature & Humidity Monitoring
//!
//! This example demonstrates EdgeGuard running on an ESP32 microcontroller
//! with minimal features for memory-constrained environments.
//!
//! ## Hardware Setup
//! - ESP32 DevKit (any variant with at least 4MB flash)
//! - DHT22/AM2302 temperature & humidity sensor on GPIO 4
//! - Optional: BME280 for pressure readings on I2C (SDA: GPIO 21, SCL: GPIO 22)
//!
//! ## Memory Usage
//! - Flash: ~150KB (with minimal features)
//! - RAM: ~20KB during operation
//!
//! ## Build Instructions
//! ```bash
//! # Install ESP toolchain
//! cargo install espup
//! espup install
//! source ~/export-esp.sh
//!
//! # Build for ESP32
//! cargo build --example esp32_basic --target xtensa-esp32-none-elf --no-default-features --features "validation-core"
//! 
//! # Flash to device
//! cargo espflash --example esp32_basic --monitor
//! ```

#![no_std]
#![no_main]

// ESP32 HAL and panic handler
use esp32_hal::{
    clock::ClockControl,
    gpio::IO,
    peripherals::Peripherals,
    prelude::*,
    timer::TimerGroup,
    Rtc,
};
use esp_backtrace as _;
use esp_println::println;

// EdgeGuard imports (minimal features)
use edgeguard_core::{
    validators::{TemperatureValidator, HumidityValidator},
    traits::{Validator, ValidationContext, TimeSource},
    time::Timestamp,
    buffer::CircularBuffer,
};

// Fixed-point math for temperature calculations (no floating point on some ESP32 variants)
use fixed::types::I16F16;

/// ESP32 RTC-based time source
struct EspTimeSource {
    rtc: Rtc<'static>,
    start_ms: u64,
}

impl EspTimeSource {
    fn new(rtc: Rtc<'static>) -> Self {
        Self { rtc, start_ms: 0 }
    }
}

impl TimeSource for EspTimeSource {
    fn now(&self) -> Timestamp {
        // Get microseconds from RTC and convert to milliseconds
        let micros = self.rtc.get_time_us();
        self.start_ms + (micros / 1000)
    }
    
    fn is_wall_clock(&self) -> bool {
        false // RTC is monotonic
    }
    
    fn precision_ms(&self) -> u32 {
        1 // Millisecond precision
    }
}

/// Simulated sensor readings (replace with actual sensor driver)
struct SensorReadings {
    temperature: f32,
    humidity: f32,
}

fn read_sensors() -> SensorReadings {
    // In real implementation, this would read from DHT22/BME280
    // For demo, we'll use simulated values
    static mut COUNTER: u32 = 0;
    
    unsafe {
        COUNTER += 1;
        
        // Simulate temperature varying between 20-25°C
        let temp_variation = ((COUNTER % 100) as f32) * 0.05;
        let temperature = 22.5 + temp_variation;
        
        // Simulate humidity varying between 40-60%
        let humidity_variation = ((COUNTER % 50) as f32) * 0.4;
        let humidity = 50.0 + humidity_variation;
        
        SensorReadings {
            temperature,
            humidity,
        }
    }
}

#[entry]
fn main() -> ! {
    // Initialize ESP32 peripherals
    let peripherals = Peripherals::take();
    let mut system = peripherals.SYSTEM.split();
    let clocks = ClockControl::boot_defaults(system.clock_control).freeze();
    
    // Initialize RTC for timekeeping
    let rtc = Rtc::new(peripherals.RTC_CNTL);
    let time_source = EspTimeSource::new(rtc);
    
    // Initialize GPIO for LED status indicator
    let io = IO::new(peripherals.GPIO, peripherals.IO_MUX);
    let mut led = io.pins.gpio2.into_push_pull_output();
    
    // Initialize timer for periodic readings
    let timer_group0 = TimerGroup::new(
        peripherals.TIMG0,
        &clocks,
        &mut system.peripheral_clock_control,
    );
    let mut wdt = timer_group0.wdt;
    let mut timer = timer_group0.timer0;
    
    // Disable watchdog
    wdt.disable();
    
    println!("ESP32 EdgeGuard Example Starting...");
    println!("Flash size: 4MB, RAM: 520KB");
    println!("EdgeGuard features: validation-core only");
    
    // Create validators with ESP32-appropriate settings
    let temp_validator = TemperatureValidator::new()
        .with_range(-10.0, 50.0)  // Indoor range
        .with_rate_limit(5.0);     // Max 5°C/second change
        
    let humidity_validator = HumidityValidator::new()
        .with_range(10.0, 95.0)    // Practical indoor range
        .with_rate_limit(10.0);    // Max 10%/second change
    
    // Pre-allocate validation context with small history buffer
    let mut temp_context = ValidationContext::with_capacity(10);
    let mut humidity_context = ValidationContext::with_capacity(10);
    
    // Statistics
    let mut total_readings = 0u32;
    let mut valid_temp_readings = 0u32;
    let mut valid_humidity_readings = 0u32;
    
    // Start periodic timer (1 second interval)
    timer.start(1_000_000u64);
    
    println!("Starting sensor monitoring loop...");
    
    loop {
        // Wait for timer
        nb::block!(timer.wait()).unwrap();
        
        // Toggle LED to show activity
        led.toggle().unwrap();
        
        // Read sensors
        let readings = read_sensors();
        let timestamp = time_source.now();
        total_readings += 1;
        
        // Update contexts
        temp_context.timestamp = timestamp;
        humidity_context.timestamp = timestamp;
        
        // Validate temperature
        match temp_validator.validate(&readings.temperature, &temp_context) {
            Ok(()) => {
                valid_temp_readings += 1;
                // Add to history for rate-of-change validation
                temp_context.add_reading(readings.temperature, timestamp);
                
                println!(
                    "[{}ms] Temperature: {:.1}°C ✓",
                    timestamp,
                    readings.temperature
                );
            }
            Err(e) => {
                println!(
                    "[{}ms] Temperature: {:.1}°C ✗ - {:?}",
                    timestamp,
                    readings.temperature,
                    e
                );
            }
        }
        
        // Validate humidity
        match humidity_validator.validate(&readings.humidity, &humidity_context) {
            Ok(()) => {
                valid_humidity_readings += 1;
                // Add to history
                humidity_context.add_reading(readings.humidity, timestamp);
                
                println!(
                    "[{}ms] Humidity: {:.1}% ✓",
                    timestamp,
                    readings.humidity
                );
            }
            Err(e) => {
                println!(
                    "[{}ms] Humidity: {:.1}% ✗ - {:?}",
                    timestamp,
                    readings.humidity,
                    e
                );
            }
        }
        
        // Print statistics every 10 readings
        if total_readings % 10 == 0 {
            let temp_success_rate = (valid_temp_readings as f32 / total_readings as f32) * 100.0;
            let humidity_success_rate = (valid_humidity_readings as f32 / total_readings as f32) * 100.0;
            
            println!("\n--- Statistics ---");
            println!("Total readings: {}", total_readings);
            println!("Temperature validation rate: {:.1}%", temp_success_rate);
            println!("Humidity validation rate: {:.1}%", humidity_success_rate);
            println!("Free heap: ~400KB"); // ESP32 typically has ~400KB free after boot
            println!("-----------------\n");
        }
        
        // In production, validated data would be:
        // - Stored in flash using ESP32's NVS
        // - Sent via WiFi/Bluetooth
        // - Aggregated for batch transmission
    }
}

/// Custom panic handler for ESP32
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    println!("PANIC: {}", info);
    
    // In production, you might want to:
    // - Write panic info to flash
    // - Trigger watchdog reset
    // - Send alert via network
    
    loop {}
}

// Example Cargo.toml configuration for ESP32:
/*
[dependencies]
edgeguard-core = { version = "0.1", default-features = false, features = ["validation-core"] }
esp32-hal = "0.13"
esp-backtrace = { version = "0.7", features = ["esp32", "panic-handler", "print-uart"] }
esp-println = { version = "0.5", features = ["esp32"] }
nb = "1.0"
fixed = { version = "1.23", default-features = false }

[profile.release]
opt-level = "z"     # Optimize for size
lto = true          # Link-time optimization
codegen-units = 1   # Better optimization
strip = true        # Strip symbols

[profile.dev]
opt-level = "s"     # Some optimization even in dev
*/