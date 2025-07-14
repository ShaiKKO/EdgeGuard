//! XIAO nRF52840 Sense Quick Start Example
//!
//! A minimal example to get started with EdgeGuard on XIAO nRF52840 Sense.
//! This demonstrates basic temperature and humidity validation using the
//! built-in sensors, perfect for learning EdgeGuard's core concepts.
//!
//! ## What You'll Learn
//! - Basic sensor validation
//! - Validation contexts and history
//! - Rate-of-change detection
//! - Simple LED feedback
//!
//! ## Hardware Required
//! - Seeed Studio XIAO nRF52840 Sense
//! - USB-C cable
//! - (Optional) BME688 sensor on Grove connector
//!
//! ## Quick Start
//! ```bash
//! # Install tools
//! cargo install probe-rs --features cli
//! 
//! # Build and flash
//! cargo run --example xiao_quickstart --release
//! ```

#![no_std]
#![no_main]

use panic_probe as _;
use defmt_rtt as _;

use cortex_m_rt::entry;
use embassy_nrf::{
    self as _,
    gpio::{Level, Output, OutputDrive},
};
use embassy_time::{Duration, Timer};

// Import only what we need from EdgeGuard
use edgeguard_core::{
    validators::{TemperatureValidator, HumidityValidator},
    traits::{Validator, ValidationContext},
    time::Timestamp,
};

/// A simple time counter (in production, use RTC)
static mut TIME_MS: u64 = 0;

/// Get current timestamp
fn get_timestamp() -> Timestamp {
    unsafe { TIME_MS }
}

/// Update timestamp (call every millisecond)
fn tick_time() {
    unsafe { TIME_MS += 1 };
}

#[entry]
fn main() -> ! {
    // Print startup message
    defmt::info!("XIAO nRF52840 Sense - EdgeGuard Quick Start");
    
    // Initialize the HAL
    let p = embassy_nrf::init(Default::default());
    
    // Set up the onboard LED (red)
    let mut led = Output::new(p.P0_26, Level::High, OutputDrive::Standard);
    
    // Create validators with sensible defaults
    let temp_validator = TemperatureValidator::new()
        .with_range(10.0, 40.0)    // Indoor temperature range
        .with_rate_limit(2.0);      // Max 2°C/second change
        
    let humidity_validator = HumidityValidator::new()
        .with_range(20.0, 80.0)     // Comfortable humidity range
        .with_rate_limit(5.0);      // Max 5%/second change
    
    // Create validation contexts to track history
    let mut temp_context = ValidationContext::with_capacity(5);
    let mut humidity_context = ValidationContext::with_capacity(5);
    
    // Track validation statistics
    let mut readings = 0u32;
    let mut temp_valid = 0u32;
    let mut humidity_valid = 0u32;
    
    defmt::info!("Starting sensor monitoring...");
    defmt::info!("LED: ON = reading valid, BLINK = validation failed");
    
    loop {
        // Simulate sensor readings (replace with actual sensor code)
        let temperature = read_temperature();
        let humidity = read_humidity();
        let timestamp = get_timestamp();
        
        readings += 1;
        
        // Update contexts with current timestamp
        temp_context.timestamp = timestamp;
        humidity_context.timestamp = timestamp;
        
        // Validate temperature
        match temp_validator.validate(&temperature, &temp_context) {
            Ok(()) => {
                // Valid reading - add to history
                temp_context.add_reading(temperature, timestamp);
                temp_valid += 1;
                
                defmt::info!(
                    "✓ Temperature: {:.1}°C",
                    temperature
                );
                
                // LED on = valid
                led.set_low();
            }
            Err(e) => {
                defmt::warn!(
                    "✗ Temperature: {:.1}°C - {:?}",
                    temperature,
                    defmt::Debug2Format(&e)
                );
                
                // Blink LED = invalid
                for _ in 0..3 {
                    led.set_low();
                    delay_ms(50);
                    led.set_high();
                    delay_ms(50);
                }
            }
        }
        
        // Validate humidity
        match humidity_validator.validate(&humidity, &humidity_context) {
            Ok(()) => {
                humidity_context.add_reading(humidity, timestamp);
                humidity_valid += 1;
                
                defmt::info!(
                    "✓ Humidity: {:.1}%",
                    humidity
                );
            }
            Err(e) => {
                defmt::warn!(
                    "✗ Humidity: {:.1}% - {:?}",
                    humidity,
                    defmt::Debug2Format(&e)
                );
            }
        }
        
        // Print statistics every 10 readings
        if readings % 10 == 0 {
            let temp_rate = (temp_valid as f32 / readings as f32) * 100.0;
            let humidity_rate = (humidity_valid as f32 / readings as f32) * 100.0;
            
            defmt::info!("");
            defmt::info!("=== Statistics ===");
            defmt::info!("Total readings: {}", readings);
            defmt::info!("Temperature valid: {}%", temp_rate as u32);
            defmt::info!("Humidity valid: {}%", humidity_rate as u32);
            defmt::info!("==================");
            defmt::info!("");
        }
        
        // Wait before next reading
        delay_ms(1000);
    }
}

/// Read temperature from sensor
fn read_temperature() -> f32 {
    // Simulate temperature readings with some variation
    static mut COUNTER: u32 = 0;
    unsafe {
        COUNTER = COUNTER.wrapping_add(1);
        
        // Base temperature with slow variation
        let base = 22.0;
        let variation = ((COUNTER as f32 * 0.1).sin() * 2.0);
        
        // Add occasional spikes to test validation
        let spike = if COUNTER % 20 == 0 { 10.0 } else { 0.0 };
        
        base + variation + spike
    }
}

/// Read humidity from sensor  
fn read_humidity() -> f32 {
    // Simulate humidity readings
    static mut COUNTER: u32 = 0;
    unsafe {
        COUNTER = COUNTER.wrapping_add(1);
        
        // Base humidity with variation
        let base = 50.0;
        let variation = ((COUNTER as f32 * 0.05).cos() * 10.0);
        
        // Add occasional invalid readings
        let invalid = if COUNTER % 15 == 0 { 50.0 } else { 0.0 };
        
        base + variation + invalid
    }
}

/// Simple blocking delay
fn delay_ms(ms: u32) {
    cortex_m::asm::delay(ms * 64_000); // Approximate for 64MHz clock
    
    // Update time counter
    for _ in 0..ms {
        tick_time();
    }
}

/// Panic handler
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    defmt::error!("PANIC: {}", defmt::Debug2Format(info));
    
    // Reset after a delay
    cortex_m::asm::delay(64_000_000); // 1 second
    cortex_m::peripheral::SCB::sys_reset();
}

// Minimal Cargo.toml for quick start:
/*
[package]
name = "xiao-edgeguard-quickstart"
version = "0.1.0"
edition = "2021"

[dependencies]
# EdgeGuard - minimal features
edgeguard-core = { 
    path = "../../../",
    default-features = false, 
    features = ["validation-core"] 
}

# nRF52840 support
embassy-nrf = { version = "0.1", features = ["defmt", "nrf52840", "time-driver-rtc1"] }
embassy-time = { version = "0.3", features = ["defmt"] }
cortex-m = { version = "0.7", features = ["critical-section-single-core"] }
cortex-m-rt = "0.7"

# Debugging
defmt = "0.3"
defmt-rtt = "0.4"
panic-probe = { version = "0.3", features = ["print-defmt"] }

[[bin]]
name = "xiao_quickstart"
test = false
bench = false

[profile.release]
debug = true
lto = true
opt-level = "z"
*/

// =============================================================================
// Next Steps After Quick Start:
// 
// 1. Connect Real Sensors:
//    - Add I2C/SPI sensor drivers
//    - Replace simulated readings
//
// 2. Add More Validators:
//    - PressureValidator for barometric pressure
//    - Custom validators for your sensors
//
// 3. Use Cross-Validation:
//    - Validate temperature vs humidity (dew point)
//    - Detect sensor failures
//
// 4. Add Data Pipeline:
//    - Use EdgeGuard's event pipeline
//    - Add filtering and aggregation
//
// 5. Enable Wireless:
//    - BLE for data transmission
//    - Power management
//
// See other examples for these advanced features!
// =============================================================================