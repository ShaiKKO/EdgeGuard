//! ESP32 Advanced Example - Multi-Sensor Fusion with Event Pipeline
//!
//! This example demonstrates advanced EdgeGuard features on ESP32:
//! - Event-driven pipeline for sensor processing
//! - Kalman filter fusion of multiple temperature sensors
//! - Cross-validation between temperature and humidity
//! - WiFi transmission of validated data (optional)
//!
//! ## Hardware Setup
//! - ESP32-WROOM-32 (4MB flash, 520KB RAM)
//! - Multiple temperature sensors:
//!   - DHT22 on GPIO 4
//!   - DS18B20 on GPIO 5 (OneWire)
//!   - Internal ESP32 temperature sensor
//! - Optional: OLED display on I2C for status
//!
//! ## Memory Usage
//! - Flash: ~250KB (with pipeline + fusion features)
//! - RAM: ~40KB during operation
//!
//! ## Build for Production
//! ```bash
//! cargo build --example esp32_advanced --release \
//!   --target xtensa-esp32-none-elf \
//!   --no-default-features \
//!   --features "validation-core,pipeline-core,fusion-core,fusion-models"
//! ```

#![no_std]
#![no_main]

extern crate alloc;
use core::mem::MaybeUninit;

// ESP32 HAL
use esp32_hal::{
    clock::ClockControl,
    cpu_control::{CpuControl, Stack},
    gpio::IO,
    peripherals::Peripherals,
    prelude::*,
    timer::TimerGroup,
    Rtc,
};
use esp_backtrace as _;
use esp_println::println;

// Heap allocator for ESP32 (required for alloc)
use linked_list_allocator::LockedHeap;

#[global_allocator]
static ALLOCATOR: LockedHeap = LockedHeap::empty();

// EdgeGuard imports
use edgeguard_core::{
    // Events and pipeline
    events::{Event, EventBuilder, SensorType, InlineString},
    pipeline::{Pipeline, ValidationStage, FusionStage, CrossValidationStage},
    
    // Validators
    validators::{TemperatureValidator, HumidityValidator},
    traits::{Validator, TimeSource},
    
    // Fusion
    fusion::{
        KalmanFilter, KalmanConfig, StateTransition,
        models::TemperatureModel,
    },
    
    // Time
    time::Timestamp,
};

use heapless::Vec;
use alloc::boxed::Box;

/// Initialize the heap allocator
fn init_heap() {
    const HEAP_SIZE: usize = 32 * 1024; // 32KB heap
    static mut HEAP: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];
    
    unsafe {
        ALLOCATOR.lock().init(HEAP.as_ptr() as usize, HEAP_SIZE);
    }
}

/// ESP32 RTC-based time source
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

/// Multi-core sensor reading task
/// Runs on APP CPU (Core 1) to avoid blocking main processing
static mut SENSOR_STACK: Stack<4096> = Stack::new();

fn sensor_reading_task() -> ! {
    println!("Sensor task running on Core 1");
    
    loop {
        // In production, this would:
        // 1. Read from actual sensors via I2C/OneWire
        // 2. Put readings in a lock-free queue
        // 3. Handle sensor errors and reconnection
        
        // For demo, just sleep
        xtensa_lx_rt::delay(1_000_000); // 1 second
    }
}

/// Simulated multi-sensor readings
struct MultiSensorData {
    dht22_temp: f32,
    ds18b20_temp: f32,
    internal_temp: f32,
    humidity: f32,
}

fn read_all_sensors() -> MultiSensorData {
    // Simulate slight variations between sensors
    static mut COUNTER: u32 = 0;
    
    unsafe {
        COUNTER = COUNTER.wrapping_add(1);
        
        let base_temp = 23.0 + ((COUNTER % 20) as f32) * 0.1;
        
        MultiSensorData {
            dht22_temp: base_temp + 0.2,      // DHT22 reads slightly high
            ds18b20_temp: base_temp - 0.1,    // DS18B20 more accurate
            internal_temp: base_temp + 1.5,   // Internal sensor affected by CPU heat
            humidity: 55.0 + ((COUNTER % 30) as f32) * 0.5,
        }
    }
}

#[entry]
fn main() -> ! {
    // Initialize heap first
    init_heap();
    
    // Initialize peripherals
    let peripherals = Peripherals::take();
    let mut system = peripherals.SYSTEM.split();
    let clocks = ClockControl::boot_defaults(system.clock_control).freeze();
    
    // Initialize RTC
    let rtc = Rtc::new(peripherals.RTC_CNTL);
    let time_source = Box::new(EspTimeSource { rtc });
    
    // Initialize CPU control for multi-core
    let mut cpu_control = CpuControl::new(peripherals.CPU_CTRL);
    
    // Initialize GPIO
    let io = IO::new(peripherals.GPIO, peripherals.IO_MUX);
    let mut status_led = io.pins.gpio2.into_push_pull_output();
    
    // Initialize timer
    let timer_group0 = TimerGroup::new(
        peripherals.TIMG0,
        &clocks,
        &mut system.peripheral_clock_control,
    );
    let mut timer = timer_group0.timer0;
    
    println!("ESP32 Advanced EdgeGuard Example");
    println!("Features: Pipeline, Fusion, Cross-validation");
    
    // Start sensor reading task on Core 1
    let _guard = unsafe {
        cpu_control
            .start_app_core(&mut SENSOR_STACK, sensor_reading_task)
            .unwrap()
    };
    
    // Build the processing pipeline
    let pipeline = Pipeline::<256>::builder()
        // Stage 1: Validate individual sensors
        .add_stage(ValidationStage::new(
            TemperatureValidator::new()
                .with_range(-10.0, 50.0)
                .with_rate_limit(5.0),
            SensorType::Temperature,
        ))
        
        // Stage 2: Fuse temperature sensors using Kalman filter
        .add_stage({
            let mut fusion_stage = FusionStage::new();
            
            // Configure Kalman filter for temperature fusion
            let kalman_config = KalmanConfig::<1, 3> {
                initial_state: [23.0],  // Initial temperature estimate
                initial_covariance: [[1.0]],
                process_noise: [[0.01]], // Temperature changes slowly
                measurement_noise: [
                    [0.5], // DHT22 noise
                    [0.1], // DS18B20 noise (more accurate)  
                    [2.0], // Internal sensor noise (affected by CPU)
                ],
                transition: StateTransition {
                    transition_matrix: [[1.0]], // Temperature doesn't change in prediction
                    control_matrix: None,
                },
                measurement_matrix: [[1.0], [1.0], [1.0]], // Direct measurements
                control_matrix: None,
                convergence_threshold: 0.01,
            };
            
            let kalman = Box::new(KalmanFilter::new(kalman_config));
            
            fusion_stage.add_group(
                edgeguard_core::fusion::SensorGroup::new("temperature", SensorType::Temperature)
                    .add_sensor("dht22_temp")
                    .add_sensor("ds18b20_temp")
                    .add_sensor("internal_temp")
                    .with_algorithm(edgeguard_core::fusion::FusionAlgorithmType::Kalman(kalman))
                    .with_min_sensors(2) // Need at least 2 sensors for fusion
            )
        })
        
        // Stage 3: Cross-validation
        .add_stage({
            let mut cross_val = CrossValidationStage::new();
            cross_val.add_pair(
                SensorType::Temperature,
                SensorType::Humidity,
                edgeguard_core::events::CrossValidationType::DewPoint,
            );
            cross_val
        })
        
        .build();
    
    // Wrap pipeline in a mutex for safe access
    let pipeline = alloc::sync::Arc::new(spin::Mutex::new(pipeline));
    
    // Statistics
    let mut stats = ProcessingStats::default();
    
    // Start main processing loop
    timer.start(1_000_000u64); // 1 second interval
    
    println!("Starting multi-sensor fusion loop...");
    
    loop {
        // Wait for timer
        nb::block!(timer.wait()).unwrap();
        status_led.toggle().unwrap();
        
        // Read all sensors
        let sensor_data = read_all_sensors();
        let timestamp = time_source.now();
        
        // Create events for each sensor
        let events = [
            EventBuilder::new(timestamp)
                .sensor("dht22_temp", SensorType::Temperature)
                .reading(sensor_data.dht22_temp, 0.8)
                .unwrap(),
                
            EventBuilder::new(timestamp)
                .sensor("ds18b20_temp", SensorType::Temperature)
                .reading(sensor_data.ds18b20_temp, 0.95)
                .unwrap(),
                
            EventBuilder::new(timestamp)
                .sensor("internal_temp", SensorType::Temperature)  
                .reading(sensor_data.internal_temp, 0.6)
                .unwrap(),
                
            EventBuilder::new(timestamp)
                .sensor("dht22_humidity", SensorType::Humidity)
                .reading(sensor_data.humidity, 0.85)
                .unwrap(),
        ];
        
        // Process through pipeline
        {
            let mut pipeline = pipeline.lock();
            
            for event in &events {
                if pipeline.push_event(event.clone()) {
                    stats.events_pushed += 1;
                } else {
                    stats.events_dropped += 1;
                }
            }
            
            // Process the batch
            match pipeline.process_batch(10) {
                Ok(processed) => stats.batches_processed += 1,
                Err(e) => {
                    println!("Pipeline error: {:?}", e);
                    stats.errors += 1;
                }
            }
            
            // Collect results
            while let Some(result) = pipeline.pop_result() {
                match result {
                    Event::ValidationResult { sensor_id, status, .. } => {
                        println!(
                            "[{}ms] {} validation: {:?}",
                            timestamp,
                            sensor_id.as_str(),
                            status
                        );
                    }
                    Event::SensorReading { sensor_id, value, quality, .. } => {
                        // Check if this is a fused reading
                        if sensor_id.as_str() == "fused" {
                            println!(
                                "[{}ms] FUSED Temperature: {:.2}°C (confidence: {:.2})",
                                timestamp,
                                value,
                                quality
                            );
                            stats.fused_readings += 1;
                            
                            // In production, send this via WiFi/MQTT
                        }
                    }
                    Event::CrossValidationResult { validation_type, status, .. } => {
                        println!(
                            "[{}ms] Cross-validation {:?}: {:?}",
                            timestamp,
                            validation_type,
                            status
                        );
                    }
                    _ => {}
                }
            }
        }
        
        // Print statistics every 10 iterations
        if stats.events_pushed % 40 == 0 && stats.events_pushed > 0 {
            stats.print();
            
            // Check heap usage
            let free_heap = ALLOCATOR.lock().free();
            let used_heap = ALLOCATOR.lock().used();
            println!("Heap: {} used, {} free", used_heap, free_heap);
        }
    }
}

#[derive(Default)]
struct ProcessingStats {
    events_pushed: u32,
    events_dropped: u32,
    batches_processed: u32,
    fused_readings: u32,
    errors: u32,
}

impl ProcessingStats {
    fn print(&self) {
        println!("\n=== Pipeline Statistics ===");
        println!("Events pushed: {}", self.events_pushed);
        println!("Events dropped: {}", self.events_dropped);
        println!("Batches processed: {}", self.batches_processed);
        println!("Fused readings: {}", self.fused_readings);
        println!("Errors: {}", self.errors);
        
        if self.events_pushed > 0 {
            let drop_rate = (self.events_dropped as f32 / self.events_pushed as f32) * 100.0;
            println!("Drop rate: {:.1}%", drop_rate);
        }
        println!("==========================\n");
    }
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    println!("PANIC: {}", info);
    
    // Print backtrace if available
    esp_backtrace::print_backtrace();
    
    // Reset after 5 seconds
    xtensa_lx_rt::delay(5_000_000);
    esp32_hal::reset::software_reset();
    
    loop {}
}

// Production Cargo.toml additions:
/*
[dependencies]
# Core EdgeGuard with selected features
edgeguard-core = { 
    version = "0.1", 
    default-features = false, 
    features = [
        "validation-core",
        "pipeline-core", 
        "fusion-core",
        "fusion-models",
        "alloc"  # Required for Box, Arc
    ]
}

# ESP32 specific
esp32-hal = "0.13"
esp-backtrace = { version = "0.7", features = ["esp32", "panic-handler", "print-uart"] }
esp-println = { version = "0.5", features = ["esp32"] }
xtensa-lx-rt = "0.15"
critical-section = "1.1"
nb = "1.0"

# Memory management
linked_list_allocator = { version = "0.10", default-features = false }
heapless = "0.8"
spin = { version = "0.9", default-features = false, features = ["mutex", "spin_mutex"] }

# Optional: WiFi support
esp-wifi = { version = "0.1", optional = true, features = ["esp32", "wifi"] }
smoltcp = { version = "0.10", optional = true, default-features = false }

[features]
default = []
wifi = ["esp-wifi", "smoltcp"]

[profile.release]
opt-level = "z"
lto = "fat"
codegen-units = 1
panic = "abort"
strip = true

# Reduce binary size further
[profile.release.package."*"]
opt-level = "z"
*/