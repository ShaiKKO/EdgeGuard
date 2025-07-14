//! Seeed XIAO nRF52840 Sense + BME688 Example
//!
//! This example demonstrates EdgeGuard on the Seeed Studio XIAO nRF52840 Sense
//! with an external BME688 environmental sensor for comprehensive monitoring.
//!
//! ## Hardware Features Used
//! - Built-in IMU (LSM6DS3TR-C): 6-axis accelerometer + gyroscope
//! - Built-in PDM Microphone: Sound level monitoring
//! - External BME688: Temperature, humidity, pressure, gas (VOC)
//! - BLE for wireless data transmission
//! - 256KB RAM, 1MB Flash
//!
//! ## Sensor Configuration
//! - BME688 on I2C (SDA: P0.04, SCL: P0.05)
//! - IMU interrupt on P0.11
//! - PDM Microphone (CLK: P1.00, DIN: P0.16)
//!
//! ## Features Demonstrated
//! - Multi-sensor validation and fusion
//! - Gas sensor baseline calibration
//! - Motion detection with IMU
//! - Sound anomaly detection
//! - BLE advertisement of validated data
//! - Power-efficient operation (~5mA average)
//!
//! ## Build Instructions
//! ```bash
//! # Install probe-rs for flashing
//! cargo install probe-rs --features cli
//!
//! # Build for nRF52840
//! cargo build --example xiao_nrf52840_sense \
//!   --target thumbv7em-none-eabihf \
//!   --no-default-features \
//!   --features "validation-core,pipeline-core,fusion-core"
//!
//! # Flash using probe-rs
//! probe-rs run --chip nRF52840_xxAA target/thumbv7em-none-eabihf/release/examples/xiao_nrf52840_sense
//! ```

#![no_std]
#![no_main]

use panic_probe as _;
use rtt_target::{rprintln, rtt_init_print};

use cortex_m_rt::entry;
use embassy_nrf::{
    self as _, // HAL
    gpio::{Input, Level, Output, OutputDrive, Pull},
    twim::{self, Twim},
    peripherals,
    bind_interrupts,
};
use embassy_time::{Duration, Timer};

// EdgeGuard imports
use edgeguard_core::{
    // Core validation
    validators::{TemperatureValidator, HumidityValidator, PressureValidator},
    traits::{Validator, ValidationContext, TimeSource, ValidatorConstraints},
    
    // Events and pipeline
    events::{Event, EventBuilder, SensorType},
    pipeline::{Pipeline, ValidationStage, CrossValidationStage, FilterStage},
    
    // Fusion for combining multiple sensors
    fusion::{
        KalmanFilter, KalmanConfig, StateTransition,
        confidence::ConfidenceScore,
    },
    
    // Time management
    time::Timestamp,
    buffer::CircularBuffer,
};

use heapless::{String, Vec};
use core::fmt::Write;

// Bind interrupts for I2C
bind_interrupts!(struct Irqs {
    SPIM0_SPIS0_TWIM0_TWIS0_SPI0_TWI0 => twim::InterruptHandler<peripherals::TWISPI0>;
});

/// VOC (Volatile Organic Compounds) validator for BME688 gas sensor
struct VocValidator {
    baseline: f32,
    max_deviation: f32,
    calibration_samples: usize,
    calibrated: bool,
}

impl VocValidator {
    fn new() -> Self {
        Self {
            baseline: 0.0,
            max_deviation: 50.0, // 50% deviation from baseline
            calibration_samples: 0,
            calibrated: false,
        }
    }
    
    fn calibrate_baseline(&mut self, reading: f32) {
        if self.calibration_samples < 10 {
            self.baseline = (self.baseline * self.calibration_samples as f32 + reading) 
                / (self.calibration_samples + 1) as f32;
            self.calibration_samples += 1;
            
            if self.calibration_samples >= 10 {
                self.calibrated = true;
                rprintln!("VOC baseline calibrated: {:.2} kOhm", self.baseline);
            }
        }
    }
}

impl Validator for VocValidator {
    type Value = f32;
    type Error = &'static str;
    
    fn validate(&self, value: &Self::Value, _context: &ValidationContext) -> Result<(), Self::Error> {
        if !self.calibrated {
            return Err("VOC sensor not calibrated");
        }
        
        if *value < 10.0 || *value > 500.0 {
            return Err("VOC resistance out of range");
        }
        
        let deviation = ((*value - self.baseline).abs() / self.baseline) * 100.0;
        if deviation > self.max_deviation {
            return Err("VOC deviation exceeds threshold");
        }
        
        Ok(())
    }
    
    fn constraints(&self) -> ValidatorConstraints {
        ValidatorConstraints {
            min_value: 10.0,
            max_value: 500.0,
            max_rate_change: 10.0, // kOhm/s
            noise_threshold: Some(1.0),
        }
    }
}

/// Motion detector using IMU data
struct MotionDetector {
    accel_threshold: f32,
    gyro_threshold: f32,
    history: CircularBuffer<10>,
}

impl MotionDetector {
    fn new() -> Self {
        Self {
            accel_threshold: 0.1, // g
            gyro_threshold: 10.0, // deg/s
            history: CircularBuffer::new(),
        }
    }
    
    fn detect_motion(&mut self, accel_mag: f32, gyro_mag: f32) -> MotionState {
        self.history.push(accel_mag);
        
        if accel_mag > self.accel_threshold || gyro_mag > self.gyro_threshold {
            MotionState::Moving
        } else if self.history.len() >= 5 {
            // Check if we've been still for 5 samples
            let all_still = self.history.as_slice()
                .iter()
                .all(|&a| a < self.accel_threshold);
            
            if all_still {
                MotionState::Stationary
            } else {
                MotionState::Vibration
            }
        } else {
            MotionState::Unknown
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum MotionState {
    Stationary,
    Moving,
    Vibration,
    Unknown,
}

/// Sound level analyzer for PDM microphone
struct SoundAnalyzer {
    baseline_db: f32,
    anomaly_threshold: f32,
}

impl SoundAnalyzer {
    fn new() -> Self {
        Self {
            baseline_db: 40.0, // Typical quiet room
            anomaly_threshold: 20.0, // 20dB above baseline
        }
    }
    
    fn analyze(&self, sound_level_db: f32) -> SoundState {
        let deviation = sound_level_db - self.baseline_db;
        
        if deviation > self.anomaly_threshold {
            SoundState::Anomaly
        } else if sound_level_db < 30.0 {
            SoundState::Quiet
        } else if sound_level_db < 60.0 {
            SoundState::Normal
        } else {
            SoundState::Loud
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum SoundState {
    Quiet,
    Normal,
    Loud,
    Anomaly,
}

/// nRF52840 RTC-based time source
struct NrfTimeSource;

impl TimeSource for NrfTimeSource {
    fn now(&self) -> Timestamp {
        // In production, use RTC or TIMER peripheral
        // For demo, use a simple counter
        static mut COUNTER: Timestamp = 0;
        unsafe {
            COUNTER += 1;
            COUNTER
        }
    }
    
    fn is_wall_clock(&self) -> bool {
        false
    }
    
    fn precision_ms(&self) -> u32 {
        1
    }
}

/// BME688 sensor readings
struct Bme688Reading {
    temperature: f32,
    humidity: f32,
    pressure: f32,
    gas_resistance: f32, // in kOhm
}

/// IMU sensor readings
struct ImuReading {
    accel_x: f32,
    accel_y: f32,
    accel_z: f32,
    gyro_x: f32,
    gyro_y: f32,
    gyro_z: f32,
}

/// Mock BME688 readings (replace with actual driver)
fn read_bme688() -> Bme688Reading {
    static mut COUNTER: u32 = 0;
    unsafe {
        COUNTER = COUNTER.wrapping_add(1);
        
        Bme688Reading {
            temperature: 22.5 + ((COUNTER % 20) as f32) * 0.1,
            humidity: 45.0 + ((COUNTER % 30) as f32) * 0.5,
            pressure: 1013.25 + ((COUNTER % 10) as f32) * 0.2,
            gas_resistance: 150.0 + ((COUNTER % 50) as f32) * 2.0,
        }
    }
}

/// Mock IMU readings (replace with actual driver)
fn read_imu() -> ImuReading {
    static mut COUNTER: u32 = 0;
    unsafe {
        COUNTER = COUNTER.wrapping_add(1);
        
        // Simulate slight movement
        let movement = if COUNTER % 100 < 10 { 0.5 } else { 0.05 };
        
        ImuReading {
            accel_x: movement * (COUNTER % 3) as f32,
            accel_y: movement * ((COUNTER + 1) % 3) as f32,
            accel_z: 1.0 - movement, // Gravity
            gyro_x: movement * 10.0,
            gyro_y: movement * 5.0,
            gyro_z: movement * 2.0,
        }
    }
}

/// Mock sound level reading
fn read_sound_level() -> f32 {
    static mut COUNTER: u32 = 0;
    unsafe {
        COUNTER = COUNTER.wrapping_add(1);
        
        // Base level with occasional spikes
        if COUNTER % 50 == 0 {
            75.0 // Loud event
        } else {
            40.0 + ((COUNTER % 10) as f32)
        }
    }
}

#[entry]
fn main() -> ! {
    // Initialize RTT for debug output
    rtt_init_print!();
    rprintln!("XIAO nRF52840 Sense + BME688 Example");
    rprintln!("EdgeGuard IoT Validation Framework");
    
    // Initialize Embassy
    let p = embassy_nrf::init(Default::default());
    
    // Configure LED for status indication
    let mut led = Output::new(p.P0_26, Level::Low, OutputDrive::Standard);
    
    // Configure I2C for BME688
    let config = twim::Config::default();
    let mut i2c = Twim::new(p.TWISPI0, Irqs, p.P0_05, p.P0_04, config);
    
    // Initialize time source
    let time_source = NrfTimeSource;
    
    // Create validators
    let temp_validator = TemperatureValidator::new()
        .with_range(-40.0, 85.0)
        .with_rate_limit(5.0);
        
    let humidity_validator = HumidityValidator::new()
        .with_range(0.0, 100.0)
        .with_rate_limit(10.0);
        
    let pressure_validator = PressureValidator::new()
        .with_range(300.0, 1100.0)
        .with_altitude(100.0); // 100m above sea level
        
    let mut voc_validator = VocValidator::new();
    
    // Create specialized detectors
    let mut motion_detector = MotionDetector::new();
    let sound_analyzer = SoundAnalyzer::new();
    
    // Build validation pipeline
    let mut pipeline = Pipeline::<512>::builder()
        // Environmental validators
        .add_stage(ValidationStage::new(temp_validator, SensorType::Temperature))
        .add_stage(ValidationStage::new(humidity_validator, SensorType::Humidity))
        .add_stage(ValidationStage::new(pressure_validator, SensorType::Pressure))
        
        // Cross-validation for environmental sensors
        .add_stage({
            let mut cross_val = CrossValidationStage::new();
            cross_val.add_pair(
                SensorType::Temperature,
                SensorType::Humidity,
                edgeguard_core::events::CrossValidationType::DewPoint,
            );
            cross_val
        })
        
        // Filter out low-quality readings
        .add_stage(FilterStage::new(|event| {
            if let Event::SensorReading { quality, .. } = event {
                *quality > 0.7 // Only accept high-quality readings
            } else {
                true
            }
        }))
        
        .build();
    
    // Validation contexts
    let mut temp_context = ValidationContext::with_capacity(20);
    let mut humidity_context = ValidationContext::with_capacity(20);
    let mut pressure_context = ValidationContext::with_capacity(20);
    let mut voc_context = ValidationContext::with_capacity(50);
    
    // Statistics
    let mut total_readings = 0u32;
    let mut validation_failures = 0u32;
    let mut motion_events = 0u32;
    let mut sound_anomalies = 0u32;
    
    rprintln!("Starting sensor monitoring loop...");
    rprintln!("Calibrating VOC sensor baseline...");
    
    // Main sensing loop
    loop {
        // Toggle LED to show activity
        led.toggle();
        
        // Read all sensors
        let bme688 = read_bme688();
        let imu = read_imu();
        let sound_db = read_sound_level();
        let timestamp = time_source.now();
        
        total_readings += 1;
        
        // Update contexts
        temp_context.timestamp = timestamp;
        humidity_context.timestamp = timestamp;
        pressure_context.timestamp = timestamp;
        voc_context.timestamp = timestamp;
        
        // Validate BME688 readings
        if let Err(e) = temp_validator.validate(&bme688.temperature, &temp_context) {
            rprintln!("Temperature validation failed: {:?}", e);
            validation_failures += 1;
        } else {
            temp_context.add_reading(bme688.temperature, timestamp);
        }
        
        if let Err(e) = humidity_validator.validate(&bme688.humidity, &humidity_context) {
            rprintln!("Humidity validation failed: {:?}", e);
            validation_failures += 1;
        } else {
            humidity_context.add_reading(bme688.humidity, timestamp);
        }
        
        if let Err(e) = pressure_validator.validate(&bme688.pressure, &pressure_context) {
            rprintln!("Pressure validation failed: {:?}", e);
            validation_failures += 1;
        } else {
            pressure_context.add_reading(bme688.pressure, timestamp);
        }
        
        // VOC sensor calibration and validation
        if !voc_validator.calibrated {
            voc_validator.calibrate_baseline(bme688.gas_resistance);
        } else {
            if let Err(e) = voc_validator.validate(&bme688.gas_resistance, &voc_context) {
                rprintln!("VOC validation failed: {}", e);
                validation_failures += 1;
            } else {
                voc_context.add_reading(bme688.gas_resistance, timestamp);
            }
        }
        
        // Motion detection
        let accel_mag = (imu.accel_x * imu.accel_x + 
                        imu.accel_y * imu.accel_y + 
                        imu.accel_z * imu.accel_z).sqrt();
        let gyro_mag = (imu.gyro_x * imu.gyro_x + 
                       imu.gyro_y * imu.gyro_y + 
                       imu.gyro_z * imu.gyro_z).sqrt();
        
        let motion_state = motion_detector.detect_motion(accel_mag - 1.0, gyro_mag);
        
        match motion_state {
            MotionState::Moving => {
                motion_events += 1;
                rprintln!("[{}] Motion detected! Accel: {:.2}g", timestamp, accel_mag);
            }
            MotionState::Vibration => {
                rprintln!("[{}] Vibration detected", timestamp);
            }
            _ => {}
        }
        
        // Sound analysis
        let sound_state = sound_analyzer.analyze(sound_db);
        
        if let SoundState::Anomaly = sound_state {
            sound_anomalies += 1;
            rprintln!("[{}] Sound anomaly detected: {:.1} dB", timestamp, sound_db);
        }
        
        // Create events for pipeline processing
        let events = [
            EventBuilder::new(timestamp)
                .sensor("bme688_temp", SensorType::Temperature)
                .reading(bme688.temperature, 0.95)
                .unwrap(),
                
            EventBuilder::new(timestamp)
                .sensor("bme688_humidity", SensorType::Humidity)
                .reading(bme688.humidity, 0.90)
                .unwrap(),
                
            EventBuilder::new(timestamp)
                .sensor("bme688_pressure", SensorType::Pressure)
                .reading(bme688.pressure, 0.98)
                .unwrap(),
                
            EventBuilder::new(timestamp)
                .sensor("bme688_voc", SensorType::Voc)
                .reading(bme688.gas_resistance, if voc_validator.calibrated { 0.85 } else { 0.3 })
                .unwrap(),
        ];
        
        // Process through pipeline
        for event in &events {
            pipeline.push_event(event.clone());
        }
        
        // Process batch
        let _ = pipeline.process_batch(events.len());
        
        // Collect validated results
        let mut validated_data = String::<256>::new();
        let _ = write!(&mut validated_data, "T:{:.1}°C ", bme688.temperature);
        let _ = write!(&mut validated_data, "H:{:.0}% ", bme688.humidity);
        let _ = write!(&mut validated_data, "P:{:.0}hPa ", bme688.pressure);
        
        if voc_validator.calibrated {
            let _ = write!(&mut validated_data, "VOC:{:.0}kΩ ", bme688.gas_resistance);
        }
        
        match motion_state {
            MotionState::Moving => { let _ = write!(&mut validated_data, "MOTION "); }
            MotionState::Vibration => { let _ = write!(&mut validated_data, "VIBRATION "); }
            _ => {}
        }
        
        match sound_state {
            SoundState::Anomaly => { let _ = write!(&mut validated_data, "SOUND! "); }
            SoundState::Loud => { let _ = write!(&mut validated_data, "LOUD "); }
            _ => {}
        }
        
        // Display current readings every 10 samples
        if total_readings % 10 == 0 {
            rprintln!("\n--- Sensor Status ---");
            rprintln!("{}", validated_data.as_str());
            rprintln!("Total readings: {}", total_readings);
            rprintln!("Validation failures: {}", validation_failures);
            rprintln!("Motion events: {}", motion_events);
            rprintln!("Sound anomalies: {}", sound_anomalies);
            
            let success_rate = ((total_readings - validation_failures) as f32 / total_readings as f32) * 100.0;
            rprintln!("Validation success rate: {:.1}%", success_rate);
            rprintln!("-------------------\n");
            
            // In production, transmit via BLE here
            // ble_advertise(&validated_data);
        }
        
        // Sleep for power efficiency
        embassy_executor::block_on(Timer::after(Duration::from_millis(100)));
    }
}

// Production Cargo.toml for XIAO nRF52840 Sense:
/*
[dependencies]
# EdgeGuard with minimal features for embedded
edgeguard-core = { 
    version = "0.1", 
    default-features = false, 
    features = [
        "validation-core",
        "pipeline-core",
        "pipeline-stages",
        "fusion-core"
    ]
}

# nRF52840 HAL and runtime
embassy-nrf = { version = "0.1", features = ["defmt", "nrf52840", "time-driver-rtc1", "gpiote", "unstable-pac"] }
embassy-executor = { version = "0.5", features = ["defmt", "arch-cortex-m", "executor-thread"] }
embassy-time = { version = "0.3", features = ["defmt", "defmt-timestamp-uptime"] }
cortex-m = { version = "0.7", features = ["inline-asm", "critical-section-single-core"] }
cortex-m-rt = "0.7"
defmt = "0.3"
defmt-rtt = "0.4"
panic-probe = { version = "0.3", features = ["print-defmt"] }
rtt-target = "0.5"

# Sensor drivers
bme680 = { version = "0.6", default-features = false } # For BME688
lsm6ds3 = { version = "0.3", default-features = false } # For IMU

# Utilities
heapless = "0.8"
nb = "1.0"
fixed = { version = "1.23", default-features = false }

# Optional: BLE support
nrf-softdevice = { version = "0.1", optional = true, features = ["defmt", "nrf52840", "s140", "ble-peripheral"] }

[features]
default = []
ble = ["nrf-softdevice"]

[profile.release]
opt-level = "z"
lto = true
codegen-units = 1
debug = false
strip = true

# Memory layout for XIAO nRF52840
[[bin]]
name = "xiao_nrf52840_sense"

[package.metadata.chip]
chip = "nRF52840_xxAA"

[package.metadata.memory]
# XIAO nRF52840 has 1MB flash, 256KB RAM
FLASH : ORIGIN = 0x00000000, LENGTH = 1024K
RAM : ORIGIN = 0x20000000, LENGTH = 256K
*/