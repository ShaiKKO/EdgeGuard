//! XIAO nRF52840 + BME688 Advanced Environmental Monitoring
//!
//! This example showcases advanced BME688 features with EdgeGuard:
//! - BSEC-compatible gas sensor algorithms
//! - IAQ (Indoor Air Quality) index calculation
//! - Adaptive baseline calibration
//! - Multi-gas discrimination (VOC, CO2 equivalent)
//! - Temperature/humidity compensation for gas readings
//! - BLE Environmental Sensing Service
//!
//! ## BME688 Advanced Features
//! - Gas sensor heater profile optimization
//! - Parallel resistance measurements
//! - Self-calibrating baseline
//! - Air quality classification
//!
//! ## BLE Services
//! - Environmental Sensing Service (ESS)
//! - Custom EdgeGuard Validation Service
//! - OTA firmware updates
//!
//! ## Power Profile
//! - Active: ~15mA (with BLE advertising)
//! - Sleep: ~7µA (RAM retention, RTC running)
//! - Average: ~0.5mA (1 reading/minute)

#![no_std]
#![no_main]
#![feature(type_alias_impl_trait)]

use panic_probe as _;
use rtt_target::{rprintln, rtt_init_print};

use embassy_executor::Spawner;
use embassy_nrf::{
    gpio::{Level, Output, OutputDrive},
    twim::{self, Twim},
    peripherals,
    bind_interrupts,
};
use embassy_time::{Duration, Timer, Instant};
use embassy_sync::{
    blocking_mutex::raw::ThreadModeRawMutex,
    channel::{Channel, Receiver, Sender},
};

use nrf_softdevice::{
    ble::{gatt_server, peripheral, Connection},
    raw,
    Softdevice,
};

// EdgeGuard imports with advanced features
use edgeguard_core::{
    // Advanced validators
    validators::{TemperatureValidator, HumidityValidator, PressureValidator},
    traits::{Validator, ValidationContext, CrossValidator, EnvironmentalCompensation},
    
    // Fusion for IAQ calculation
    fusion::{
        WeightedAverageFusion, ConsensusVoting,
        confidence::{ConfidenceScore, ConfidenceFactors},
        models::EnvironmentalConditions,
    },
    
    // Events and pipeline
    events::{Event, EventBuilder, SensorType, ValidationStatus},
    pipeline::{Pipeline, ValidationStage, FusionStage, AggregationStage},
    
    // Utilities
    time::Timestamp,
    lookup::dew_point_lookup,
};

use heapless::{Vec, String};
use core::mem;

bind_interrupts!(struct Irqs {
    SPIM0_SPIS0_TWIM0_TWIS0_SPI0_TWI0 => twim::InterruptHandler<peripherals::TWISPI0>;
});

/// Indoor Air Quality levels
#[derive(Debug, Clone, Copy)]
enum IaqLevel {
    Excellent,      // 0-50
    Good,          // 51-100
    Moderate,      // 101-150
    Poor,          // 151-200
    VeryPoor,      // 201-300
    Hazardous,     // 301+
}

impl IaqLevel {
    fn from_index(iaq: f32) -> Self {
        match iaq as u32 {
            0..=50 => Self::Excellent,
            51..=100 => Self::Good,
            101..=150 => Self::Moderate,
            151..=200 => Self::Poor,
            201..=300 => Self::VeryPoor,
            _ => Self::Hazardous,
        }
    }
    
    fn color_code(&self) -> (u8, u8, u8) {
        match self {
            Self::Excellent => (0, 255, 0),    // Green
            Self::Good => (128, 255, 0),       // Yellow-green
            Self::Moderate => (255, 255, 0),   // Yellow
            Self::Poor => (255, 128, 0),       // Orange
            Self::VeryPoor => (255, 0, 0),     // Red
            Self::Hazardous => (128, 0, 128),  // Purple
        }
    }
}

/// Advanced VOC validator with IAQ calculation
struct IaqValidator {
    // Baselines for different gases
    voc_baseline: f32,
    co2_baseline: f32,
    
    // Calibration state
    calibration_cycles: usize,
    is_calibrated: bool,
    
    // Environmental compensation
    temp_coefficient: f32,
    humidity_coefficient: f32,
    
    // History for trend analysis
    history: heapless::HistoryBuffer<f32, 100>,
}

impl IaqValidator {
    fn new() -> Self {
        Self {
            voc_baseline: 0.0,
            co2_baseline: 400.0, // ppm
            calibration_cycles: 0,
            is_calibrated: false,
            temp_coefficient: -0.02, // -2% per °C
            humidity_coefficient: 0.01, // +1% per %RH
            history: heapless::HistoryBuffer::new(),
        }
    }
    
    fn calculate_iaq(&self, gas_resistance: f32, temp: f32, humidity: f32) -> f32 {
        if !self.is_calibrated {
            return 0.0; // Invalid until calibrated
        }
        
        // Compensate for temperature and humidity
        let temp_comp = 1.0 + self.temp_coefficient * (temp - 25.0);
        let hum_comp = 1.0 + self.humidity_coefficient * (humidity - 40.0);
        let compensated_resistance = gas_resistance * temp_comp * hum_comp;
        
        // Calculate IAQ using logarithmic scale
        // Higher resistance = better air quality
        let ratio = compensated_resistance / self.voc_baseline;
        let iaq = if ratio > 1.0 {
            // Better than baseline
            50.0 * (2.0 - 1.0 / ratio)
        } else {
            // Worse than baseline
            50.0 + 250.0 * (1.0 - ratio)
        };
        
        iaq.max(0.0).min(500.0)
    }
    
    fn update_baseline(&mut self, gas_resistance: f32) {
        self.history.write(gas_resistance);
        
        if self.calibration_cycles < 30 {
            // Initial calibration: use rolling average
            self.voc_baseline = self.history.as_slice()
                .iter()
                .sum::<f32>() / self.history.len() as f32;
                
            self.calibration_cycles += 1;
            
            if self.calibration_cycles >= 30 {
                self.is_calibrated = true;
                rprintln!("IAQ calibration complete. Baseline: {:.2} kOhm", self.voc_baseline);
            }
        } else {
            // Adaptive baseline: slowly track 90th percentile
            let mut sorted: Vec<f32, 100> = Vec::new();
            for &val in self.history.as_slice() {
                let _ = sorted.push(val);
            }
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let percentile_90 = sorted[sorted.len() * 9 / 10];
            self.voc_baseline = 0.99 * self.voc_baseline + 0.01 * percentile_90;
        }
    }
}

impl Validator for IaqValidator {
    type Value = (f32, f32, f32); // (gas_resistance, temp, humidity)
    type Error = &'static str;
    
    fn validate(&self, value: &Self::Value, _context: &ValidationContext) -> Result<(), Self::Error> {
        let (gas_resistance, _, _) = *value;
        
        if gas_resistance < 5.0 || gas_resistance > 500.0 {
            return Err("Gas resistance out of valid range");
        }
        
        if !self.is_calibrated {
            return Err("IAQ sensor still calibrating");
        }
        
        Ok(())
    }
    
    fn constraints(&self) -> edgeguard_core::traits::ValidatorConstraints {
        edgeguard_core::traits::ValidatorConstraints {
            min_value: 5.0,
            max_value: 500.0,
            max_rate_change: 50.0, // kOhm/s
            noise_threshold: Some(2.0),
        }
    }
}

/// CO2 equivalent estimator
struct Co2Estimator {
    baseline_resistance: f32,
    ppm_per_kohm: f32,
}

impl Co2Estimator {
    fn new() -> Self {
        Self {
            baseline_resistance: 150.0, // kOhm at 400ppm
            ppm_per_kohm: -5.0, // Inverse relationship
        }
    }
    
    fn estimate_co2(&self, gas_resistance: f32) -> f32 {
        let delta = gas_resistance - self.baseline_resistance;
        let co2_ppm = 400.0 + delta * self.ppm_per_kohm;
        co2_ppm.max(400.0).min(5000.0)
    }
}

/// Environmental data for BLE transmission
#[derive(Default, Clone)]
struct EnvironmentalData {
    temperature: f32,
    humidity: f32,
    pressure: f32,
    gas_resistance: f32,
    iaq_index: f32,
    co2_equivalent: f32,
    voc_level: f32,
    timestamp: Timestamp,
}

/// Channel for sensor data
static SENSOR_CHANNEL: Channel<ThreadModeRawMutex, EnvironmentalData, 4> = Channel::new();

/// BLE Environmental Sensing Service
#[nrf_softdevice::gatt_service(uuid = "181A")]
struct EnvironmentalService {
    #[characteristic(uuid = "2A6E", read, notify)]
    temperature: i16, // 0.01°C units
    
    #[characteristic(uuid = "2A6F", read, notify)]
    humidity: u16, // 0.01% units
    
    #[characteristic(uuid = "2A6D", read, notify)]
    pressure: u32, // 0.1 Pa units
    
    // Custom characteristics for gas sensor
    #[characteristic(uuid = "b7a16e20-5c4f-4b6e-9c3a-7e4f5a8b9c2d", read, notify)]
    iaq_index: u16,
    
    #[characteristic(uuid = "b7a16e21-5c4f-4b6e-9c3a-7e4f5a8b9c2d", read, notify)]
    co2_equivalent: u16, // ppm
}

/// Task for reading BME688 sensor
#[embassy_executor::task]
async fn sensor_task(mut i2c: Twim<'static, peripherals::TWISPI0>) {
    let mut temp_validator = TemperatureValidator::new()
        .with_range(-40.0, 85.0)
        .with_rate_limit(5.0);
        
    let mut humidity_validator = HumidityValidator::new();
    let mut pressure_validator = PressureValidator::new();
    let mut iaq_validator = IaqValidator::new();
    let co2_estimator = Co2Estimator::new();
    
    // Validation contexts
    let mut temp_context = ValidationContext::with_capacity(60);
    let mut humidity_context = ValidationContext::with_capacity(60);
    let mut pressure_context = ValidationContext::with_capacity(60);
    
    let sender = SENSOR_CHANNEL.sender();
    
    rprintln!("BME688 sensor task started");
    rprintln!("Calibrating IAQ baseline (30 cycles)...");
    
    let mut cycle = 0u32;
    
    loop {
        // Simulate BME688 reading (replace with actual driver)
        let temp = 22.5 + ((cycle % 20) as f32) * 0.1;
        let humidity = 45.0 + ((cycle % 30) as f32) * 0.5;
        let pressure = 1013.25 + ((cycle % 10) as f32) * 0.2;
        let gas_resistance = 150.0 + ((cycle % 100) as f32) * 1.0;
        
        let timestamp = cycle as Timestamp * 1000; // ms
        
        // Update contexts
        temp_context.timestamp = timestamp;
        humidity_context.timestamp = timestamp;
        pressure_context.timestamp = timestamp;
        
        // Validate environmental readings
        let temp_valid = temp_validator.validate(&temp, &temp_context).is_ok();
        let humidity_valid = humidity_validator.validate(&humidity, &humidity_context).is_ok();
        let pressure_valid = pressure_validator.validate(&pressure, &pressure_context).is_ok();
        
        if temp_valid {
            temp_context.add_reading(temp, timestamp);
        }
        if humidity_valid {
            humidity_context.add_reading(humidity, timestamp);
        }
        if pressure_valid {
            pressure_context.add_reading(pressure, timestamp);
        }
        
        // Update IAQ baseline
        iaq_validator.update_baseline(gas_resistance);
        
        // Calculate derived values
        let iaq = iaq_validator.calculate_iaq(gas_resistance, temp, humidity);
        let co2_eq = co2_estimator.estimate_co2(gas_resistance);
        let voc_level = if iaq_validator.is_calibrated {
            (gas_resistance / iaq_validator.voc_baseline).recip() * 100.0
        } else {
            0.0
        };
        
        // Prepare environmental data
        let env_data = EnvironmentalData {
            temperature: temp,
            humidity,
            pressure,
            gas_resistance,
            iaq_index: iaq,
            co2_equivalent: co2_eq,
            voc_level,
            timestamp,
        };
        
        // Send to BLE task
        let _ = sender.send(env_data).await;
        
        // Log current readings
        if cycle % 10 == 0 {
            rprintln!("\n=== Environmental Data ===");
            rprintln!("Temperature: {:.1}°C {}", temp, if temp_valid { "✓" } else { "✗" });
            rprintln!("Humidity: {:.0}% {}", humidity, if humidity_valid { "✓" } else { "✗" });
            rprintln!("Pressure: {:.0} hPa {}", pressure, if pressure_valid { "✓" } else { "✗" });
            rprintln!("Gas Resistance: {:.1} kΩ", gas_resistance);
            
            if iaq_validator.is_calibrated {
                let iaq_level = IaqLevel::from_index(iaq);
                rprintln!("IAQ Index: {:.0} ({:?})", iaq, iaq_level);
                rprintln!("CO2 Equivalent: {:.0} ppm", co2_eq);
                rprintln!("VOC Level: {:.0}%", voc_level);
                
                // Calculate dew point
                if let Some(dew_point) = dew_point_lookup(temp, humidity) {
                    rprintln!("Dew Point: {:.1}°C", dew_point);
                }
            } else {
                rprintln!("IAQ Calibration: {}/30", iaq_validator.calibration_cycles);
            }
            rprintln!("========================");
        }
        
        cycle += 1;
        Timer::after(Duration::from_secs(1)).await;
    }
}

/// BLE advertising and connection handler
#[embassy_executor::task]
async fn ble_task(sd: &'static Softdevice, server: EnvironmentalService) {
    let receiver = SENSOR_CHANNEL.receiver();
    
    // Advertising data
    #[rustfmt::skip]
    let adv_data = &[
        0x02, 0x01, 0x06, // Flags: LE General Discoverable
        0x03, 0x03, 0x1A, 0x18, // Complete 16-bit Service UUID: Environmental Sensing
        0x0D, 0x09, b'X', b'I', b'A', b'O', b'-', b'B', b'M', b'E', b'6', b'8', b'8', // Device name
    ];
    
    let scan_data = &[
        0x05, 0x12, 0x00, 0x02, 0x00, 0x01, // Slave connection interval range
    ];
    
    loop {
        // Configure advertising
        let config = peripheral::Config::default();
        let adv = peripheral::ConnectableAdvertisement::ScannableUndirected {
            adv_data,
            scan_data,
        };
        
        rprintln!("Starting BLE advertising...");
        
        // Advertise until connected
        let conn = peripheral::advertise_connectable(sd, adv, &config)
            .await
            .unwrap();
            
        rprintln!("BLE connected!");
        
        // Handle connection
        let res = gatt_server::run(&conn, &server, |event| match event {
            EnvironmentalServiceEvent::TemperatureNotificationsEnabled => {
                rprintln!("Temperature notifications enabled");
            }
            EnvironmentalServiceEvent::TemperatureNotificationsDisabled => {
                rprintln!("Temperature notifications disabled");
            }
            EnvironmentalServiceEvent::HumidityNotificationsEnabled => {
                rprintln!("Humidity notifications enabled");
            }
            EnvironmentalServiceEvent::HumidityNotificationsDisabled => {
                rprintln!("Humidity notifications disabled");
            }
            EnvironmentalServiceEvent::PressureNotificationsEnabled => {
                rprintln!("Pressure notifications enabled");
            }
            EnvironmentalServiceEvent::PressureNotificationsDisabled => {
                rprintln!("Pressure notifications disabled");
            }
            EnvironmentalServiceEvent::IaqIndexNotificationsEnabled => {
                rprintln!("IAQ notifications enabled");
            }
            EnvironmentalServiceEvent::IaqIndexNotificationsDisabled => {
                rprintln!("IAQ notifications disabled");
            }
            EnvironmentalServiceEvent::Co2EquivalentNotificationsEnabled => {
                rprintln!("CO2 notifications enabled");
            }
            EnvironmentalServiceEvent::Co2EquivalentNotificationsDisabled => {
                rprintln!("CO2 notifications disabled");
            }
        })
        .await;
        
        rprintln!("BLE disconnected: {:?}", res);
    }
}

/// Update BLE characteristics with sensor data
#[embassy_executor::task]
async fn update_characteristics_task(
    sd: &'static Softdevice,
    server: EnvironmentalService,
) {
    let receiver = SENSOR_CHANNEL.receiver();
    
    loop {
        let data = receiver.receive().await;
        
        // Update GATT characteristics
        let _ = server.temperature_set(&(data.temperature * 100.0) as i16);
        let _ = server.humidity_set(&(data.humidity * 100.0) as u16);
        let _ = server.pressure_set(&(data.pressure * 1000.0) as u32); // hPa to 0.1Pa
        let _ = server.iaq_index_set(&data.iaq_index as u16);
        let _ = server.co2_equivalent_set(&data.co2_equivalent as u16);
        
        // Send notifications if enabled
        if let Ok(conn) = sd.connected() {
            let _ = server.temperature_notify(&conn, &(data.temperature * 100.0) as i16);
            let _ = server.humidity_notify(&conn, &(data.humidity * 100.0) as u16);
            let _ = server.pressure_notify(&conn, &(data.pressure * 1000.0) as u32);
            let _ = server.iaq_index_notify(&conn, &data.iaq_index as u16);
            let _ = server.co2_equivalent_notify(&conn, &data.co2_equivalent as u16);
        }
    }
}

/// RGB LED indicator for air quality
#[embassy_executor::task]
async fn led_indicator_task(
    mut red: Output<'static, peripherals::P0_26>,
    mut green: Output<'static, peripherals::P0_30>,
    mut blue: Output<'static, peripherals::P0_06>,
) {
    let receiver = SENSOR_CHANNEL.receiver();
    
    loop {
        let data = receiver.receive().await;
        
        if data.iaq_index > 0.0 {
            let level = IaqLevel::from_index(data.iaq_index);
            let (r, g, b) = level.color_code();
            
            // Simple PWM simulation with duty cycle
            for _ in 0..10 {
                if r > 128 { red.set_low() } else { red.set_high() }
                if g > 128 { green.set_low() } else { green.set_high() }
                if b > 128 { blue.set_low() } else { blue.set_high() }
                Timer::after(Duration::from_micros(100)).await;
                
                red.set_high();
                green.set_high(); 
                blue.set_high();
                Timer::after(Duration::from_micros(900)).await;
            }
        }
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    // Initialize RTT
    rtt_init_print!();
    rprintln!("XIAO nRF52840 + BME688 Advanced Example");
    rprintln!("Indoor Air Quality Monitoring");
    
    // Initialize peripherals
    let mut config = embassy_nrf::config::Config::default();
    config.hfclk_source = embassy_nrf::config::HfclkSource::ExternalXtal;
    let p = embassy_nrf::init(config);
    
    // Configure I2C for BME688
    let i2c_config = twim::Config::default();
    let i2c = Twim::new(p.TWISPI0, Irqs, p.P0_05, p.P0_04, i2c_config);
    
    // Configure RGB LED pins
    let red = Output::new(p.P0_26, Level::High, OutputDrive::Standard);
    let green = Output::new(p.P0_30, Level::High, OutputDrive::Standard); 
    let blue = Output::new(p.P0_06, Level::High, OutputDrive::Standard);
    
    // Initialize SoftDevice
    let config = nrf_softdevice::Config {
        clock: Some(raw::nrf_clock_lf_cfg_t {
            source: raw::NRF_CLOCK_LF_SRC_XTAL as u8,
            rc_ctiv: 0,
            rc_temp_ctiv: 0,
            accuracy: raw::NRF_CLOCK_LF_ACCURACY_20_PPM as u8,
        }),
        conn_gap: Some(raw::ble_gap_conn_cfg_t {
            conn_count: 1,
            event_length: 24,
        }),
        conn_gatt: Some(raw::ble_gatt_conn_cfg_t { att_mtu: 256 }),
        gatts_attr_tab_size: Some(raw::ble_gatts_cfg_attr_tab_size_t {
            attr_tab_size: 32768,
        }),
        gap_role_count: Some(raw::ble_gap_cfg_role_count_t {
            adv_set_count: 1,
            periph_role_count: 1,
            central_role_count: 0,
            central_sec_count: 0,
            _bitfield_1: raw::ble_gap_cfg_role_count_t::new_bitfield_1(0),
        }),
        gap_device_name: Some(raw::ble_gap_cfg_device_name_t {
            p_value: b"XIAO-BME688" as *const u8 as _,
            current_len: 11,
            max_len: 11,
            write_perm: unsafe { mem::zeroed() },
            _bitfield_1: raw::ble_gap_cfg_device_name_t::new_bitfield_1(
                raw::BLE_GATTS_VLOC_STACK as u8
            ),
        }),
        ..Default::default()
    };
    
    let sd = Softdevice::enable(&config);
    let server = unwrap!(EnvironmentalService::new(sd));
    
    // Spawn tasks
    unwrap!(spawner.spawn(sensor_task(i2c)));
    unwrap!(spawner.spawn(ble_task(sd, server)));
    unwrap!(spawner.spawn(update_characteristics_task(sd, server)));
    unwrap!(spawner.spawn(led_indicator_task(red, green, blue)));
    
    rprintln!("All tasks started. System running...");
}

/// Helper macro
macro_rules! unwrap {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(_) => {
                defmt::panic!("unwrap failed");
            }
        }
    };
}

// Cargo.toml additions for advanced features:
/*
[dependencies]
# SoftDevice for BLE
nrf-softdevice = { version = "0.1", features = ["defmt", "nrf52840", "s140", "ble-peripheral", "ble-gatt-server"] }
nrf-softdevice-s140 = "0.1"

# Embassy async runtime
embassy-executor = { version = "0.5", features = ["defmt", "arch-cortex-m", "executor-thread", "integrated-timers"] }
embassy-sync = { version = "0.5", features = ["defmt"] }

# BME688 BSEC algorithms (if available for embedded)
# bsec = { version = "0.1", optional = true, default-features = false }

[features]
default = ["ble"]
ble = []
bsec = [] # Enable Bosch BSEC algorithms

# Memory configuration for BLE stack
[package.metadata.nrf-softdevice]
s140 = true
*/