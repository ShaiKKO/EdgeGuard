//! Common test utilities and data generators for integration tests
//!
//! This module provides:
//! - Realistic sensor data generators with physics-aware constraints
//! - Test fixtures and scenarios for edge cases
//! - Assertion helpers for validation testing
//! - Performance measurement utilities
//!
//! All utilities are no_std compatible for embedded testing

#![allow(dead_code)]
#![cfg_attr(not(test), no_std)]

extern crate alloc;
use alloc::vec::Vec;

use edgeguard_core::{
    events::{Event, EventBuilder, SensorType},
    time::{Timestamp, TimeSource, MockTimeSource},
    lookup::sin_lookup,
};

// Re-export submodules
pub mod generators;
pub mod harness;
pub mod scenarios;

/// Test data generator for realistic sensor scenarios
pub struct TestDataGenerator {
    time_source: MockTimeSource,
    seed: u32,
}

impl TestDataGenerator {
    /// Create new test data generator
    pub fn new(start_time: Timestamp) -> Self {
        Self {
            time_source: MockTimeSource::new(start_time),
            seed: 42,
        }
    }
    
    /// Generate temperature sensor data with realistic patterns
    /// 
    /// Simulates:
    /// - Diurnal temperature variation
    /// - Random noise
    /// - Occasional spikes/anomalies
    /// - Sensor drift over time
    pub fn generate_temperature_series(
        &mut self,
        sensor_id: &str,
        base_temp: f32,
        duration_hours: u32,
        samples_per_hour: u32,
    ) -> Vec<Event> {
        let mut events = Vec::new();
        let total_samples = duration_hours * samples_per_hour;
        let sample_interval_ms = 3600 * 1000 / samples_per_hour;
        
        for i in 0..total_samples {
            let hours_elapsed = i as f32 / samples_per_hour as f32;
            
            // Diurnal variation (sine wave with 24h period)
            // Using lookup table for no_std compatibility
            let angle_rad = 2.0 * 3.14159 * hours_elapsed / 24.0;
            let diurnal = 2.0 * sin_lookup(angle_rad).unwrap_or(0.0);
            
            // Add some noise
            let noise = self.random_noise(0.5);
            
            // Occasional anomalies (1% chance)
            let anomaly = if self.random_float() < 0.01 {
                self.random_float() * 10.0 - 5.0 // ±5°C spikes
            } else {
                0.0
            };
            
            // Sensor drift (0.01°C per hour)
            let drift = hours_elapsed * 0.01;
            
            let temperature = base_temp + diurnal + noise + anomaly + drift;
            let quality = if anomaly != 0.0 { 0.5 } else { 0.95 };
            
            let timestamp = self.time_source.now() + (i * sample_interval_ms) as u64;
            
            events.push(
                EventBuilder::new(timestamp)
                    .sensor(sensor_id, SensorType::Temperature)
                    .reading(temperature, quality)
                    .unwrap()
            );
        }
        
        events
    }
    
    /// Generate correlated temperature and humidity data
    /// 
    /// Simulates realistic correlation where:
    /// - Higher temperature generally means lower relative humidity
    /// - Both follow similar diurnal patterns
    /// - Humidity has hysteresis effects
    pub fn generate_temp_humidity_series(
        &mut self,
        temp_sensor: &str,
        humidity_sensor: &str,
        base_temp: f32,
        base_humidity: f32,
        duration_hours: u32,
        samples_per_hour: u32,
    ) -> Vec<Event> {
        let mut events = Vec::new();
        let total_samples = duration_hours * samples_per_hour;
        let sample_interval_ms = 3600 * 1000 / samples_per_hour;
        
        let mut last_humidity = base_humidity;
        
        for i in 0..total_samples {
            let hours_elapsed = i as f32 / samples_per_hour as f32;
            let timestamp = self.time_source.now() + (i * sample_interval_ms) as u64;
            
            // Temperature calculation (same as before)
            let angle_rad = 2.0 * 3.14159 * hours_elapsed / 24.0;
            let diurnal = 2.0 * sin_lookup(angle_rad).unwrap_or(0.0);
            let temp_noise = self.random_noise(0.5);
            let temperature = base_temp + diurnal + temp_noise;
            
            // Humidity inversely correlated with temperature
            // For every 1°C increase, humidity drops ~2%
            let temp_effect = -2.0 * (temperature - base_temp);
            let humidity_target = base_humidity + temp_effect + self.random_noise(2.0);
            
            // Add hysteresis - humidity changes slowly
            let humidity_change = (humidity_target - last_humidity) * 0.1;
            last_humidity += humidity_change;
            
            // Clamp humidity to valid range
            let humidity = last_humidity.clamp(0.0, 100.0);
            
            // Add temperature event
            events.push(
                EventBuilder::new(timestamp)
                    .sensor(temp_sensor, SensorType::Temperature)
                    .reading(temperature, 0.95)
                    .unwrap()
            );
            
            // Add humidity event
            events.push(
                EventBuilder::new(timestamp)
                    .sensor(humidity_sensor, SensorType::Humidity)
                    .reading(humidity, 0.93)
                    .unwrap()
            );
        }
        
        events
    }
    
    /// Generate pressure sensor data with altitude simulation
    pub fn generate_pressure_series(
        &mut self,
        sensor_id: &str,
        base_pressure: f32,
        altitude_changes: &[(u32, f32)], // (hour, altitude_m)
        duration_hours: u32,
        samples_per_hour: u32,
    ) -> Vec<Event> {
        let mut events = Vec::new();
        let total_samples = duration_hours * samples_per_hour;
        let sample_interval_ms = 3600 * 1000 / samples_per_hour;
        
        for i in 0..total_samples {
            let hours_elapsed = i as f32 / samples_per_hour as f32;
            let timestamp = self.time_source.now() + (i * sample_interval_ms) as u64;
            
            // Find current altitude based on altitude_changes
            let mut current_altitude = 0.0;
            for &(hour, altitude) in altitude_changes {
                if hours_elapsed >= hour as f32 {
                    current_altitude = altitude;
                }
            }
            
            // Pressure decreases ~12 hPa per 100m altitude
            let altitude_effect = -0.12 * current_altitude;
            
            // Add weather variation (slower changes)
            let weather_angle = 2.0 * 3.14159 * hours_elapsed / 48.0;
            let weather = 5.0 * sin_lookup(weather_angle).unwrap_or(0.0);
            
            let pressure = base_pressure + altitude_effect + weather + self.random_noise(0.5);
            
            events.push(
                EventBuilder::new(timestamp)
                    .sensor(sensor_id, SensorType::Pressure)
                    .reading(pressure, 0.98)
                    .unwrap()
            );
        }
        
        events
    }
    
    /// Generate multi-sensor scenario with failures
    pub fn generate_failure_scenario(
        &mut self,
        sensors: &[(&str, SensorType)],
        failure_periods: &[(usize, usize, usize)], // (sensor_idx, start_hour, end_hour)
        duration_hours: u32,
        samples_per_hour: u32,
    ) -> Vec<Event> {
        let mut events = Vec::new();
        let total_samples = duration_hours * samples_per_hour;
        let sample_interval_ms = 3600 * 1000 / samples_per_hour;
        
        for i in 0..total_samples {
            let hours_elapsed = i / samples_per_hour;
            let timestamp = self.time_source.now() + (i * sample_interval_ms) as u64;
            
            for (idx, &(sensor_id, sensor_type)) in sensors.iter().enumerate() {
                // Check if sensor is in failure period
                let mut is_failing = false;
                for &(fail_idx, start, end) in failure_periods {
                    if fail_idx == idx && hours_elapsed as usize >= start && (hours_elapsed as usize) < end {
                        is_failing = true;
                        break;
                    }
                }
                
                let (value, quality) = if is_failing {
                    // Generate bad data during failure
                    match self.random_int(0, 3) {
                        0 => (0.0, 0.0),           // Zero reading
                        1 => (999.9, 0.1),         // Maxed out
                        2 => (-999.9, 0.0),        // Invalid negative
                        _ => continue,              // No reading
                    }
                } else {
                    // Normal operation
                    match sensor_type {
                        SensorType::Temperature => (25.0 + self.random_noise(1.0), 0.95),
                        SensorType::Humidity => (60.0 + self.random_noise(5.0), 0.93),
                        SensorType::Pressure => (1013.0 + self.random_noise(2.0), 0.98),
                        _ => (0.0, 0.0),
                    }
                };
                
                events.push(
                    EventBuilder::new(timestamp)
                        .sensor(sensor_id, sensor_type)
                        .reading(value, quality)
                        .unwrap()
                );
            }
        }
        
        events
    }
    
    // Helper methods
    
    fn random_noise(&mut self, std_dev: f32) -> f32 {
        // Simple pseudo-random noise
        self.seed = self.seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let uniform = (self.seed as f32) / (u32::MAX as f32);
        
        // Box-Muller transform for Gaussian noise
        let z = (uniform - 0.5) * 2.0 * std_dev;
        z
    }
    
    fn random_float(&mut self) -> f32 {
        self.seed = self.seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.seed as f32) / (u32::MAX as f32)
    }
    
    fn random_int(&mut self, min: u32, max: u32) -> u32 {
        let range = max - min;
        min + (self.random_float() * range as f32) as u32
    }
}

/// Test scenario builder for complex integration tests
pub struct TestScenario {
    pub name: &'static str,
    pub description: &'static str,
    pub events: Vec<Event>,
    pub expected_valid: usize,
    pub expected_invalid: usize,
    pub expected_cross_valid: usize,
}

impl TestScenario {
    /// Create standard test scenarios
    pub fn standard_scenarios() -> Vec<Self> {
        let mut generator = TestDataGenerator::new(1000);
        
        vec![
            // Scenario 1: Normal operation
            Self {
                name: "normal_operation",
                description: "24 hours of normal sensor operation",
                events: generator.generate_temperature_series("temp1", 22.0, 24, 60),
                expected_valid: 1440, // All should be valid
                expected_invalid: 0,
                expected_cross_valid: 0,
            },
            
            // Scenario 2: Correlated sensors
            Self {
                name: "temp_humidity_correlation",
                description: "Temperature and humidity with proper correlation",
                events: generator.generate_temp_humidity_series(
                    "temp1", "hum1", 22.0, 65.0, 12, 60
                ),
                expected_valid: 1440,
                expected_invalid: 0,
                expected_cross_valid: 720, // All temp/humidity pairs should cross-validate
            },
            
            // Add more scenarios...
        ]
    }
}

/// Performance measurement utilities
pub struct PerformanceMetrics {
    pub events_processed: usize,
    pub processing_time_us: u64,
    pub memory_used: usize,
    pub events_per_second: f32,
}

impl PerformanceMetrics {
    pub fn calculate(events: usize, start_us: u64, end_us: u64) -> Self {
        let duration_us = end_us.saturating_sub(start_us);
        let events_per_second = if duration_us > 0 {
            (events as f64 * 1_000_000.0 / duration_us as f64) as f32
        } else {
            0.0
        };
        
        Self {
            events_processed: events,
            processing_time_us: duration_us,
            memory_used: 0, // Would need platform-specific measurement
            events_per_second,
        }
    }
    
    pub fn print_summary(&self) {
        println!("Performance Metrics:");
        println!("  Events processed: {}", self.events_processed);
        println!("  Processing time: {} µs", self.processing_time_us);
        println!("  Throughput: {:.0} events/second", self.events_per_second);
        if self.memory_used > 0 {
            println!("  Memory used: {} bytes", self.memory_used);
        }
    }
}

/// Test assertion helpers
#[macro_export]
macro_rules! assert_validation_stats {
    ($results:expr, valid: $valid:expr, invalid: $invalid:expr) => {
        let stats = count_validation_results(&$results);
        assert_eq!(
            stats.0, $valid,
            "Expected {} valid results, got {}", $valid, stats.0
        );
        assert_eq!(
            stats.1, $invalid,
            "Expected {} invalid results, got {}", $invalid, stats.1
        );
    };
}

/// Count validation results
pub fn count_validation_results(events: &[Event]) -> (usize, usize) {
    let mut valid = 0;
    let mut invalid = 0;
    
    for event in events {
        if let Event::ValidationResult { status, .. } = event {
            // Check if status indicates valid reading
            match status {
                edgeguard_core::events::ValidationStatus::Valid => valid += 1,
                _ => invalid += 1,
            }
        }
    }
    
    (valid, invalid)
}