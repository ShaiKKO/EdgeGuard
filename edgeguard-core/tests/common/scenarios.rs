//! Pre-built test scenarios for comprehensive integration testing
//!
//! This module provides realistic test scenarios that cover:
//! - Normal operation patterns
//! - Edge cases and failure modes
//! - Multi-sensor interactions
//! - Long-term stability testing

#![cfg_attr(not(test), no_std)]

extern crate alloc;
use alloc::{vec::Vec, string::String};

use edgeguard_core::{
    events::{Event, SensorType},
    time::Timestamp,
};

use super::generators::{PhysicsAwareGenerator, SensorModel, EnvironmentScenario, Season};

/// Test scenario configuration
pub struct ScenarioConfig {
    pub name: &'static str,
    pub description: &'static str,
    pub duration_hours: u32,
    pub samples_per_hour: u32,
    pub start_time: Timestamp,
}

impl Default for ScenarioConfig {
    fn default() -> Self {
        Self {
            name: "default",
            description: "Default test scenario",
            duration_hours: 24,
            samples_per_hour: 60,
            start_time: 1000,
        }
    }
}

/// Expected test outcomes
#[derive(Default)]
pub struct ExpectedOutcomes {
    pub valid_readings: usize,
    pub invalid_readings: usize,
    pub cross_validation_passes: usize,
    pub cross_validation_failures: usize,
    pub anomalies_detected: usize,
    pub fusion_events: usize,
}

/// Complete test scenario with data and expectations
pub struct TestScenario {
    pub config: ScenarioConfig,
    pub events: Vec<Event>,
    pub expected: ExpectedOutcomes,
}

/// Pre-built scenario definitions
pub struct Scenarios;

impl Scenarios {
    /// Normal 24-hour home environment
    pub fn home_environment() -> TestScenario {
        let config = ScenarioConfig {
            name: "home_environment",
            description: "Typical home with HVAC, day/night cycles",
            duration_hours: 24,
            samples_per_hour: 60,
            start_time: 0,
        };
        
        let mut generator = PhysicsAwareGenerator::new(config.start_time);
        
        // Generate correlated temp/humidity data
        let events = generator.generate_temp_humidity_with_dewpoint(
            "lr_temp",
            "lr_humidity",
            EnvironmentScenario::IndoorHVAC {
                setpoint: 22.0,
                deadband: 1.0,
            },
            config.duration_hours,
            config.samples_per_hour,
            &SensorModel::consumer_grade(),
            &SensorModel::consumer_grade(),
        );
        
        let expected = ExpectedOutcomes {
            valid_readings: events.len(),
            invalid_readings: 0,
            cross_validation_passes: events.len() / 2, // Each temp/humidity pair
            cross_validation_failures: 0,
            anomalies_detected: 0,
            fusion_events: 0,
        };
        
        TestScenario { config, events, expected }
    }
    
    /// Industrial monitoring with heat sources
    pub fn industrial_process() -> TestScenario {
        let config = ScenarioConfig {
            name: "industrial_process",
            description: "Industrial furnace with cycling and anomalies",
            duration_hours: 48,
            samples_per_hour: 120, // Higher frequency
            start_time: 0,
        };
        
        let mut generator = PhysicsAwareGenerator::new(config.start_time);
        
        // High-quality sensors near furnace
        let temp_events = generator.generate_temperature_with_thermal_mass(
            "furnace_temp",
            300.0, // Initial temp
            &[(0, 300.0), (6, 500.0), (12, 500.0), (18, 300.0), (24, 300.0), (30, 500.0), (36, 500.0), (42, 300.0)],
            50.0, // 50kg thermal mass
            config.duration_hours,
            config.samples_per_hour,
            &SensorModel::high_quality(),
        );
        
        let expected = ExpectedOutcomes {
            valid_readings: temp_events.len(),
            invalid_readings: 0,
            cross_validation_passes: 0,
            cross_validation_failures: 0,
            anomalies_detected: 10, // Expect some during rapid transitions
            fusion_events: 0,
        };
        
        TestScenario {
            config,
            events: temp_events,
            expected,
        }
    }
    
    /// Outdoor weather station
    pub fn weather_station() -> TestScenario {
        let config = ScenarioConfig {
            name: "weather_station",
            description: "Outdoor station with weather patterns",
            duration_hours: 72,
            samples_per_hour: 30,
            start_time: 0,
        };
        
        let mut generator = PhysicsAwareGenerator::new(config.start_time);
        let mut all_events = Vec::new();
        
        // Temperature and humidity
        let mut temp_humidity = generator.generate_temp_humidity_with_dewpoint(
            "out_temp",
            "out_humidity",
            EnvironmentScenario::OutdoorWeather {
                latitude: 45.0, // Mid-latitude
                season: Season::Summer,
            },
            config.duration_hours,
            config.samples_per_hour,
            &SensorModel::consumer_grade(),
            &SensorModel::consumer_grade(),
        );
        
        // Pressure with weather systems
        let mut pressure = generator.generate_pressure_with_weather(
            "out_pressure",
            100.0, // 100m altitude
            &[(12, -10.0, 24.0), (48, 5.0, 12.0)], // Weather fronts
            config.duration_hours,
            config.samples_per_hour,
            &SensorModel::high_quality(),
        );
        
        all_events.append(&mut temp_humidity);
        all_events.append(&mut pressure);
        
        // Sort by timestamp
        all_events.sort_by_key(|e| e.timestamp());
        
        let expected = ExpectedOutcomes {
            valid_readings: all_events.len(),
            invalid_readings: 0,
            cross_validation_passes: config.duration_hours as usize * config.samples_per_hour as usize,
            cross_validation_failures: 0,
            anomalies_detected: 5, // Some during weather fronts
            fusion_events: 0,
        };
        
        TestScenario {
            config,
            events: all_events,
            expected,
        }
    }
    
    /// Sensor failure scenario
    pub fn sensor_degradation() -> TestScenario {
        let config = ScenarioConfig {
            name: "sensor_degradation",
            description: "Multiple sensors with progressive failures",
            duration_hours: 48,
            samples_per_hour: 60,
            start_time: 0,
        };
        
        let mut generator = PhysicsAwareGenerator::new(config.start_time);
        
        // Multi-sensor setup with failures
        let sensors = vec![
            ("temp_1", SensorType::Temperature, SensorModel::consumer_grade()),
            ("temp_2", SensorType::Temperature, SensorModel::low_quality()),
            ("temp_3", SensorType::Temperature, SensorModel::high_quality()),
            ("hum_1", SensorType::Humidity, SensorModel::consumer_grade()),
        ];
        
        // Correlation matrix (temp sensors correlated, humidity independent)
        let correlation = vec![
            &[1.0, 0.8, 0.8, 0.2][..],
            &[0.8, 1.0, 0.8, 0.2][..],
            &[0.8, 0.8, 1.0, 0.2][..],
            &[0.2, 0.2, 0.2, 1.0][..],
        ];
        
        // Anomaly periods (sensor failures)
        let anomalies = vec![
            (1, 12, 18), // temp_2 fails for 6 hours
            (0, 36, 40), // temp_1 fails for 4 hours
        ];
        
        let events = generator.generate_fusion_scenario(
            &sensors,
            &correlation,
            config.duration_hours,
            config.samples_per_hour,
            &anomalies,
        );
        
        let expected = ExpectedOutcomes {
            valid_readings: events.len() - 10 * 60, // Subtract failure periods
            invalid_readings: 10 * 60, // Failure periods
            cross_validation_passes: 0,
            cross_validation_failures: 10 * 60,
            anomalies_detected: 10 * 60,
            fusion_events: config.duration_hours as usize * config.samples_per_hour as usize,
        };
        
        TestScenario { config, events, expected }
    }
    
    /// Edge case stress test
    pub fn edge_cases() -> TestScenario {
        let config = ScenarioConfig {
            name: "edge_cases",
            description: "Physical limits and edge conditions",
            duration_hours: 12,
            samples_per_hour: 120,
            start_time: 0,
        };
        
        let mut generator = PhysicsAwareGenerator::new(config.start_time);
        let mut all_events = Vec::new();
        
        // Extreme temperature variations
        let temp_extremes = generator.generate_temperature_with_thermal_mass(
            "ext_temp",
            -40.0, // Start at minimum
            &[(0, -40.0), (2, 85.0), (4, -40.0), (6, 20.0), (8, 85.0), (10, 20.0)],
            0.1, // Very low thermal mass for rapid changes
            config.duration_hours,
            config.samples_per_hour,
            &SensorModel::high_quality(),
        );
        
        // Humidity at dew point limits
        let mut humidity_events = Vec::new();
        let mut time = config.start_time;
        let interval = 3600 * 1000 / config.samples_per_hour;
        
        // Generate events that push dew point constraints
        for hour in 0..config.duration_hours {
            for _ in 0..config.samples_per_hour {
                time += interval as u64;
                
                // Alternate between valid and edge cases
                let (temp, humidity) = match hour % 4 {
                    0 => (20.0, 100.0), // At dew point
                    1 => (0.0, 50.0),   // Low temp, moderate humidity
                    2 => (40.0, 10.0),  // High temp, low humidity
                    _ => (25.0, 95.0),  // Near dew point
                };
                
                humidity_events.push(
                    edgeguard_core::events::EventBuilder::new(time)
                        .sensor("ext_humidity", SensorType::Humidity)
                        .reading(humidity, 0.95)
                        .unwrap()
                );
            }
        }
        
        all_events.extend(temp_extremes);
        all_events.extend(humidity_events);
        all_events.sort_by_key(|e| e.timestamp());
        
        let expected = ExpectedOutcomes {
            valid_readings: all_events.len() * 3 / 4, // Some will fail validation
            invalid_readings: all_events.len() / 4,
            cross_validation_passes: 0,
            cross_validation_failures: config.duration_hours as usize * config.samples_per_hour as usize / 4,
            anomalies_detected: 100, // Many during extreme transitions
            fusion_events: 0,
        };
        
        TestScenario { config, events: all_events, expected }
    }
    
    /// Long-term stability test
    pub fn long_term_drift() -> TestScenario {
        let config = ScenarioConfig {
            name: "long_term_drift",
            description: "Week-long test with sensor drift",
            duration_hours: 168, // 1 week
            samples_per_hour: 6,  // Every 10 minutes
            start_time: 0,
        };
        
        let mut generator = PhysicsAwareGenerator::new(config.start_time);
        
        // Create sensors with different drift rates
        let mut drifting_sensor = SensorModel::consumer_grade();
        drifting_sensor.drift_rate = 0.5; // 0.5°C/hour drift
        
        let events = generator.generate_temperature_with_thermal_mass(
            "drifting_sensor",
            20.0,
            &[(0, 20.0), (24, 22.0), (48, 20.0), (72, 22.0), (96, 20.0), (120, 22.0), (144, 20.0)],
            10.0,
            config.duration_hours,
            config.samples_per_hour,
            &drifting_sensor,
        );
        
        // After a week, drift should be ~84°C, causing failures
        let drift_hours = 84;
        let samples_before_failure = drift_hours * config.samples_per_hour as usize;
        
        let expected = ExpectedOutcomes {
            valid_readings: samples_before_failure,
            invalid_readings: events.len() - samples_before_failure,
            cross_validation_passes: 0,
            cross_validation_failures: 0,
            anomalies_detected: 50, // Gradual increase as drift worsens
            fusion_events: 0,
        };
        
        TestScenario { config, events, expected }
    }
    
    /// Get all standard scenarios
    pub fn all_scenarios() -> Vec<TestScenario> {
        vec![
            Self::home_environment(),
            Self::industrial_process(),
            Self::weather_station(),
            Self::sensor_degradation(),
            Self::edge_cases(),
            Self::long_term_drift(),
        ]
    }
}

/// Scenario validation helpers
impl TestScenario {
    /// Check if events match expected sensor types
    pub fn validate_sensor_types(&self) -> Result<(), String> {
        let mut sensor_types: alloc::collections::BTreeMap<String, SensorType> = alloc::collections::BTreeMap::new();
        
        for event in &self.events {
            if let Some(sensor_id) = event.sensor_id() {
                let sensor_type = match event {
                    Event::SensorReading { sensor_type, .. } => Some(sensor_type),
                    _ => None,
                };
                
                if let Some(st) = sensor_type {
                    sensor_types.insert(String::from(sensor_id), *st);
                }
            }
        }
        
        // Verify consistency
        for event in &self.events {
            if let Event::SensorReading { sensor_id, sensor_type, .. } = event {
                if let Some(&expected_type) = sensor_types.get(sensor_id.as_str()) {
                    if expected_type != *sensor_type {
                        return Err(format!(
                            "Sensor {} has inconsistent types: {:?} vs {:?}",
                            sensor_id.as_str(), expected_type, sensor_type
                        ));
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Check if events are properly time-ordered
    pub fn validate_time_ordering(&self) -> Result<(), String> {
        let mut last_time = 0;
        
        for event in &self.events {
            let timestamp = event.timestamp();
            if timestamp < last_time {
                return Err(format!(
                    "Time ordering violation: {} < {}",
                    timestamp, last_time
                ));
            }
            last_time = timestamp;
        }
        
        Ok(())
    }
    
    /// Get events for a specific sensor
    pub fn events_for_sensor(&self, sensor_id: &str) -> Vec<&Event> {
        self.events
            .iter()
            .filter(|e| e.sensor_id().map(|id| id == sensor_id).unwrap_or(false))
            .collect()
    }
    
    /// Get events in a time window
    pub fn events_in_window(&self, start: Timestamp, end: Timestamp) -> Vec<&Event> {
        self.events
            .iter()
            .filter(|e| {
                let t = e.timestamp();
                t >= start && t <= end
            })
            .collect()
    }
}