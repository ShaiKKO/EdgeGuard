//! Enhanced test data generators with physics-aware constraints
//!
//! This module provides sophisticated data generation that respects:
//! - Physical laws (thermal mass, atmospheric pressure)
//! - Sensor characteristics (noise models, drift patterns)
//! - Environmental correlations (temp/humidity, pressure/altitude)
//! - Real-world anomaly patterns

#![cfg_attr(not(test), no_std)]

extern crate alloc;
use alloc::vec::Vec;

use edgeguard_core::{
    events::{Event, EventBuilder, SensorType},
    time::Timestamp,
    lookup::{sin_lookup, cos_lookup, dew_point_lookup},
};

// Define constants locally since validator modules are private
const KELVIN_OFFSET: f32 = 273.15;
const TEMP_MIN_CELSIUS: f32 = -40.0;
const TEMP_MAX_CELSIUS: f32 = 85.0;
const HUMIDITY_MIN: f32 = 0.0;
const HUMIDITY_MAX: f32 = 100.0;
const PRESSURE_MIN_HPA: f32 = 540.0;
const PRESSURE_MAX_HPA: f32 = 1080.0;

/// Physics constants for realistic simulation
mod physics {
    /// Thermal time constant in seconds for typical sensor
    pub const THERMAL_TIME_CONSTANT: f32 = 30.0;
    
    /// Maximum rate of temperature change (°C/s) for air
    pub const MAX_TEMP_RATE_AIR: f32 = 1.0;
    
    /// Pressure change per meter altitude (hPa/m)
    pub const PRESSURE_LAPSE_RATE: f32 = 0.12;
    
    /// Humidity time constant (slower than temperature)
    pub const HUMIDITY_TIME_CONSTANT: f32 = 60.0;
}

/// Sensor-specific characteristics
pub struct SensorModel {
    /// Base noise standard deviation
    pub noise_std: f32,
    /// Drift rate per hour
    pub drift_rate: f32,
    /// Response time constant
    pub time_constant: f32,
    /// Quantization step (ADC resolution)
    pub quantization: f32,
    /// Probability of spike/glitch
    pub glitch_probability: f32,
}

impl SensorModel {
    /// High-quality sensor model
    pub fn high_quality() -> Self {
        Self {
            noise_std: 0.1,
            drift_rate: 0.001,
            time_constant: 10.0,
            quantization: 0.01,
            glitch_probability: 0.0001,
        }
    }
    
    /// Typical consumer-grade sensor
    pub fn consumer_grade() -> Self {
        Self {
            noise_std: 0.5,
            drift_rate: 0.01,
            time_constant: 30.0,
            quantization: 0.1,
            glitch_probability: 0.001,
        }
    }
    
    /// Low-quality or degraded sensor
    pub fn low_quality() -> Self {
        Self {
            noise_std: 2.0,
            drift_rate: 0.1,
            time_constant: 60.0,
            quantization: 0.5,
            glitch_probability: 0.01,
        }
    }
}

/// Environmental scenario for correlated sensor generation
pub enum EnvironmentScenario {
    /// Indoor HVAC-controlled environment
    IndoorHVAC {
        setpoint: f32,
        deadband: f32,
    },
    /// Outdoor with weather patterns
    OutdoorWeather {
        latitude: f32,
        season: Season,
    },
    /// Industrial process with heat sources
    Industrial {
        process_temp: f32,
        cycle_time_hours: f32,
    },
    /// Agricultural greenhouse
    Greenhouse {
        day_temp: f32,
        night_temp: f32,
        humidity_target: f32,
    },
}

#[derive(Clone, Copy)]
pub enum Season {
    Spring,
    Summer,
    Autumn,
    Winter,
}

/// Advanced physics-aware data generator
pub struct PhysicsAwareGenerator {
    seed: u32,
    time: Timestamp,
}

impl PhysicsAwareGenerator {
    pub fn new(start_time: Timestamp) -> Self {
        Self {
            seed: 0x12345678,
            time: start_time,
        }
    }
    
    /// Generate temperature data with thermal mass simulation
    pub fn generate_temperature_with_thermal_mass(
        &mut self,
        sensor_id: &str,
        initial_temp: f32,
        ambient_profile: &[(u32, f32)], // (hour, ambient_temp)
        thermal_mass: f32, // kg
        duration_hours: u32,
        samples_per_hour: u32,
        sensor_model: &SensorModel,
    ) -> Vec<Event> {
        let mut events = Vec::new();
        let mut actual_temp = initial_temp;
        let mut sensor_drift = 0.0;
        
        let sample_interval_s = 3600.0 / samples_per_hour as f32;
        let sample_interval_ms = (sample_interval_s * 1000.0) as u64;
        
        for hour in 0..duration_hours {
            // Find ambient temperature for this hour
            let ambient = self.interpolate_profile(ambient_profile, hour as f32);
            
            for sample in 0..samples_per_hour {
                // Update time
                self.time += sample_interval_ms;
                
                // Thermal dynamics: T(t+dt) = T(t) + dt/τ * (T_ambient - T(t))
                let tau = thermal_mass * sensor_model.time_constant;
                let temp_change = sample_interval_s / tau * (ambient - actual_temp);
                
                // Limit rate of change to physical maximum
                let limited_change = temp_change.clamp(
                    -physics::MAX_TEMP_RATE_AIR * sample_interval_s,
                    physics::MAX_TEMP_RATE_AIR * sample_interval_s
                );
                
                actual_temp += limited_change;
                
                // Add sensor effects
                sensor_drift += sensor_model.drift_rate * sample_interval_s / 3600.0;
                let noise = self.gaussian_noise(sensor_model.noise_std);
                let glitch = if self.random_float() < sensor_model.glitch_probability {
                    self.random_float() * 10.0 - 5.0
                } else {
                    0.0
                };
                
                // Quantize to ADC resolution
                let measured = actual_temp + sensor_drift + noise + glitch;
                let quantized = (measured / sensor_model.quantization).round() * sensor_model.quantization;
                
                // Determine quality based on sensor state
                let quality = if glitch != 0.0 {
                    0.5
                } else if noise.abs() > sensor_model.noise_std * 3.0 {
                    0.8
                } else {
                    0.95
                };
                
                events.push(
                    EventBuilder::new(self.time)
                        .sensor(sensor_id, SensorType::Temperature)
                        .reading(quantized, quality)
                        .unwrap()
                );
            }
        }
        
        events
    }
    
    /// Generate correlated temperature and humidity with dew point constraints
    pub fn generate_temp_humidity_with_dewpoint(
        &mut self,
        temp_sensor: &str,
        humidity_sensor: &str,
        environment: EnvironmentScenario,
        duration_hours: u32,
        samples_per_hour: u32,
        temp_model: &SensorModel,
        humidity_model: &SensorModel,
    ) -> Vec<Event> {
        let mut events = Vec::new();
        
        // Initialize based on environment
        let (mut temp, mut humidity) = match environment {
            EnvironmentScenario::IndoorHVAC { setpoint, .. } => (setpoint, 50.0),
            EnvironmentScenario::OutdoorWeather { .. } => (20.0, 65.0),
            EnvironmentScenario::Industrial { process_temp, .. } => (process_temp * 0.7, 30.0),
            EnvironmentScenario::Greenhouse { day_temp, humidity_target, .. } => (day_temp, humidity_target),
        };
        
        let sample_interval_ms = 3600 * 1000 / samples_per_hour;
        
        for hour in 0..duration_hours {
            for _sample in 0..samples_per_hour {
                self.time += sample_interval_ms as u64;
                
                // Update temperature based on environment
                let (temp_target, humidity_target) = self.get_environmental_targets(&environment, hour as f32);
                
                // Temperature dynamics
                temp += (temp_target - temp) / temp_model.time_constant;
                temp += self.gaussian_noise(temp_model.noise_std);
                
                // Humidity dynamics with dew point constraint
                let dew_point = dew_point_lookup(temp, humidity).unwrap_or(temp - 5.0);
                let max_humidity = 100.0 * (dew_point + 5.0) / temp; // Rough approximation
                
                humidity += (humidity_target - humidity) / humidity_model.time_constant;
                humidity += self.gaussian_noise(humidity_model.noise_std);
                humidity = humidity.clamp(HUMIDITY_MIN, max_humidity.min(HUMIDITY_MAX));
                
                // Add sensor readings
                events.push(
                    EventBuilder::new(self.time)
                        .sensor(temp_sensor, SensorType::Temperature)
                        .reading(temp, 0.95)
                        .unwrap()
                );
                
                events.push(
                    EventBuilder::new(self.time)
                        .sensor(humidity_sensor, SensorType::Humidity)
                        .reading(humidity, 0.93)
                        .unwrap()
                );
            }
        }
        
        events
    }
    
    /// Generate pressure data with realistic weather patterns
    pub fn generate_pressure_with_weather(
        &mut self,
        sensor_id: &str,
        base_altitude_m: f32,
        weather_systems: &[(u32, f32, f32)], // (hour, pressure_change, duration_hours)
        duration_hours: u32,
        samples_per_hour: u32,
        sensor_model: &SensorModel,
    ) -> Vec<Event> {
        let mut events = Vec::new();
        
        // Base pressure at altitude
        let base_pressure = 1013.25 - base_altitude_m * physics::PRESSURE_LAPSE_RATE;
        let sample_interval_ms = 3600 * 1000 / samples_per_hour;
        
        for hour in 0..duration_hours {
            for _sample in 0..samples_per_hour {
                self.time += sample_interval_ms as u64;
                
                // Calculate weather effect
                let mut weather_effect = 0.0;
                for &(system_hour, pressure_change, duration) in weather_systems {
                    if hour >= system_hour && hour < system_hour + duration as u32 {
                        // Smooth transition using cosine
                        let phase = (hour - system_hour) as f32 / duration;
                        let angle = phase * 3.14159;
                        weather_effect += pressure_change * (1.0 - cos_lookup(angle).unwrap_or(1.0)) / 2.0;
                    }
                }
                
                // Add diurnal variation (small)
                let diurnal_angle = 2.0 * 3.14159 * hour as f32 / 24.0;
                let diurnal = 2.0 * sin_lookup(diurnal_angle).unwrap_or(0.0);
                
                let pressure = base_pressure + weather_effect + diurnal + self.gaussian_noise(sensor_model.noise_std);
                let clamped = pressure.clamp(PRESSURE_MIN_HPA, PRESSURE_MAX_HPA);
                
                events.push(
                    EventBuilder::new(self.time)
                        .sensor(sensor_id, SensorType::Pressure)
                        .reading(clamped, 0.98)
                        .unwrap()
                );
            }
        }
        
        events
    }
    
    /// Generate multi-sensor fusion test scenario
    pub fn generate_fusion_scenario(
        &mut self,
        sensors: &[(&str, SensorType, SensorModel)],
        correlation_matrix: &[&[f32]], // Correlation between sensors
        duration_hours: u32,
        samples_per_hour: u32,
        anomaly_periods: &[(usize, u32, u32)], // (sensor_idx, start_hour, end_hour)
    ) -> Vec<Event> {
        let mut events = Vec::new();
        let n_sensors = sensors.len();
        let sample_interval_ms = 3600 * 1000 / samples_per_hour;
        
        // State for each sensor
        let mut states: Vec<f32> = sensors.iter().map(|(_, sensor_type, _)| {
            match sensor_type {
                SensorType::Temperature => 25.0,
                SensorType::Humidity => 60.0,
                SensorType::Pressure => 1013.0,
                _ => 0.0,
            }
        }).collect();
        
        for hour in 0..duration_hours {
            for _sample in 0..samples_per_hour {
                self.time += sample_interval_ms as u64;
                
                // Generate correlated noise
                let mut noise = Vec::new();
                for i in 0..n_sensors {
                    let mut corr_noise = 0.0;
                    for j in 0..n_sensors {
                        if i != j && j < correlation_matrix[i].len() {
                            corr_noise += correlation_matrix[i][j] * self.gaussian_noise(1.0);
                        }
                    }
                    noise.push(self.gaussian_noise(1.0) + corr_noise);
                }
                
                // Update each sensor
                for (idx, ((sensor_id, sensor_type, model), noise_val)) in 
                    sensors.iter().zip(noise.iter()).enumerate() 
                {
                    // Check if in anomaly period
                    let mut in_anomaly = false;
                    for &(anomaly_idx, start, end) in anomaly_periods {
                        if anomaly_idx == idx && hour >= start && hour < end {
                            in_anomaly = true;
                            break;
                        }
                    }
                    
                    // Update state with drift and noise
                    states[idx] += model.drift_rate / 3600.0;
                    let value = if in_anomaly {
                        // Generate anomalous reading
                        states[idx] + self.random_float() * 20.0 - 10.0
                    } else {
                        states[idx] + noise_val * model.noise_std
                    };
                    
                    let quality = if in_anomaly { 0.5 } else { 0.95 };
                    
                    events.push(
                        EventBuilder::new(self.time)
                            .sensor(sensor_id, *sensor_type)
                            .reading(value, quality)
                            .unwrap()
                    );
                }
            }
        }
        
        events
    }
    
    // Helper methods
    
    fn interpolate_profile(&self, profile: &[(u32, f32)], hour: f32) -> f32 {
        if profile.is_empty() {
            return 0.0;
        }
        
        // Find surrounding points
        let mut prev = (0u32, profile[0].1);
        let mut next = profile[profile.len() - 1];
        
        for &point in profile {
            if point.0 as f32 <= hour {
                prev = point;
            }
            if point.0 as f32 >= hour {
                next = point;
                break;
            }
        }
        
        // Linear interpolation
        if prev.0 == next.0 {
            prev.1
        } else {
            let t = (hour - prev.0 as f32) / (next.0 - prev.0) as f32;
            prev.1 + t * (next.1 - prev.1)
        }
    }
    
    fn get_environmental_targets(&self, scenario: &EnvironmentScenario, hour: f32) -> (f32, f32) {
        match scenario {
            EnvironmentScenario::IndoorHVAC { setpoint, deadband } => {
                // Simple HVAC cycling
                let cycle = (hour * 2.0) % 1.0;
                let temp = if cycle < 0.5 {
                    setpoint - deadband
                } else {
                    setpoint + deadband
                };
                (temp, 45.0 + 10.0 * sin_lookup(hour * 0.1).unwrap_or(0.0))
            }
            EnvironmentScenario::OutdoorWeather { latitude, season } => {
                // Diurnal temperature variation based on season and latitude
                let day_length = match season {
                    Season::Summer => 14.0 + latitude.abs() / 90.0 * 4.0,
                    Season::Winter => 10.0 - latitude.abs() / 90.0 * 4.0,
                    _ => 12.0,
                };
                
                let sunrise = 12.0 - day_length / 2.0;
                let sunset = 12.0 + day_length / 2.0;
                
                let (base_temp, temp_range) = match season {
                    Season::Summer => (25.0, 10.0),
                    Season::Winter => (5.0, 5.0),
                    Season::Spring | Season::Autumn => (15.0, 8.0),
                };
                
                let temp = if hour >= sunrise && hour <= sunset {
                    let day_phase = (hour - sunrise) / day_length;
                    base_temp + temp_range * sin_lookup(day_phase * 3.14159).unwrap_or(0.5)
                } else {
                    base_temp - 2.0
                };
                
                let humidity = 80.0 - temp; // Rough inverse correlation
                (temp, humidity)
            }
            EnvironmentScenario::Industrial { process_temp, cycle_time_hours } => {
                let cycle_phase = (hour % cycle_time_hours) / cycle_time_hours;
                let temp = process_temp * (0.5 + 0.5 * sin_lookup(cycle_phase * 2.0 * 3.14159).unwrap_or(0.0));
                let humidity = 30.0 - temp / process_temp * 10.0;
                (temp, humidity)
            }
            EnvironmentScenario::Greenhouse { day_temp, night_temp, humidity_target } => {
                let is_day = hour % 24.0 >= 6.0 && hour % 24.0 < 18.0;
                let temp = if is_day { day_temp } else { night_temp };
                (*temp, *humidity_target)
            }
        }
    }
    
    fn gaussian_noise(&mut self, std_dev: f32) -> f32 {
        // Box-Muller transform for Gaussian distribution
        let u1 = self.random_float();
        let u2 = self.random_float();
        
        // Approximate sqrt using Newton's method for no_std
        let sqrt_val = self.fast_sqrt(-2.0 * self.fast_ln(u1));
        let angle = 2.0 * 3.14159 * u2;
        
        sqrt_val * cos_lookup(angle).unwrap_or(0.0) * std_dev
    }
    
    fn random_float(&mut self) -> f32 {
        // Linear congruential generator
        self.seed = self.seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.seed >> 8) as f32 / 16777216.0
    }
    
    fn fast_sqrt(&self, x: f32) -> f32 {
        // Newton's method approximation
        if x <= 0.0 { return 0.0; }
        let mut guess = x;
        for _ in 0..4 {
            guess = (guess + x / guess) * 0.5;
        }
        guess
    }
    
    fn fast_ln(&self, x: f32) -> f32 {
        // Rough approximation for ln(x) around 0.5-1.0
        // ln(x) ≈ (x - 1) - (x - 1)²/2 + (x - 1)³/3
        let t = x - 1.0;
        t - t * t / 2.0 + t * t * t / 3.0
    }
}