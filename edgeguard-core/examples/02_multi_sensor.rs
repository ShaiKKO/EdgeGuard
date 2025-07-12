//! Multi-Sensor Validation with Cross-Validation Example
//!
//! This example demonstrates how EdgeGuard handles multiple sensors
//! and validates relationships between them using physics laws.
//!
//! ## What You'll Learn
//!
//! - Working with multiple sensor types
//! - Cross-sensor validation (e.g., dew point)
//! - Building contexts with environmental data
//! - Understanding cross-validation failures
//!
//! ## Physical Relationships
//!
//! 1. **Dew Point**: Temperature at which water vapor condenses
//!    - Must be <= air temperature (physics law)
//!    - Calculated from temperature and humidity
//!
//! 2. **Pressure-Altitude**: Pressure decreases with altitude
//!    - Sea level: ~1013 hPa
//!    - Every 100m up: ~12 hPa drop
//!
//! ## Running the Example
//!
//! ```bash
//! cargo run --example 02_multi_sensor
//! ```

use edgeguard_core::{
    validators::{TemperatureValidator, HumidityValidator, PressureValidator},
    traits::{Validator, ValidationContext},
};

fn main() {
    println!("EdgeGuard Multi-Sensor Validation Example");
    println!("=========================================\n");

    // Create validators for different sensor types
    let temp_validator = TemperatureValidator::default();
    let humidity_validator = HumidityValidator::default();
    let pressure_validator = PressureValidator::default();

    // Create contexts for each sensor (in real use, might share context)
    let mut temp_context = ValidationContext::default();
    let mut humidity_context = ValidationContext::default();
    let mut pressure_context = ValidationContext::default();

    // Simulate some historical data
    println!("Setting up sensor history...\n");
    
    // Temperature history: gradual warming
    temp_context.add_reading(20.0, 1000);
    temp_context.add_reading(20.5, 2000);
    temp_context.add_reading(21.0, 3000);
    temp_context.timestamp = 4000;
    
    // Humidity history: increasing humidity
    humidity_context.add_reading(50.0, 1000);
    humidity_context.add_reading(55.0, 2000);
    humidity_context.add_reading(60.0, 3000);
    humidity_context.timestamp = 4000;
    
    // Pressure history: stable pressure
    pressure_context.add_reading(1013.0, 1000);
    pressure_context.add_reading(1013.5, 2000);
    pressure_context.add_reading(1013.2, 3000);
    pressure_context.timestamp = 4000;

    // Test individual sensor readings
    println!("Individual Sensor Validation:");
    println!("-----------------------------");
    
    let temperature = 22.0;
    let humidity = 65.0;
    let pressure = 1013.0;
    
    match temp_validator.validate(temperature, &temp_context) {
        Ok(()) => println!("✓ Temperature: {:.1}°C - VALID", temperature),
        Err(e) => println!("✗ Temperature: {:.1}°C - INVALID: {:?}", temperature, e),
    }
    
    match humidity_validator.validate(humidity, &humidity_context) {
        Ok(()) => println!("✓ Humidity: {:.1}% - VALID", humidity),
        Err(e) => println!("✗ Humidity: {:.1}% - INVALID: {:?}", humidity, e),
    }
    
    match pressure_validator.validate(pressure, &pressure_context) {
        Ok(()) => println!("✓ Pressure: {:.1} hPa - VALID", pressure),
        Err(e) => println!("✗ Pressure: {:.1} hPa - INVALID: {:?}", pressure, e),
    }

    // Cross-sensor validation scenarios
    println!("\n\nCross-Sensor Validation Scenarios:");
    println!("----------------------------------\n");

    // Scenario 1: Valid conditions
    println!("Scenario 1: Normal summer day");
    println!("  Temperature: 25°C, Humidity: 60%");
    validate_dew_point(25.0, 60.0);
    
    // Scenario 2: Impossible dew point
    println!("\nScenario 2: Impossible conditions");
    println!("  Temperature: 10°C, Humidity: 100%");
    println!("  (Dew point would exceed air temperature)");
    validate_dew_point(10.0, 100.0);
    
    // Scenario 3: Freezing with high humidity
    println!("\nScenario 3: Freezing conditions");
    println!("  Temperature: -5°C, Humidity: 90%");
    println!("  (Risk of ice formation)");
    validate_dew_point(-5.0, 90.0);
    
    // Scenario 4: Desert conditions
    println!("\nScenario 4: Desert conditions");
    println!("  Temperature: 45°C, Humidity: 10%");
    validate_dew_point(45.0, 10.0);

    // Demonstrate pressure-altitude relationship
    println!("\n\nPressure-Altitude Validation:");
    println!("-----------------------------");
    
    println!("Sea level pressure: 1013.25 hPa");
    validate_altitude_pressure(0.0, 1013.25);
    
    println!("\nMountain station (2000m): ~795 hPa expected");
    validate_altitude_pressure(2000.0, 795.0);
    
    println!("\nImpossible reading (2000m with sea-level pressure):");
    validate_altitude_pressure(2000.0, 1013.0);

    println!("\n{}", "=".repeat(60));
    println!("Key Insights:");
    println!("- Individual sensors may pass but together violate physics");
    println!("- Dew point must never exceed air temperature");
    println!("- Pressure and altitude have a fixed relationship");
    println!("- Cross-validation catches subtle environment errors");
}

fn validate_dew_point(temperature: f32, humidity: f32) {
    // Simplified dew point calculation (Magnus formula)
    let a = 17.27;
    let b = 237.7;
    let alpha = ((a * temperature) / (b + temperature)) + (humidity / 100.0).ln();
    let dew_point = (b * alpha) / (a - alpha);
    
    print!("  Calculated dew point: {:.1}°C ... ", dew_point);
    
    if dew_point > temperature {
        println!("✗ INVALID: Dew point exceeds air temperature!");
    } else if temperature < 0.0 && dew_point > temperature - 2.0 {
        println!("⚠ WARNING: Ice formation likely");
    } else {
        println!("✓ VALID");
    }
}

fn validate_altitude_pressure(altitude_m: f32, pressure_hpa: f32) {
    // Barometric formula (simplified)
    let sea_level_pressure = 1013.25;
    let expected_pressure = sea_level_pressure * (1.0 - 0.0065 * altitude_m / 288.15).powf(5.255);
    let tolerance = 20.0; // hPa tolerance for weather variations
    
    print!("  Altitude: {:.0}m, Pressure: {:.1} hPa ... ", altitude_m, pressure_hpa);
    
    if (pressure_hpa - expected_pressure).abs() < tolerance {
        println!("✓ VALID (expected ~{:.1} hPa)", expected_pressure);
    } else {
        println!("✗ INVALID (expected ~{:.1} hPa, diff: {:.1} hPa)", 
                 expected_pressure, 
                 (pressure_hpa - expected_pressure).abs());
    }
}