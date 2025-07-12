//! Basic Sensor Validation Example
//!
//! This example demonstrates the simplest use case of EdgeGuard:
//! validating temperature sensor readings using physics-based constraints.
//!
//! ## What You'll Learn
//!
//! - Creating a validator with physical constraints
//! - Building validation context with sensor history
//! - Validating individual readings
//! - Understanding validation errors
//!
//! ## Running the Example
//!
//! ```bash
//! cargo run --example 01_basic_validation
//! ```

use edgeguard_core::{
    validators::TemperatureValidator,
    traits::{Validator, ValidationContext},
    errors::ValidationError,
};

fn main() {
    println!("EdgeGuard Basic Validation Example");
    println!("==================================\n");

    // Create a temperature validator with realistic constraints
    // Min: -40°C (cold weather)
    // Max: 50°C (hot weather)
    // Max rate: 2°C/second (reasonable for air temperature)
    let validator = TemperatureValidator::new_with_limits(-40.0, 50.0, 2.0);

    // Show the validator's constraints
    let constraints = validator.constraints();
    println!("Temperature Validator Constraints:");
    println!("  Min value: {}°C", constraints.min_value);
    println!("  Max value: {}°C", constraints.max_value);
    println!("  Max rate of change: {}°C/s", constraints.max_rate_change);
    println!();

    // Create a validation context to track sensor history
    let mut context = ValidationContext::default();
    
    // Simulate some historical readings
    println!("Adding historical readings:");
    let historical_readings = [
        (20.0, 1000),  // 20°C at t=1000ms
        (20.5, 2000),  // 20.5°C at t=2000ms
        (21.0, 3000),  // 21°C at t=3000ms
    ];
    
    for (temp, time) in &historical_readings {
        context.add_reading(*temp, *time);
        println!("  t={:4}ms: {:.1}°C", time, temp);
    }
    
    // Set current timestamp and sensor quality
    context.timestamp = 4000;
    context.sensor_quality = 0.95; // 95% quality (good sensor)
    
    println!("\nCurrent context:");
    println!("  Timestamp: {}ms", context.timestamp);
    println!("  Sensor quality: {:.0}%", context.sensor_quality * 100.0);
    println!();

    // Test various readings
    println!("Testing various readings:\n");
    
    let test_cases = [
        (21.5, "Normal reading (0.5°C increase)"),
        (22.5, "Acceptable rate (1.5°C/s)"),
        (26.0, "Too fast! (5.5°C/s)"),
        (-50.0, "Too cold! (below minimum)"),
        (60.0, "Too hot! (above maximum)"),
        (f32::NAN, "Invalid value (NaN)"),
    ];
    
    for (value, description) in &test_cases {
        print!("{:.<50} ", description);
        
        match validator.validate(*value, &context) {
            Ok(()) => {
                println!("✓ VALID ({:.1}°C)", value);
                // In real usage, you would add this reading to context
                // context.add_reading(*value, context.timestamp);
            }
            Err(e) => {
                println!("✗ INVALID ({:.1}°C)", value);
                println!("    Reason: {}", format_error(&e));
            }
        }
    }
    
    println!("\n{}", "=".repeat(60));
    println!("Key Insights:");
    println!("- EdgeGuard validates both absolute limits and rate of change");
    println!("- Historical context is crucial for detecting anomalies");
    println!("- Invalid values (NaN, Inf) are caught automatically");
    println!("- Physics-based validation prevents impossible readings");
}

fn format_error(error: &ValidationError) -> String {
    match error {
        ValidationError::OutOfRange { value, min, max } => {
            format!("Out of range: {} not in [{}, {}]", value, min, max)
        }
        ValidationError::RateExceeded { rate, max_rate } => {
            format!(
                "Rate exceeded: {:.1}°C/s (max: {}°C/s)",
                rate, max_rate
            )
        }
        ValidationError::InvalidValue => "Invalid value (NaN or Infinity)".to_string(),
        _ => format!("{:?}", error),
    }
}