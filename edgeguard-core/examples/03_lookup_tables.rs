//! Lookup Tables Example
//!
//! This example demonstrates how EdgeGuard uses pre-computed lookup tables
//! to achieve high performance on resource-constrained embedded devices.
//!
//! ## What You'll Learn
//!
//! - How lookup tables replace expensive computations
//! - Performance comparison: lookup vs calculation
//! - Accuracy trade-offs with different table sizes
//! - Memory usage optimization strategies
//!
//! ## Why Lookup Tables?
//!
//! On embedded devices without FPU (Floating Point Unit):
//! - `ln()` can take ~1000 cycles
//! - `exp()` can take ~1500 cycles
//! - Lookup table access: ~10 cycles
//!
//! That's a 100x speedup for critical calculations!
//!
//! ## Running the Example
//!
//! ```bash
//! cargo run --example 03_lookup_tables
//! ```

use edgeguard_core::{
    lookup::{
        DEW_POINT_STANDARD, AltitudeTable,
        DEW_POINT_TABLE_ROWS, DEW_POINT_TABLE_COLS,
        LookupError,
    },
};
use std::time::Instant;

fn main() {
    println!("EdgeGuard Lookup Tables Example");
    println!("===============================\n");

    // Demonstrate dew point calculation
    println!("Dew Point Lookup Table:");
    println!("-----------------------");
    demo_dew_point_lookup();
    
    println!("\n\nAltitude Compensation Table:");
    println!("----------------------------");
    demo_altitude_compensation();
    
    println!("\n\nPerformance Comparison:");
    println!("----------------------");
    benchmark_lookup_vs_calculation();
    
    println!("\n\nMemory Usage Analysis:");
    println!("---------------------");
    analyze_memory_usage();
    
    println!("\n\nAccuracy Analysis:");
    println!("-----------------");
    analyze_accuracy();
}

fn demo_dew_point_lookup() {
    let test_cases = [
        (25.0, 60.0, "Comfortable room"),
        (30.0, 80.0, "Hot and humid"),
        (10.0, 40.0, "Cool and dry"),
        (35.0, 90.0, "Tropical conditions"),
        (-5.0, 80.0, "Cold with frost risk"),
    ];
    
    for (temp, humidity, condition) in &test_cases {
        match DEW_POINT_STANDARD.lookup(*temp, *humidity) {
            Ok(dew_point) => {
                println!("  {} - Temp: {:.1}°C, RH: {:.0}%", condition, temp, humidity);
                println!("    → Dew point: {:.1}°C", dew_point);
                
                // Check for condensation risk
                let margin = temp - dew_point;
                if margin < 2.0 {
                    println!("    ⚠ WARNING: Condensation risk! (margin: {:.1}°C)", margin);
                }
            }
            Err(e) => {
                println!("  Error for {}°C, {}%: {:?}", temp, humidity, e);
            }
        }
    }
}

fn demo_altitude_compensation() {
    println!("  Pressure compensation factors for different altitudes:");
    println!("  (Used to adjust pressure sensor readings)\n");
    
    let altitudes = [0.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0];
    
    let altitude_table = &AltitudeTable::STANDARD;
    
    for altitude in &altitudes {
        match altitude_table.get_adjustment(*altitude) {
            Ok(adjustment) => {
                let sea_level_pressure = 1013.25;
                let actual_pressure = sea_level_pressure + adjustment;
                let factor = actual_pressure / sea_level_pressure;
                println!("    {:4.0}m: adjustment = {:.1} hPa, pressure ≈ {:.1} hPa", 
                         altitude, adjustment, actual_pressure);
            }
            Err(e) => {
                println!("    {:4.0}m: Error - {:?}", altitude, e);
            }
        }
    }
}

fn benchmark_lookup_vs_calculation() {
    const ITERATIONS: usize = 100_000;
    
    // Benchmark lookup table
    let start = Instant::now();
    for i in 0..ITERATIONS {
        let temp = 20.0 + (i as f32 % 20.0);
        let humidity = 40.0 + (i as f32 % 50.0);
        let _ = DEW_POINT_STANDARD.lookup(temp, humidity);
    }
    let lookup_time = start.elapsed();
    
    // Benchmark calculation (simplified Magnus formula)
    let start = Instant::now();
    for i in 0..ITERATIONS {
        let temp = 20.0 + (i as f32 % 20.0);
        let humidity = 40.0 + (i as f32 % 50.0);
        let _ = calculate_dew_point(temp, humidity);
    }
    let calc_time = start.elapsed();
    
    println!("  {} iterations:", ITERATIONS);
    println!("  Lookup table: {:?}", lookup_time);
    println!("  Calculation:  {:?}", calc_time);
    println!("  Speedup:      {:.1}x faster", 
             calc_time.as_secs_f64() / lookup_time.as_secs_f64());
}

fn analyze_memory_usage() {
    // Calculate memory usage for our tables
    let dew_point_size = DEW_POINT_TABLE_ROWS * DEW_POINT_TABLE_COLS * 4; // f32 = 4 bytes
    let altitude_size = 500 * 4; // Approximate size based on standard altitude table
    
    println!("  Dew Point Table:");
    println!("    Dimensions: {}x{} (temp × humidity)", DEW_POINT_TABLE_ROWS, DEW_POINT_TABLE_COLS);
    println!("    Memory:     {} bytes", dew_point_size);
    println!("    Coverage:   -10°C to 40°C, 0% to 100% RH");
    
    println!("\n  Altitude Table:");
    println!("    Entries:    ~125");
    println!("    Memory:     {} bytes", altitude_size);
    println!("    Coverage:   0m to 4000m");
    
    let total = dew_point_size + altitude_size;
    println!("\n  Total lookup table memory: {} bytes ({:.1} KB)", total, total as f32 / 1024.0);
    
    // Compare to code size
    println!("\n  For comparison:");
    println!("    - libm (math library): ~50 KB");
    println!("    - Our tables: {:.1} KB", total as f32 / 1024.0);
    println!("    - Savings: ~{:.0} KB", 50.0 - (total as f32 / 1024.0));
}

fn analyze_accuracy() {
    println!("  Comparing lookup table accuracy to exact calculation:\n");
    
    let test_points = [
        (20.0, 50.0),
        (25.0, 60.0),
        (30.0, 70.0),
        (15.0, 45.0),
        (35.0, 85.0),
    ];
    
    for (temp, humidity) in &test_points {
        let lookup_result = DEW_POINT_STANDARD.lookup(*temp, *humidity).unwrap_or(f32::NAN);
        let exact_result = calculate_dew_point(*temp, *humidity);
        let error = (lookup_result - exact_result).abs();
        
        println!("    T={:.0}°C, RH={:.0}%:", temp, humidity);
        println!("      Lookup: {:.2}°C", lookup_result);
        println!("      Exact:  {:.2}°C", exact_result);
        println!("      Error:  {:.2}°C ({:.1}%)", error, error / exact_result.abs() * 100.0);
    }
    
    println!("\n  Note: Errors < 0.5°C are negligible for most applications");
}

// Helper function for exact dew point calculation
fn calculate_dew_point(temp: f32, humidity: f32) -> f32 {
    let a = 17.27;
    let b = 237.7;
    let alpha = ((a * temp) / (b + temp)) + (humidity / 100.0).ln();
    (b * alpha) / (a - alpha)
}