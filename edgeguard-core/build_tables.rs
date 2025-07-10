#!/usr/bin/env rust-script
//! Generate lookup tables for EdgeGuard
//!
//! This script generates the lookup tables used in `src/lookup.rs`.
//! Run with: `cargo script build_tables.rs -- [options]`
//!
//! Options:
//!   --config <name>     Table configuration: standard, high_precision, low_memory
//!   --temp-min <val>    Minimum temperature in Celsius
//!   --temp-max <val>    Maximum temperature in Celsius
//!   --temp-step <val>   Temperature step size
//!   --rh-step <val>     Humidity step size
//!   --alt-min <val>     Minimum altitude in meters
//!   --alt-max <val>     Maximum altitude in meters
//!   --alt-step <val>    Altitude step size
//!   --output <path>     Output file path
//!   --append            Append to existing file instead of overwriting

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::env;
use std::process;

/// Magnus formula constants
const MAGNUS_A: f64 = 17.27;
const MAGNUS_B: f64 = 237.7;

/// Calculate dew point using Magnus formula
fn calculate_dew_point(temp_c: f64, rh_percent: f64) -> f64 {
    if rh_percent == 0.0 {
        // Return arbitrary low value since dew point is undefined at 0% RH
        // -100°C is well below any realistic temperature
        return -100.0;
    }
    
    let gamma = (rh_percent / 100.0).ln() + (MAGNUS_A * temp_c) / (MAGNUS_B + temp_c);
    (MAGNUS_B * gamma) / (MAGNUS_A - gamma)
}

/// Calculate pressure adjustment for altitude using standard atmosphere
fn calculate_pressure_adjustment(altitude_m: f64) -> f64 {
    const SEA_LEVEL_PRESSURE: f64 = 1013.25;
    const TEMP_LAPSE_RATE: f64 = 0.0065; // K/m
    const SEA_LEVEL_TEMP: f64 = 288.15; // K
    const GAS_CONSTANT: f64 = 8.3144598;
    const MOLAR_MASS_AIR: f64 = 0.0289644; // kg/mol
    const GRAVITY: f64 = 9.80665; // m/s²
    
    // Standard atmosphere model: P = P0 * (1 - L*h/T0)^(g*M/R*L)
    // Where the exponent (g*M/R*L) ≈ 5.255 represents how pressure
    // decreases with altitude due to decreasing air density
    let pressure = SEA_LEVEL_PRESSURE * 
        (1.0 - (TEMP_LAPSE_RATE * altitude_m) / SEA_LEVEL_TEMP)
        .powf((GRAVITY * MOLAR_MASS_AIR) / (GAS_CONSTANT * TEMP_LAPSE_RATE));
    
    SEA_LEVEL_PRESSURE - pressure
}

/// Table configuration presets
#[derive(Debug, Clone)]
struct TableConfig {
    name: String,
    temp_min: i32,
    temp_max: i32,
    temp_step: i32,
    rh_min: i32,
    rh_max: i32,
    rh_step: i32,
    alt_min: i32,
    alt_max: i32,
    alt_step: i32,
}

impl TableConfig {
    fn standard() -> Self {
        Self {
            name: "STANDARD".to_string(),
            temp_min: -40,
            temp_max: 50,
            temp_step: 5,
            rh_min: 0,
            rh_max: 100,
            rh_step: 10,
            alt_min: -500,
            alt_max: 5000,
            alt_step: 100,
        }
    }
    
    fn high_precision() -> Self {
        Self {
            name: "FINE_STEP".to_string(),  // Align with lookup.rs naming
            temp_min: -40,
            temp_max: 50,
            temp_step: 2,
            rh_min: 0,
            rh_max: 100,
            rh_step: 5,
            alt_min: -500,
            alt_max: 5000,
            alt_step: 50,
        }
    }
    
    fn low_memory() -> Self {
        Self {
            name: "LOW_MEMORY".to_string(),
            temp_min: -40,
            temp_max: 50,
            temp_step: 10,
            rh_min: 0,
            rh_max: 100,
            rh_step: 20,
            alt_min: -500,
            alt_max: 5000,
            alt_step: 200,
        }
    }
    
    fn validate(&self) -> Result<(), String> {
        if self.temp_step <= 0 {
            return Err("Temperature step must be positive".to_string());
        }
        if self.temp_min >= self.temp_max {
            return Err("Temperature min must be less than max".to_string());
        }
        if self.rh_step <= 0 {
            return Err("Humidity step must be positive".to_string());
        }
        if self.alt_step <= 0 {
            return Err("Altitude step must be positive".to_string());
        }
        if self.alt_min >= self.alt_max {
            return Err("Altitude min must be less than max".to_string());
        }
        Ok(())
    }
}

fn parse_args() -> Result<(TableConfig, String, bool), String> {
    let args: Vec<String> = env::args().collect();
    let mut config = TableConfig::standard();
    let mut output_path = "generated_tables.rs".to_string();
    let mut append = false;
    
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                i += 1;
                if i >= args.len() {
                    return Err("--config requires a value".to_string());
                }
                config = match args[i].as_str() {
                    "standard" => TableConfig::standard(),
                    "high_precision" => TableConfig::high_precision(),
                    "low_memory" => TableConfig::low_memory(),
                    _ => return Err(format!("Unknown config: {}", args[i])),
                };
            }
            "--temp-min" => {
                i += 1;
                config.temp_min = args.get(i)
                    .ok_or("--temp-min requires a value")?
                    .parse()
                    .map_err(|_| "Invalid temp-min value")?;
            }
            "--temp-max" => {
                i += 1;
                config.temp_max = args.get(i)
                    .ok_or("--temp-max requires a value")?
                    .parse()
                    .map_err(|_| "Invalid temp-max value")?;
            }
            "--temp-step" => {
                i += 1;
                config.temp_step = args.get(i)
                    .ok_or("--temp-step requires a value")?
                    .parse()
                    .map_err(|_| "Invalid temp-step value")?;
            }
            "--rh-step" => {
                i += 1;
                config.rh_step = args.get(i)
                    .ok_or("--rh-step requires a value")?
                    .parse()
                    .map_err(|_| "Invalid rh-step value")?;
            }
            "--alt-min" => {
                i += 1;
                config.alt_min = args.get(i)
                    .ok_or("--alt-min requires a value")?
                    .parse()
                    .map_err(|_| "Invalid alt-min value")?;
            }
            "--alt-max" => {
                i += 1;
                config.alt_max = args.get(i)
                    .ok_or("--alt-max requires a value")?
                    .parse()
                    .map_err(|_| "Invalid alt-max value")?;
            }
            "--alt-step" => {
                i += 1;
                config.alt_step = args.get(i)
                    .ok_or("--alt-step requires a value")?
                    .parse()
                    .map_err(|_| "Invalid alt-step value")?;
            }
            "--output" => {
                i += 1;
                output_path = args.get(i)
                    .ok_or("--output requires a value")?
                    .to_string();
            }
            "--append" => {
                append = true;
            }
            _ => {
                return Err(format!("Unknown argument: {}", args[i]));
            }
        }
        i += 1;
    }
    
    config.validate()?;
    Ok((config, output_path, append))
}

fn main() {
    let (config, output_path, append) = match parse_args() {
        Ok(result) => result,
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!("Usage: build_tables.rs [options]");
            eprintln!("Run with --help for more information");
            process::exit(1);
        }
    };
    
    // Calculate table dimensions
    let dew_point_rows = ((config.temp_max - config.temp_min) / config.temp_step + 1) as usize;
    let dew_point_cols = ((config.rh_max - config.rh_min) / config.rh_step + 1) as usize;
    let altitude_entries = ((config.alt_max - config.alt_min) / config.alt_step + 1) as usize;
    
    println!("Generating {} tables...", config.name);
    println!("Temperature range: {}°C to {}°C, step: {}°C", 
        config.temp_min, config.temp_max, config.temp_step);
    println!("Humidity range: {}% to {}%, step: {}%", 
        config.rh_min, config.rh_max, config.rh_step);
    println!("Altitude range: {}m to {}m, step: {}m", 
        config.alt_min, config.alt_max, config.alt_step);
    
    // Pre-allocate table data for better performance
    let mut dew_point_table = Vec::with_capacity(dew_point_rows);
    let mut altitude_table = Vec::with_capacity(altitude_entries);
    
    // Show progress for large tables
    let show_progress = dew_point_rows * dew_point_cols > 1000;
    
    // Generate dew point table with progress
    println!("\nGenerating dew point table...");
    for (i, temp) in (config.temp_min..=config.temp_max)
        .step_by(config.temp_step as usize)
        .enumerate() 
    {
        if show_progress && i % 10 == 0 {
            println!("  Progress: {}/{} rows", i, dew_point_rows);
        }
        
        let mut row = Vec::with_capacity(dew_point_cols);
        for rh in (config.rh_min..=config.rh_max).step_by(config.rh_step as usize) {
            let dp = calculate_dew_point(temp as f64, rh as f64);
            row.push(dp.round() as i8);
        }
        dew_point_table.push(row);
    }
    
    // Generate altitude table
    println!("\nGenerating altitude table...");
    for (i, alt) in (config.alt_min..=config.alt_max)
        .step_by(config.alt_step as usize)
        .enumerate() 
    {
        if show_progress && i % 20 == 0 {
            println!("  Progress: {}/{} entries", i, altitude_entries);
        }
        let adj = -calculate_pressure_adjustment(alt as f64);
        altitude_table.push(adj as f32);
    }
    
    // Write output
    println!("\nWriting to {}...", output_path);
    let mut output = String::new();
    
    // Header
    output.push_str(&format!(
        "// Auto-generated lookup tables for EdgeGuard
\
        // Configuration: {}
\
        // Temperature: {}..{} step {}, Humidity: {}..{} step {}
\
        // Generated on: {}
\n",
        config.name,
        config.temp_min, config.temp_max, config.temp_step,
        config.rh_min, config.rh_max, config.rh_step,
        "(use --timestamp for current time)"
    ));
    
    // Size constants
    output.push_str(&format!(
        "/// Dew point table dimensions for {}
\
        pub const DEW_POINT_{}_ROWS: usize = {};
\
        pub const DEW_POINT_{}_COLS: usize = {};
\n",
        config.name, config.name, dew_point_rows, config.name, dew_point_cols
    ));
    
    // Dew point table
    output.push_str(&format!(
        "/// Dew point values for {} configuration
\
        pub const DEW_POINT_VALUES_{}: [[i8; {}]; {}] = [",
        config.name, config.name, dew_point_cols, dew_point_rows
    ));
    
    for (i, row) in dew_point_table.iter().enumerate() {
        let temp = config.temp_min + (i as i32) * config.temp_step;
        output.push_str(&format!("\n    // {}°C\n    [", temp));
        
        for (j, &value) in row.iter().enumerate() {
            output.push_str(&format!("{}", value));
            if j < row.len() - 1 {
                output.push_str(", ");
            }
        }
        
        output.push(']');
        if i < dew_point_table.len() - 1 {
            output.push(',');
        }
    }
    output.push_str("\n];\n");
    
    // Altitude table size constant
    output.push_str(&format!(
        "\n/// Altitude adjustment table size for {}\n\
        pub const ALTITUDE_{}_ENTRIES: usize = {};\n\n",
        config.name, config.name, altitude_entries
    ));
    
    // Altitude adjustment table
    output.push_str(&format!(
        "/// Altitude pressure adjustments for {} configuration\n\
        pub const ALTITUDE_ADJUSTMENTS_{}: [f32; {}] = [",
        config.name, config.name, altitude_entries
    ));
    
    let mut line_items = 0;
    for (i, &adj) in altitude_table.iter().enumerate() {
        if line_items == 0 {
            output.push_str("\n    ");
        }
        
        output.push_str(&format!("{:.1}", adj));
        
        if i < altitude_table.len() - 1 {
            output.push_str(", ");
            line_items += 1;
            
            if line_items >= 5 {
                let alt = config.alt_min + (i as i32) * config.alt_step;
                let start_alt = alt - 4 * config.alt_step;
                output.push_str(&format!(" // {}m to {}m", start_alt, alt));
                line_items = 0;
            }
        }
    }
    
    if line_items > 0 {
        let last_alt = config.alt_max;
        let start_alt = last_alt - (line_items - 1) * config.alt_step;
        output.push_str(&format!(" // {}m to {}m", start_alt, last_alt));
    }
    
    output.push_str("\n];\n");
    
    // Write to file
    let result = if append {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&output_path)
            .and_then(|mut file| file.write_all(output.as_bytes()))
    } else {
        File::create(&output_path)
            .and_then(|mut file| file.write_all(output.as_bytes()))
    };
    
    match result {
        Ok(_) => {
            println!("\nSuccessfully {} tables to {}", 
                if append { "appended" } else { "wrote" }, 
                output_path);
            println!("Generated {} dew point entries and {} altitude entries",
                dew_point_rows * dew_point_cols, altitude_entries);
        }
        Err(e) => {
            eprintln!("\nError writing to {}: {}", output_path, e);
            eprintln!("Details: {:?}", e);
            process::exit(1);
        }
    }
}