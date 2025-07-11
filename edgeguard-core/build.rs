//! Build script for EdgeGuard Core
//!
//! Automatically generates lookup tables if they don't exist.
//! Re-runs if build_tables.rs changes.

use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    // Tell cargo to re-run this build script if build_tables.rs changes
    println!("cargo:rerun-if-changed=build_tables.rs");
    
    // Get the manifest directory (where Cargo.toml is)
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let tables_script = Path::new(&manifest_dir).join("build_tables.rs");
    
    // Check if we need to generate standard tables
    let generated_tables = Path::new(&manifest_dir).join("generated_tables.rs");
    if !generated_tables.exists() {
        println!("cargo:warning=Generating standard lookup tables...");
        
        let output = Command::new("rustc")
            .args(&[
                tables_script.to_str().unwrap(),
                "-o",
                "build_tables",
            ])
            .current_dir(&manifest_dir)
            .output()
            .expect("Failed to compile build_tables.rs");
            
        if !output.status.success() {
            panic!("Failed to compile build_tables.rs: {}", 
                String::from_utf8_lossy(&output.stderr));
        }
        
        let output = Command::new("./build_tables")
            .args(&[
                "--config", "standard",
                "--output", "generated_tables.rs",
            ])
            .current_dir(&manifest_dir)
            .output()
            .expect("Failed to run build_tables");
            
        if !output.status.success() {
            panic!("Failed to generate standard tables: {}", 
                String::from_utf8_lossy(&output.stderr));
        }
    }
    
    // Check if we need to generate high precision tables
    let high_precision_tables = Path::new(&manifest_dir).join("high_precision_tables.rs");
    if !high_precision_tables.exists() {
        println!("cargo:warning=Generating high precision lookup tables...");
        
        let output = Command::new("./build_tables")
            .args(&[
                "--config", "high_precision",
                "--output", "high_precision_tables.rs",
            ])
            .current_dir(&manifest_dir)
            .output()
            .expect("Failed to run build_tables");
            
        if !output.status.success() {
            panic!("Failed to generate high precision tables: {}", 
                String::from_utf8_lossy(&output.stderr));
        }
    }
    
    // Check if we need to generate low memory tables
    let low_memory_tables = Path::new(&manifest_dir).join("low_memory_tables.rs");
    if !low_memory_tables.exists() {
        println!("cargo:warning=Generating low memory lookup tables...");
        
        let output = Command::new("./build_tables")
            .args(&[
                "--config", "low_memory",
                "--output", "low_memory_tables.rs",
            ])
            .current_dir(&manifest_dir)
            .output()
            .expect("Failed to run build_tables");
            
        if !output.status.success() {
            panic!("Failed to generate low memory tables: {}", 
                String::from_utf8_lossy(&output.stderr));
        }
    }
    
    // Clean up the build_tables executable
    let _ = std::fs::remove_file(Path::new(&manifest_dir).join("build_tables"));
}