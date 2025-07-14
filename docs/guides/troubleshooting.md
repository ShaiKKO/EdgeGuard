# Troubleshooting Guide

This guide covers common issues and solutions when working with EdgeGuard.

## Table of Contents

1. [Compilation Issues](#compilation-issues)
2. [Runtime Issues](#runtime-issues)
3. [Performance Issues](#performance-issues)
4. [Memory Issues](#memory-issues)
5. [Network Issues](#network-issues)
6. [Platform-Specific Issues](#platform-specific-issues)
7. [Debugging Tools](#debugging-tools)
8. [Getting Help](#getting-help)

## Compilation Issues

### Rust Toolchain Problems

#### "rustc not found" or "cargo not found"
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Update Rust
rustup update

# Verify installation
rustc --version
cargo --version
```

#### "can't find crate for `std`"
This typically occurs when building for `no_std` targets.

**Solution:**
```toml
# In Cargo.toml
[dependencies]
edgeguard = { version = "0.1.0", default-features = false, features = ["embedded"] }
```

```rust
// In your code
#![no_std]
#![no_main]

use panic_halt as _; // or another panic handler
```

### Feature Flag Issues

#### "undefined reference to symbol" errors
Usually caused by missing or conflicting feature flags.

**Solution:**
```bash
# Check what features are available
cargo metadata --format-version 1 | jq '.packages[] | select(.name == "edgeguard") | .features'

# Use correct feature combination
cargo build --no-default-features --features "embedded,validation-core"
```

#### "multiple definition of symbol" errors
Caused by enabling conflicting features.

**Solution:**
```toml
# Don't enable both std and no_std features
[dependencies]
edgeguard = { version = "0.1.0", default-features = false, features = ["embedded"] }
# NOT: features = ["std", "embedded"]
```

### ESP32 Compilation Issues

#### "xtensa-esp32-none-elf not found"
ESP32 toolchain not installed.

**Solution:**
```bash
# Install ESP32 toolchain
cargo install espup
espup install

# Source environment variables
source ~/export-esp.sh

# Add to shell profile
echo 'source ~/export-esp.sh' >> ~/.bashrc
```

#### "undefined reference to `__sync_*`"
Atomic operations not available on ESP32.

**Solution:**
```toml
[dependencies]
atomic-polyfill = "1.0"
```

```rust
// In main.rs
#![no_std]
#![no_main]

use atomic_polyfill as _;
```

### Raspberry Pi Cross-Compilation Issues

#### "linker not found"
Cross-compilation linker not installed.

**Solution:**
```bash
# Install cross-compilation tools
sudo apt-get install gcc-arm-linux-gnueabihf

# Or for 64-bit
sudo apt-get install gcc-aarch64-linux-gnu

# Add to ~/.cargo/config.toml
[target.armv7-unknown-linux-gnueabihf]
linker = "arm-linux-gnueabihf-gcc"
```

## Runtime Issues

### Validation Errors

#### "OutOfRange" errors for valid values
Check validator configuration and sensor specifications.

**Debug:**
```rust
let validator = TemperatureValidator::new();
let constraints = validator.constraints();
println!("Range: {} to {}", constraints.min_value, constraints.max_value);
println!("Max rate: {}", constraints.max_rate_change);

// Check actual sensor specs
let sensor_reading = 25.0;
if sensor_reading < constraints.min_value || sensor_reading > constraints.max_value {
    println!("Reading {} outside validator range", sensor_reading);
}
```

**Solution:**
```rust
// Adjust validator to match sensor specifications
let validator = TemperatureValidator::new()
    .with_range(-40.0, 85.0)  // Match sensor operating range
    .with_rate_limit(10.0);   // Increase rate limit if needed
```

#### "RateOfChangeExceeded" errors
Sensor readings changing too rapidly.

**Causes:**
- Sensor malfunction
- Incorrect rate limit configuration
- Missing thermal mass considerations

**Solution:**
```rust
// Increase rate limit
let validator = TemperatureValidator::new()
    .with_rate_limit(20.0);  // Increase from default 10.0

// Or add thermal mass modeling
let validator = TemperatureValidator::new()
    .with_thermal_mass(0.5);  // 0.5kg thermal mass
```

### Pipeline Issues

#### "QueueFull" errors
Pipeline input queue is full.

**Causes:**
- Processing too slow
- Input rate too high
- Pipeline stages blocking

**Debug:**
```rust
let metrics = pipeline.metrics();
println!("Queue depth: {}", metrics.current_depth);
println!("Events processed: {}", metrics.events_processed);
println!("Events dropped: {}", metrics.events_dropped);
```

**Solution:**
```rust
// Increase buffer size
let pipeline = Pipeline::<1024>::builder()  // Increase from 64
    .add_stage(...)
    .build();

// Or use backpressure strategy
let pipeline = Pipeline::<64>::builder()
    .backpressure(BackpressureStrategy::DropOldest)
    .add_stage(...)
    .build();
```

### Fusion Issues

#### "NumericalInstability" errors
Fusion algorithms becoming unstable.

**Causes:**
- Poor sensor quality
- Incorrect noise parameters
- Insufficient measurements

**Debug:**
```rust
let confidence = fusion_result.confidence();
println!("Confidence: {:.2}", confidence.as_f32());

if confidence.as_f32() < 0.5 {
    println!("Low confidence - check sensor quality");
}
```

**Solution:**
```rust
// Increase measurement noise
let config = KalmanConfig {
    measurement_noise: [[1.0]],  // Increase from 0.5
    // ... other config
};

// Or reset on instability
match fusion.update(&measurements, timestamp) {
    Ok(result) => use_result(result),
    Err(FusionError::NumericalInstability) => {
        fusion.reset();
        println!("Fusion reset due to instability");
    }
}
```

## Performance Issues

### Slow Processing

#### Validation taking too long
Check validator configuration and sensor history.

**Debug:**
```rust
use std::time::Instant;

let start = Instant::now();
let result = validator.validate(reading);
let duration = start.elapsed();

if duration > Duration::from_micros(100) {
    println!("Validation took: {:?}", duration);
}
```

**Solution:**
```rust
// Reduce history buffer size
let validator = TemperatureValidator::new()
    .with_history_size(10);  // Reduce from default 100

// Or disable expensive checks
let validator = TemperatureValidator::new()
    .with_rate_limiting(false);  // Disable rate limiting
```

#### Pipeline bottlenecks
Identify slow stages in processing pipeline.

**Debug:**
```rust
let metrics = pipeline.metrics();
for (i, stage_metrics) in metrics.stage_metrics.iter().enumerate() {
    println!("Stage {}: {} events, {} errors", 
        i, stage_metrics.events_processed, stage_metrics.errors);
}
```

**Solution:**
```rust
// Process events in batches
let processed = pipeline.process_batch(100)?;  // Increase batch size

// Or optimize stage ordering
let pipeline = Pipeline::<256>::builder()
    .add_stage(FilterStage::new(...))     // Filter first
    .add_stage(ValidationStage::new(...))  // Then validate
    .add_stage(FusionStage::new(...))     // Fusion last
    .build();
```

### High CPU Usage

#### Excessive computation
Profile to identify hotspots.

**Debug:**
```bash
# Install profiling tools
cargo install cargo-profiler flamegraph

# Profile application
cargo profiler callgrind --bin your_app
```

**Solution:**
```rust
// Use lookup tables instead of calculations
use edgeguard::lookup::dew_point_lookup;

let dew_point = dew_point_lookup(temperature, humidity);
// Instead of: calculate_dew_point(temperature, humidity)
```

## Memory Issues

### Out of Memory (OOM)

#### Heap exhaustion
Common on embedded systems with limited RAM.

**Debug:**
```rust
// Monitor heap usage
#[cfg(feature = "std")]
fn print_memory_usage() {
    use std::alloc::{GlobalAlloc, System};
    // Print heap statistics
}

// For embedded systems
extern crate linked_list_allocator;
use linked_list_allocator::LockedHeap;

#[global_allocator]
static ALLOCATOR: LockedHeap = LockedHeap::empty();
```

**Solution:**
```rust
// Reduce buffer sizes
let pipeline = Pipeline::<32>::builder()  // Reduce from 64
    .add_stage(...)
    .build();

// Use stack allocation
use heapless::Vec;
let mut events: Vec<Event, 32> = Vec::new();
```

#### Stack overflow
Large local variables or deep recursion.

**Debug:**
```rust
// Check stack usage
fn check_stack_usage() {
    let stack_var = [0u8; 1024];  // 1KB stack usage
    // ... function logic
}
```

**Solution:**
```rust
// Move large data to heap or static
static mut LARGE_BUFFER: [u8; 4096] = [0; 4096];

// Or use Box for heap allocation
let large_data = Box::new([0u8; 4096]);
```

### Memory Leaks

#### Growing memory usage
Memory not being freed properly.

**Debug:**
```bash
# Use memory profiler
cargo install cargo-valgrind
cargo valgrind run --bin your_app

# Or use AddressSanitizer
RUSTFLAGS="-Z sanitizer=address" cargo run
```

**Solution:**
```rust
// Ensure proper cleanup
impl Drop for MyStruct {
    fn drop(&mut self) {
        // Clean up resources
    }
}

// Use RAII patterns
{
    let _guard = acquire_resource();
    // Resource automatically released at end of scope
}
```

## Network Issues

### MQTT Connection Problems

#### Connection refused
MQTT broker not available or credentials incorrect.

**Debug:**
```bash
# Test broker connectivity
mosquitto_pub -h broker_host -t test -m "hello"

# Check credentials
mosquitto_pub -h broker_host -u username -P password -t test -m "hello"
```

**Solution:**
```rust
let config = MqttConfig::new("client_id", "mqtt://broker:1883")
    .with_credentials("username", "password")
    .with_keep_alive(Duration::from_secs(60))
    .with_max_reconnect_attempts(10);

// Add error handling
match client.connect() {
    Ok(()) => println!("Connected to MQTT broker"),
    Err(e) => eprintln!("Connection failed: {}", e),
}
```

#### Message loss
Messages not being delivered or received.

**Debug:**
```rust
// Check connection status
if !client.is_connected() {
    println!("MQTT client disconnected");
}

// Monitor message statistics
let stats = client.statistics();
println!("Messages sent: {}", stats.messages_sent);
println!("Messages received: {}", stats.messages_received);
```

**Solution:**
```rust
// Use higher QoS for important messages
client.publish("sensors/data", payload, 2)?;  // QoS 2 for exactly once

// Enable persistent sessions
let config = MqttConfig::new("client_id", "mqtt://broker:1883")
    .with_clean_session(false);
```

### HTTP Issues

#### Connection timeouts
HTTP requests timing out.

**Debug:**
```rust
// Check network connectivity
use std::net::TcpStream;
use std::time::Duration;

match TcpStream::connect_timeout(&"api.example.com:443".parse()?, Duration::from_secs(5)) {
    Ok(_) => println!("Network connection OK"),
    Err(e) => println!("Network error: {}", e),
}
```

**Solution:**
```rust
let config = HttpConfig::new("https://api.example.com")
    .with_timeout(Duration::from_secs(30))  // Increase timeout
    .with_retry_policy(RetryPolicy::exponential(3));  // Add retries
```

#### TLS errors
SSL/TLS certificate verification failures.

**Solution:**
```rust
// For development only - disable TLS verification
let config = HttpConfig::new("https://api.example.com")
    .with_tls_verification(false);

// For production - add CA certificates
let config = HttpConfig::new("https://api.example.com")
    .with_ca_certificate("ca.pem")
    .with_tls_verification(true);
```

## Platform-Specific Issues

### ESP32 Issues

#### Constant reboots
Usually caused by stack overflow or panic.

**Debug:**
```rust
// Add panic handler with debugging
use esp_backtrace as _;

// Or custom panic handler
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    esp_println::println!("Panic: {}", info);
    loop {}
}
```

**Solution:**
```rust
// Increase stack size
fn main() -> ! {
    let app = App::new()
        .stack_size(16384);  // Increase from default 8192
    app.run();
}

// Add watchdog timer
use esp32_hal::watchdog::Watchdog;
let mut wdt = Watchdog::new();
wdt.start(5_000_000);  // 5 second timeout

loop {
    wdt.feed();
    // ... main loop
}
```

#### WiFi connection issues
WiFi not connecting or frequent disconnections.

**Solution:**
```rust
// Add WiFi error handling
use esp_wifi::wifi::{WifiError, WifiState};

match wifi.connect() {
    Ok(()) => println!("WiFi connected"),
    Err(WifiError::InternalError) => {
        println!("WiFi internal error - restart");
        esp32_hal::reset();
    }
    Err(e) => println!("WiFi error: {:?}", e),
}

// Enable power management
wifi.set_power_save_mode(PowerSaveMode::Min);
```

### Raspberry Pi Issues

#### GPIO permission errors
GPIO access denied.

**Solution:**
```bash
# Add user to gpio group
sudo usermod -a -G gpio $USER

# Or run with sudo (not recommended for production)
sudo ./your_program
```

#### I2C/SPI issues
I2C or SPI devices not responding.

**Debug:**
```bash
# Check I2C devices
i2cdetect -y 1

# Check SPI devices
ls /dev/spi*
```

**Solution:**
```bash
# Enable I2C/SPI in raspi-config
sudo raspi-config
# Advanced Options -> I2C/SPI -> Enable

# Set correct permissions
sudo chown root:i2c /dev/i2c-1
sudo chmod 664 /dev/i2c-1
```

## Debugging Tools

### Logging

#### Enable debug logging
```rust
// Add to Cargo.toml
[dependencies]
log = "0.4"
env_logger = "0.10"

// In main.rs
use log::{debug, info, warn, error};

fn main() {
    env_logger::init();
    
    debug!("Debug message");
    info!("Info message");
    warn!("Warning message");
    error!("Error message");
}
```

#### Set log level
```bash
# Run with debug logging
RUST_LOG=debug cargo run

# Or specific module
RUST_LOG=edgeguard::validators=debug cargo run
```

### Performance Profiling

#### CPU profiling
```bash
# Install profiling tools
cargo install cargo-profiler

# Profile with callgrind
cargo profiler callgrind --bin your_app

# View results
kcachegrind callgrind.out.*
```

#### Memory profiling
```bash
# Install memory profiler
cargo install cargo-valgrind

# Run with valgrind
cargo valgrind run --bin your_app

# Or use heaptrack
heaptrack ./target/release/your_app
heaptrack_gui heaptrack.your_app.*
```

### Testing

#### Unit testing
```bash
# Run all tests
cargo test

# Run specific test
cargo test test_temperature_validator

# Run with logging
RUST_LOG=debug cargo test
```

#### Integration testing
```bash
# Run integration tests
cargo test --test integration

# Run with specific features
cargo test --features "embedded,validation-core"
```

### Hardware Testing

#### Oscilloscope/Logic Analyzer
For ESP32 and embedded systems:
- Monitor GPIO signals
- Check I2C/SPI communication
- Measure timing and jitter

#### Multimeter
- Check power supply voltages
- Verify sensor connections
- Test pull-up resistors

## Getting Help

### Documentation
- [API Reference](../api/README.md)
- [Architecture Guide](architecture.md)
- [Deployment Guide](deployment.md)
- [Examples](../examples/README.md)

### Community
- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Ask questions and share ideas
- **Stack Overflow**: Tag questions with `edgeguard`

### Professional Support
For commercial applications, professional support is available:
- Architecture consulting
- Performance optimization
- Custom feature development
- Training and workshops

### Bug Reports

When reporting bugs, include:

1. **Platform information**: OS, Rust version, hardware
2. **EdgeGuard version**: Output of `cargo tree | grep edgeguard`
3. **Minimal reproduction**: Smallest code that reproduces the issue
4. **Expected vs actual behavior**: What should happen vs what happens
5. **Error messages**: Complete error output with stack traces
6. **Configuration**: Relevant parts of Cargo.toml and code

### Feature Requests

For feature requests, provide:

1. **Use case**: What problem does this solve?
2. **Proposed solution**: How should it work?
3. **Alternatives**: What workarounds exist?
4. **Impact**: How important is this feature?

### Contributing

The best way to get help is often to contribute:

1. **Fix documentation**: Improve unclear sections
2. **Add examples**: Show how to solve common problems
3. **Report issues**: Help identify and fix bugs
4. **Implement features**: Add functionality you need

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for contribution guidelines.

## Common Error Messages

### "undefined reference to `__sync_*`"
Add atomic polyfill for embedded targets:
```toml
[dependencies]
atomic-polyfill = "1.0"
```

### "can't find crate for `std`"
Use no_std configuration:
```toml
[dependencies]
edgeguard = { version = "0.1.0", default-features = false, features = ["embedded"] }
```

### "stack overflow"
Increase stack size or reduce stack usage:
```rust
// Increase stack size
const STACK_SIZE: usize = 16384;

// Or reduce local variables
fn process_data() {
    let data = Box::new([0u8; 4096]);  // Move to heap
    // ... process data
}
```

### "out of memory"
Reduce heap usage or increase available memory:
```rust
// Use fixed-size collections
use heapless::Vec;
let mut events: Vec<Event, 32> = Vec::new();
```

This troubleshooting guide should help resolve most common issues with EdgeGuard. If you encounter problems not covered here, please consult the documentation or seek help from the community.