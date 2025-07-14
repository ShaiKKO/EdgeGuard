# EdgeGuard ESP32 Examples

This directory contains examples demonstrating EdgeGuard on ESP32 microcontrollers. These examples show how to use EdgeGuard's validation, fusion, and streaming capabilities in memory-constrained embedded environments.

## Examples Overview

### 1. `esp32_basic.rs` - Basic Sensor Validation
- **Features**: Core validation only
- **Memory**: ~150KB flash, ~20KB RAM
- **Hardware**: DHT22 sensor, basic ESP32
- **Concepts**: Temperature/humidity validation, rate limiting, history tracking

### 2. `esp32_advanced.rs` - Multi-Sensor Fusion
- **Features**: Pipeline, Kalman fusion, cross-validation
- **Memory**: ~250KB flash, ~40KB RAM  
- **Hardware**: Multiple temperature sensors, dual-core processing
- **Concepts**: Sensor fusion, event pipeline, multi-core tasks

### 3. `esp32_wifi_stream.rs` - Network Streaming
- **Features**: Stream adapters, batching, backpressure, WiFi/MQTT
- **Memory**: ~300KB flash, ~64KB RAM
- **Hardware**: BME280, WiFi-enabled ESP32
- **Concepts**: Real-time streaming, network backpressure, data aggregation

## Prerequisites

### 1. Install Rust ESP toolchain
```bash
# Install espup
cargo install espup

# Install ESP toolchain
espup install

# Source the environment (add to .bashrc/.zshrc)
source ~/export-esp.sh
```

### 2. Install flashing tools
```bash
cargo install espflash
cargo install espmonitor
```

### 3. Install build dependencies
```bash
# Linux
sudo apt-get install libudev-dev

# macOS
brew install libusb
```

## Building Examples

### Basic Example (Minimal Features)
```bash
cd edgeguard-core
cargo build --example esp32_basic \
  --target xtensa-esp32-none-elf \
  --no-default-features \
  --features "validation-core"
```

### Advanced Example (With Fusion)
```bash
cargo build --example esp32_advanced \
  --target xtensa-esp32-none-elf \
  --release \
  --no-default-features \
  --features "validation-core,pipeline-core,fusion-core,fusion-models,alloc"
```

### WiFi Streaming Example
```bash
# Set WiFi credentials
export WIFI_SSID="your_network"
export WIFI_PASSWORD="your_password"
export MQTT_BROKER="192.168.1.100:1883"

cargo build --example esp32_wifi_stream \
  --target xtensa-esp32-none-elf \
  --release \
  --no-default-features \
  --features "validation-core,pipeline-core,stream-core,stream-adapters,alloc"
```

## Flashing to ESP32

### Flash and monitor
```bash
cargo espflash --example esp32_basic --monitor
```

### Flash with specific port
```bash
cargo espflash --example esp32_basic --monitor --port /dev/ttyUSB0
```

### Just monitor existing firmware
```bash
espmonitor /dev/ttyUSB0
```

## Hardware Connections

### Basic Example - DHT22 Wiring
```
ESP32          DHT22
-----          -----
3.3V    -->    VCC
GND     -->    GND
GPIO4   -->    DATA
              (10kΩ pullup between DATA and VCC)
```

### Advanced Example - Multiple Sensors
```
ESP32          Sensors
-----          -------
GPIO4   -->    DHT22 DATA
GPIO5   -->    DS18B20 DATA (with 4.7kΩ pullup)
GPIO21  -->    BME280 SDA
GPIO22  -->    BME280 SCL
```

## Memory Optimization Tips

### 1. Feature Selection
Choose only the features you need:
```toml
# Minimal validation only
features = ["validation-core"]

# Add pipeline support
features = ["validation-core", "pipeline-core"]

# Add fusion algorithms
features = ["validation-core", "pipeline-core", "fusion-core"]
```

### 2. Optimization Flags
```toml
[profile.release]
opt-level = "z"     # Optimize for size
lto = true          # Link-time optimization
codegen-units = 1   # Better optimization
strip = true        # Remove debug symbols
panic = "abort"     # Smaller panic handler
```

### 3. Memory Usage Guidelines

| Configuration | Flash Usage | RAM Usage | Features |
|--------------|-------------|-----------|----------|
| Minimal | ~150KB | ~20KB | Basic validation |
| Standard | ~250KB | ~40KB | + Pipeline, fusion |
| Full | ~350KB | ~64KB | + Streaming, WiFi |

### 4. Buffer Sizing

Adjust buffer sizes based on your ESP32 variant:

```rust
// ESP32-C3 (400KB RAM)
const HEAP_SIZE: usize = 32 * 1024;  // 32KB heap
const QUEUE_SIZE: usize = 128;       // Event queue

// ESP32-S3 (512KB RAM)  
const HEAP_SIZE: usize = 64 * 1024;  // 64KB heap
const QUEUE_SIZE: usize = 256;       // Larger queue
```

## Troubleshooting

### Build Errors

1. **"can't find crate for `std`"**
   - Ensure you're using `--no-default-features`
   - Add `#![no_std]` to your code

2. **"undefined reference to `__sync_*`"**
   - Add atomic emulation: `atomic-polyfill = "1.0"`

3. **Linker errors**
   - Check target: `--target xtensa-esp32-none-elf`
   - Ensure `espup` environment is sourced

### Runtime Issues

1. **Constant reboots**
   - Check stack sizes (default 8KB may be too small)
   - Disable watchdog in development
   - Check for panic messages

2. **WiFi connection failures**
   - Increase heap size for WiFi (needs ~40KB)
   - Check power supply (WiFi needs 300mA+)
   - Verify credentials in environment

3. **Sensor reading errors**
   - Add delays between readings
   - Check pull-up resistors
   - Verify voltage levels (3.3V vs 5V)

## Production Deployment

### 1. Error Handling
```rust
// Don't panic in production
match sensor.read() {
    Ok(value) => process(value),
    Err(_) => {
        // Log error and continue
        error_count += 1;
        if error_count > 10 {
            // Restart sensor
        }
    }
}
```

### 2. Watchdog Timer
```rust
// Feed watchdog in main loop
wdt.feed();

// Set appropriate timeout
wdt.start(5_000_000); // 5 seconds
```

### 3. OTA Updates
Consider using `esp-ota` for over-the-air updates:
```toml
esp-ota = { version = "0.1", features = ["esp32"] }
```

### 4. Power Management
```rust
// Enable WiFi power save
wifi.set_power_save(PowerSave::MinModem);

// Deep sleep between readings
esp32_hal::sleep::deep_sleep(Duration::from_secs(60));
```

## Resources

- [ESP32 Datasheet](https://www.espressif.com/sites/default/files/documentation/esp32_datasheet_en.pdf)
- [esp-rs Book](https://esp-rs.github.io/book/)
- [EdgeGuard Documentation](https://github.com/edgeguard-dev/edgeguard)
- [ESP32 Examples](https://github.com/esp-rs/esp32-hal/tree/main/examples)

## License

These examples are part of EdgeGuard and licensed under Apache-2.0.