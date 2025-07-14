# Time API

Time sources and synchronization for sensor data processing on embedded systems.

## TimeSource Trait

Core abstraction for time management across different platforms and environments.

### Trait Definition

```rust
pub trait TimeSource: Send {
    fn now(&self) -> Timestamp;
    fn is_wall_clock(&self) -> bool;
    fn precision_ms(&self) -> u32;
    fn is_monotonic(&self) -> bool;
}
```

### Implementation Requirements

```rust
// All time sources must be Send for multi-threaded use
// Timestamps are u64 milliseconds since epoch or boot
pub type Timestamp = u64;

impl TimeSource for CustomTimeSource {
    fn now(&self) -> Timestamp {
        // Return current timestamp in milliseconds
        self.get_current_time_ms()
    }
    
    fn is_wall_clock(&self) -> bool {
        // Return true if this represents wall clock time
        true
    }
    
    fn precision_ms(&self) -> u32 {
        // Return precision in milliseconds
        1  // 1ms precision
    }
    
    fn is_monotonic(&self) -> bool {
        // Return true if time never goes backwards
        true
    }
}
```

## SystemTime

Wall clock time source using system time (requires `std` feature).

### Constructor

```rust
impl SystemTime {
    pub fn new() -> Self;
    pub fn now_utc() -> Timestamp;
    pub fn now_local() -> Timestamp;
    pub fn from_unix_timestamp(timestamp: u64) -> Self;
}
```

### Usage

```rust
// Standard system time
let time_source = SystemTime::new();
let current_time = time_source.now();

println!("Current time: {} ms since epoch", current_time);
println!("Wall clock: {}", time_source.is_wall_clock());
println!("Precision: {} ms", time_source.precision_ms());

// Direct access to current time
let utc_time = SystemTime::now_utc();
let local_time = SystemTime::now_local();
```

### Platform Support

```rust
// Linux/macOS/Windows
let system_time = SystemTime::new();

// Get actual system time
let timestamp = system_time.now();

// Convert to human-readable format
let datetime = SystemTime::from_timestamp(timestamp);
println!("System time: {:?}", datetime);
```

## MonotonicTime

Monotonic time source that never goes backwards, ideal for embedded systems.

### Constructor

```rust
impl MonotonicTime {
    pub fn new() -> Self;
    pub fn from_boot() -> Self;
    pub fn with_offset(offset_ms: u64) -> Self;
    pub fn elapsed_ms(&self) -> u64;
    pub fn reset(&mut self);
}
```

### Usage

```rust
// Monotonic time from system boot
let monotonic = MonotonicTime::new();

// Get monotonic timestamp
let start_time = monotonic.now();
perform_operation();
let end_time = monotonic.now();

let duration = end_time - start_time;
println!("Operation took {} ms", duration);

// Monotonic time properties
assert!(monotonic.is_monotonic());
assert!(!monotonic.is_wall_clock());
```

### Embedded Systems

```rust
// ESP32 example
#[cfg(target_arch = "xtensa")]
use esp32_hal::systimer::SystemTimer;

let timer = SystemTimer::new();
let monotonic = MonotonicTime::from_boot();

// Get milliseconds since boot
let uptime = monotonic.elapsed_ms();
println!("Device uptime: {} ms", uptime);
```

## FixedTime

Fixed time source for testing and deterministic behavior.

### Constructor

```rust
impl FixedTime {
    pub fn new(timestamp: Timestamp) -> Self;
    pub fn set(&mut self, timestamp: Timestamp);
    pub fn advance(&mut self, delta_ms: u64);
    pub fn freeze(&mut self);
    pub fn unfreeze(&mut self);
}
```

### Testing Usage

```rust
// Create fixed time for testing
let mut fixed_time = FixedTime::new(1640995200000);  // 2022-01-01 00:00:00

// Use in tests
let validator = TemperatureValidator::new();
let mut context = ValidationContext::new(fixed_time.now());

// Advance time for rate limiting tests
fixed_time.advance(1000);  // Advance by 1 second
let new_context = ValidationContext::new(fixed_time.now());

// Test rate limiting
let result1 = validator.validate_with_context(25.0, &context);
let result2 = validator.validate_with_context(35.0, &new_context);  // 10°C change in 1s
```

### Deterministic Testing

```rust
// Create deterministic test scenarios
let mut test_time = FixedTime::new(0);

let events = vec![
    (0, 20.0),      // t=0: 20°C
    (1000, 21.0),   // t=1s: 21°C
    (2000, 22.0),   // t=2s: 22°C
    (3000, 30.0),   // t=3s: 30°C (potential rate violation)
];

for (timestamp, temperature) in events {
    test_time.set(timestamp);
    let context = ValidationContext::new(test_time.now());
    
    let result = validator.validate_with_context(temperature, &context);
    println!("t={}: {}°C -> {:?}", timestamp, temperature, result);
}
```

## TimeManager

Centralized time management with fallback sources and synchronization.

### Constructor

```rust
impl TimeManager {
    pub fn new(primary: Box<dyn TimeSource>) -> Self;
    pub fn with_fallback(mut self, fallback: Box<dyn TimeSource>) -> Self;
    pub fn with_sync_interval(mut self, interval_ms: u64) -> Self;
    pub fn with_max_drift(mut self, max_drift_ms: u64) -> Self;
    pub fn sync_now(&mut self) -> Result<(), TimeError>;
}
```

### Multi-Source Management

```rust
// Create time manager with primary and fallback sources
let primary = Box::new(SystemTime::new());
let fallback = Box::new(MonotonicTime::new());

let mut time_manager = TimeManager::new(primary)
    .with_fallback(fallback)
    .with_sync_interval(60000)  // Sync every minute
    .with_max_drift(5000);      // Max 5 second drift

// Get time with automatic fallback
let timestamp = time_manager.now();

// Check source status
match time_manager.current_source() {
    TimeSourceStatus::Primary => println!("Using primary time source"),
    TimeSourceStatus::Fallback => println!("Using fallback time source"),
    TimeSourceStatus::Unavailable => println!("No time source available"),
}
```

### Synchronization

```rust
// Automatic synchronization
let mut time_manager = TimeManager::new(Box::new(SystemTime::new()))
    .with_sync_interval(300000);  // Sync every 5 minutes

// Manual synchronization
match time_manager.sync_now() {
    Ok(()) => println!("Time synchronized"),
    Err(TimeError::SyncFailed) => println!("Synchronization failed"),
    Err(e) => println!("Time error: {:?}", e),
}

// Check synchronization status
let sync_status = time_manager.sync_status();
println!("Last sync: {} ms ago", sync_status.last_sync_ms);
println!("Drift: {} ms", sync_status.drift_ms);
```

## Time Utilities

### Time Conversion

```rust
// Timestamp conversion utilities
pub fn timestamp_to_unix(timestamp: Timestamp) -> u64;
pub fn unix_to_timestamp(unix_time: u64) -> Timestamp;
pub fn timestamp_to_string(timestamp: Timestamp) -> String;
pub fn parse_timestamp(time_str: &str) -> Result<Timestamp, TimeError>;

// Usage
let timestamp = 1640995200000;
let unix_time = timestamp_to_unix(timestamp);
let time_str = timestamp_to_string(timestamp);
println!("Timestamp: {} -> Unix: {} -> String: {}", timestamp, unix_time, time_str);
```

### Duration Calculations

```rust
// Duration and rate calculations
pub fn duration_ms(start: Timestamp, end: Timestamp) -> u64;
pub fn rate_per_second(value_delta: f32, time_delta_ms: u64) -> f32;
pub fn time_since(timestamp: Timestamp, reference: Timestamp) -> u64;

// Usage
let start = time_source.now();
perform_operation();
let end = time_source.now();

let duration = duration_ms(start, end);
let rate = rate_per_second(100.0, duration);
println!("Processed 100 items in {} ms ({:.2} items/sec)", duration, rate);
```

### Time Windows

```rust
// Time window management
pub struct TimeWindow {
    start: Timestamp,
    duration_ms: u64,
}

impl TimeWindow {
    pub fn new(start: Timestamp, duration_ms: u64) -> Self;
    pub fn contains(&self, timestamp: Timestamp) -> bool;
    pub fn expired(&self, current_time: Timestamp) -> bool;
    pub fn remaining(&self, current_time: Timestamp) -> u64;
}

// Usage
let window = TimeWindow::new(time_source.now(), 5000);  // 5 second window

if window.contains(event_timestamp) {
    // Event is within time window
    process_event(event);
} else if window.expired(time_source.now()) {
    // Window has expired
    close_window();
}
```

## Embedded Time Sources

### Hardware Timer Integration

```rust
// ESP32 hardware timer
#[cfg(target_arch = "xtensa")]
pub struct HardwareTimer {
    timer: esp32_hal::timer::Timer,
}

impl TimeSource for HardwareTimer {
    fn now(&self) -> Timestamp {
        self.timer.get_counter_value() / 1000  // Convert to milliseconds
    }
    
    fn is_wall_clock(&self) -> bool {
        false  // Hardware timers are not wall clock
    }
    
    fn precision_ms(&self) -> u32 {
        1  // 1ms precision
    }
    
    fn is_monotonic(&self) -> bool {
        true  // Hardware timers are monotonic
    }
}
```

### RTC Integration

```rust
// Real-Time Clock integration
pub struct RtcTimeSource {
    rtc: Rtc,
}

impl RtcTimeSource {
    pub fn new(rtc: Rtc) -> Self;
    pub fn set_time(&mut self, timestamp: Timestamp) -> Result<(), TimeError>;
    pub fn calibrate(&mut self) -> Result<(), TimeError>;
}

impl TimeSource for RtcTimeSource {
    fn now(&self) -> Timestamp {
        self.rtc.get_timestamp()
    }
    
    fn is_wall_clock(&self) -> bool {
        true  // RTC provides wall clock time
    }
    
    fn precision_ms(&self) -> u32 {
        1000  // 1 second precision typical for RTC
    }
}
```

### Network Time Protocol

```rust
// NTP client for time synchronization
pub struct NtpTimeSource {
    server: &'static str,
    last_sync: Timestamp,
    offset: i64,
}

impl NtpTimeSource {
    pub fn new(server: &'static str) -> Self;
    pub async fn sync(&mut self) -> Result<(), TimeError>;
    pub fn time_since_sync(&self) -> u64;
}

impl TimeSource for NtpTimeSource {
    fn now(&self) -> Timestamp {
        let local_time = MonotonicTime::new().now();
        ((local_time as i64) + self.offset) as Timestamp
    }
    
    fn is_wall_clock(&self) -> bool {
        true  // NTP provides wall clock time
    }
    
    fn precision_ms(&self) -> u32 {
        50  // Typical NTP precision
    }
}
```

## Time Validation

### Timestamp Validation

```rust
// Validate timestamps for consistency
pub fn validate_timestamp(timestamp: Timestamp, reference: Timestamp) -> Result<(), TimeError>;
pub fn validate_time_sequence(timestamps: &[Timestamp]) -> Result<(), TimeError>;
pub fn detect_clock_skew(local_time: Timestamp, reference_time: Timestamp) -> Option<i64>;

// Usage
let events = collect_events();
let timestamps: Vec<Timestamp> = events.iter().map(|e| e.timestamp()).collect();

match validate_time_sequence(&timestamps) {
    Ok(()) => println!("Time sequence valid"),
    Err(TimeError::NonMonotonic { index }) => {
        println!("Non-monotonic timestamp at index {}", index);
    }
    Err(TimeError::InvalidTimestamp { timestamp }) => {
        println!("Invalid timestamp: {}", timestamp);
    }
}
```

### Clock Synchronization

```rust
// Synchronize clocks across devices
pub struct ClockSynchronizer {
    reference_source: Box<dyn TimeSource>,
    local_sources: Vec<Box<dyn TimeSource>>,
    sync_interval: u64,
}

impl ClockSynchronizer {
    pub fn new(reference: Box<dyn TimeSource>) -> Self;
    pub fn add_source(&mut self, source: Box<dyn TimeSource>);
    pub fn sync_all(&mut self) -> Result<SyncReport, TimeError>;
    pub fn drift_report(&self) -> Vec<ClockDrift>;
}

// Usage
let mut synchronizer = ClockSynchronizer::new(Box::new(SystemTime::new()));
synchronizer.add_source(Box::new(MonotonicTime::new()));
synchronizer.add_source(Box::new(RtcTimeSource::new(rtc)));

let sync_report = synchronizer.sync_all()?;
println!("Synchronized {} sources", sync_report.sources_synced);
```

## Performance Characteristics

### Time Source Performance

- **SystemTime**: <1μs per call (platform dependent)
- **MonotonicTime**: <100ns per call (hardware dependent)
- **FixedTime**: <10ns per call (constant time)
- **RTC**: <1ms per call (I2C/SPI dependent)

### Memory Usage

- **Basic time sources**: <100 bytes
- **TimeManager**: <1KB including buffers
- **Synchronization**: <512 bytes per source

### Precision Comparison

| Time Source | Precision | Accuracy | Drift |
|------------|-----------|----------|--------|
| SystemTime | 1ms | ±1ms | <1ppm |
| MonotonicTime | 1ms | ±10ms | <50ppm |
| FixedTime | Perfect | N/A | None |
| RTC | 1s | ±20s/month | <50ppm |
| NTP | 1-50ms | ±10ms | <1ppm |

## Error Handling

### TimeError

```rust
pub enum TimeError {
    SystemTimeUnavailable,
    ClockNotSet,
    InvalidTimestamp { timestamp: Timestamp },
    NonMonotonic { index: usize },
    SyncFailed,
    NetworkError(String),
    HardwareError(String),
    Overflow,
}
```

### Error Recovery

```rust
// Robust time handling with fallback
let mut time_manager = TimeManager::new(Box::new(SystemTime::new()))
    .with_fallback(Box::new(MonotonicTime::new()));

match time_manager.now() {
    Ok(timestamp) => {
        // Use timestamp normally
        process_with_time(timestamp);
    }
    Err(TimeError::SystemTimeUnavailable) => {
        // Fallback to monotonic time
        let monotonic = MonotonicTime::new();
        process_with_time(monotonic.now());
    }
    Err(e) => {
        log::error!("Time error: {:?}", e);
        // Use fixed time as last resort
        let fixed = FixedTime::new(0);
        process_with_time(fixed.now());
    }
}
```

## Integration Examples

### Validation Context

```rust
use edgeguard::validators::ValidationContext;

// Create validation context with time
let time_source = SystemTime::new();
let context = ValidationContext {
    timestamp: time_source.now(),
    quality: 0.95,
    sensor_id: InlineString::new("temp_001").unwrap(),
};

// Validate with temporal context
let result = validator.validate_with_context(23.5, &context);
```

### Rate Limiting

```rust
// Rate limiting with time sources
let mut rate_limiter = RateLimiter::new(100.0, time_source);  // 100 events/sec

if rate_limiter.allow() {
    process_event(event);
} else {
    // Rate limit exceeded
    drop_event(event);
}
```

### Pipeline Integration

```rust
// Pipeline with time-aware processing
let pipeline = Pipeline::<256>::builder()
    .with_time_source(Box::new(SystemTime::new()))
    .add_stage(ValidationStage::new(
        TemperatureValidator::new(),
        SensorType::Temperature
    ))
    .add_stage(TimeWindowStage::new(5000))  // 5 second window
    .build();
```

## Best Practices

### Time Source Selection

```rust
// Choose appropriate time source for use case
let time_source: Box<dyn TimeSource> = if cfg!(feature = "std") {
    Box::new(SystemTime::new())  // Wall clock for logging
} else {
    Box::new(MonotonicTime::new())  // Monotonic for embedded
};
```

### Testing Strategy

```rust
// Use fixed time for deterministic tests
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rate_limiting() {
        let mut fixed_time = FixedTime::new(0);
        let mut validator = TemperatureValidator::new();
        
        // Test sequence with known timestamps
        let test_cases = vec![
            (0, 20.0, true),      // Initial reading
            (100, 21.0, true),    // 1°C in 100ms - OK
            (200, 30.0, false),   // 9°C in 100ms - rate exceeded
        ];
        
        for (timestamp, temperature, expected_valid) in test_cases {
            fixed_time.set(timestamp);
            let context = ValidationContext::new(fixed_time.now());
            
            let result = validator.validate_with_context(temperature, &context);
            assert_eq!(result.is_ok(), expected_valid);
        }
    }
}
```

### Synchronization Strategy

```rust
// Implement robust time synchronization
let mut time_manager = TimeManager::new(Box::new(SystemTime::new()))
    .with_fallback(Box::new(MonotonicTime::new()))
    .with_sync_interval(300000);  // 5 minutes

// Periodic synchronization
tokio::spawn(async move {
    let mut interval = tokio::time::interval(Duration::from_secs(300));
    loop {
        interval.tick().await;
        if let Err(e) = time_manager.sync_now() {
            log::warn!("Time sync failed: {:?}", e);
        }
    }
});
```

### Embedded Optimization

```rust
// Optimize for embedded systems
let time_source = if cfg!(target_arch = "xtensa") {
    // ESP32 - use hardware timer
    Box::new(HardwareTimer::new()) as Box<dyn TimeSource>
} else if cfg!(target_arch = "arm") {
    // ARM Cortex-M - use systick
    Box::new(SysTickTimer::new()) as Box<dyn TimeSource>
} else {
    // Desktop - use system time
    Box::new(SystemTime::new()) as Box<dyn TimeSource>
};
```

This time API provides comprehensive time management capabilities with platform-specific optimizations and robust error handling suitable for embedded and real-time applications.