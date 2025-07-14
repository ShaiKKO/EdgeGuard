//! Buffer Sizes and Memory Constraints
//!
//! This module defines buffer sizes and memory limits optimized for
//! embedded systems and IoT devices with constrained resources.

// ===== EVENT BUFFER SIZES =====

/// Default event buffer size for standard operations.
/// 
/// Sized for typical IoT gateway or edge server:
/// - 1024 events × ~64 bytes/event = ~64KB
/// - Handles burst traffic and processing delays
/// - Suitable for devices with >512KB RAM
/// 
/// Source: ESP32, Raspberry Pi deployment experience
pub const DEFAULT_EVENT_BUFFER_SIZE: usize = 1024;

/// Small event buffer for memory-constrained devices.
/// 
/// Sized for microcontrollers and embedded systems:
/// - 256 events × ~64 bytes/event = ~16KB
/// - Minimal RAM footprint
/// - Suitable for devices with 128-256KB RAM
/// 
/// Source: STM32, nRF52 deployment constraints
pub const SMALL_EVENT_BUFFER_SIZE: usize = 256;

/// Tiny event buffer for ultra-constrained devices.
/// 
/// Sized for minimal embedded systems:
/// - 64 events × ~64 bytes/event = ~4KB
/// - Bare minimum for operation
/// - Suitable for devices with <64KB RAM
/// 
/// Source: Arduino, ATmega constraints
pub const TINY_EVENT_BUFFER_SIZE: usize = 64;

/// Large event buffer for high-throughput applications.
/// 
/// Sized for industrial gateways and servers:
/// - 4096 events × ~64 bytes/event = ~256KB
/// - Handles multiple high-frequency sensors
/// - Requires devices with >1MB RAM
/// 
/// Source: Industrial IoT gateway requirements
pub const LARGE_EVENT_BUFFER_SIZE: usize = 4096;

// ===== CIRCULAR BUFFER SIZES =====

/// Default history buffer size for validation.
/// 
/// Stores recent sensor readings for trend analysis:
/// - 100 samples × 4 bytes/sample = 400 bytes
/// - Covers 100 seconds at 1Hz sampling
/// - Sufficient for rate-of-change validation
/// 
/// Source: Validation algorithm requirements
pub const DEFAULT_HISTORY_SIZE: usize = 100;

/// Extended history buffer for analytics.
/// 
/// For advanced trending and pattern detection:
/// - 1000 samples provides better statistics
/// - 10-15 minutes of data at 1Hz
/// - Used in ML feature extraction
/// 
/// Source: Time-series analysis practices
pub const EXTENDED_HISTORY_SIZE: usize = 1000;

/// Minimal history buffer for basic validation.
/// 
/// Absolute minimum for rate checking:
/// - 10 samples for simple differentiation
/// - ~400 bytes total memory
/// - For severely constrained devices
/// 
/// Source: Embedded validation requirements
pub const MINIMAL_HISTORY_SIZE: usize = 10;

// ===== PIPELINE CONFIGURATION LIMITS =====

/// Maximum pipeline stages allowed.
/// 
/// Limits pipeline complexity:
/// - Each stage ~1KB memory
/// - 16 stages = ~16KB overhead
/// - Prevents excessive memory use
/// 
/// Source: Practical pipeline configurations
pub const MAX_PIPELINE_STAGES: usize = 16;

/// Maximum sensor groups for fusion.
/// 
/// Limits fusion algorithm complexity:
/// - Each group tracks multiple sensors
/// - 8 groups handles most applications
/// - Matrix operations scale O(n²)
/// 
/// Source: Fusion algorithm analysis
pub const MAX_SENSOR_GROUPS: usize = 8;

/// Maximum sensors per fusion group.
/// 
/// Practical limit for sensor redundancy:
/// - 8 sensors provides good redundancy
/// - Larger groups have diminishing returns
/// - Kalman filter matrices grow quadratically
/// 
/// Source: Sensor array deployments
pub const MAX_SENSORS_PER_GROUP: usize = 8;

/// Maximum routing rules in pipeline.
/// 
/// For RouterStage complexity:
/// - 8 routes handle all sensor types
/// - Each route ~100 bytes overhead
/// - Prevents routing table explosion
/// 
/// Source: EdgeGuard sensor type count
pub const MAX_ROUTES: usize = 8;

/// Maximum sensor pairs for cross-validation.
/// 
/// Limits cross-validation complexity:
/// - 4 pairs = temp/humidity, pressure/altitude, etc.
/// - Each pair requires correlation tracking
/// - O(n²) comparison complexity
/// 
/// Source: Physical sensor relationships
pub const MAX_SENSOR_PAIRS: usize = 4;

// ===== QUEUE AND BATCH SIZES =====

/// Default batch processing size.
/// 
/// Events processed per iteration:
/// - 100 events balances latency and efficiency
/// - ~100ms processing time on embedded CPU
/// - Prevents UI/network blocking
/// 
/// Source: Real-time processing requirements
pub const DEFAULT_BATCH_SIZE: usize = 100;

/// Small batch size for real-time processing.
/// 
/// Minimizes latency for control applications:
/// - 10 events = ~10ms maximum latency
/// - Used in PID loops and safety systems
/// - Trades efficiency for responsiveness
/// 
/// Source: Control system requirements
pub const REALTIME_BATCH_SIZE: usize = 10;

/// Large batch size for bulk processing.
/// 
/// Maximizes throughput for offline processing:
/// - 1000 events per batch
/// - Used for historical data analysis
/// - Requires adequate memory buffers
/// 
/// Source: Batch processing optimization
pub const BULK_BATCH_SIZE: usize = 1000;

// ===== STRING AND IDENTIFIER LIMITS =====

/// Maximum sensor ID length (characters).
/// 
/// Fits in inline string optimization:
/// - 15 chars + null terminator = 16 bytes
/// - Avoids heap allocation
/// - Covers "building_floor_room_sensor" patterns
/// 
/// Source: Memory optimization analysis
pub const MAX_SENSOR_ID_LENGTH: usize = 15;

/// Inline string buffer size (bytes).
/// 
/// For zero-allocation string storage:
/// - 32 bytes total structure size
/// - 31 chars + length byte
/// - Fits in single cache line
/// 
/// Source: Cache line optimization
pub const INLINE_STRING_SIZE: usize = 32;

// ===== NETWORK BUFFER SIZES =====

/// Default network packet buffer size.
/// 
/// For MQTT, CoAP, HTTP payloads:
/// - 1024 bytes handles most IoT messages
/// - Fits in single network packet
/// - Avoids fragmentation
/// 
/// Source: IoT protocol analysis
pub const NETWORK_BUFFER_SIZE: usize = 1024;

/// Large network buffer for bulk transfers.
/// 
/// For firmware updates, file transfers:
/// - 4KB allows efficient transfers
/// - Still fits in embedded RAM
/// - Multiple of flash page size
/// 
/// Source: OTA update requirements
pub const LARGE_NETWORK_BUFFER_SIZE: usize = 4096;

// ===== MEMORY ALLOCATION LIMITS =====

/// Maximum heap allocation size (bytes).
/// 
/// Prevents single allocation from exhausting memory:
/// - 64KB limit for embedded systems
/// - Forces efficient data structures
/// - Leaves room for stack and other uses
/// 
/// Source: Embedded memory management
pub const MAX_HEAP_ALLOCATION: usize = 65536;

/// Stack size for worker threads (bytes).
/// 
/// Adequate for sensor processing tasks:
/// - 4KB handles most operations
/// - Includes interrupt handling overhead
/// - Prevents stack overflow
/// 
/// Source: RTOS configuration experience
pub const WORKER_STACK_SIZE: usize = 4096;

/// Reserved memory for system use (bytes).
/// 
/// Ensures system stability:
/// - 16KB reserved for OS, interrupts
/// - Prevents memory exhaustion
/// - Critical for embedded reliability
/// 
/// Source: Embedded system best practices
pub const SYSTEM_RESERVED_MEMORY: usize = 16384;