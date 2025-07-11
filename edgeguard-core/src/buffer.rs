//! Fixed-Size Circular Buffer for Sensor History Tracking
//!
//! ## Overview
//! 
//! This module provides a circular (ring) buffer implementation specifically designed for
//! tracking sensor reading history on memory-constrained embedded systems. Unlike traditional
//! collections that allocate memory dynamically, this buffer has a fixed size determined
//! at compile time through const generics.
//!
//! ## Design Rationale
//!
//! ### Why a Circular Buffer?
//! 
//! Sensor validation often requires historical context:
//! - Rate-of-change calculations need previous readings
//! - Anomaly detection benefits from recent history
//! - Trend analysis requires a sliding window of data
//!
//! A circular buffer provides constant-time operations while using fixed memory:
//! - O(1) insertion (overwrites oldest when full)
//! - O(1) access to most recent reading
//! - O(n) iteration over all readings
//! - Zero heap allocations
//!
//! ### Why Not Use `heapless::Vec`?
//! 
//! We initially considered `heapless::Vec` but chose a custom implementation because:
//! 
//! 1. **Automatic Overwrite**: When full, we want to automatically discard old data
//!    rather than returning an error. This matches sensor use cases where recent
//!    data is more valuable than old data.
//!
//! 2. **Optimized for Time Series**: Our implementation maintains chronological order
//!    and provides efficient access to the most recent reading.
//!
//! 3. **Minimal Dependencies**: One less dependency for embedded targets where every
//!    byte counts.
//!
//! ### Memory Layout
//!
//! The buffer uses an array of `Option<TimestampedReading>` for storage:
//! 
//! ```text
//! CircularBuffer<5> memory layout:
//! ┌─────┬─────┬─────┬─────┬─────┐
//! │  0  │  1  │  2  │  3  │  4  │  ← Array indices
//! └─────┴─────┴─────┴─────┴─────┘
//!    ↑                       ↑
//!    └── write_pos = 0      └── After 5 writes, wraps here
//! 
//! Each slot: Option<TimestampedReading> = 16 bytes
//! - Some variant: 1 byte discriminant + 4 bytes value + 8 bytes timestamp + 3 bytes padding
//! - None variant: 1 byte discriminant + 15 bytes padding
//! 
//! Total size = 16 * N + 16 bytes (for write_pos and len)
//! ```
//!
//! ### Performance Characteristics
//!
//! All operations are deterministic with no hidden allocations:
//! - `push()`: ~20 CPU cycles (simple array write + modulo)
//! - `last()`: ~10 CPU cycles (array access with bounds check)
//! - `len()`: ~3 CPU cycles (field access)
//! - `iter()`: ~5 CPU cycles per element
//!
//! The modulo operation in `push()` compiles to efficient bit manipulation when
//! N is a power of 2, so prefer sizes like 8, 16, 32, 64.
//!
//! ## Usage Example
//!
//! ```rust
//! use edgeguard_core::buffer::CircularBuffer;
//! use edgeguard_core::traits::TimestampedReading;
//! 
//! // Create buffer for 10 readings
//! let mut history: CircularBuffer<10> = CircularBuffer::new();
//! 
//! // Add readings (timestamp could be from RTC or monotonic timer)
//! history.push(TimestampedReading { value: 23.5, timestamp: 1000 });
//! history.push(TimestampedReading { value: 23.7, timestamp: 2000 });
//! 
//! // Access most recent
//! if let Some(latest) = history.last() {
//!     println!("Latest: {}°C at t={}", latest.value, latest.timestamp);
//! }
//! 
//! // Calculate rate of change
//! let readings: Vec<_> = history.iter().collect();
//! if readings.len() >= 2 {
//!     let dt = readings[1].timestamp - readings[0].timestamp;
//!     let dv = readings[1].value - readings[0].value;
//!     let rate = dv / (dt as f32 / 1000.0); // °C/sec
//! }
//! ```

use crate::traits::TimestampedReading;

/// Fixed-size circular buffer for time-series sensor data
/// 
/// This structure implements a ring buffer that automatically overwrites the oldest
/// data when full, making it ideal for maintaining a sliding window of sensor readings
/// without dynamic memory allocation.
/// 
/// ## Type Parameter
/// 
/// - `N`: The maximum number of readings to store. This is a compile-time constant,
///   allowing the compiler to optimize the memory layout and eliminate bounds checks
///   where possible. For best performance, use powers of 2 (8, 16, 32, etc.).
/// 
/// ## Internal Invariants
/// 
/// The implementation maintains these invariants:
/// - `write_pos < N` (next write position is always valid)
/// - `len <= N` (never claim to have more items than capacity)
/// - Items are stored in chronological order when iterating
/// 
/// ## Thread Safety
/// 
/// This type is not thread-safe. In concurrent environments, wrap it in a mutex
/// or use a lock-free ring buffer implementation instead.
#[derive(Clone)]
pub struct CircularBuffer<const N: usize> {
    /// Storage array using Option for uninitialized slots
    /// We use Option instead of MaybeUninit to avoid unsafe code
    data: [Option<TimestampedReading>; N],
    
    /// Index where the next write will occur
    /// Wraps around to 0 when reaching N
    write_pos: usize,
    
    /// Current number of valid readings
    /// Increases until N, then stays constant
    len: usize,
}

impl<const N: usize> CircularBuffer<N> {
    /// Creates a new empty circular buffer
    /// 
    /// This is a const function, allowing creation in static contexts:
    /// ```rust
    /// use edgeguard_core::buffer::CircularBuffer;
    /// static SENSOR_HISTORY: CircularBuffer<100> = CircularBuffer::new();
    /// ```
    /// 
    /// The buffer starts empty with all slots initialized to `None`.
    pub const fn new() -> Self {
        Self {
            data: [None; N],
            write_pos: 0,
            len: 0,
        }
    }
    
    /// Adds a reading to the buffer
    /// 
    /// When the buffer is full, this overwrites the oldest reading.
    /// This behavior ensures we always keep the most recent N readings.
    /// 
    /// ## Performance Note
    /// 
    /// The modulo operation `% N` is optimized by the compiler when N is a
    /// power of 2, becoming a simple bit mask operation. For other values,
    /// it may require a division instruction.
    /// 
    /// ## Example
    /// 
    /// ```rust
    /// # use edgeguard_core::buffer::CircularBuffer;
    /// # use edgeguard_core::traits::TimestampedReading;
    /// let mut buf = CircularBuffer::<3>::new();
    /// 
    /// // First 3 pushes fill the buffer
    /// buf.push(TimestampedReading { value: 1.0, timestamp: 100 });
    /// buf.push(TimestampedReading { value: 2.0, timestamp: 200 });
    /// buf.push(TimestampedReading { value: 3.0, timestamp: 300 });
    /// 
    /// // Fourth push overwrites the first
    /// buf.push(TimestampedReading { value: 4.0, timestamp: 400 });
    /// 
    /// // Buffer now contains [2.0, 3.0, 4.0]
    /// ```
    pub fn push(&mut self, reading: TimestampedReading) {
        self.data[self.write_pos] = Some(reading);
        self.write_pos = (self.write_pos + 1) % N;
        
        if self.len < N {
            self.len += 1;
        }
    }
    
    /// Get number of stored readings
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.len == N
    }
    
    /// Get the most recent reading
    pub fn last(&self) -> Option<&TimestampedReading> {
        if self.is_empty() {
            return None;
        }
        
        // Most recent is one before write position
        let idx = if self.write_pos == 0 { N - 1 } else { self.write_pos - 1 };
        
        self.data[idx].as_ref()
    }
    
    /// Iterate over readings from oldest to newest
    pub fn iter(&self) -> CircularBufferIter<N> {
        CircularBufferIter {
            buffer: self,
            index: 0,
            count: 0,
        }
    }
    
    /// Clear all readings
    pub fn clear(&mut self) {
        self.write_pos = 0;
        self.len = 0;
    }
    
    /// Gets a reading by its logical index (0 = oldest, len-1 = newest)
    /// 
    /// This method handles the circular nature of the buffer by translating
    /// logical indices to physical array positions.
    /// 
    /// ## Algorithm
    /// 
    /// When the buffer is not full, logical and physical indices match.
    /// When full, the oldest element is at `write_pos`, so we offset:
    /// 
    /// ```text
    /// Physical array:  [D, E, A, B, C]  (write_pos = 2)
    ///                   0  1  2  3  4
    /// 
    /// Logical view:    [A, B, C, D, E]  (chronological order)
    ///                   0  1  2  3  4
    /// 
    /// Mapping: logical[0] = physical[(2+0)%5] = physical[2] = A
    /// ```
    fn get(&self, index: usize) -> Option<&TimestampedReading> {
        if index >= self.len {
            return None;
        }
        
        let actual_index = if self.len < N {
            // Buffer not full yet, data starts at 0
            index
        } else {
            // Buffer is full, oldest data is at write_pos
            (self.write_pos + index) % N
        };
        
        self.data[actual_index].as_ref()
    }
}

/// Iterator over circular buffer contents
pub struct CircularBufferIter<'a, const N: usize> {
    buffer: &'a CircularBuffer<N>,
    index: usize,
    count: usize,
}

impl<'a, const N: usize> Iterator for CircularBufferIter<'a, N> {
    type Item = &'a TimestampedReading;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.count >= self.buffer.len() {
            return None;
        }
        
        let item = self.buffer.get(self.index)?;
        self.index += 1;
        self.count += 1;
        Some(item)
    }
}

impl<const N: usize> Default for CircularBuffer<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn empty_buffer() {
        let buffer: CircularBuffer<5> = CircularBuffer::new();
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
        assert!(buffer.last().is_none());
    }
    
    #[test]
    fn push_and_retrieve() {
        let mut buffer = CircularBuffer::<5>::new();
        
        let reading = TimestampedReading {
            value: 25.0,
            timestamp: 1000,
        };
        
        buffer.push(reading);
        assert_eq!(buffer.len(), 1);
        assert!(!buffer.is_empty());
        
        let last = buffer.last().unwrap();
        assert_eq!(last.value, 25.0);
        assert_eq!(last.timestamp, 1000);
    }
    
    #[test]
    fn circular_overwrite() {
        let mut buffer = CircularBuffer::<3>::new();
        
        // Fill buffer
        for i in 0..5 {
            buffer.push(TimestampedReading {
                value: i as f32,
                timestamp: i as u64 * 1000,
            });
        }
        
        // Should only have 3 items
        assert_eq!(buffer.len(), 3);
        assert!(buffer.is_full());
        
        // Should have values 2, 3, 4 (oldest 0, 1 were overwritten)
        let values: Vec<f32> = buffer.iter().map(|r| r.value).collect();
        assert_eq!(values, vec![2.0, 3.0, 4.0]);
    }
    
    #[test]
    fn iterator_order() {
        let mut buffer = CircularBuffer::<4>::new();
        
        for i in 0..4 {
            buffer.push(TimestampedReading {
                value: i as f32,
                timestamp: i as u64,
            });
        }
        
        let timestamps: Vec<u64> = buffer.iter().map(|r| r.timestamp).collect();
        assert_eq!(timestamps, vec![0, 1, 2, 3]);
    }
}