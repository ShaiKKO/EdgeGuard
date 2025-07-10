//! Circular buffer for sensor history
//!
//! Fixed-size ring buffer that overwrites old data when full.
//! Optimized for embedded systems with no heap allocation.

use crate::traits::TimestampedReading;

/// Ring buffer for sensor readings
/// 
/// Stores up to N readings in a circular fashion.
/// When full, oldest readings are overwritten.
#[derive(Clone)]
pub struct CircularBuffer<const N: usize> {
    /// Storage for readings
    data: [Option<TimestampedReading>; N],
    /// Write position (next slot to write)
    write_pos: usize,
    /// Number of valid items
    len: usize,
}

impl<const N: usize> CircularBuffer<N> {
    /// Create empty buffer
    pub const fn new() -> Self {
        Self {
            data: [None; N],
            write_pos: 0,
            len: 0,
        }
    }
    
    /// Add a reading to the buffer
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
    
    /// Get reading at index (0 = oldest)
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