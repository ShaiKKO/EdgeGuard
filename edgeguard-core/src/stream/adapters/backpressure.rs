//! Backpressure control for streams
//!
//! Provides mechanisms to handle backpressure when consumers
//! can't keep up with producers.

use crate::stream::{Stream, StreamError, BackpressureStream};
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Backpressure control mechanism
/// 
/// Monitors queue depth and signals when to pause/resume based on
/// configurable watermarks.
/// 
/// ## Example
/// ```rust
/// use edgeguard_core::stream::BackpressureControl;
/// 
/// let control = BackpressureControl::new(80, 20); // Pause at 80%, resume at 20%
/// control.update(current_depth, max_capacity);
/// 
/// if control.should_pause() {
///     // Stop producing
/// }
/// ```
pub struct BackpressureControl {
    /// High watermark (percentage)
    high_watermark: u8,
    /// Low watermark (percentage) 
    low_watermark: u8,
    /// Current state
    paused: AtomicBool,
    /// Current depth
    current_depth: AtomicUsize,
    /// Maximum capacity
    capacity: AtomicUsize,
}

impl BackpressureControl {
    /// Create new backpressure control
    /// 
    /// ## Parameters
    /// - `high_watermark`: Percentage (0-100) at which to pause
    /// - `low_watermark`: Percentage (0-100) at which to resume
    pub fn new(high_watermark: u8, low_watermark: u8) -> Self {
        debug_assert!(high_watermark > low_watermark);
        debug_assert!(high_watermark <= 100);
        
        Self {
            high_watermark,
            low_watermark,
            paused: AtomicBool::new(false),
            current_depth: AtomicUsize::new(0),
            capacity: AtomicUsize::new(100), // Default capacity
        }
    }
    
    /// Update current state
    pub fn update(&self, current: usize, capacity: usize) {
        self.current_depth.store(current, Ordering::Relaxed);
        self.capacity.store(capacity, Ordering::Relaxed);
        
        let percentage = (current * 100) / capacity.max(1);
        
        if percentage >= self.high_watermark as usize {
            self.paused.store(true, Ordering::Relaxed);
        } else if percentage <= self.low_watermark as usize {
            self.paused.store(false, Ordering::Relaxed);
        }
    }
    
    /// Check if should pause
    pub fn should_pause(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }
    
    /// Force resume
    pub fn resume(&self) {
        self.paused.store(false, Ordering::Relaxed);
    }
    
    /// Get current stats
    pub fn stats(&self) -> (usize, usize) {
        (
            self.current_depth.load(Ordering::Relaxed),
            self.capacity.load(Ordering::Relaxed),
        )
    }
}

impl Default for BackpressureControl {
    fn default() -> Self {
        Self::new(80, 20)
    }
}

/// Stream wrapper that adds backpressure support
/// 
/// Wraps any stream and adds backpressure signaling based on
/// downstream queue depth.
pub struct BackpressureWrapper<S: Stream> {
    /// Inner stream
    inner: S,
    /// Backpressure control
    control: BackpressureControl,
    /// Pending items when paused
    pending: heapless::Vec<S::Item, 16>,
}

impl<S: Stream> BackpressureWrapper<S> {
    /// Create new backpressure wrapper
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            control: BackpressureControl::default(),
            pending: heapless::Vec::new(),
        }
    }
    
    /// Create with custom watermarks
    pub fn with_watermarks(inner: S, high: u8, low: u8) -> Self {
        Self {
            inner,
            control: BackpressureControl::new(high, low),
            pending: heapless::Vec::new(),
        }
    }
    
    /// Update queue depth
    pub fn update_depth(&mut self, current: usize, capacity: usize) {
        self.control.update(current, capacity);
    }
}

impl<S: Stream> Stream for BackpressureWrapper<S>
where
    S::Item: Clone,
{
    type Item = S::Item;
    type Error = S::Error;
    
    fn poll_next(&mut self) -> nb::Result<Self::Item, Self::Error> {
        // First, try to drain pending items
        if let Some(item) = self.pending.pop() {
            return Ok(item);
        }
        
        // Check backpressure
        if self.control.should_pause() {
            return Err(nb::Error::WouldBlock);
        }
        
        // Pull from inner stream
        match self.inner.poll_next() {
            Ok(item) => {
                // Update depth based on pending count
                let (_, capacity) = self.control.stats();
                self.control.update(self.pending.len(), capacity);
                Ok(item)
            }
            Err(e) => Err(e),
        }
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (inner_min, inner_max) = self.inner.size_hint();
        let pending = self.pending.len();
        
        (
            inner_min.saturating_add(pending),
            inner_max.map(|max| max.saturating_add(pending)),
        )
    }
}

impl<S: Stream> BackpressureStream for BackpressureWrapper<S>
where
    S::Item: Clone,
{
    fn should_pause(&self) -> bool {
        self.control.should_pause()
    }
    
    fn resume(&mut self) {
        self.control.resume();
    }
    
    fn backpressure_stats(&self) -> (usize, usize) {
        self.control.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn backpressure_control() {
        let control = BackpressureControl::new(80, 20);
        
        // Initially not paused
        assert!(!control.should_pause());
        
        // Update to high watermark
        control.update(85, 100);
        assert!(control.should_pause());
        
        // Still paused at middle
        control.update(50, 100);
        assert!(control.should_pause());
        
        // Resume at low watermark
        control.update(15, 100);
        assert!(!control.should_pause());
    }
}