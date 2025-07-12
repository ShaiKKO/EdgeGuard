//! No-std compatible test harness for embedded and integration testing
//!
//! Provides:
//! - Custom test runner for embedded targets
//! - Memory usage tracking and watermarking
//! - Cycle-accurate performance measurement
//! - Deterministic test execution

#![cfg_attr(not(test), no_std)]

extern crate alloc;
use alloc::{vec::Vec, string::String, format};
use core::mem;

use edgeguard_core::{
    time::{Timestamp, TimeSource},
};

/// Test result tracking
#[derive(Debug, Clone)]
pub struct TestResult {
    pub name: &'static str,
    pub passed: bool,
    pub duration_us: u64,
    pub memory_used: usize,
    pub error_message: Option<String>,
}

/// Memory usage tracker
pub struct MemoryTracker {
    initial_usage: usize,
    peak_usage: usize,
    allocations: usize,
    deallocations: usize,
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            initial_usage: Self::current_usage(),
            peak_usage: 0,
            allocations: 0,
            deallocations: 0,
        }
    }
    
    pub fn update(&mut self) {
        let current = Self::current_usage();
        if current > self.peak_usage {
            self.peak_usage = current;
        }
    }
    
    pub fn record_allocation(&mut self, size: usize) {
        self.allocations += 1;
        self.update();
    }
    
    pub fn record_deallocation(&mut self, size: usize) {
        self.deallocations += 1;
    }
    
    pub fn peak_delta(&self) -> usize {
        self.peak_usage.saturating_sub(self.initial_usage)
    }
    
    #[cfg(feature = "std")]
    fn current_usage() -> usize {
        // In std environments, we can't accurately track heap usage
        // This would need platform-specific implementation
        0
    }
    
    #[cfg(not(feature = "std"))]
    fn current_usage() -> usize {
        // On embedded, this would read from allocator statistics
        // For now, return 0 as placeholder
        0
    }
}

/// Performance measurement using cycle counters
pub struct CycleCounter {
    start_cycles: u64,
}

impl CycleCounter {
    pub fn start() -> Self {
        Self {
            start_cycles: Self::read_cycles(),
        }
    }
    
    pub fn elapsed_cycles(&self) -> u64 {
        Self::read_cycles().saturating_sub(self.start_cycles)
    }
    
    pub fn elapsed_us(&self, cpu_freq_mhz: u32) -> u64 {
        self.elapsed_cycles() / (cpu_freq_mhz as u64)
    }
    
    #[cfg(target_arch = "arm")]
    fn read_cycles() -> u64 {
        // Read ARM cycle counter (DWT CYCCNT)
        // This requires DWT to be enabled in startup code
        #[cfg(feature = "cortex-m")]
        unsafe {
            use cortex_m::peripheral::DWT;
            DWT::get_cycle_count() as u64
        }
        #[cfg(not(feature = "cortex-m"))]
        0
    }
    
    #[cfg(target_arch = "riscv32")]
    fn read_cycles() -> u64 {
        // Read RISC-V cycle counter
        let cycles: u32;
        unsafe {
            core::arch::asm!("rdcycle {}", out(reg) cycles);
        }
        cycles as u64
    }
    
    #[cfg(not(any(target_arch = "arm", target_arch = "riscv32")))]
    fn read_cycles() -> u64 {
        // Fallback for other architectures
        // In tests, we can use a mock time source
        0
    }
}

/// Test harness for running integration tests
pub struct TestHarness {
    results: Vec<TestResult>,
    memory_tracker: MemoryTracker,
    cpu_freq_mhz: u32,
}

impl TestHarness {
    pub fn new(cpu_freq_mhz: u32) -> Self {
        Self {
            results: Vec::new(),
            memory_tracker: MemoryTracker::new(),
            cpu_freq_mhz,
        }
    }
    
    /// Run a single test case
    pub fn run_test<F>(&mut self, name: &'static str, test_fn: F) 
    where
        F: FnOnce() -> Result<(), String>
    {
        // Reset memory tracking
        self.memory_tracker = MemoryTracker::new();
        
        // Start timing
        let counter = CycleCounter::start();
        
        // Run test
        let result = test_fn();
        
        // Calculate metrics
        let duration_us = counter.elapsed_us(self.cpu_freq_mhz);
        let memory_used = self.memory_tracker.peak_delta();
        
        // Record result
        self.results.push(TestResult {
            name,
            passed: result.is_ok(),
            duration_us,
            memory_used,
            error_message: result.err(),
        });
    }
    
    /// Run a parameterized test
    pub fn run_parameterized_test<T, F>(
        &mut self,
        name: &'static str,
        params: &[T],
        test_fn: F,
    )
    where
        T: core::fmt::Debug,
        F: Fn(&T) -> Result<(), String>,
    {
        for (i, param) in params.iter().enumerate() {
            let test_name = format!("{}[{}]", name, i);
            self.run_test(
                // Leak the string to get 'static lifetime (okay for tests)
                alloc::boxed::Box::leak(test_name.into_boxed_str()),
                || test_fn(param),
            );
        }
    }
    
    /// Print test results summary
    pub fn print_summary(&self) {
        let total = self.results.len();
        let passed = self.results.iter().filter(|r| r.passed).count();
        let failed = total - passed;
        
        println!("\nTest Results:");
        println!("============");
        println!("Total:  {}", total);
        println!("Passed: {} ✓", passed);
        println!("Failed: {} ✗", failed);
        
        if failed > 0 {
            println!("\nFailed Tests:");
            for result in self.results.iter().filter(|r| !r.passed) {
                println!("  ✗ {}", result.name);
                if let Some(msg) = &result.error_message {
                    println!("    Error: {}", msg);
                }
            }
        }
        
        // Performance summary
        println!("\nPerformance Summary:");
        let total_us: u64 = self.results.iter().map(|r| r.duration_us).sum();
        let max_us = self.results.iter().map(|r| r.duration_us).max().unwrap_or(0);
        let total_memory: usize = self.results.iter().map(|r| r.memory_used).sum();
        
        println!("  Total time:   {} µs", total_us);
        println!("  Slowest test: {} µs", max_us);
        println!("  Total memory: {} bytes", total_memory);
    }
    
    /// Check if all tests passed
    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.passed)
    }
}

/// Assertion helpers for physics-aware testing
#[macro_export]
macro_rules! assert_physics_valid {
    ($value:expr, $validator:expr, $context:expr) => {
        match $validator.validate($value, $context) {
            Ok(_) => {},
            Err(e) => panic!("Physics validation failed: {:?}", e),
        }
    };
}

#[macro_export]
macro_rules! assert_thermally_plausible {
    ($current:expr, $previous:expr, $time_delta_s:expr) => {
        let rate = ($current - $previous).abs() / $time_delta_s;
        if rate > 5.0 { // Max 5°C/s for air
            panic!(
                "Thermally implausible rate: {:.2}°C/s (current: {}, previous: {}, dt: {}s)",
                rate, $current, $previous, $time_delta_s
            );
        }
    };
}

#[macro_export]
macro_rules! assert_within_tolerance {
    ($actual:expr, $expected:expr, $tolerance:expr) => {
        let diff = ($actual - $expected).abs();
        if diff > $tolerance {
            panic!(
                "Value {} not within tolerance {} of expected {} (diff: {})",
                $actual, $tolerance, $expected, diff
            );
        }
    };
}

/// Deterministic random number generator for tests
pub struct TestRng {
    state: u32,
}

impl TestRng {
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }
    
    pub fn next_u32(&mut self) -> u32 {
        // Xorshift algorithm
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        self.state
    }
    
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16777216.0
    }
    
    pub fn gen_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }
}

/// Test timing utilities
pub struct TestTimer<T: TimeSource> {
    time_source: T,
    checkpoints: Vec<(String, Timestamp)>,
}

impl<T: TimeSource> TestTimer<T> {
    pub fn new(time_source: T) -> Self {
        Self {
            time_source,
            checkpoints: Vec::new(),
        }
    }
    
    pub fn checkpoint(&mut self, name: &str) {
        let now = self.time_source.now();
        self.checkpoints.push((String::from(name), now));
    }
    
    pub fn print_intervals(&self) {
        if self.checkpoints.len() < 2 {
            return;
        }
        
        println!("\nTiming Intervals:");
        for i in 1..self.checkpoints.len() {
            let (name, time) = &self.checkpoints[i];
            let (_, prev_time) = &self.checkpoints[i - 1];
            let delta = time - prev_time;
            println!("  {}: {} ms", name, delta);
        }
    }
}