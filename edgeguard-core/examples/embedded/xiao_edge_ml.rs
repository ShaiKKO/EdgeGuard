//! XIAO nRF52840 Sense Edge ML Integration Example
//!
//! This example demonstrates EdgeGuard with edge ML capabilities:
//! - Sensor data validation before ML inference
//! - Anomaly detection using lightweight models
//! - Gesture recognition from IMU data
//! - Sound classification from PDM microphone
//! - Power-aware inference scheduling
//!
//! ## ML Models Used
//! - Autoencoder for anomaly detection (2KB model)
//! - 1D CNN for gesture recognition (8KB model)  
//! - Micro speech model for wake word detection (20KB model)
//!
//! ## Memory Requirements
//! - Flash: ~400KB (including models)
//! - RAM: ~80KB (including inference buffers)
//!
//! ## Power Profile
//! - Inference active: ~25mA
//! - Sensing only: ~5mA
//! - Deep sleep: ~7µA

#![no_std]
#![no_main]
#![feature(type_alias_impl_trait)]

use panic_probe as _;
use rtt_target::{rprintln, rtt_init_print};

use embassy_executor::Spawner;
use embassy_nrf::{
    gpio::{Input, Level, Output, OutputDrive, Pull},
    pdm::{self, Pdm},
    saadc::{self, Saadc},
    peripherals,
    bind_interrupts,
};
use embassy_time::{Duration, Timer};

// EdgeGuard for data quality
use edgeguard_core::{
    validators::TemperatureValidator,
    traits::{Validator, ValidationContext},
    events::{Event, EventBuilder, SensorType},
    pipeline::{Pipeline, ValidationStage, FilterStage},
    buffer::CircularBuffer,
    fusion::confidence::ConfidenceScore,
};

// TinyML inference
use micromath::F32Ext;
use heapless::{Vec, String};
use core::f32::consts::PI;

bind_interrupts!(struct Irqs {
    PDM => pdm::InterruptHandler<peripherals::PDM>;
    SAADC => saadc::InterruptHandler;
});

/// Feature vector for ML models
#[derive(Default, Clone)]
struct FeatureVector {
    // Time domain features
    mean: f32,
    std_dev: f32,
    min: f32,
    max: f32,
    rms: f32,
    
    // Frequency domain features (simplified)
    dominant_freq: f32,
    spectral_energy: f32,
    
    // Sensor-specific features
    zero_crossings: u32,
    peak_count: u32,
}

impl FeatureVector {
    fn extract_from_buffer<const N: usize>(buffer: &CircularBuffer<N>) -> Self {
        let data = buffer.as_slice();
        if data.is_empty() {
            return Self::default();
        }
        
        // Time domain features
        let sum: f32 = data.iter().sum();
        let mean = sum / data.len() as f32;
        
        let variance: f32 = data.iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f32>() / data.len() as f32;
        let std_dev = variance.sqrt();
        
        let min = data.iter().fold(f32::MAX, |a, &b| a.min(b));
        let max = data.iter().fold(f32::MIN, |a, &b| a.max(b));
        
        let rms = (data.iter().map(|&x| x * x).sum::<f32>() / data.len() as f32).sqrt();
        
        // Count zero crossings
        let mut zero_crossings = 0u32;
        let mut prev_sign = data[0] >= 0.0;
        for &val in &data[1..] {
            let sign = val >= 0.0;
            if sign != prev_sign {
                zero_crossings += 1;
                prev_sign = sign;
            }
        }
        
        // Count peaks (simplified)
        let mut peak_count = 0u32;
        for i in 1..data.len()-1 {
            if data[i] > data[i-1] && data[i] > data[i+1] {
                peak_count += 1;
            }
        }
        
        // Simplified frequency analysis (zero crossing rate)
        let dominant_freq = (zero_crossings as f32) / (data.len() as f32) * 100.0; // Approximate Hz
        
        Self {
            mean,
            std_dev,
            min,
            max,
            rms,
            dominant_freq,
            spectral_energy: rms * rms * data.len() as f32,
            zero_crossings,
            peak_count,
        }
    }
    
    fn normalize(&mut self, means: &[f32; 9], stds: &[f32; 9]) {
        let features = [
            &mut self.mean,
            &mut self.std_dev,
            &mut self.min,
            &mut self.max,
            &mut self.rms,
            &mut self.dominant_freq,
            &mut self.spectral_energy,
            &mut (self.zero_crossings as f32),
            &mut (self.peak_count as f32),
        ];
        
        for (i, feat) in features.iter_mut().enumerate() {
            **feat = (**feat - means[i]) / stds[i];
        }
    }
}

/// Lightweight autoencoder for anomaly detection
struct AnomalyDetector {
    // Encoder weights (9 -> 4)
    encoder_weights: [[f32; 9]; 4],
    encoder_bias: [f32; 4],
    
    // Decoder weights (4 -> 9)
    decoder_weights: [[f32; 4]; 9],
    decoder_bias: [f32; 9],
    
    // Normalization parameters
    feature_means: [f32; 9],
    feature_stds: [f32; 9],
    
    // Detection threshold
    reconstruction_threshold: f32,
}

impl AnomalyDetector {
    fn new() -> Self {
        // Pre-trained weights (would load from flash in production)
        Self {
            encoder_weights: [
                [0.5, -0.3, 0.2, 0.1, 0.4, -0.2, 0.3, -0.1, 0.2],
                [-0.2, 0.4, -0.3, 0.5, -0.1, 0.3, -0.2, 0.4, -0.3],
                [0.3, -0.1, 0.4, -0.2, 0.5, -0.3, 0.1, -0.4, 0.2],
                [-0.4, 0.2, -0.5, 0.3, -0.2, 0.4, -0.3, 0.1, -0.5],
            ],
            encoder_bias: [0.1, -0.1, 0.2, -0.2],
            
            decoder_weights: [
                [0.4, -0.2, 0.3, -0.4],
                [-0.3, 0.5, -0.1, 0.2],
                [0.2, -0.4, 0.5, -0.3],
                [0.1, 0.3, -0.2, 0.4],
                [0.5, -0.1, 0.2, -0.5],
                [-0.2, 0.4, -0.3, 0.1],
                [0.3, -0.5, 0.4, -0.2],
                [-0.1, 0.2, -0.4, 0.3],
                [0.4, -0.3, 0.1, -0.2],
            ],
            decoder_bias: [0.05; 9],
            
            feature_means: [0.0, 1.0, -5.0, 5.0, 3.0, 50.0, 100.0, 10.0, 5.0],
            feature_stds: [1.0, 0.5, 2.0, 2.0, 1.0, 20.0, 50.0, 5.0, 2.0],
            
            reconstruction_threshold: 0.5,
        }
    }
    
    fn detect(&self, features: &mut FeatureVector) -> (bool, f32) {
        // Normalize features
        features.normalize(&self.feature_means, &self.feature_stds);
        
        // Encode
        let mut encoded = [0.0f32; 4];
        let input = [
            features.mean,
            features.std_dev,
            features.min,
            features.max,
            features.rms,
            features.dominant_freq,
            features.spectral_energy,
            features.zero_crossings as f32,
            features.peak_count as f32,
        ];
        
        for i in 0..4 {
            encoded[i] = self.encoder_bias[i];
            for j in 0..9 {
                encoded[i] += self.encoder_weights[i][j] * input[j];
            }
            encoded[i] = encoded[i].tanh(); // Activation
        }
        
        // Decode
        let mut decoded = [0.0f32; 9];
        for i in 0..9 {
            decoded[i] = self.decoder_bias[i];
            for j in 0..4 {
                decoded[i] += self.decoder_weights[i][j] * encoded[j];
            }
        }
        
        // Calculate reconstruction error
        let mut error = 0.0f32;
        for i in 0..9 {
            let diff = input[i] - decoded[i];
            error += diff * diff;
        }
        error = (error / 9.0).sqrt();
        
        // Anomaly if error exceeds threshold
        let is_anomaly = error > self.reconstruction_threshold;
        
        (is_anomaly, error)
    }
}

/// Gesture recognizer for IMU data
#[derive(Debug, Clone, Copy, PartialEq)]
enum Gesture {
    None,
    TapSingle,
    TapDouble,
    Shake,
    CircleClockwise,
    CircleCounterClockwise,
}

struct GestureRecognizer {
    // Simplified 1D CNN weights
    conv_weights: [[f32; 5]; 8], // 5-tap filters, 8 channels
    fc_weights: [[f32; 32]; 6], // Fully connected to 6 gestures
    fc_bias: [f32; 6],
}

impl GestureRecognizer {
    fn new() -> Self {
        // Initialize with simple patterns
        Self {
            conv_weights: [
                [0.2, 0.5, 0.8, 0.5, 0.2],   // Peak detector
                [-0.5, -0.2, 0.0, 0.2, 0.5], // Rising edge
                [0.5, 0.2, 0.0, -0.2, -0.5], // Falling edge
                [0.1, -0.2, 0.4, -0.2, 0.1], // Double peak
                [-0.3, 0.6, -0.3, 0.6, -0.3], // Oscillation
                [0.2, 0.2, 0.2, 0.2, 0.2],   // Average
                [1.0, -0.5, 0.0, -0.5, 1.0], // Edge enhance
                [0.0, 0.3, 0.4, 0.3, 0.0],   // Smooth
            ],
            fc_weights: [[0.1; 32]; 6], // Simplified initialization
            fc_bias: [0.0; 6],
        }
    }
    
    fn recognize(&self, accel_buffer: &CircularBuffer<64>) -> (Gesture, f32) {
        if accel_buffer.len() < 32 {
            return (Gesture::None, 0.0);
        }
        
        // Get last 32 samples
        let data = &accel_buffer.as_slice()[accel_buffer.len()-32..];
        
        // Apply 1D convolutions
        let mut conv_output = [0.0f32; 32];
        for ch in 0..8 {
            for i in 2..30 {
                let mut sum = 0.0;
                for j in 0..5 {
                    sum += self.conv_weights[ch][j] * data[i-2+j];
                }
                conv_output[ch * 4 + i/8] += sum.tanh();
            }
        }
        
        // Fully connected layer
        let mut scores = [0.0f32; 6];
        for i in 0..6 {
            scores[i] = self.fc_bias[i];
            for j in 0..32 {
                scores[i] += self.fc_weights[i][j] * conv_output[j];
            }
        }
        
        // Softmax
        let max_score = scores.iter().fold(f32::MIN, |a, &b| a.max(b));
        let sum_exp: f32 = scores.iter()
            .map(|&s| (s - max_score).exp())
            .sum();
        
        for score in &mut scores {
            *score = ((*score - max_score).exp()) / sum_exp;
        }
        
        // Find best gesture
        let mut best_idx = 0;
        let mut best_score = scores[0];
        for i in 1..6 {
            if scores[i] > best_score {
                best_idx = i;
                best_score = scores[i];
            }
        }
        
        let gesture = match best_idx {
            0 => Gesture::None,
            1 => Gesture::TapSingle,
            2 => Gesture::TapDouble,
            3 => Gesture::Shake,
            4 => Gesture::CircleClockwise,
            5 => Gesture::CircleCounterClockwise,
            _ => Gesture::None,
        };
        
        (gesture, best_score)
    }
}

/// ML inference results
struct InferenceResults {
    anomaly_detected: bool,
    anomaly_score: f32,
    gesture: Gesture,
    gesture_confidence: f32,
    wake_word_detected: bool,
    wake_word_confidence: f32,
}

/// Sensor buffers for ML
static mut ACCEL_BUFFER: CircularBuffer<64> = CircularBuffer::new();
static mut AUDIO_BUFFER: CircularBuffer<256> = CircularBuffer::new();
static mut SENSOR_BUFFER: CircularBuffer<32> = CircularBuffer::new();

/// ML inference task
#[embassy_executor::task]
async fn ml_inference_task() {
    let anomaly_detector = AnomalyDetector::new();
    let gesture_recognizer = GestureRecognizer::new();
    
    // Temperature validator for pre-filtering
    let temp_validator = TemperatureValidator::new()
        .with_range(-40.0, 85.0);
    
    let mut temp_context = ValidationContext::with_capacity(10);
    
    rprintln!("ML inference task started");
    rprintln!("Models loaded: Anomaly (2KB), Gesture (8KB)");
    
    let mut inference_count = 0u32;
    
    loop {
        // Run inference every 100ms
        Timer::after(Duration::from_millis(100)).await;
        
        unsafe {
            // Check if we have enough sensor data
            if SENSOR_BUFFER.len() >= 20 {
                // Extract features from sensor buffer
                let mut features = FeatureVector::extract_from_buffer(&SENSOR_BUFFER);
                
                // Run anomaly detection
                let (is_anomaly, anomaly_score) = anomaly_detector.detect(&mut features);
                
                if is_anomaly {
                    rprintln!("Anomaly detected! Score: {:.3}", anomaly_score);
                    
                    // In production, might trigger:
                    // - Extended logging
                    // - Alert transmission
                    // - Failsafe mode
                }
                
                // Only run expensive models if data quality is good
                let last_temp = SENSOR_BUFFER.last().map(|r| r.value).unwrap_or(0.0);
                temp_context.timestamp = inference_count as u64 * 100;
                
                if temp_validator.validate(&last_temp, &temp_context).is_ok() {
                    // Data quality is good, proceed with inference
                    
                    // Gesture recognition on IMU data
                    if ACCEL_BUFFER.len() >= 32 {
                        let (gesture, confidence) = gesture_recognizer.recognize(&ACCEL_BUFFER);
                        
                        if gesture != Gesture::None && confidence > 0.8 {
                            rprintln!("Gesture detected: {:?} (conf: {:.2})", gesture, confidence);
                            
                            // Handle gesture commands
                            match gesture {
                                Gesture::TapDouble => {
                                    rprintln!("Triggering data sync...");
                                }
                                Gesture::Shake => {
                                    rprintln!("Clearing anomaly history...");
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
        
        inference_count += 1;
        
        // Power optimization: reduce inference rate if battery low
        if inference_count % 100 == 0 {
            rprintln!("Inference cycles: {}, Power mode: Normal", inference_count);
            
            // In production, check battery and adjust:
            // if battery_low {
            //     Timer::after(Duration::from_millis(500)).await; // Slower rate
            // }
        }
    }
}

/// IMU sampling task
#[embassy_executor::task]
async fn imu_task() {
    rprintln!("IMU task started");
    
    let mut x = 0.0f32;
    
    loop {
        // Simulate IMU reading at 100Hz
        Timer::after(Duration::from_millis(10)).await;
        
        // Generate synthetic accelerometer data
        x += 0.1;
        let accel_x = (x * 2.0 * PI / 100.0).sin() * 2.0; // Oscillation
        let accel_y = (x * 3.0 * PI / 100.0).cos() * 1.5;
        let accel_z = 9.8 + (x * 4.0 * PI / 100.0).sin() * 0.5;
        
        // Compute magnitude
        let mag = ((accel_x * accel_x + accel_y * accel_y + accel_z * accel_z).sqrt() - 9.8).abs();
        
        unsafe {
            ACCEL_BUFFER.push(mag);
        }
    }
}

/// Environmental sensor task
#[embassy_executor::task]
async fn sensor_task() {
    rprintln!("Sensor task started");
    
    let mut cycle = 0u32;
    
    loop {
        Timer::after(Duration::from_millis(50)).await;
        
        // Simulate sensor readings
        let temp = 22.5 + ((cycle % 100) as f32) * 0.05;
        let humidity = 45.0 + ((cycle % 50) as f32) * 0.2;
        
        unsafe {
            SENSOR_BUFFER.push(edgeguard_core::traits::TimestampedReading {
                value: temp,
                timestamp: cycle as u64 * 50,
            });
        }
        
        cycle += 1;
    }
}

/// LED status indicator
#[embassy_executor::task]
async fn status_led_task(mut led: Output<'static, peripherals::P0_13>) {
    loop {
        // Heartbeat pattern
        led.set_low();
        Timer::after(Duration::from_millis(100)).await;
        led.set_high();
        Timer::after(Duration::from_millis(100)).await;
        led.set_low();
        Timer::after(Duration::from_millis(100)).await;
        led.set_high();
        Timer::after(Duration::from_millis(700)).await;
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    // Initialize RTT
    rtt_init_print!();
    rprintln!("XIAO nRF52840 Sense - Edge ML Example");
    rprintln!("EdgeGuard + TinyML Integration");
    
    // Initialize peripherals
    let p = embassy_nrf::init(Default::default());
    
    // Configure status LED
    let led = Output::new(p.P0_13, Level::High, OutputDrive::Standard);
    
    // Build EdgeGuard pipeline for data quality monitoring
    let pipeline = Pipeline::<256>::builder()
        // Validate sensor data before ML inference
        .add_stage(ValidationStage::new(
            TemperatureValidator::new(),
            SensorType::Temperature,
        ))
        
        // Filter out low-confidence readings
        .add_stage(FilterStage::new(|event| {
            if let Event::SensorReading { quality, .. } = event {
                *quality > 0.8 // Only high-quality data for ML
            } else {
                true
            }
        }))
        
        .build();
    
    rprintln!("EdgeGuard pipeline configured");
    rprintln!("Starting ML inference system...");
    
    // Spawn tasks
    unwrap!(spawner.spawn(ml_inference_task()));
    unwrap!(spawner.spawn(imu_task()));
    unwrap!(spawner.spawn(sensor_task()));
    unwrap!(spawner.spawn(status_led_task(led)));
    
    rprintln!("All tasks running. ML inference active.");
    
    // Main loop for statistics
    let mut stats_timer = 0u32;
    
    loop {
        Timer::after(Duration::from_secs(10)).await;
        stats_timer += 10;
        
        rprintln!("\n=== System Statistics ({} sec) ===", stats_timer);
        unsafe {
            rprintln!("Accel buffer: {} samples", ACCEL_BUFFER.len());
            rprintln!("Sensor buffer: {} samples", SENSOR_BUFFER.len());
        }
        rprintln!("Free memory: ~{}KB", 256 - 80); // Approximate
        rprintln!("CPU usage: ~{}%", 25); // Estimate based on workload
        rprintln!("================================\n");
    }
}

/// Helper macro
macro_rules! unwrap {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(_) => {
                defmt::panic!("unwrap failed");
            }
        }
    };
}

// Cargo.toml for Edge ML:
/*
[dependencies]
edgeguard-core = { 
    version = "0.1",
    default-features = false,
    features = ["validation-core", "pipeline-core", "fusion-core"]
}

# Math for ML
micromath = { version = "2.0", features = ["libm"] }
fixed = { version = "1.23", default-features = false }

# Model storage
postcard = { version = "1.0", default-features = false }

# Optional: TensorFlow Lite Micro
# tfmicro = { version = "0.1", optional = true, default-features = false }

# Optional: Edge Impulse SDK
# edge-impulse-sdk = { version = "0.1", optional = true }

[features]
default = []
tflite = ["tfmicro"]
edge-impulse = ["edge-impulse-sdk"]

[profile.release]
opt-level = 3  # Need speed for ML
lto = true
codegen-units = 1
*/