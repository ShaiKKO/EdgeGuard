//! Built-in pipeline stages for common processing patterns
//!
//! This module provides pre-built stages for validation, filtering,
//! routing, cross-validation, and aggregation of sensor events.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

#[cfg(feature = "std")]
use std::boxed::Box;

use heapless::{Vec, FnvIndexMap};

use crate::{
    events::{Event, SensorType, ValidationStatus, ConstraintFlags, CrossValidationType, 
             CrossValidationDetails, InlineString, EventBuilder},
    errors::ValidationError,
    traits::{Validator, ValidationContext},
    time::Timestamp,
    constants::pipeline::{
        MAX_ROUTES, MAX_SENSOR_PAIRS, MAX_AGGREGATION_WINDOW,
        MAX_RECENT_READINGS, CROSS_VALIDATION_TIME_WINDOW_MS,
        SQRT_EPSILON, SQRT_MAX_ITERATIONS, NEWTON_SQRT_DIVISOR,
        MIN_STDDEV_SAMPLES, VARIANCE_DIVISOR_OFFSET, MEDIAN_EVEN_DIVISOR,
        AGG_MIN_INITIAL, AGG_MAX_INITIAL, EMPTY_BUFFER_VALUE, DEFAULT_INTERVAL_MS,
        DEW_POINT_VALIDATION_MARGIN,
    },
};

use super::{
    PipelineStage, PipelineError, PipelineResult, StageOutput,
};

// No need to implement BitOr - we'll use the methods provided

/// Window specification for aggregation
#[derive(Debug, Clone, Copy)]
pub enum WindowSpec {
    /// Time-based window
    Time { duration_ms: u32 },
    /// Count-based window
    Count { size: u16 },
    /// Tumbling window (non-overlapping)
    Tumbling { duration_ms: u32 },
}

/// Aggregation method for windowed data
#[derive(Debug, Clone, Copy)]
pub enum AggregationMethod {
    /// Average of values
    Mean,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Sum of values
    Sum,
    /// Standard deviation
    StdDev,
    /// Median value
    Median,
}

// ============================================================================
// ValidationStage - Applies validators to sensor readings
// ============================================================================

/// Stage that applies validation to sensor readings
pub struct ValidationStage<V: Validator + Send> {
    validator: V,
    sensor_type: SensorType,
    constraints: ConstraintFlags,
}

impl<V: Validator<Value = f32> + Send> ValidationStage<V> {
    pub fn new(validator: V, sensor_type: SensorType) -> Self {
        Self {
            validator,
            sensor_type,
            constraints: {
                let mut flags = ConstraintFlags::empty();
                flags.set(ConstraintFlags::RANGE);
                flags.set(ConstraintFlags::RATE);
                flags
            },
        }
    }
}

impl<V: Validator<Value = f32, Error = ValidationError> + Send> PipelineStage for ValidationStage<V> {
    fn process(&mut self, event: Event, output: &mut StageOutput) -> PipelineResult<()> {
        // Check if this is a sensor reading we should process
        let should_validate = matches!(&event, 
            Event::SensorReading { sensor_type, .. } if *sensor_type == self.sensor_type
        );
        
        if !should_validate {
            output.push(event);
            return Ok(());
        }
        
        // Clone the event to work with it
        let event_clone = event.clone();
        
        // Pass through original event
        output.push(event);
        
        // Now work with the clone
        match event_clone {
            Event::SensorReading { sensor_id, sensor_type, value, timestamp, quality } => {
                // Create validation context
                let context = ValidationContext {
                    history: crate::buffer::CircularBuffer::new(),
                    timestamp,
                    ambient_temp: None,
                    ambient_humidity: None,
                    sensor_quality: quality,
                };
                
                // Validate the reading
                let status = match self.validator.validate(&value, &context) {
                    Ok(_) => ValidationStatus::Valid,
                    Err(ValidationError::OutOfRange { .. }) => ValidationStatus::OutOfRange,
                    Err(ValidationError::RateExceeded { .. }) => ValidationStatus::RateExceeded,
                    Err(ValidationError::InvalidValue) => ValidationStatus::InvalidValue,
                    Err(ValidationError::SensorQualityBad { .. }) => ValidationStatus::SensorQualityBad,
                    Err(ValidationError::CrossValidationFailed { .. }) => ValidationStatus::CrossValidationFailed,
                    Err(ValidationError::InsufficientData { .. }) => ValidationStatus::InvalidValue,
                };
                
                // Emit validation result
                let validation_event = EventBuilder::new(timestamp)
                    .sensor(sensor_id.as_str(), sensor_type)
                    .validation(status, self.constraints);
        
                if let Some(ve) = validation_event {
                    output.push(ve);
                }
            }
            _ => unreachable!(),
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "ValidationStage"
    }
}

// ============================================================================
// FilterStage - Filters events based on predicate
// ============================================================================

/// Stage that filters events based on a predicate function
pub struct FilterStage<F> {
    predicate: F,
    name: &'static str,
}

impl<F> FilterStage<F>
where
    F: Fn(&Event) -> bool + Send,
{
    pub fn new(predicate: F) -> Self {
        Self { predicate, name: "FilterStage" }
    }
}

impl<F> PipelineStage for FilterStage<F>
where
    F: Fn(&Event) -> bool + Send,
{
    fn process(&mut self, event: Event, output: &mut StageOutput) -> PipelineResult<()> {
        if (self.predicate)(&event) {
            output.push(event);
        }
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        self.name
    }
}

// ============================================================================
// RouterStage - Routes events to specific handlers
// ============================================================================

/// Stage that routes events to different handlers based on sensor type
pub struct RouterStage {
    /// Maps sensor types to their processing stages
    routes: FnvIndexMap<SensorType, Box<dyn PipelineStage>, MAX_ROUTES>,
    /// Default handler for unmatched types
    default_handler: Option<Box<dyn PipelineStage>>,
}

impl RouterStage {
    pub fn new() -> Self {
        Self {
            routes: FnvIndexMap::new(),
            default_handler: None,
        }
    }
    
    /// Add a route for a specific sensor type
    pub fn add_route(
        &mut self, 
        sensor_type: SensorType, 
        stage: Box<dyn PipelineStage>
    ) -> Result<(), PipelineError> {
        self.routes.insert(sensor_type, stage)
            .map(|_| ())
            .map_err(|_| PipelineError::ResourceExhausted)
    }
    
    /// Set default handler for unmatched sensor types
    pub fn set_default(&mut self, stage: Box<dyn PipelineStage>) {
        self.default_handler = Some(stage);
    }
}

impl PipelineStage for RouterStage {
    fn process(&mut self, event: Event, output: &mut StageOutput) -> PipelineResult<()> {
        let sensor_type = match &event {
            Event::SensorReading { sensor_type, .. } => Some(*sensor_type),
            Event::BatchReading { sensor_type, .. } => Some(*sensor_type),
            _ => None,
        };
        
        if let Some(st) = sensor_type {
            if let Some(handler) = self.routes.get_mut(&st) {
                return handler.process(event, output);
            }
        }
        
        // Use default handler or pass through
        if let Some(ref mut handler) = self.default_handler {
            handler.process(event, output)
        } else {
            output.push(event);
            Ok(())
        }
    }
    
    fn name(&self) -> &'static str {
        "RouterStage"
    }
}

impl Default for RouterStage {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CrossValidationStage - Validates relationships between sensors
// ============================================================================

/// Cross-validation configuration
struct CrossValidationPair {
    primary: SensorType,
    secondary: SensorType,
    validation_type: CrossValidationType,
}

/// Stage that performs cross-sensor validation
pub struct CrossValidationStage {
    /// Configured sensor pairs for cross-validation
    pairs: Vec<CrossValidationPair, MAX_SENSOR_PAIRS>,
    /// Recent readings buffer for each sensor type
    recent_readings: FnvIndexMap<SensorType, (f32, Timestamp, InlineString), MAX_RECENT_READINGS>,
    /// Time window for considering readings as "recent"
    time_window_ms: u32,
}

impl CrossValidationStage {
    pub fn new() -> Self {
        Self {
            pairs: Vec::new(),
            recent_readings: FnvIndexMap::new(),
            time_window_ms: CROSS_VALIDATION_TIME_WINDOW_MS,
        }
    }
    
    /// Add a sensor pair for cross-validation
    pub fn add_pair(
        &mut self,
        primary: SensorType,
        secondary: SensorType,
        validation_type: CrossValidationType,
    ) -> Result<(), PipelineError> {
        let pair = CrossValidationPair {
            primary,
            secondary,
            validation_type,
        };
        
        self.pairs.push(pair)
            .map_err(|_| PipelineError::ResourceExhausted)
    }
    
    /// Set time window for recent readings
    pub fn set_time_window(&mut self, window_ms: u32) {
        self.time_window_ms = window_ms;
    }
    
    /// Validate dew point constraint
    fn validate_dew_point(
        &self,
        temp: f32,
        humidity: f32,
    ) -> ValidationStatus {
        // Use lookup table for dew point calculation
        if let Some(dew_point) = crate::lookup::dew_point_lookup(temp, humidity) {
            if dew_point > temp {
                ValidationStatus::CrossValidationFailed
            } else {
                ValidationStatus::Valid
            }
        } else {
            ValidationStatus::InvalidValue
        }
    }
    
    /// Check if two timestamps are within the time window
    fn within_window(&self, t1: Timestamp, t2: Timestamp) -> bool {
        let diff = if t1 > t2 { t1 - t2 } else { t2 - t1 };
        diff <= self.time_window_ms as u64
    }
}

impl PipelineStage for CrossValidationStage {
    fn process(&mut self, event: Event, output: &mut StageOutput) -> PipelineResult<()> {
        // Always pass through the original event
        output.push(event.clone());
        
        // Extract sensor reading data if applicable
        if let Event::SensorReading { sensor_id, sensor_type, value, timestamp, .. } = &event {
            // Store this reading
            self.recent_readings.insert(
                *sensor_type,
                (*value, *timestamp, sensor_id.clone())
            ).ok(); // Ignore if map is full
            
            // Check all pairs involving this sensor type
            for pair in &self.pairs {
                let (other_type, is_primary) = if pair.primary == *sensor_type {
                    (pair.secondary, true)
                } else if pair.secondary == *sensor_type {
                    (pair.primary, false)
                } else {
                    continue;
                };
                
                // Look for recent reading from the other sensor
                if let Some(&(other_value, other_timestamp, ref other_id)) = 
                    self.recent_readings.get(&other_type) 
                {
                    // Check if readings are within time window
                    if !self.within_window(*timestamp, other_timestamp) {
                        continue;
                    }
                    
                    // Perform validation based on type
                    let status = match pair.validation_type {
                        CrossValidationType::DewPoint => {
                            if is_primary {
                                // Temperature is primary, humidity is secondary
                                self.validate_dew_point(*value, other_value)
                            } else {
                                // Humidity is primary, temperature is secondary
                                self.validate_dew_point(other_value, *value)
                            }
                        }
                        _ => ValidationStatus::Valid, // Other types not implemented yet
                    };
                    
                    // Emit cross-validation result
                    let details = CrossValidationDetails {
                        expected_value: if is_primary { *value } else { other_value },
                        actual_value: if is_primary { other_value } else { *value },
                        deviation_percent: DEW_POINT_VALIDATION_MARGIN, // Could be enhanced
                    };
                    
                    let primary_id: InlineString;
                    let secondary_id: InlineString;
                    if is_primary {
                        primary_id = sensor_id.clone();
                        secondary_id = other_id.clone();
                    } else {
                        primary_id = other_id.clone();
                        secondary_id = sensor_id.clone();
                    }
                    
                    // Create cross validation event manually
                    let cross_event = Some(Event::CrossValidationResult {
                        primary_sensor: primary_id,
                        related_sensor: secondary_id,
                        validation_type: pair.validation_type,
                        status,
                        details,
                        timestamp: *timestamp,
                    });
                    
                    if let Some(ce) = cross_event {
                        output.push(ce);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "CrossValidationStage"
    }
}

impl Default for CrossValidationStage {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// AggregationStage - Aggregates sensor readings over windows
// ============================================================================

/// Buffer for aggregating sensor readings
struct AggregationBuffer {
    values: Vec<f32, MAX_AGGREGATION_WINDOW>,
    timestamps: Vec<Timestamp, MAX_AGGREGATION_WINDOW>,
    sensor_id: InlineString,
    sensor_type: SensorType,
    window_start: Timestamp,
}

/// Stage that aggregates sensor readings over time or count windows
pub struct AggregationStage {
    window: WindowSpec,
    method: AggregationMethod,
    sensor_type: SensorType,
    buffer: Option<AggregationBuffer>,
}

impl AggregationStage {
    pub fn new(window: WindowSpec, method: AggregationMethod, sensor_type: SensorType) -> Self {
        Self {
            window,
            method,
            sensor_type,
            buffer: None,
        }
    }
    
    /// Calculate mean of values
    fn calculate_mean(values: &[f32]) -> f32 {
        if values.is_empty() {
            return EMPTY_BUFFER_VALUE;
        }
        let sum: f32 = values.iter().sum();
        sum / values.len() as f32
    }
    
    /// Calculate standard deviation
    fn calculate_std_dev(values: &[f32], mean: f32) -> f32 {
        if values.len() < MIN_STDDEV_SAMPLES {
            return EMPTY_BUFFER_VALUE;
        }
        let variance: f32 = values.iter()
            .map(|&v| {
                let diff = v - mean;
                diff * diff
            })
            .sum::<f32>() / (values.len() - VARIANCE_DIVISOR_OFFSET) as f32;
        // Newton's method for square root
        let mut x = variance;
        let mut last_x = 0.0;
        let mut iterations = 0;
        
        while (x - last_x).abs() > SQRT_EPSILON && iterations < SQRT_MAX_ITERATIONS {
            last_x = x;
            x = (x + variance / x) / NEWTON_SQRT_DIVISOR;
            iterations += 1;
        }
        x
    }
    
    /// Calculate median (requires sorted values)
    fn calculate_median(values: &mut [f32]) -> f32 {
        if values.is_empty() {
            return EMPTY_BUFFER_VALUE;
        }
        
        // Simple bubble sort for small arrays
        for i in 0..values.len() {
            for j in 0..values.len() - i - 1 {
                if values[j] > values[j + 1] {
                    values.swap(j, j + 1);
                }
            }
        }
        
        let mid = values.len() / 2;
        if values.len() % 2 == 0 {
            (values[mid - 1] + values[mid]) / MEDIAN_EVEN_DIVISOR
        } else {
            values[mid]
        }
    }
    
    /// Process the buffered values and emit aggregated event
    fn process_buffer(&mut self, output: &mut StageOutput) -> PipelineResult<()> {
        if let Some(ref mut buffer) = self.buffer {
            if buffer.values.is_empty() {
                return Ok(());
            }
            
            let result = match self.method {
                AggregationMethod::Mean => Self::calculate_mean(&buffer.values),
                AggregationMethod::Min => buffer.values.iter().cloned().fold(AGG_MIN_INITIAL, f32::min),
                AggregationMethod::Max => buffer.values.iter().cloned().fold(AGG_MAX_INITIAL, f32::max),
                AggregationMethod::Sum => buffer.values.iter().sum(),
                AggregationMethod::StdDev => {
                    let mean = Self::calculate_mean(&buffer.values);
                    Self::calculate_std_dev(&buffer.values, mean)
                }
                AggregationMethod::Median => {
                    let mut values = Vec::<f32, MAX_AGGREGATION_WINDOW>::new();
                    for &v in &buffer.values {
                        values.push(v).ok();
                    }
                    Self::calculate_median(values.as_mut())
                }
            };
            
            // Create batch reading event
            let interval_ms = if buffer.timestamps.len() > 1 {
                let duration = buffer.timestamps.last().unwrap() - buffer.timestamps.first().unwrap();
                (duration / (buffer.timestamps.len() as u64 - 1)) as u16
            } else {
                DEFAULT_INTERVAL_MS
            };
            
            let min_val = buffer.values.iter().cloned().fold(AGG_MIN_INITIAL, f32::min);
            let max_val = buffer.values.iter().cloned().fold(AGG_MAX_INITIAL, f32::max);
            
            // Create batch event manually
            let batch_event = Some(Event::BatchReading {
                sensor_id: buffer.sensor_id.clone(),
                sensor_type: buffer.sensor_type,
                base_timestamp: buffer.window_start,
                count: buffer.values.len() as u16,
                interval_ms,
                mean_value: result,
                min_value: min_val,
                max_value: max_val,
            });
            
            if let Some(be) = batch_event {
                output.push(be);
            }
            
            // Clear buffer for next window
            buffer.values.clear();
            buffer.timestamps.clear();
        }
        
        Ok(())
    }
}

impl PipelineStage for AggregationStage {
    fn process(&mut self, event: Event, output: &mut StageOutput) -> PipelineResult<()> {
        match &event {
            Event::SensorReading { sensor_id, sensor_type, value, timestamp, .. } => {
                // Only aggregate our sensor type
                if *sensor_type != self.sensor_type {
                    output.push(event);
                    return Ok(());
                }
                
                // Initialize buffer if needed
                if self.buffer.is_none() {
                    self.buffer = Some(AggregationBuffer {
                        values: Vec::new(),
                        timestamps: Vec::new(),
                        sensor_id: sensor_id.clone(),
                        sensor_type: *sensor_type,
                        window_start: *timestamp,
                    });
                }
                
                // Check if buffer is full or window complete
                let should_process = {
                    let buffer = self.buffer.as_ref().unwrap();
                    let buffer_full = buffer.values.is_full() || buffer.timestamps.is_full();
                    let window_complete = match self.window {
                        WindowSpec::Time { duration_ms } => {
                            *timestamp - buffer.window_start >= duration_ms as u64
                        }
                        WindowSpec::Count { size } => {
                            buffer.values.len() >= size as usize
                        }
                        WindowSpec::Tumbling { duration_ms } => {
                            *timestamp - buffer.window_start >= duration_ms as u64
                        }
                    };
                    (buffer_full, window_complete)
                };
                
                // Process buffer if needed
                match should_process {
                    (true, _) => {
                        // Buffer full, process it
                        self.process_buffer(output)?;
                        
                        // Reinitialize buffer
                        if let Some(ref mut buffer) = self.buffer {
                            buffer.values.clear();
                            buffer.timestamps.clear();
                            buffer.window_start = *timestamp;
                            buffer.values.push(*value).ok();
                            buffer.timestamps.push(*timestamp).ok();
                        }
                    }
                    (false, true) => {
                        // Window complete, process it
                        self.process_buffer(output)?;
                        
                        // Update window start based on type
                        if let Some(ref mut buffer) = self.buffer {
                            buffer.window_start = match self.window {
                                WindowSpec::Tumbling { duration_ms } => {
                                    ((timestamp / duration_ms as u64) + 1) * duration_ms as u64
                                }
                                _ => *timestamp,
                            };
                        }
                    }
                    (false, false) => {
                        // Normal case, just add to buffer
                        if let Some(ref mut buffer) = self.buffer {
                            buffer.values.push(*value).ok();
                            buffer.timestamps.push(*timestamp).ok();
                        }
                    }
                }
                
                Ok(())
            }
            _ => {
                // Pass through non-reading events
                output.push(event);
                Ok(())
            }
        }
    }
    
    fn name(&self) -> &'static str {
        "AggregationStage"
    }
    
    fn reset(&mut self) {
        self.buffer = None;
    }
}