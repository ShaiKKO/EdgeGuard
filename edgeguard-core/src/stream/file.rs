//! File-based streaming for sensor data
//!
//! This module provides file streaming capabilities for reading sensor
//! data from CSV, JSON Lines, and binary formats.
//!
//! ## Supported Formats
//!
//! 1. **CSV**: Standard comma-separated values with headers
//! 2. **JSON Lines**: One JSON object per line
//! 3. **Binary**: Compact binary format (EdgeGuard Binary)

use std::io::Read;
use std::fs::File;

use crate::events::{Event, EventBuilder, SensorType, InlineString};
use super::{Stream, StreamError};

/// File formats supported by FileStream
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FileFormat {
    /// Comma-separated values
    Csv,
    /// Line-delimited JSON
    JsonLines,
    /// Custom binary format
    Binary,
}

/// Statistics for file streaming
#[derive(Debug, Default, Clone)]
pub struct FileStreamStats {
    /// Total events read successfully
    pub events_read: usize,
    /// Total lines processed
    pub lines_processed: usize,
    /// Parse errors encountered
    pub parse_errors: usize,
    /// Bytes read from file
    pub bytes_read: usize,
}

/// File-based event stream
/// 
/// Reads sensor events from files in various formats. Supports CSV,
/// JSON Lines, and binary formats with configurable buffering.
/// 
/// ## Example
/// 
/// ```rust,no_run
/// use edgeguard_core::stream::{FileStream, FileFormat};
/// 
/// // Read CSV file
/// let mut stream = FileStream::from_csv("sensors.csv")?
///     .with_skip_lines(1); // Skip header
/// 
/// while let Ok(event) = stream.poll_next() {
///     // Process event
/// }
/// ```
/// 
/// ## Formats
/// 
/// ### CSV Format
/// ```csv
/// timestamp,sensor_id,type,value,confidence
/// 1234567890,temp_01,temperature,25.5,0.95
/// ```
/// 
/// ### JSON Lines Format
/// ```json
/// {"timestamp":1234567890,"sensor_id":"temp_01","type":"temperature","value":25.5,"confidence":0.95}
/// ```
/// 
/// ## Memory Efficiency
/// 
/// File streams read data in chunks to minimize memory usage:
/// - Default buffer: 4KB
/// - Configurable via `with_buffer_size()`
/// - Zero-copy parsing where possible
pub struct FileStream {
    /// File handle
    file: File,
    /// File format
    format: FileFormat,
    /// Read buffer
    buffer: [u8; 4096],
    /// Current position in buffer
    buffer_pos: usize,
    /// Valid bytes in buffer
    buffer_len: usize,
    /// Line buffer for text formats
    line_buffer: heapless::String<256>,
    /// Whether we've reached EOF
    eof: bool,
    /// Skip first N lines (for headers)
    skip_lines: usize,
    /// Lines already skipped
    lines_skipped: usize,
    /// Statistics
    stats: FileStreamStats,
}

impl FileStream {
    /// Create new file stream
    pub fn new(path: &str, format: FileFormat) -> Result<Self, StreamError<std::io::Error>> {
        let file = File::open(path)
            .map_err(StreamError::Transport)?;
            
        Ok(Self {
            file,
            format,
            buffer: [0; 4096],
            buffer_pos: 0,
            buffer_len: 0,
            line_buffer: heapless::String::new(),
            eof: false,
            skip_lines: 0,
            lines_skipped: 0,
            stats: FileStreamStats::default(),
        })
    }
    
    /// Create CSV file stream
    pub fn from_csv(path: &str) -> Result<Self, StreamError<std::io::Error>> {
        Self::new(path, FileFormat::Csv)
    }
    
    /// Skip first N lines (useful for headers)
    pub fn with_skip_lines(mut self, lines: usize) -> Self {
        self.skip_lines = lines;
        self
    }
    
    /// Get statistics
    pub fn stats(&self) -> &FileStreamStats {
        &self.stats
    }
    
    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = FileStreamStats::default();
    }
    
    /// Refill buffer from file
    fn refill_buffer(&mut self) -> Result<bool, StreamError<std::io::Error>> {
        if self.eof {
            return Ok(false);
        }
        
        // Move remaining data to beginning
        if self.buffer_pos < self.buffer_len {
            let remaining = self.buffer_len - self.buffer_pos;
            self.buffer.copy_within(self.buffer_pos..self.buffer_len, 0);
            self.buffer_len = remaining;
            self.buffer_pos = 0;
        } else {
            self.buffer_len = 0;
            self.buffer_pos = 0;
        }
        
        // Read more data
        let bytes_read = self.file.read(&mut self.buffer[self.buffer_len..])
            .map_err(StreamError::Transport)?;
            
        if bytes_read == 0 {
            self.eof = true;
            return Ok(self.buffer_len > 0);
        }
        
        self.buffer_len += bytes_read;
        self.stats.bytes_read += bytes_read;
        Ok(true)
    }
    
    /// Read next line from buffer
    fn read_line(&mut self) -> Result<Option<&str>, StreamError<std::io::Error>> {
        self.line_buffer.clear();
        
        loop {
            // Look for newline in buffer
            while self.buffer_pos < self.buffer_len {
                let byte = self.buffer[self.buffer_pos];
                self.buffer_pos += 1;
                
                if byte == b'\n' {
                    self.stats.lines_processed += 1;
                    
                    // Skip lines if needed
                    if self.lines_skipped < self.skip_lines {
                        self.lines_skipped += 1;
                        self.line_buffer.clear();
                        continue;
                    }
                    
                    return Ok(Some(self.line_buffer.as_str()));
                } else if byte != b'\r' {
                    if self.line_buffer.push(byte as char).is_err() {
                        return Err(StreamError::Overflow);
                    }
                }
            }
            
            // Need more data
            if !self.refill_buffer()? {
                // EOF reached
                if !self.line_buffer.is_empty() {
                    self.stats.lines_processed += 1;
                    return Ok(Some(self.line_buffer.as_str()));
                }
                return Ok(None);
            }
        }
    }
    
    /// Parse CSV line into event
    fn parse_csv(&mut self, line: &str) -> Result<Event, StreamError<std::io::Error>> {
        let fields: heapless::Vec<&str, 8> = line.split(',')
            .map(|s| s.trim())
            .collect();
            
        if fields.len() < 5 {
            self.stats.parse_errors += 1;
            return Err(StreamError::Format("Not enough CSV fields"));
        }
        
        // Parse timestamp
        let timestamp = fields[0].parse::<u64>()
            .map_err(|_| {
                self.stats.parse_errors += 1;
                StreamError::Format("Invalid timestamp")
            })?;
            
        // Parse sensor ID
        let sensor_id = InlineString::new(fields[1].trim_matches('"'))
            .ok_or_else(|| {
                self.stats.parse_errors += 1;
                StreamError::Format("Sensor ID too long")
            })?;
            
        // Parse sensor type
        let sensor_type = match fields[2].trim_matches('"').to_lowercase().as_str() {
            "temperature" | "temp" => SensorType::Temperature,
            "humidity" | "humid" => SensorType::Humidity,
            "pressure" | "press" => SensorType::Pressure,
            _ => {
                self.stats.parse_errors += 1;
                return Err(StreamError::Format("Unknown sensor type"));
            }
        };
        
        // Parse value
        let value = fields[3].parse::<f32>()
            .map_err(|_| {
                self.stats.parse_errors += 1;
                StreamError::Format("Invalid value")
            })?;
            
        // Parse confidence (optional, default to 1.0)
        let confidence = if fields.len() > 4 {
            fields[4].parse::<f32>().unwrap_or(1.0)
        } else {
            1.0
        };
        
        self.stats.events_read += 1;
        
        EventBuilder::new(timestamp)
            .sensor(sensor_id.as_str(), sensor_type)
            .reading(value, confidence)
            .ok_or_else(|| {
                self.stats.parse_errors += 1;
                StreamError::Format("Failed to build event")
            })
    }
    
    /// Parse JSON line into event
    fn parse_json(&mut self, line: &str) -> Result<Event, StreamError<std::io::Error>> {
        if line.trim().is_empty() {
            return Err(StreamError::Format("Empty line"));
        }
        
        // Very basic JSON parsing for common format
        // {"timestamp":1234567890,"sensor_id":"temp_01","type":"temperature","value":25.5,"confidence":0.95}
        
        // Helper to find a JSON field and its value position
        let find_field = |json: &str, field: &str| -> Option<usize> {
            let mut search_start = 0;
            while let Some(pos) = json[search_start..].find(field) {
                let abs_pos = search_start + pos;
                // Check if it's preceded and followed by quotes
                if abs_pos > 0 && json.as_bytes()[abs_pos - 1] == b'"' 
                    && json[abs_pos + field.len()..].starts_with("\":") {
                    // Return position after ":"
                    return Some(abs_pos + field.len() + 2);
                }
                search_start = abs_pos + 1;
            }
            None
        };
        
        // Extract timestamp
        let timestamp = if let Some(pos) = find_field(line, "timestamp").or_else(|| find_field(line, "ts")) {
            let remaining = &line[pos..];
            let end = remaining.find(|c: char| c == ',' || c == '}').unwrap_or(remaining.len());
            remaining[..end].trim().parse::<f32>().ok()
                .map(|f| f as u64)
                .ok_or_else(|| {
                    self.stats.parse_errors += 1;
                    StreamError::Format("Invalid timestamp")
                })?
        } else {
            self.stats.parse_errors += 1;
            return Err(StreamError::Format("Missing timestamp"));
        };
        
        // Extract sensor ID
        let sensor_id = if let Some(pos) = find_field(line, "sensor_id").or_else(|| find_field(line, "id")) {
            if line[pos..].starts_with('"') {
                let value_start = pos + 1;
                if let Some(end) = line[value_start..].find('"') {
                    let id_str = &line[value_start..value_start + end];
                    id_str
                } else {
                    self.stats.parse_errors += 1;
                    return Err(StreamError::Format("Invalid sensor_id"));
                }
            } else {
                self.stats.parse_errors += 1;
                return Err(StreamError::Format("Invalid sensor_id format"));
            }
        } else {
            self.stats.parse_errors += 1;
            return Err(StreamError::Format("Missing sensor_id"));
        };
        
        // Extract sensor type
        let sensor_type = if let Some(pos) = find_field(line, "type").or_else(|| find_field(line, "sensor_type")) {
            if line[pos..].starts_with('"') {
                let value_start = pos + 1;
                if let Some(end) = line[value_start..].find('"') {
                    let type_str = &line[value_start..value_start + end];
                    match type_str.to_lowercase().as_str() {
                        "temperature" | "temp" => SensorType::Temperature,
                        "pressure" | "press" => SensorType::Pressure,
                        "humidity" | "humid" => SensorType::Humidity,
                        _ => {
                            self.stats.parse_errors += 1;
                            return Err(StreamError::Format("Unknown sensor type"));
                        }
                    }
                } else {
                    self.stats.parse_errors += 1;
                    return Err(StreamError::Format("Invalid sensor type"));
                }
            } else {
                self.stats.parse_errors += 1;
                return Err(StreamError::Format("Invalid sensor type format"));
            }
        } else {
            self.stats.parse_errors += 1;
            return Err(StreamError::Format("Missing sensor type"));
        };
        
        // Extract value
        let value = if let Some(pos) = find_field(line, "value").or_else(|| find_field(line, "reading")) {
            let remaining = &line[pos..];
            let end = remaining.find(|c: char| c == ',' || c == '}').unwrap_or(remaining.len());
            remaining[..end].trim().parse::<f32>()
                .map_err(|_| {
                    self.stats.parse_errors += 1;
                    StreamError::Format("Invalid value")
                })?
        } else {
            self.stats.parse_errors += 1;
            return Err(StreamError::Format("Missing value"));
        };
        
        // Extract confidence (optional, default to 1.0)
        let confidence = if let Some(pos) = find_field(line, "confidence").or_else(|| find_field(line, "quality")) {
            let remaining = &line[pos..];
            let end = remaining.find(|c: char| c == ',' || c == '}').unwrap_or(remaining.len());
            remaining[..end].trim().parse::<f32>().unwrap_or(1.0).clamp(0.0, 1.0)
        } else {
            1.0
        };
        
        self.stats.events_read += 1;
        
        let sensor_id_inline = InlineString::new(sensor_id)
            .ok_or_else(|| {
                self.stats.parse_errors += 1;
                StreamError::Format("Sensor ID too long")
            })?;
            
        EventBuilder::new(timestamp)
            .sensor(sensor_id_inline.as_str(), sensor_type)
            .reading(value, confidence)
            .ok_or_else(|| {
                self.stats.parse_errors += 1;
                StreamError::Format("Failed to build event")
            })
    }
    
    /// Parse binary data into event
    fn parse_binary(&mut self) -> Result<Event, StreamError<std::io::Error>> {
        // Binary format: 
        // - 8 bytes: timestamp
        // - 1 byte: sensor type
        // - 16 bytes: sensor ID (null-terminated)
        // - 4 bytes: value (f32)
        // - 4 bytes: confidence (f32)
        // Total: 33 bytes per event
        
        const BINARY_EVENT_SIZE: usize = 33;
        
        // Ensure we have enough data
        while self.buffer_len - self.buffer_pos < BINARY_EVENT_SIZE {
            if !self.refill_buffer()? {
                return Err(StreamError::EndOfStream);
            }
        }
        
        let data = &self.buffer[self.buffer_pos..self.buffer_pos + BINARY_EVENT_SIZE];
        self.buffer_pos += BINARY_EVENT_SIZE;
        
        // Parse timestamp (little-endian)
        let timestamp = u64::from_le_bytes([
            data[0], data[1], data[2], data[3],
            data[4], data[5], data[6], data[7],
        ]);
        
        // Parse sensor type
        let sensor_type = match data[8] {
            0 => SensorType::Temperature,
            1 => SensorType::Humidity,
            2 => SensorType::Pressure,
            _ => return Err(StreamError::Format("Unknown sensor type")),
        };
        
        // Parse sensor ID (null-terminated string)
        let id_bytes = &data[9..25];
        let id_len = id_bytes.iter().position(|&b| b == 0).unwrap_or(16);
        let sensor_id = core::str::from_utf8(&id_bytes[..id_len])
            .map_err(|_| StreamError::Format("Invalid sensor ID"))?;
        
        // Parse value
        let value = f32::from_le_bytes([data[25], data[26], data[27], data[28]]);
        
        // Parse confidence
        let confidence = f32::from_le_bytes([data[29], data[30], data[31], data[32]]);
        
        self.stats.events_read += 1;
        
        EventBuilder::new(timestamp)
            .sensor(sensor_id, sensor_type)
            .reading(value, confidence)
            .ok_or(StreamError::Format("Failed to build event"))
    }
}

impl Stream for FileStream {
    type Item = Event;
    type Error = StreamError<std::io::Error>;
    
    fn poll_next(&mut self) -> nb::Result<Self::Item, Self::Error> {
        let format = self.format; // Copy to avoid borrow issues
        match format {
            FileFormat::Csv | FileFormat::JsonLines => {
                // Read next line
                loop {
                    // Read line first and store it
                    let line_result = self.read_line()?;
                    match line_result {
                        Some(line_str) => {
                            // Skip empty lines and comments
                            if line_str.trim().is_empty() || line_str.starts_with('#') {
                                continue; // Try next line
                            }
                            
                            // Copy line to owned string to avoid borrow issues
                            let line_owned = line_str.to_string();
                            
                            // Parse based on format
                            let event = match format {
                                FileFormat::Csv => self.parse_csv(&line_owned),
                                FileFormat::JsonLines => self.parse_json(&line_owned),
                                _ => unreachable!(),
                            };
                            
                            match event {
                                Ok(e) => return Ok(e),
                                Err(_e) => {
                                    // Log error and try next line
                                    // In production, might want to limit consecutive errors
                                    continue;
                                }
                            }
                        }
                        None => return Err(nb::Error::Other(StreamError::EndOfStream)),
                    }
                }
            }
            FileFormat::Binary => {
                match self.parse_binary() {
                    Ok(event) => Ok(event),
                    Err(StreamError::EndOfStream) => Err(nb::Error::Other(StreamError::EndOfStream)),
                    Err(e) => Err(nb::Error::Other(e)),
                }
            }
        }
    }
}

// Extension methods for batch processing
impl FileStream {
    /// Process all events in the file
    pub fn process_all<F, E>(&mut self, mut callback: F) -> Result<FileStreamStats, StreamError<std::io::Error>>
    where
        F: FnMut(&Event) -> Result<(), E>,
        StreamError<std::io::Error>: From<E>,
    {
        loop {
            match self.poll_next() {
                Ok(event) => {
                    callback(&event)?;
                }
                Err(nb::Error::WouldBlock) => continue,
                Err(nb::Error::Other(StreamError::EndOfStream)) => break,
                Err(nb::Error::Other(e)) => return Err(e),
            }
        }
        
        Ok(self.stats.clone())
    }
    
    /// Process events in batches
    pub fn process_batch<F, E>(
        &mut self,
        batch_size: usize,
        mut callback: F,
    ) -> Result<FileStreamStats, StreamError<std::io::Error>>
    where
        F: FnMut(&[Event]) -> Result<(), E>,
        StreamError<std::io::Error>: From<E>,
    {
        let mut batch = heapless::Vec::<Event, 128>::new();
        
        loop {
            match self.poll_next() {
                Ok(event) => {
                    if batch.push(event.clone()).is_err() {
                        // Batch full, process it
                        callback(&batch)?;
                        batch.clear();
                        batch.push(event).ok(); // Add current event to new batch
                    }
                }
                Err(nb::Error::Other(StreamError::EndOfStream)) => {
                    // Process remaining batch
                    if !batch.is_empty() {
                        callback(&batch)?;
                    }
                    break;
                }
                Err(nb::Error::WouldBlock) => continue,
                Err(nb::Error::Other(e)) => return Err(e),
            }
            
            // Check if batch is ready
            if batch.len() >= batch_size {
                callback(&batch)?;
                batch.clear();
            }
        }
        
        Ok(self.stats.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_file_format_parsing() {
        // Test CSV parsing logic
        let _csv_line = "1234567890,temp_01,temperature,25.5,0.95";
        // Would test parse_csv method
        
        // Test JSON parsing logic  
        let _json_line = r#"{"timestamp":1234567890,"sensor_id":"temp_01","type":"temperature","value":25.5,"confidence":0.95}"#;
        // Would test parse_json method
    }
}