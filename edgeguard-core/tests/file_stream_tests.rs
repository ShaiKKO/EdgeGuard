//! Tests for FileStream implementation

#[cfg(all(test, feature = "std"))]
mod tests {
    use edgeguard_core::{
        stream::{FileStream, FileFormat, Stream},
        events::{Event, SensorType},
    };
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_csv_parsing() {
        // Create temporary CSV file
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "timestamp,sensor_id,type,value,confidence").unwrap();
        writeln!(temp_file, "1000,temp_01,temperature,25.5,0.95").unwrap();
        writeln!(temp_file, "2000,humid_01,humidity,65.0,0.92").unwrap();
        writeln!(temp_file, "3000,press_01,pressure,1013.25,0.98").unwrap();
        temp_file.flush().unwrap();
        
        // Read and parse
        let mut stream = FileStream::from_csv(temp_file.path().to_str().unwrap())
            .unwrap()
            .with_skip_lines(1);
        
        // First event
        let event1 = stream.poll_next().unwrap();
        if let Event::SensorReading { sensor_id, sensor_type, value, timestamp, .. } = event1 {
            assert_eq!(sensor_id.as_str(), "temp_01");
            assert_eq!(sensor_type, SensorType::Temperature);
            assert_eq!(value, 25.5);
            assert_eq!(timestamp, 1000);
        } else {
            panic!("Expected SensorReading event");
        }
        
        // Second event
        let event2 = stream.poll_next().unwrap();
        if let Event::SensorReading { sensor_type, value, .. } = event2 {
            assert_eq!(sensor_type, SensorType::Humidity);
            assert_eq!(value, 65.0);
        }
        
        // Third event
        let event3 = stream.poll_next().unwrap();
        if let Event::SensorReading { sensor_type, value, .. } = event3 {
            assert_eq!(sensor_type, SensorType::Pressure);
            assert_eq!(value, 1013.25);
        }
        
        // EOF
        match stream.poll_next() {
            Err(nb::Error::Other(edgeguard_core::stream::StreamError::EndOfStream)) => {},
            _ => panic!("Expected EndOfStream"),
        }
    }

    #[test]
    fn test_csv_with_errors() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "timestamp,sensor_id,type,value,confidence").unwrap();
        writeln!(temp_file, "1000,temp_01,temperature,25.5,0.95").unwrap();
        writeln!(temp_file, "invalid_line").unwrap(); // Invalid format
        writeln!(temp_file, "2000,temp_02,unknown_type,30.0,0.90").unwrap(); // Unknown type
        writeln!(temp_file, "3000,temp_03,temperature,not_a_number,0.95").unwrap(); // Invalid value
        writeln!(temp_file, "4000,temp_04,temperature,35.0,0.93").unwrap(); // Valid again
        temp_file.flush().unwrap();
        
        let mut stream = FileStream::from_csv(temp_file.path().to_str().unwrap())
            .unwrap()
            .with_skip_lines(1);
        
        let mut valid_count = 0;
        let mut events_processed = 0;
        
        loop {
            match stream.poll_next() {
                Ok(_) => {
                    valid_count += 1;
                    events_processed += 1;
                }
                Err(nb::Error::Other(edgeguard_core::stream::StreamError::EndOfStream)) => break,
                Err(_) => events_processed += 1,
            }
            
            if events_processed > 10 {
                panic!("Too many events");
            }
        }
        
        assert_eq!(valid_count, 2); // Only first and last are valid
        
        let stats = stream.stats();
        assert!(stats.parse_errors > 0);
    }

    #[test]
    fn test_json_parsing() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, r#"{{"timestamp":1000,"sensor_id":"temp_01","type":"temperature","value":25.5,"confidence":0.95}}"#).unwrap();
        writeln!(temp_file, r#"{{"timestamp":2000,"sensor_id":"humid_01","type":"humidity","value":65.0,"confidence":0.92}}"#).unwrap();
        temp_file.flush().unwrap();
        
        let mut stream = FileStream::new(
            temp_file.path().to_str().unwrap(),
            FileFormat::JsonLines
        ).unwrap();
        
        // Parse first event
        let event1 = stream.poll_next().unwrap();
        if let Event::SensorReading { sensor_id, sensor_type, value, .. } = event1 {
            assert_eq!(sensor_id.as_str(), "temp_01");
            assert_eq!(sensor_type, SensorType::Temperature);
            assert_eq!(value, 25.5);
        } else {
            panic!("Expected SensorReading");
        }
        
        // Parse second event
        let event2 = stream.poll_next().unwrap();
        if let Event::SensorReading { sensor_type, value, .. } = event2 {
            assert_eq!(sensor_type, SensorType::Humidity);
            assert_eq!(value, 65.0);
        }
    }

    #[test]
    fn test_batch_processing() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "timestamp,sensor_id,type,value,confidence").unwrap();
        for i in 0..25 {
            writeln!(temp_file, "{},temp_01,temperature,{},0.95", 
                     1000 + i * 100, 20.0 + i as f32 * 0.1).unwrap();
        }
        temp_file.flush().unwrap();
        
        let mut stream = FileStream::from_csv(temp_file.path().to_str().unwrap())
            .unwrap()
            .with_skip_lines(1);
        
        let mut batch_count = 0;
        let total = stream.process_batch(10, |batch| {
            batch_count += 1;
            assert!(batch.len() <= 10);
            assert!(batch.len() > 0);
            Ok(())
        }).unwrap();
        
        assert_eq!(total, 25);
        assert_eq!(batch_count, 3); // 10 + 10 + 5
    }

    #[test]
    fn test_process_all() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "timestamp,sensor_id,type,value,confidence").unwrap();
        for i in 0..10 {
            writeln!(temp_file, "{},temp_01,temperature,{},0.95", 
                     1000 + i * 100, 20.0 + i as f32).unwrap();
        }
        temp_file.flush().unwrap();
        
        let mut stream = FileStream::from_csv(temp_file.path().to_str().unwrap())
            .unwrap()
            .with_skip_lines(1);
        
        let mut count = 0;
        let stats = stream.process_all(|_event| {
            count += 1;
            Ok(())
        }).unwrap();
        
        assert_eq!(count, 10);
        assert_eq!(stats.events_read, 10);
        assert_eq!(stats.lines_processed, 11); // Including header
    }

    #[test]
    fn test_skip_lines() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "# Comment line 1").unwrap();
        writeln!(temp_file, "# Comment line 2").unwrap();
        writeln!(temp_file, "timestamp,sensor_id,type,value,confidence").unwrap();
        writeln!(temp_file, "1000,temp_01,temperature,25.5,0.95").unwrap();
        temp_file.flush().unwrap();
        
        let mut stream = FileStream::from_csv(temp_file.path().to_str().unwrap())
            .unwrap()
            .with_skip_lines(3); // Skip comments and header
        
        let event = stream.poll_next().unwrap();
        if let Event::SensorReading { value, .. } = event {
            assert_eq!(value, 25.5);
        }
    }

    #[test]
    fn test_empty_file() {
        let temp_file = NamedTempFile::new().unwrap();
        
        let mut stream = FileStream::from_csv(temp_file.path().to_str().unwrap()).unwrap();
        
        match stream.poll_next() {
            Err(nb::Error::Other(edgeguard_core::stream::StreamError::EndOfStream)) => {},
            _ => panic!("Expected EndOfStream for empty file"),
        }
    }

    #[test]
    fn test_mixed_case_sensor_types() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "1000,t1,Temperature,25.5,0.95").unwrap();
        writeln!(temp_file, "2000,h1,HUMIDITY,65.0,0.92").unwrap();
        writeln!(temp_file, "3000,p1,pressure,1013.0,0.98").unwrap();
        temp_file.flush().unwrap();
        
        let mut stream = FileStream::from_csv(temp_file.path().to_str().unwrap()).unwrap();
        
        // All should parse successfully
        assert!(matches!(stream.poll_next(), Ok(_)));
        assert!(matches!(stream.poll_next(), Ok(_)));
        assert!(matches!(stream.poll_next(), Ok(_)));
    }
}