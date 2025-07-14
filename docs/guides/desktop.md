# Desktop Deployment Guide

Complete guide for deploying EdgeGuard on desktop and server environments with multi-threading, database integration, and production-scale configurations.

## System Requirements

### Minimum Requirements
- **Operating System**: Linux, macOS, or Windows 10+
- **RAM**: 512MB available for EdgeGuard processes
- **Storage**: 100MB for installation, 1GB+ for data storage
- **Network**: Internet connectivity for remote sensors/cloud integration

### Recommended Configuration
- **CPU**: 4+ cores for multi-sensor fusion
- **RAM**: 2GB+ for large-scale deployments
- **Storage**: SSD for optimal I/O performance
- **Network**: Gigabit Ethernet for high-throughput scenarios

### Supported Platforms
- **Linux**: Ubuntu 20.04+, CentOS 8+, Debian 11+
- **macOS**: macOS 11+ (Big Sur and later)
- **Windows**: Windows 10/11 with WSL2 or native

## Installation

### Rust Installation

```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Verify installation
rustc --version
cargo --version
```

### System Dependencies

#### Linux (Ubuntu/Debian)
```bash
# Update package list
sudo apt update

# Install build dependencies
sudo apt install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    libsqlite3-dev \
    cmake \
    git

# Install optional dependencies
sudo apt install -y \
    postgresql-client \
    redis-tools \
    mosquitto-clients
```

#### macOS
```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install openssl cmake pkg-config sqlite3

# Install optional dependencies
brew install postgresql redis mosquitto
```

#### Windows
```powershell
# Install dependencies via vcpkg or use WSL2
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install
.\vcpkg install openssl:x64-windows sqlite3:x64-windows
```

## Project Setup

### Basic Project Structure

```
desktop-edgeguard/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── config/
│   │   ├── mod.rs
│   │   └── settings.rs
│   ├── sensors/
│   │   ├── mod.rs
│   │   ├── network.rs
│   │   ├── file.rs
│   │   └── database.rs
│   ├── storage/
│   │   ├── mod.rs
│   │   ├── sqlite.rs
│   │   ├── postgresql.rs
│   │   └── redis.rs
│   ├── services/
│   │   ├── mod.rs
│   │   ├── web_server.rs
│   │   ├── mqtt_bridge.rs
│   │   └── metrics.rs
│   └── utils/
│       ├── mod.rs
│       └── logging.rs
├── config/
│   └── settings.toml
├── migrations/
│   └── 001_initial.sql
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

### Cargo.toml Configuration

```toml
[package]
name = "desktop-edgeguard"
version = "0.1.0"
edition = "2021"

[dependencies]
# EdgeGuard core with all features
edgeguard = { version = "0.1.0", features = [
    "std",
    "validation-core",
    "pipeline-core",
    "fusion-core",
    "connectors",
    "schemas",
    "ml"
] }

# Async runtime
tokio = { version = "1.40", features = ["full"] }

# Database integrations
sqlx = { version = "0.7", features = ["runtime-tokio-native-tls", "sqlite", "postgres", "chrono"] }
redis = { version = "0.23", features = ["tokio-comp"] }

# Web framework
axum = { version = "0.7", features = ["macros"] }
tower = "0.4"
tower-http = { version = "0.4", features = ["cors", "trace"] }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
toml = "0.8"

# Configuration
config = "0.13"
clap = { version = "4.0", features = ["derive"] }

# Logging and metrics
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
metrics = "0.22"
metrics-exporter-prometheus = "0.13"

# Utilities
uuid = { version = "1.0", features = ["v4"] }
chrono = { version = "0.4", features = ["serde"] }
anyhow = "1.0"
thiserror = "1.0"

[dev-dependencies]
tempfile = "3.0"
criterion = "0.5"

[[bench]]
name = "pipeline_benchmark"
harness = false
```

## Basic Implementation

### Main Application Structure

```rust
// src/main.rs
use anyhow::Result;
use clap::Parser;
use desktop_edgeguard::{
    config::Settings,
    services::EdgeGuardService,
    storage::StorageManager,
    utils::logging::setup_logging,
};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "config/settings.toml")]
    config: String,
    
    #[arg(short, long, default_value = "info")]
    log_level: String,
    
    #[arg(short, long)]
    daemon: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    // Initialize logging
    setup_logging(&args.log_level)?;
    
    // Load configuration
    let settings = Settings::from_file(&args.config)?;
    
    // Initialize storage
    let storage = StorageManager::new(&settings.storage).await?;
    
    // Create and start EdgeGuard service
    let service = EdgeGuardService::new(settings, storage).await?;
    
    if args.daemon {
        service.run_daemon().await?;
    } else {
        service.run_interactive().await?;
    }
    
    Ok(())
}
```

### Configuration Management

```rust
// src/config/settings.rs
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub server: ServerConfig,
    pub storage: StorageConfig,
    pub sensors: SensorConfig,
    pub pipeline: PipelineConfig,
    pub fusion: FusionConfig,
    pub connectors: ConnectorConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: usize,
    pub metrics_port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub database_url: String,
    pub redis_url: Option<String>,
    pub retention_days: u32,
    pub batch_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorConfig {
    pub sources: Vec<SensorSource>,
    pub polling_interval: u64,
    pub timeout: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorSource {
    pub id: String,
    pub source_type: String,
    pub config: HashMap<String, String>,
    pub validators: ValidatorConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorConfig {
    pub temperature: Option<TemperatureValidatorConfig>,
    pub humidity: Option<HumidityValidatorConfig>,
    pub pressure: Option<PressureValidatorConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureValidatorConfig {
    pub min: f32,
    pub max: f32,
    pub rate_limit: f32,
    pub thermal_mass: Option<f32>,
}

impl Settings {
    pub fn from_file(path: &str) -> Result<Self, config::ConfigError> {
        let settings = config::Config::builder()
            .add_source(config::File::with_name(path))
            .add_source(config::Environment::with_prefix("EDGEGUARD"))
            .build()?;
        
        settings.try_deserialize()
    }
}
```

### Multi-threaded Pipeline Processing

```rust
// src/services/mod.rs
use edgeguard::{
    pipeline::{Pipeline, ValidationStage, FusionStage},
    validators::{TemperatureValidator, HumidityValidator, PressureValidator},
    events::{Event, EventBuilder, SensorType},
    time::SystemTime,
};
use tokio::sync::{broadcast, mpsc};
use std::sync::Arc;
use anyhow::Result;

pub struct EdgeGuardService {
    settings: Settings,
    storage: Arc<StorageManager>,
    pipeline: Arc<Pipeline<2048>>,
    event_tx: broadcast::Sender<Event>,
    shutdown_rx: broadcast::Receiver<()>,
}

impl EdgeGuardService {
    pub async fn new(settings: Settings, storage: StorageManager) -> Result<Self> {
        let pipeline = create_pipeline(&settings)?;
        let (event_tx, _) = broadcast::channel(1000);
        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        
        Ok(Self {
            settings,
            storage: Arc::new(storage),
            pipeline: Arc::new(pipeline),
            event_tx,
            shutdown_rx,
        })
    }
    
    pub async fn run_interactive(&self) -> Result<()> {
        tracing::info!("Starting EdgeGuard in interactive mode");
        
        // Start sensor collection
        let sensor_handle = self.start_sensor_collection().await?;
        
        // Start pipeline processing
        let pipeline_handle = self.start_pipeline_processing().await?;
        
        // Start web server
        let web_handle = self.start_web_server().await?;
        
        // Start metrics server
        let metrics_handle = self.start_metrics_server().await?;
        
        // Wait for shutdown signal
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("Received shutdown signal");
            }
            _ = self.shutdown_rx.recv() => {
                tracing::info!("Received internal shutdown signal");
            }
        }
        
        // Graceful shutdown
        sensor_handle.abort();
        pipeline_handle.abort();
        web_handle.abort();
        metrics_handle.abort();
        
        Ok(())
    }
    
    async fn start_sensor_collection(&self) -> Result<tokio::task::JoinHandle<()>> {
        let settings = self.settings.clone();
        let event_tx = self.event_tx.clone();
        let time_source = SystemTime::new();
        
        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                tokio::time::Duration::from_millis(settings.sensors.polling_interval)
            );
            
            loop {
                interval.tick().await;
                
                for sensor in &settings.sensors.sources {
                    match collect_sensor_data(sensor, &time_source).await {
                        Ok(events) => {
                            for event in events {
                                if let Err(e) = event_tx.send(event) {
                                    tracing::error!("Failed to send event: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!("Failed to collect sensor data: {}", e);
                        }
                    }
                }
            }
        });
        
        Ok(handle)
    }
    
    async fn start_pipeline_processing(&self) -> Result<tokio::task::JoinHandle<()>> {
        let pipeline = Arc::clone(&self.pipeline);
        let storage = Arc::clone(&self.storage);
        let mut event_rx = self.event_tx.subscribe();
        
        let handle = tokio::spawn(async move {
            let mut batch = Vec::with_capacity(100);
            let mut flush_interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
            
            loop {
                tokio::select! {
                    // Collect events
                    event = event_rx.recv() => {
                        match event {
                            Ok(event) => {
                                batch.push(event);
                                
                                // Process batch when full
                                if batch.len() >= 100 {
                                    process_event_batch(&pipeline, &storage, &mut batch).await;
                                }
                            }
                            Err(e) => {
                                tracing::error!("Event receive error: {}", e);
                            }
                        }
                    }
                    
                    // Flush remaining events periodically
                    _ = flush_interval.tick() => {
                        if !batch.is_empty() {
                            process_event_batch(&pipeline, &storage, &mut batch).await;
                        }
                    }
                }
            }
        });
        
        Ok(handle)
    }
}

async fn process_event_batch(
    pipeline: &Pipeline<2048>,
    storage: &StorageManager,
    batch: &mut Vec<Event>,
) {
    // Process events through pipeline
    for event in batch.drain(..) {
        if !pipeline.push_event(event) {
            tracing::warn!("Pipeline queue full, dropping event");
        }
    }
    
    // Process pipeline
    match pipeline.process_batch(1000) {
        Ok(processed) => {
            tracing::debug!("Processed {} events", processed);
            
            // Store results
            let mut results = Vec::new();
            while let Some(result) = pipeline.pop_result() {
                results.push(result);
            }
            
            if !results.is_empty() {
                if let Err(e) = storage.store_events(&results).await {
                    tracing::error!("Failed to store events: {}", e);
                }
            }
        }
        Err(e) => {
            tracing::error!("Pipeline processing error: {}", e);
        }
    }
}

fn create_pipeline(settings: &Settings) -> Result<Pipeline<2048>> {
    let mut builder = Pipeline::<2048>::builder();
    
    // Add validation stages for each sensor type
    for sensor in &settings.sensors.sources {
        match sensor.source_type.as_str() {
            "temperature" => {
                if let Some(temp_config) = &sensor.validators.temperature {
                    let validator = TemperatureValidator::new()
                        .with_range(temp_config.min, temp_config.max)
                        .with_rate_limit(temp_config.rate_limit);
                    
                    builder = builder.add_stage(ValidationStage::new(
                        validator,
                        SensorType::Temperature
                    ));
                }
            }
            "humidity" => {
                if let Some(hum_config) = &sensor.validators.humidity {
                    let validator = HumidityValidator::new()
                        .with_range(hum_config.min, hum_config.max)
                        .with_rate_limit(hum_config.rate_limit);
                    
                    builder = builder.add_stage(ValidationStage::new(
                        validator,
                        SensorType::Humidity
                    ));
                }
            }
            _ => {}
        }
    }
    
    // Add fusion stage if configured
    if settings.fusion.enabled {
        let fusion_stage = create_fusion_stage(&settings.fusion)?;
        builder = builder.add_stage(fusion_stage);
    }
    
    Ok(builder.build())
}
```

### Database Integration

```rust
// src/storage/sqlite.rs
use sqlx::{Pool, Sqlite, SqlitePool};
use edgeguard::events::Event;
use chrono::{DateTime, Utc};
use anyhow::Result;

pub struct SqliteStorage {
    pool: SqlitePool,
}

impl SqliteStorage {
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = SqlitePool::connect(database_url).await?;
        
        // Run migrations
        sqlx::migrate!("./migrations").run(&pool).await?;
        
        Ok(Self { pool })
    }
    
    pub async fn store_events(&self, events: &[Event]) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        
        for event in events {
            match event {
                Event::SensorReading {
                    sensor_id,
                    sensor_type,
                    value,
                    timestamp,
                    quality,
                } => {
                    sqlx::query!(
                        r#"
                        INSERT INTO sensor_readings (
                            sensor_id, sensor_type, value, timestamp, quality
                        ) VALUES (?, ?, ?, ?, ?)
                        "#,
                        sensor_id.as_str(),
                        *sensor_type as i32,
                        value,
                        timestamp,
                        quality
                    )
                    .execute(&mut *tx)
                    .await?;
                }
                
                Event::ValidationResult {
                    sensor_id,
                    status,
                    timestamp,
                    ..
                } => {
                    sqlx::query!(
                        r#"
                        INSERT INTO validation_results (
                            sensor_id, status, timestamp
                        ) VALUES (?, ?, ?)
                        "#,
                        sensor_id.as_str(),
                        *status as i32,
                        timestamp
                    )
                    .execute(&mut *tx)
                    .await?;
                }
                
                _ => {
                    // Handle other event types
                }
            }
        }
        
        tx.commit().await?;
        Ok(())
    }
    
    pub async fn get_recent_readings(
        &self,
        sensor_id: &str,
        limit: u32,
    ) -> Result<Vec<SensorReading>> {
        let readings = sqlx::query_as!(
            SensorReading,
            r#"
            SELECT sensor_id, sensor_type, value, timestamp, quality
            FROM sensor_readings
            WHERE sensor_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            "#,
            sensor_id,
            limit
        )
        .fetch_all(&self.pool)
        .await?;
        
        Ok(readings)
    }
    
    pub async fn cleanup_old_data(&self, retention_days: u32) -> Result<u64> {
        let cutoff = Utc::now() - chrono::Duration::days(retention_days as i64);
        
        let result = sqlx::query!(
            "DELETE FROM sensor_readings WHERE timestamp < ?",
            cutoff.timestamp()
        )
        .execute(&self.pool)
        .await?;
        
        Ok(result.rows_affected())
    }
}

#[derive(Debug, Clone)]
pub struct SensorReading {
    pub sensor_id: String,
    pub sensor_type: i32,
    pub value: f32,
    pub timestamp: i64,
    pub quality: f32,
}
```

### Web API Server

```rust
// src/services/web_server.rs
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::net::TcpListener;

pub struct WebServer {
    storage: Arc<StorageManager>,
    pipeline: Arc<Pipeline<2048>>,
}

impl WebServer {
    pub fn new(storage: Arc<StorageManager>, pipeline: Arc<Pipeline<2048>>) -> Self {
        Self { storage, pipeline }
    }
    
    pub async fn serve(self, host: &str, port: u16) -> Result<()> {
        let app = Router::new()
            .route("/api/sensors", get(list_sensors))
            .route("/api/sensors/:id/readings", get(get_sensor_readings))
            .route("/api/sensors/:id/readings", post(add_sensor_reading))
            .route("/api/pipeline/metrics", get(get_pipeline_metrics))
            .route("/api/pipeline/status", get(get_pipeline_status))
            .route("/api/health", get(health_check))
            .with_state(Arc::new(self));
        
        let listener = TcpListener::bind(format!("{}:{}", host, port)).await?;
        tracing::info!("Web server listening on {}:{}", host, port);
        
        axum::serve(listener, app).await?;
        Ok(())
    }
}

// API handlers
async fn list_sensors(State(server): State<Arc<WebServer>>) -> Result<Json<Vec<String>>, StatusCode> {
    // Implementation
    Ok(Json(vec!["sensor1".to_string(), "sensor2".to_string()]))
}

async fn get_sensor_readings(
    Path(sensor_id): Path<String>,
    Query(params): Query<ReadingQuery>,
    State(server): State<Arc<WebServer>>,
) -> Result<Json<Vec<SensorReading>>, StatusCode> {
    match server.storage.get_recent_readings(&sensor_id, params.limit.unwrap_or(100)).await {
        Ok(readings) => Ok(Json(readings)),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

async fn add_sensor_reading(
    Path(sensor_id): Path<String>,
    State(server): State<Arc<WebServer>>,
    Json(reading): Json<AddReadingRequest>,
) -> Result<StatusCode, StatusCode> {
    let event = EventBuilder::new(SystemTime::new().now())
        .sensor(&sensor_id, reading.sensor_type)
        .reading(reading.value, reading.quality)
        .ok_or(StatusCode::BAD_REQUEST)?;
    
    if server.pipeline.push_event(event) {
        Ok(StatusCode::CREATED)
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

async fn get_pipeline_metrics(
    State(server): State<Arc<WebServer>>,
) -> Result<Json<PipelineMetrics>, StatusCode> {
    let metrics = server.pipeline.metrics();
    Ok(Json(PipelineMetrics {
        events_processed: metrics.events_processed,
        events_dropped: metrics.events_dropped,
        processing_errors: metrics.processing_errors,
        current_depth: metrics.current_depth,
    }))
}

async fn health_check() -> StatusCode {
    StatusCode::OK
}

#[derive(Deserialize)]
struct ReadingQuery {
    limit: Option<u32>,
}

#[derive(Deserialize)]
struct AddReadingRequest {
    sensor_type: SensorType,
    value: f32,
    quality: f32,
}

#[derive(Serialize)]
struct PipelineMetrics {
    events_processed: u64,
    events_dropped: u64,
    processing_errors: u64,
    current_depth: usize,
}
```

## Advanced Features

### Machine Learning Integration

```rust
// src/services/ml_service.rs
use edgeguard::ml::{IsolationForest, FeatureExtractor, MLAnomalyStage};
use tokio::sync::RwLock;
use std::sync::Arc;

pub struct MLService {
    isolation_forest: Arc<RwLock<IsolationForest>>,
    feature_extractor: FeatureExtractor,
    is_trained: Arc<RwLock<bool>>,
}

impl MLService {
    pub fn new() -> Self {
        Self {
            isolation_forest: Arc::new(RwLock::new(
                IsolationForest::new(100, 256)
                    .with_contamination(0.1)
                    .with_random_seed(42)
            )),
            feature_extractor: FeatureExtractor::new(),
            is_trained: Arc::new(RwLock::new(false)),
        }
    }
    
    pub async fn train_on_historical_data(&self, storage: &StorageManager) -> Result<()> {
        tracing::info!("Training ML models on historical data");
        
        // Fetch training data
        let training_data = storage.get_training_data(10000).await?;
        
        // Extract features
        let features: Vec<_> = training_data.iter()
            .map(|reading| self.feature_extractor.extract_features(reading))
            .collect();
        
        // Train isolation forest
        let feature_refs: Vec<_> = features.iter().collect();
        {
            let mut forest = self.isolation_forest.write().await;
            forest.train(&feature_refs)?;
        }
        
        // Mark as trained
        {
            let mut is_trained = self.is_trained.write().await;
            *is_trained = true;
        }
        
        tracing::info!("ML model training completed");
        Ok(())
    }
    
    pub async fn detect_anomalies(&self, events: &[Event]) -> Result<Vec<AnomalyResult>> {
        let is_trained = *self.is_trained.read().await;
        if !is_trained {
            return Ok(vec![]);
        }
        
        let mut results = Vec::new();
        let forest = self.isolation_forest.read().await;
        
        for event in events {
            if let Event::SensorReading { sensor_id, value, timestamp, .. } = event {
                let features = self.feature_extractor.extract_features(&SensorReading {
                    sensor_id: sensor_id.clone(),
                    value: *value,
                    timestamp: *timestamp,
                });
                
                let anomaly_score = forest.predict(&features)?;
                
                results.push(AnomalyResult {
                    sensor_id: sensor_id.clone(),
                    timestamp: *timestamp,
                    value: *value,
                    anomaly_score,
                    is_anomaly: anomaly_score > 0.5,
                });
            }
        }
        
        Ok(results)
    }
    
    pub async fn retrain_if_needed(&self, storage: &StorageManager) -> Result<()> {
        // Check if retraining is needed based on model performance
        let should_retrain = self.should_retrain(storage).await?;
        
        if should_retrain {
            tracing::info!("Retraining ML model");
            self.train_on_historical_data(storage).await?;
        }
        
        Ok(())
    }
    
    async fn should_retrain(&self, storage: &StorageManager) -> Result<bool> {
        // Logic to determine if retraining is needed
        // Based on model age, performance metrics, data drift, etc.
        Ok(false)
    }
}

#[derive(Debug, Clone)]
pub struct AnomalyResult {
    pub sensor_id: String,
    pub timestamp: u64,
    pub value: f32,
    pub anomaly_score: f32,
    pub is_anomaly: bool,
}
```

### Real-time Dashboard

```rust
// src/services/dashboard.rs
use axum::{
    extract::ws::{WebSocket, WebSocketUpgrade},
    response::Response,
};
use serde_json::json;
use tokio::sync::broadcast;

pub struct DashboardService {
    event_rx: broadcast::Receiver<Event>,
}

impl DashboardService {
    pub fn new(event_rx: broadcast::Receiver<Event>) -> Self {
        Self { event_rx }
    }
    
    pub async fn websocket_handler(
        ws: WebSocketUpgrade,
        State(dashboard): State<Arc<DashboardService>>,
    ) -> Response {
        ws.on_upgrade(|socket| dashboard.handle_websocket(socket))
    }
    
    async fn handle_websocket(&self, mut socket: WebSocket) {
        let mut event_rx = self.event_rx.resubscribe();
        
        // Send initial dashboard data
        if let Err(_) = socket.send(axum::extract::ws::Message::Text(
            json!({
                "type": "init",
                "data": {
                    "status": "connected",
                    "timestamp": chrono::Utc::now().timestamp()
                }
            }).to_string()
        )).await {
            return;
        }
        
        // Stream real-time updates
        while let Ok(event) = event_rx.recv().await {
            let message = match event {
                Event::SensorReading { sensor_id, value, timestamp, .. } => {
                    json!({
                        "type": "sensor_reading",
                        "sensor_id": sensor_id.as_str(),
                        "value": value,
                        "timestamp": timestamp
                    })
                }
                Event::ValidationResult { sensor_id, status, timestamp, .. } => {
                    json!({
                        "type": "validation_result",
                        "sensor_id": sensor_id.as_str(),
                        "status": status,
                        "timestamp": timestamp
                    })
                }
                _ => continue,
            };
            
            if let Err(_) = socket.send(axum::extract::ws::Message::Text(
                message.to_string()
            )).await {
                break;
            }
        }
    }
}
```

## Performance Optimization

### Connection Pooling

```rust
// src/storage/connection_pool.rs
use sqlx::{Pool, Sqlite, SqlitePool};
use std::sync::Arc;
use tokio::sync::Semaphore;

pub struct ConnectionPoolManager {
    pool: Arc<SqlitePool>,
    semaphore: Arc<Semaphore>,
}

impl ConnectionPoolManager {
    pub async fn new(database_url: &str, max_connections: usize) -> Result<Self> {
        let pool = SqlitePool::connect_with(
            sqlx::sqlite::SqliteConnectOptions::new()
                .filename(database_url)
                .create_if_missing(true)
        ).await?;
        
        // Set pool configuration
        pool.set_max_connections(max_connections as u32);
        
        Ok(Self {
            pool: Arc::new(pool),
            semaphore: Arc::new(Semaphore::new(max_connections)),
        })
    }
    
    pub async fn execute_with_connection<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&SqlitePool) -> T,
    {
        let _permit = self.semaphore.acquire().await?;
        Ok(f(&self.pool))
    }
}
```

### Caching Layer

```rust
// src/storage/cache.rs
use redis::{Client, Commands, Connection};
use serde::{Deserialize, Serialize};
use std::time::Duration;

pub struct RedisCache {
    client: Client,
}

impl RedisCache {
    pub fn new(redis_url: &str) -> Result<Self> {
        let client = Client::open(redis_url)?;
        Ok(Self { client })
    }
    
    pub async fn get<T>(&self, key: &str) -> Result<Option<T>>
    where
        T: for<'de> Deserialize<'de>,
    {
        let mut conn = self.client.get_async_connection().await?;
        let value: Option<String> = conn.get(key).await?;
        
        match value {
            Some(json_str) => {
                let item: T = serde_json::from_str(&json_str)?;
                Ok(Some(item))
            }
            None => Ok(None),
        }
    }
    
    pub async fn set<T>(&self, key: &str, value: &T, ttl: Duration) -> Result<()>
    where
        T: Serialize,
    {
        let mut conn = self.client.get_async_connection().await?;
        let json_str = serde_json::to_string(value)?;
        
        conn.set_ex(key, json_str, ttl.as_secs() as usize).await?;
        Ok(())
    }
    
    pub async fn invalidate(&self, pattern: &str) -> Result<()> {
        let mut conn = self.client.get_async_connection().await?;
        let keys: Vec<String> = conn.keys(pattern).await?;
        
        if !keys.is_empty() {
            conn.del(&keys).await?;
        }
        
        Ok(())
    }
}
```

## Testing and Monitoring

### Integration Tests

```rust
// tests/integration_tests.rs
use desktop_edgeguard::{
    config::Settings,
    services::EdgeGuardService,
    storage::StorageManager,
};
use tempfile::TempDir;
use tokio::time::{sleep, Duration};

#[tokio::test]
async fn test_full_pipeline_integration() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.db");
    
    let settings = Settings {
        storage: StorageConfig {
            database_url: format!("sqlite://{}", db_path.display()),
            redis_url: None,
            retention_days: 7,
            batch_size: 100,
        },
        // ... other config
    };
    
    let storage = StorageManager::new(&settings.storage).await.unwrap();
    let service = EdgeGuardService::new(settings, storage).await.unwrap();
    
    // Test sensor data processing
    let test_events = create_test_events();
    for event in test_events {
        service.process_event(event).await.unwrap();
    }
    
    // Verify data was stored correctly
    sleep(Duration::from_millis(100)).await;
    let readings = service.get_recent_readings("test_sensor", 10).await.unwrap();
    assert!(!readings.is_empty());
}

#[tokio::test]
async fn test_anomaly_detection() {
    let ml_service = MLService::new();
    
    // Create normal and anomalous data
    let normal_events = create_normal_events();
    let anomalous_events = create_anomalous_events();
    
    // Train on normal data
    ml_service.train_on_events(&normal_events).await.unwrap();
    
    // Test anomaly detection
    let results = ml_service.detect_anomalies(&anomalous_events).await.unwrap();
    
    // Verify anomalies were detected
    assert!(results.iter().any(|r| r.is_anomaly));
}

fn create_test_events() -> Vec<Event> {
    // Create test events
    vec![]
}
```

### Metrics Collection

```rust
// src/services/metrics.rs
use metrics::{counter, gauge, histogram, register_counter, register_gauge, register_histogram};
use metrics_exporter_prometheus::PrometheusBuilder;
use std::net::SocketAddr;
use tokio::net::TcpListener;

pub struct MetricsService;

impl MetricsService {
    pub fn new() -> Self {
        // Register metrics
        register_counter!("edgeguard_events_processed_total", "Total events processed");
        register_counter!("edgeguard_events_dropped_total", "Total events dropped");
        register_counter!("edgeguard_validation_errors_total", "Total validation errors");
        register_gauge!("edgeguard_pipeline_depth", "Current pipeline depth");
        register_histogram!("edgeguard_processing_duration_seconds", "Processing duration");
        
        Self
    }
    
    pub async fn start_server(&self, addr: SocketAddr) -> Result<()> {
        let recorder = PrometheusBuilder::new().build_recorder();
        metrics::set_boxed_recorder(Box::new(recorder.clone()))?;
        
        let listener = TcpListener::bind(addr).await?;
        tracing::info!("Metrics server listening on {}", addr);
        
        loop {
            let (stream, _) = listener.accept().await?;
            let recorder = recorder.clone();
            
            tokio::spawn(async move {
                if let Err(e) = handle_metrics_request(stream, recorder).await {
                    tracing::error!("Metrics request error: {}", e);
                }
            });
        }
    }
    
    pub fn record_event_processed(&self) {
        counter!("edgeguard_events_processed_total").increment(1);
    }
    
    pub fn record_event_dropped(&self) {
        counter!("edgeguard_events_dropped_total").increment(1);
    }
    
    pub fn record_validation_error(&self) {
        counter!("edgeguard_validation_errors_total").increment(1);
    }
    
    pub fn record_pipeline_depth(&self, depth: usize) {
        gauge!("edgeguard_pipeline_depth").set(depth as f64);
    }
    
    pub fn record_processing_duration(&self, duration: Duration) {
        histogram!("edgeguard_processing_duration_seconds").record(duration.as_secs_f64());
    }
}

async fn handle_metrics_request(
    stream: tokio::net::TcpStream,
    recorder: PrometheusRecorder,
) -> Result<()> {
    // Handle Prometheus metrics request
    Ok(())
}
```

## Deployment

### Docker Configuration

```dockerfile
# docker/Dockerfile
FROM rust:1.75 as builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    libssl3 \
    libsqlite3-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/desktop-edgeguard /usr/local/bin/
COPY config/settings.toml /etc/edgeguard/

EXPOSE 8080 9090

CMD ["desktop-edgeguard", "--config", "/etc/edgeguard/settings.toml"]
```

### Docker Compose

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  edgeguard:
    build: .
    ports:
      - "8080:8080"
      - "9090:9090"
    environment:
      - EDGEGUARD_DATABASE_URL=postgres://user:password@postgres:5432/edgeguard
      - EDGEGUARD_REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    volumes:
      - ./config:/etc/edgeguard
      - ./data:/var/lib/edgeguard

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=edgeguard
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  redis_data:
  grafana_data:
```

### Systemd Service

```ini
# systemd/edgeguard.service
[Unit]
Description=EdgeGuard IoT Data Validation Service
After=network.target
Requires=network.target

[Service]
Type=simple
User=edgeguard
Group=edgeguard
WorkingDirectory=/opt/edgeguard
ExecStart=/opt/edgeguard/bin/desktop-edgeguard --config /etc/edgeguard/settings.toml
Restart=always
RestartSec=10
KillMode=mixed
KillSignal=SIGTERM

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/edgeguard
ReadWritePaths=/var/log/edgeguard

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
```

## Configuration Example

```toml
# config/settings.toml
[server]
host = "0.0.0.0"
port = 8080
workers = 4
metrics_port = 9090

[storage]
database_url = "sqlite:///var/lib/edgeguard/data.db"
redis_url = "redis://localhost:6379"
retention_days = 30
batch_size = 1000

[sensors]
polling_interval = 5000
timeout = 30000

[[sensors.sources]]
id = "outdoor_temp"
source_type = "temperature"
config = { url = "http://sensor1/api/temperature" }

[sensors.sources.validators.temperature]
min = -40.0
max = 60.0
rate_limit = 5.0
thermal_mass = 5.0

[[sensors.sources]]
id = "indoor_humidity"
source_type = "humidity"
config = { url = "http://sensor2/api/humidity" }

[sensors.sources.validators.humidity]
min = 0.0
max = 100.0
rate_limit = 10.0

[pipeline]
max_depth = 2048
backpressure_strategy = "DropOldest"
processing_timeout = 1000

[fusion]
enabled = true
algorithm = "KalmanFilter"
confidence_threshold = 0.8

[connectors]
mqtt_enabled = true
mqtt_broker = "mqtt://localhost:1883"
http_enabled = true
http_endpoints = ["http://api.example.com/sensors"]
```

This desktop deployment guide provides comprehensive coverage of EdgeGuard implementation in desktop and server environments with enterprise-grade features, monitoring, and deployment strategies.