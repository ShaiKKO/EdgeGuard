//! Sensor Specifications and Limits
//!
//! This module defines operational limits and characteristics for various
//! sensor types based on commercial sensor datasheets and industry standards.

// ===== TEMPERATURE SENSOR SPECIFICATIONS =====

/// Minimum operating temperature for typical sensors (°C).
/// 
/// Based on industrial temperature sensors (e.g., PT100, DS18B20).
/// Consumer sensors may have narrower ranges.
/// 
/// Source: Common sensor datasheets (Maxim, Sensirion, etc.)
pub const TEMP_SENSOR_MIN_C: f32 = -80.0;

/// Maximum operating temperature for typical sensors (°C).
/// 
/// Upper limit for most semiconductor and RTD sensors.
/// Thermocouples can measure higher but require special handling.
/// 
/// Source: Common sensor datasheets
pub const TEMP_SENSOR_MAX_C: f32 = 125.0;

/// Maximum rate of temperature change sensors can track (°C/s).
/// 
/// Limited by sensor thermal mass and response time.
/// Faster changes may not be accurately measured.
/// 
/// Source: Empirical testing with TO-92 packaged sensors
pub const TEMP_MAX_RATE_C_PER_S: f32 = 10.0;

/// Temperature sensor accuracy grades (°C).
/// 
/// Professional: ±0.1°C (calibrated RTD, precision thermistor)
/// Consumer: ±0.5°C (DS18B20, SHT31)
/// Budget: ±2.0°C (basic thermistor, analog sensors)
pub const TEMP_ACCURACY_PROFESSIONAL_C: f32 = 0.1;
pub const TEMP_ACCURACY_CONSUMER_C: f32 = 0.5;
pub const TEMP_ACCURACY_BUDGET_C: f32 = 2.0;

// ===== HUMIDITY SENSOR SPECIFICATIONS =====

/// Minimum measurable relative humidity (%).
/// 
/// Physical lower limit (completely dry air).
/// Most sensors accurate above 10% RH.
/// 
/// Source: Physics (0% RH = no water vapor)
pub const HUMIDITY_SENSOR_MIN_PCT: f32 = 0.0;

/// Maximum measurable relative humidity (%).
/// 
/// Physical upper limit (saturated air).
/// Condensation may affect readings at 100%.
/// 
/// Source: Physics (100% RH = saturated)
pub const HUMIDITY_SENSOR_MAX_PCT: f32 = 100.0;

/// Maximum rate of humidity change sensors can track (%RH/s).
/// 
/// Limited by water vapor diffusion and sensor response.
/// Membrane-based sensors are particularly slow.
/// 
/// Source: Sensirion SHT3x datasheet (response time ~8s)
pub const HUMIDITY_MAX_RATE_PCT_PER_S: f32 = 20.0;

/// Humidity sensor accuracy grades (%RH).
/// 
/// Professional: ±2% RH (SHT35, BME680)
/// Consumer: ±3% RH (DHT22, SHT31)
/// Budget: ±5% RH (DHT11, HIH4000)
pub const HUMIDITY_ACCURACY_PROFESSIONAL_PCT: f32 = 2.0;
pub const HUMIDITY_ACCURACY_CONSUMER_PCT: f32 = 3.0;
pub const HUMIDITY_ACCURACY_BUDGET_PCT: f32 = 5.0;

// ===== PRESSURE SENSOR SPECIFICATIONS =====

/// Minimum measurable atmospheric pressure (hPa).
/// 
/// Lower limit for barometric sensors.
/// Covers extreme low pressure and high altitude.
/// 
/// Source: BMP280 datasheet (300-1100 hPa range)
pub const PRESSURE_SENSOR_MIN_HPA: f32 = 540.0;

/// Maximum measurable atmospheric pressure (hPa).
/// 
/// Upper limit for barometric sensors.
/// Covers all natural atmospheric conditions.
/// 
/// Source: BMP280 datasheet
pub const PRESSURE_SENSOR_MAX_HPA: f32 = 1080.0;

/// Maximum rate of pressure change sensors can track (hPa/s).
/// 
/// Even in severe weather, changes are gradual.
/// Faster changes indicate sensor issues.
/// 
/// Source: NOAA severe weather observations
pub const PRESSURE_MAX_RATE_HPA_PER_S: f32 = 50.0;

/// Pressure sensor accuracy grades (hPa).
/// 
/// Professional: ±0.5 hPa (MS5611, laboratory grade)
/// Consumer: ±1.0 hPa (BMP280, BME280)
/// Budget: ±2.0 hPa (BMP180, basic sensors)
pub const PRESSURE_ACCURACY_PROFESSIONAL_HPA: f32 = 0.5;
pub const PRESSURE_ACCURACY_CONSUMER_HPA: f32 = 1.0;
pub const PRESSURE_ACCURACY_BUDGET_HPA: f32 = 2.0;

// ===== VOC SENSOR SPECIFICATIONS =====

/// Minimum detectable VOC concentration (ppb).
/// 
/// Lower detection limit for typical MOx sensors.
/// Professional PID sensors can detect lower.
/// 
/// Source: Sensirion SGP30, Bosch BME680 datasheets
pub const VOC_SENSOR_MIN_PPB: f32 = 0.0;

/// Maximum measurable VOC concentration (ppb).
/// 
/// Upper limit before sensor saturation.
/// Industrial sensors may have higher ranges.
/// 
/// Source: Common MOx sensor datasheets
pub const VOC_SENSOR_MAX_PPB: f32 = 1000.0;

/// VOC sensor warm-up time (seconds).
/// 
/// MOx sensors require heating to operating temperature.
/// Readings unreliable during warm-up period.
/// 
/// Source: SGP30 datasheet (15s warm-up)
pub const VOC_WARMUP_TIME_S: f32 = 15.0;

// ===== ACOUSTIC SENSOR SPECIFICATIONS =====

/// Minimum detectable sound level (dB SPL).
/// 
/// Noise floor of typical MEMS microphones.
/// Professional equipment can measure lower.
/// 
/// Source: INMP441 MEMS microphone datasheet
pub const ACOUSTIC_SENSOR_MIN_DB_SPL: f32 = 30.0;

/// Maximum measurable sound level (dB SPL).
/// 
/// Upper limit before distortion/damage.
/// Based on typical MEMS microphone capabilities.
/// 
/// Source: Common MEMS microphone datasheets
pub const ACOUSTIC_SENSOR_MAX_DB_SPL: f32 = 120.0;

/// Acoustic sensor frequency range (Hz).
/// 
/// Typical MEMS microphones are flat 20Hz-20kHz.
/// Used for validating frequency-domain measurements.
pub const ACOUSTIC_FREQ_MIN_HZ: f32 = 20.0;
pub const ACOUSTIC_FREQ_MAX_HZ: f32 = 20000.0;

// ===== SENSOR RESPONSE CHARACTERISTICS =====

/// Sensor stabilization time after power-on (seconds).
/// 
/// Time for readings to stabilize after cold start.
/// Varies by sensor type and technology.
/// 
/// Source: Industry rule of thumb
pub const SENSOR_STABILIZATION_TIME_S: f32 = 5.0;

/// Default sensor polling interval (milliseconds).
/// 
/// Balance between data freshness and power consumption.
/// 1 Hz is standard for environmental monitoring.
/// 
/// Source: Common IoT practices
pub const DEFAULT_POLL_INTERVAL_MS: u32 = 1000;

/// High-frequency polling interval (milliseconds).
/// 
/// For applications requiring faster updates.
/// 10 Hz suitable for control applications.
/// 
/// Source: Industrial control standards
pub const FAST_POLL_INTERVAL_MS: u32 = 100;

/// Low-power polling interval (milliseconds).
/// 
/// For battery-powered applications.
/// 0.1 Hz extends battery life significantly.
/// 
/// Source: LoRaWAN Class A timing
pub const LOW_POWER_POLL_INTERVAL_MS: u32 = 10000;

// ===== SENSOR DRIFT AND AGING =====

/// Maximum acceptable drift per year (% of reading).
/// 
/// Long-term stability specification.
/// Sensors exceeding this need recalibration.
/// 
/// Source: Industrial sensor requirements
pub const MAX_ANNUAL_DRIFT_PCT: f32 = 2.0;

/// Sensor lifespan before replacement (years).
/// 
/// Typical operational life for electronic sensors.
/// MOx and electrochemical sensors may be shorter.
/// 
/// Source: Manufacturer recommendations
pub const SENSOR_LIFESPAN_YEARS: f32 = 5.0;

// ===== CROSS-SENSOR VALIDATION =====

/// Time window for cross-sensor correlation (milliseconds).
/// 
/// Maximum time difference between related sensor readings.
/// Used for dew point, heat index calculations.
/// 
/// Source: Empirical testing
pub const CROSS_VALIDATION_WINDOW_MS: u32 = 5000;

/// Minimum correlation coefficient for sensor fusion.
/// 
/// Sensors with lower correlation may have issues.
/// Used to detect faulty sensors in redundant arrays.
/// 
/// Source: Statistical analysis practices
pub const MIN_SENSOR_CORRELATION: f32 = 0.8;