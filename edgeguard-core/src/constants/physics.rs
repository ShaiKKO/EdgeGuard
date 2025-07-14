//! Physical Constants for EdgeGuard
//!
//! This module defines fundamental physical constants and environmental limits
//! used throughout the sensor validation system. All values are based on
//! established physics principles and industry standards.

// ===== FUNDAMENTAL PHYSICS CONSTANTS =====

/// Absolute zero in Celsius (°C).
/// 
/// The theoretical lower limit of temperature where molecular motion ceases.
/// No physical system can reach temperatures below this value.
/// 
/// Source: NIST Special Publication 330 (2019)
pub const ABSOLUTE_ZERO_CELSIUS: f32 = -273.15;

/// Standard atmospheric pressure at sea level (hPa/mbar).
/// 
/// Reference pressure used for altitude calculations and weather systems.
/// Actual pressure varies with weather patterns and altitude.
/// 
/// Source: International Standard Atmosphere (ISA)
pub const SEA_LEVEL_PRESSURE_HPA: f32 = 1013.25;

/// Pressure change per meter of altitude (hPa/m).
/// 
/// Approximate rate of pressure decrease with altitude in the troposphere.
/// Used for altitude estimation from pressure readings.
/// 
/// Source: Barometric formula, valid up to ~11km altitude
pub const PRESSURE_LAPSE_RATE_HPA_PER_M: f32 = 0.12;

/// Pressure drop per 100 meters of altitude (hPa/100m).
/// 
/// Convenient unit for altitude-based pressure validation.
/// Equivalent to PRESSURE_LAPSE_RATE_HPA_PER_M * 100.
/// 
/// Source: Standard atmosphere calculations
pub const PRESSURE_DROP_PER_100M_HPA: f32 = 12.0;

/// Speed of sound in air at 20°C (m/s).
/// 
/// Used for acoustic sensor calculations and distance measurements.
/// Varies with temperature: c = 331.3 + 0.606 * T(°C).
/// 
/// Source: ISO 9613-1:1993
pub const SPEED_OF_SOUND_AIR_20C_M_PER_S: f32 = 343.2;

// ===== ENVIRONMENTAL LIMITS =====

/// Minimum survivable environmental temperature (°C).
/// 
/// Coldest natural temperature recorded on Earth (-89.2°C at Vostok Station).
/// Sensors operating below -80°C require special consideration.
/// 
/// Source: World Meteorological Organization
pub const ENVIRONMENT_TEMP_MIN_C: f32 = -80.0;

/// Maximum survivable environmental temperature (°C).
/// 
/// Hottest reliable temperature recorded on Earth (54.4°C in Death Valley).
/// Industrial environments may exceed this locally.
/// 
/// Source: World Meteorological Organization
pub const ENVIRONMENT_TEMP_MAX_C: f32 = 60.0;

/// Minimum possible atmospheric pressure at Earth's surface (hPa).
/// 
/// Lowest pressure recorded in strongest typhoons/hurricanes.
/// Below this indicates sensor malfunction.
/// 
/// Source: WMO records (Typhoon Tip, 1979: 870 hPa)
pub const STORM_PRESSURE_MIN_HPA: f32 = 870.0;

/// Maximum possible atmospheric pressure at Earth's surface (hPa).
/// 
/// Highest pressures occur in Siberian winter high pressure systems.
/// Above this indicates sensor malfunction.
/// 
/// Source: WMO records (Agata, Siberia: 1083.8 hPa)
pub const HIGH_PRESSURE_MAX_HPA: f32 = 1085.0;

// ===== INDOOR ENVIRONMENT STANDARDS =====

/// Minimum comfortable indoor temperature (°C).
/// 
/// Lower bound of ASHRAE comfort zone for sedentary activity.
/// Below this, occupants typically feel cold.
/// 
/// Source: ASHRAE Standard 55-2020, Section 5.3
pub const INDOOR_TEMP_MIN_C: f32 = 20.0;

/// Maximum comfortable indoor temperature (°C).
/// 
/// Upper bound of ASHRAE comfort zone for sedentary activity.
/// Above this, occupants typically feel warm.
/// 
/// Source: ASHRAE Standard 55-2020, Section 5.3
pub const INDOOR_TEMP_MAX_C: f32 = 24.0;

/// Nominal indoor temperature setpoint (°C).
/// 
/// Typical HVAC setpoint for comfort and energy efficiency.
/// Used as default for indoor environment simulations.
/// 
/// Source: ENERGY STAR recommendations
pub const INDOOR_TEMP_NOMINAL_C: f32 = 22.0;

/// Minimum recommended indoor relative humidity (%).
/// 
/// Below this, occupants may experience dry skin, irritation.
/// Also increases static electricity and dust.
/// 
/// Source: ASHRAE Standard 55-2020, Section 5.2
pub const INDOOR_HUMIDITY_MIN_PCT: f32 = 30.0;

/// Maximum recommended indoor relative humidity (%).
/// 
/// Above this, risk of mold growth and condensation increases.
/// Also reduces cooling effectiveness in summer.
/// 
/// Source: ASHRAE Standard 55-2020, Section 5.2
pub const INDOOR_HUMIDITY_MAX_PCT: f32 = 60.0;

// ===== RATE OF CHANGE LIMITS =====

/// Maximum rate of temperature change in air (°C/s).
/// 
/// Based on thermal mass and heat transfer limits in air.
/// Faster changes indicate sensor error or extreme events (fire).
/// 
/// Calculation: Even with 1000W/m³ heating, air temp rises ~1°C/s
pub const AIR_TEMP_MAX_RATE_C_PER_S: f32 = 10.0;

/// Maximum rate of temperature change in water (°C/s).
/// 
/// Water has 4x the heat capacity of air, limiting rate of change.
/// Used for aquatic sensor validation.
/// 
/// Source: Specific heat capacity ratio (4.18 kJ/kg·K vs 1.0 kJ/kg·K)
pub const WATER_TEMP_MAX_RATE_C_PER_S: f32 = 2.5;

/// Maximum rate of humidity change (% RH/s).
/// 
/// Limited by water vapor diffusion and air mixing rates.
/// Instantaneous changes indicate sensor malfunction.
/// 
/// Source: Empirical observations in HVAC systems
pub const HUMIDITY_MAX_RATE_PCT_PER_S: f32 = 20.0;

/// Maximum rate of atmospheric pressure change (hPa/s).
/// 
/// Even in severe weather, pressure changes are gradual.
/// Limit based on fastest recorded pressure drops in tornadoes.
/// 
/// Source: NOAA severe weather data
pub const PRESSURE_MAX_RATE_HPA_PER_S: f32 = 50.0;

// ===== THERMAL PROPERTIES =====

/// Specific heat capacity of air at 20°C (J/kg·K).
/// 
/// Energy required to raise 1kg of air by 1°C.
/// Used for thermal mass calculations.
/// 
/// Source: NIST Chemistry WebBook
pub const AIR_SPECIFIC_HEAT_J_PER_KG_K: f32 = 1005.0;

/// Density of air at 20°C, 1 atm (kg/m³).
/// 
/// Used for thermal mass and acoustic calculations.
/// Varies with temperature and pressure.
/// 
/// Source: International Standard Atmosphere
pub const AIR_DENSITY_20C_KG_PER_M3: f32 = 1.204;

/// Thermal time constant for small sensors in air (seconds).
/// 
/// Time for sensor to reach 63.2% of ambient temperature change.
/// Varies with sensor size and construction.
/// 
/// Source: Typical for TO-92 packaged sensors
pub const SENSOR_THERMAL_TIME_CONSTANT_AIR_S: f32 = 30.0;

/// Thermal time constant for sensors in water (seconds).
/// 
/// Faster response due to better thermal conductivity of water.
/// Important for aquatic monitoring applications.
/// 
/// Source: Empirical data for waterproof sensors
pub const SENSOR_THERMAL_TIME_CONSTANT_WATER_S: f32 = 5.0;

// ===== ACOUSTIC LIMITS =====

/// Threshold of hearing at 1 kHz (dB SPL).
/// 
/// Quietest sound audible to average human ear.
/// Reference level for acoustic measurements.
/// 
/// Source: ISO 226:2003
pub const HEARING_THRESHOLD_DB_SPL: f32 = 0.0;

/// Pain threshold for human hearing (dB SPL).
/// 
/// Sound levels above this cause immediate pain and damage.
/// Upper limit for acoustic sensor validation.
/// 
/// Source: OSHA hearing conservation standards
pub const HEARING_PAIN_THRESHOLD_DB_SPL: f32 = 140.0;

/// Theoretical maximum sound pressure in air at 1 atm (dB SPL).
/// 
/// At this level, sound waves become shock waves.
/// Absolute upper limit for acoustic sensors.
/// 
/// Source: Acoustic physics (pressure = 1 atm)
pub const MAX_SOUND_PRESSURE_AIR_DB_SPL: f32 = 194.0;

// ===== HUMIDITY AND DEW POINT PHYSICS =====

/// Dew point calculation error margin (°C).
/// 
/// Accounts for sensor accuracy when validating dew point
/// against air temperature. Prevents false positives.
/// 
/// Source: Typical temperature sensor accuracy
pub const DEW_POINT_MARGIN_C: f32 = 0.5;

/// Temperature threshold for high humidity warning (°C).
/// 
/// Above this temperature, high humidity (>85%) is unusual
/// except in tropical environments.
/// 
/// Source: Human comfort and weather patterns
pub const HIGH_TEMP_HUMIDITY_THRESHOLD_C: f32 = 35.0;

/// Humidity threshold at high temperatures (%).
/// 
/// Above 35°C, humidity >85% is uncommon and may indicate
/// sensor error or extreme tropical conditions.
/// 
/// Source: Global weather data analysis
pub const HIGH_TEMP_MAX_HUMIDITY_PCT: f32 = 85.0;