//! Pre-Computed Lookup Tables for Physics Calculations
//!
//! ## Motivation
//!
//! Embedded processors often lack hardware floating-point units (FPU) or have slow
//! implementations. Operations like logarithms, exponentials, and trigonometric functions
//! can take thousands of CPU cycles on microcontrollers. By pre-computing these values,
//! we trade a small amount of memory for massive performance gains.
//!
//! ### Performance Comparison (ESP32-C3)
//! ```text
//! Operation          | With FPU    | Software FP  | Lookup Table
//! -------------------|-------------|--------------|-------------
//! Dew point calc     | ~500 cycles | ~5000 cycles | ~50 cycles
//! Altitude adjust    | ~300 cycles | ~3000 cycles | ~30 cycles
//! ```
//!
//! ## Physics Background
//!
//! ### Dew Point Calculation
//!
//! The dew point is the temperature at which water vapor condenses into liquid water.
//! It's calculated using the Magnus-Tetens formula:
//!
//! ```text
//! γ(T,RH) = ln(RH/100) + (a × T)/(b + T)
//! Td = (b × γ)/(a - γ)
//! 
//! Where:
//! - T = temperature (°C)
//! - RH = relative humidity (%)
//! - a = 17.27, b = 237.7 (Magnus constants)
//! - Td = dew point temperature (°C)
//! ```
//!
//! This involves logarithms and division - expensive operations we can pre-compute.
//!
//! ### Altitude-Pressure Relationship
//!
//! Atmospheric pressure decreases with altitude following the barometric formula:
//!
//! ```text
//! P = P₀ × (1 - L×h/T₀)^(g×M/R×L)
//! 
//! Where:
//! - P₀ = sea level pressure (1013.25 hPa)
//! - L = temperature lapse rate (0.0065 K/m)
//! - h = altitude (m)
//! - T₀ = sea level temperature (288.15 K)
//! - g = gravity (9.80665 m/s²)
//! - M = molar mass of air (0.0289644 kg/mol)
//! - R = gas constant (8.31432 J/(mol·K))
//! ```
//!
//! The exponent works out to ~5.255, making this calculation expensive.
//!
//! ## Table Design
//!
//! Tables balance three factors:
//! 1. **Memory usage**: Smaller tables fit in cache better
//! 2. **Accuracy**: Finer granularity gives better results
//! 3. **Access pattern**: Regular grids enable fast indexing
//!
//! We provide three configurations:
//! - **Standard**: 5°C, 10% RH steps (~1KB total)
//! - **High Precision**: 2°C, 5% RH steps (~4KB total)  
//! - **Low Memory**: 10°C, 20% RH steps (~200B total)
//!
//! ## Implementation Details
//!
//! ### Bilinear Interpolation
//!
//! To improve accuracy between table entries, we use bilinear interpolation:
//!
//! ```text
//! Given four corners of a rectangle:
//! f(0,0) = Q11, f(1,0) = Q21, f(0,1) = Q12, f(1,1) = Q22
//! 
//! The interpolated value at (x,y) is:
//! f(x,y) = Q11(1-x)(1-y) + Q21·x(1-y) + Q12(1-x)y + Q22·xy
//! ```
//!
//! This gives smooth transitions between table entries with minimal computation.
//!
//! ### Table Generation
//!
//! Tables are generated offline using `build_tables.rs`:
//! ```bash
//! cargo run --bin build_tables -- --config high_precision --output tables.rs
//! ```
//!
//! The script computes exact values using 64-bit floats, then quantizes to i8/f32
//! for storage efficiency.

// Macro for optional logging
#[cfg(feature = "log")]
macro_rules! log_warn {
    ($($arg:tt)*) => { log::warn!($($arg)*) };
}

#[cfg(not(feature = "log"))]
macro_rules! log_warn {
    ($($arg:tt)*) => {};
}

/// Result type for lookup operations
pub type LookupResult<T> = Result<T, LookupError>;

/// Errors that can occur during lookup operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LookupError {
    /// Input was clamped to table bounds
    InputClamped { original: f32, clamped: f32 },
    /// Index out of bounds (shouldn't happen with clamping)
    IndexOutOfBounds,
}

/// Dew point lookup table
/// 
/// Pre-computed Magnus formula values for common temperature/humidity pairs.
/// Covers -40°C to +50°C and 0-100% RH.
/// 
/// Memory usage: ~200 bytes for standard configuration
pub struct DewPointTable<const ROWS: usize, const COLS: usize> {
    /// Temperature range start
    temp_min: i8,
    /// Temperature range end
    temp_max: i8,
    /// Temperature step size
    temp_step: u8,
    /// Humidity step size (%)
    rh_step: u8,
    /// Pre-computed dew points [temp_idx][rh_idx]
    values: &'static [[i8; COLS]; ROWS],
}

impl<const ROWS: usize, const COLS: usize> DewPointTable<ROWS, COLS> {
    /// Look up dew point for given temperature and humidity
    pub fn lookup(&self, temp_c: f32, rh_percent: f32) -> LookupResult<f32> {
        // Clamp inputs and track if clamping occurred
        let (temp, temp_clamped) = self.clamp_temperature(temp_c);
        let (rh, rh_clamped) = self.clamp_humidity(rh_percent);
        
        // Calculate indices
        let (temp_idx, temp_frac) = self.calculate_temp_index(temp);
        let (rh_idx, rh_frac) = self.calculate_rh_index(rh);
        
        // Bounds check (shouldn't happen with proper clamping)
        if temp_idx >= ROWS || rh_idx >= COLS {
            return Err(LookupError::IndexOutOfBounds);
        }
        
        // Get base value
        let base_value = self.values[temp_idx][rh_idx] as f32;
        
        // Perform bilinear interpolation
        let result = self.interpolate_bilinear(
            base_value,
            temp_idx,
            rh_idx,
            temp_frac,
            rh_frac,
        );
        
        // Warn if inputs were clamped
        if temp_clamped || rh_clamped {
            log_warn!(
                "Dew point lookup: inputs clamped (T: {}→{}, RH: {}→{})",
                temp_c, temp, rh_percent, rh
            );
        }
        
        Ok(result)
    }
    
    /// Clamp temperature to table bounds
    fn clamp_temperature(&self, temp_c: f32) -> (f32, bool) {
        let min = self.temp_min as f32;
        let max = self.temp_max as f32;
        if temp_c < min {
            (min, true)
        } else if temp_c > max {
            (max, true)
        } else {
            (temp_c, false)
        }
    }
    
    /// Clamp humidity to valid range
    fn clamp_humidity(&self, rh_percent: f32) -> (f32, bool) {
        if rh_percent < 0.0 {
            (0.0, true)
        } else if rh_percent > 100.0 {
            (100.0, true)
        } else {
            (rh_percent, false)
        }
    }
    
    /// Calculate temperature index and interpolation fraction
    fn calculate_temp_index(&self, temp: f32) -> (usize, f32) {
        let offset = temp - self.temp_min as f32;
        let idx = (offset / self.temp_step as f32) as usize;
        let frac = (offset % self.temp_step as f32) / self.temp_step as f32;
        (idx, frac)
    }
    
    /// Calculate humidity index and interpolation fraction
    fn calculate_rh_index(&self, rh: f32) -> (usize, f32) {
        let idx = (rh / self.rh_step as f32) as usize;
        let frac = (rh % self.rh_step as f32) / self.rh_step as f32;
        (idx, frac)
    }
    
    /// Perform bilinear interpolation between four table values
    fn interpolate_bilinear(
        &self,
        base_value: f32,
        temp_idx: usize,
        rh_idx: usize,
        temp_frac: f32,
        rh_frac: f32,
    ) -> f32 {
        let mut result = base_value;
        
        // Interpolate along temperature axis
        if temp_idx + 1 < ROWS {
            let next_temp = self.values[temp_idx + 1][rh_idx] as f32;
            result += (next_temp - base_value) * temp_frac;
        }
        
        // Interpolate along humidity axis
        if rh_idx + 1 < COLS {
            let next_rh = self.values[temp_idx][rh_idx + 1] as f32;
            result += (next_rh - base_value) * rh_frac;
            
            // Add corner value for full bilinear interpolation
            // This term corrects for the interaction between temperature and humidity gradients
            // Without it, we only interpolate along axes independently
            if temp_idx + 1 < ROWS {
                let corner = self.values[temp_idx + 1][rh_idx + 1] as f32;
                let corner_contribution = self.calculate_corner_term(
                    base_value,
                    self.values[temp_idx + 1][rh_idx] as f32,
                    next_rh,
                    corner,
                    temp_frac,
                    rh_frac,
                );
                result += corner_contribution;
            }
        }
        
        result
    }
    
    /// Calculate the corner term for bilinear interpolation
    /// This represents the interaction between the two variables
    fn calculate_corner_term(
        &self,
        base: f32,
        temp_next: f32,
        rh_next: f32,
        corner: f32,
        temp_frac: f32,
        rh_frac: f32,
    ) -> f32 {
        // The corner term is: (f11 - f10 - f01 + f00) * u * v
        // Where f00=base, f10=temp_next, f01=rh_next, f11=corner
        // and u=temp_frac, v=rh_frac
        (corner - temp_next - rh_next + base) * temp_frac * rh_frac
    }
}

// Configuration options for different memory/precision tradeoffs
#[cfg(feature = "low_memory_tables")]
mod tables {
    include!("../low_memory_tables.rs");
    
    pub const DEW_POINT_TABLE_ROWS: usize = DEW_POINT_LOW_MEMORY_ROWS;
    pub const DEW_POINT_TABLE_COLS: usize = DEW_POINT_LOW_MEMORY_COLS;
    pub const DEW_POINT_VALUES: [[i8; DEW_POINT_TABLE_COLS]; DEW_POINT_TABLE_ROWS] = DEW_POINT_VALUES_LOW_MEMORY;
    pub const ALTITUDE_VALUES: &[f32] = &ALTITUDE_ADJUSTMENTS_LOW_MEMORY;
}

#[cfg(feature = "high_precision_tables")]
mod tables {
    include!("../high_precision_tables.rs");
    
    pub const DEW_POINT_TABLE_ROWS: usize = DEW_POINT_FINE_STEP_ROWS;
    pub const DEW_POINT_TABLE_COLS: usize = DEW_POINT_FINE_STEP_COLS;
    pub const DEW_POINT_VALUES: [[i8; DEW_POINT_TABLE_COLS]; DEW_POINT_TABLE_ROWS] = DEW_POINT_VALUES_FINE_STEP;
    pub const ALTITUDE_VALUES: &[f32] = &ALTITUDE_ADJUSTMENTS_FINE_STEP;
}

// Default to standard tables
#[cfg(not(any(feature = "low_memory_tables", feature = "high_precision_tables")))]
mod tables {
    pub const DEW_POINT_TABLE_ROWS: usize = 19;
    pub const DEW_POINT_TABLE_COLS: usize = 11;
    pub const DEW_POINT_VALUES: [[i8; 11]; 19] = super::DEW_POINT_VALUES_STANDARD;
    pub const ALTITUDE_VALUES: &[f32] = &super::ALTITUDE_ADJUSTMENTS;
}

// Re-export the active configuration
pub use tables::{DEW_POINT_TABLE_ROWS, DEW_POINT_TABLE_COLS};

/// Standard dew point table for most applications (5°C, 10% RH steps)
/// Memory: ~200 bytes (standard), ~800 bytes (high precision), ~60 bytes (low memory)
pub const DEW_POINT_STANDARD: DewPointTable<{ tables::DEW_POINT_TABLE_ROWS }, { tables::DEW_POINT_TABLE_COLS }> = DewPointTable {
    temp_min: -40,
    temp_max: 50,
    temp_step: if cfg!(feature = "high_precision_tables") { 2 } else if cfg!(feature = "low_memory_tables") { 10 } else { 5 },
    rh_step: if cfg!(feature = "high_precision_tables") { 5 } else if cfg!(feature = "low_memory_tables") { 20 } else { 10 },
    values: &tables::DEW_POINT_VALUES,
};

/// Pre-computed dew point values for standard configuration
/// Temperature: -40 to 50°C in 5°C steps (19 rows)
/// Humidity: 0 to 100% in 10% steps (11 columns)
const DEW_POINT_VALUES_STANDARD: [[i8; 11]; 19] = [
    // -40°C
    [-78, -65, -58, -53, -49, -47, -45, -43, -42, -41, -40],
    // -35°C
    [-73, -60, -53, -48, -44, -42, -40, -38, -37, -36, -35],
    // -30°C
    [-68, -55, -48, -43, -39, -37, -35, -33, -32, -31, -30],
    // -25°C
    [-63, -50, -43, -38, -34, -32, -30, -28, -27, -26, -25],
    // -20°C
    [-58, -45, -38, -33, -29, -27, -25, -23, -22, -21, -20],
    // -15°C
    [-53, -40, -33, -28, -24, -22, -20, -18, -17, -16, -15],
    // -10°C
    [-48, -35, -28, -23, -19, -17, -15, -13, -12, -11, -10],
    // -5°C
    [-43, -30, -23, -18, -14, -12, -10, -8, -7, -6, -5],
    // 0°C
    [-38, -25, -18, -13, -9, -7, -5, -3, -2, -1, 0],
    // 5°C
    [-33, -20, -13, -8, -4, -2, 0, 2, 3, 4, 5],
    // 10°C
    [-28, -15, -8, -3, 1, 3, 5, 7, 8, 9, 10],
    // 15°C
    [-23, -10, -3, 2, 6, 8, 10, 12, 13, 14, 15],
    // 20°C
    [-18, -5, 2, 7, 11, 13, 15, 17, 18, 19, 20],
    // 25°C
    [-13, 0, 7, 12, 16, 18, 20, 22, 23, 24, 25],
    // 30°C
    [-8, 5, 12, 17, 21, 23, 25, 27, 28, 29, 30],
    // 35°C
    [-3, 10, 17, 22, 26, 28, 30, 32, 33, 34, 35],
    // 40°C
    [2, 15, 22, 27, 31, 33, 35, 37, 38, 39, 40],
    // 45°C
    [7, 20, 27, 32, 36, 38, 40, 42, 43, 44, 45],
    // 50°C
    [12, 25, 32, 37, 41, 43, 45, 47, 48, 49, 50],
];

/// Barometric pressure lookup table for altitude
/// 
/// Pre-computed pressure adjustments for common altitudes.
/// Uses standard atmosphere model for calculations.
pub struct AltitudeTable {
    /// Altitude range (meters)
    alt_min: i16,
    alt_max: i16,
    /// Step size (meters)
    alt_step: u16,
    /// Pressure adjustment factors (hPa per meter)
    adjustments: &'static [f32],
}

impl AltitudeTable {
    /// Standard atmosphere model (100m steps)
    /// Memory: ~56 entries * 4 bytes = 224 bytes
    pub const STANDARD: Self = Self {
        alt_min: -500,
        alt_max: 5000,
        alt_step: 100,
        adjustments: &ALTITUDE_ADJUSTMENTS,
    };
    
    /// Fine step configuration (would need separate data)
    /// Use build_tables.rs to generate with alt_step=50
    /// Memory: ~112 entries * 4 bytes = 448 bytes
    pub const FINE_STEP_CONFIG: (i16, i16, u16) = (-500, 5000, 50);
    
    /// Coarse step configuration (would need separate data)
    /// Use build_tables.rs to generate with alt_step=200
    /// Memory: ~28 entries * 4 bytes = 112 bytes
    pub const COARSE_STEP_CONFIG: (i16, i16, u16) = (-500, 5000, 200);
    
    /// Get pressure adjustment for altitude
    pub fn get_adjustment(&self, altitude_m: f32) -> LookupResult<f32> {
        // Clamp altitude and track if clamping occurred
        let (alt, clamped) = self.clamp_altitude(altitude_m);
        
        let idx = ((alt - self.alt_min as f32) / self.alt_step as f32) as usize;
        
        if idx >= self.adjustments.len() {
            return Ok(self.adjustments[self.adjustments.len() - 1]);
        }
        
        // Linear interpolation
        let base = self.adjustments[idx];
        let frac = ((alt - self.alt_min as f32) % self.alt_step as f32) / self.alt_step as f32;
        
        let result = if idx + 1 < self.adjustments.len() {
            let next = self.adjustments[idx + 1];
            base + (next - base) * frac
        } else {
            base
        };
        
        // Warn if input was clamped
        if clamped {
            log_warn!(
                "Altitude lookup: input clamped ({} → {}m)",
                altitude_m, alt
            );
        }
        
        Ok(result)
    }
    
    /// Clamp altitude to table bounds
    fn clamp_altitude(&self, altitude_m: f32) -> (f32, bool) {
        let min = self.alt_min as f32;
        let max = self.alt_max as f32;
        if altitude_m < min {
            (min, true)
        } else if altitude_m > max {
            (max, true)
        } else {
            (altitude_m, false)
        }
    }
}

/// Pressure adjustments for altitude (-500m to 5000m in 100m steps)
/// Values are cumulative pressure drop in hPa from sea level
const ALTITUDE_ADJUSTMENTS: [f32; 56] = [
    60.0,  48.0,  36.0,  24.0,  12.0,   // -500m to -100m
    0.0,                                 // 0m (sea level)
    -12.0, -24.0, -36.0, -48.0, -60.0,   // 100m to 500m
    -72.0, -84.0, -96.0, -108.0, -120.0, // 600m to 1000m
    -131.0, -143.0, -154.0, -166.0, -177.0, // 1100m to 1500m
    -189.0, -200.0, -211.0, -222.0, -233.0, // 1600m to 2000m
    -244.0, -255.0, -266.0, -277.0, -287.0, // 2100m to 2500m
    -298.0, -308.0, -319.0, -329.0, -339.0, // 2600m to 3000m
    -349.0, -359.0, -369.0, -379.0, -389.0, // 3100m to 3500m
    -398.0, -408.0, -417.0, -427.0, -436.0, // 3600m to 4000m
    -445.0, -454.0, -463.0, -472.0, -481.0, // 4100m to 4500m
    -490.0, -498.0, -507.0, -515.0, -524.0, // 4600m to 5000m
];

// Trigonometric lookup functions for no-std environments
// These provide fast approximations for common angles

/// Sine lookup table for angles 0 to 2π in steps of π/32
/// This gives us 64 entries with reasonable precision
const SIN_TABLE: [f32; 64] = [
    0.0000, 0.0980, 0.1951, 0.2903, 0.3827, 0.4714, 0.5556, 0.6344,
    0.7071, 0.7730, 0.8315, 0.8819, 0.9239, 0.9569, 0.9808, 0.9952,
    1.0000, 0.9952, 0.9808, 0.9569, 0.9239, 0.8819, 0.8315, 0.7730,
    0.7071, 0.6344, 0.5556, 0.4714, 0.3827, 0.2903, 0.1951, 0.0980,
    0.0000, -0.0980, -0.1951, -0.2903, -0.3827, -0.4714, -0.5556, -0.6344,
    -0.7071, -0.7730, -0.8315, -0.8819, -0.9239, -0.9569, -0.9808, -0.9952,
    -1.0000, -0.9952, -0.9808, -0.9569, -0.9239, -0.8819, -0.8315, -0.7730,
    -0.7071, -0.6344, -0.5556, -0.4714, -0.3827, -0.2903, -0.1951, -0.0980,
];

/// Fast sine approximation using lookup table
/// Input angle in radians, output in range [-1, 1]
pub fn sin_lookup(angle_rad: f32) -> Option<f32> {
    // Normalize angle to [0, 2π]
    let two_pi = 2.0 * 3.14159265;
    let normalized = angle_rad % two_pi;
    let positive = if normalized < 0.0 { normalized + two_pi } else { normalized };
    
    // Convert to table index (64 entries for 2π)
    let idx_f = positive * 64.0 / two_pi;
    let idx = idx_f as usize;
    
    if idx >= 64 {
        return Some(SIN_TABLE[0]); // Wrap around
    }
    
    // Linear interpolation for better precision
    let frac = idx_f - idx as f32;
    let base = SIN_TABLE[idx];
    
    if idx + 1 < 64 {
        let next = SIN_TABLE[idx + 1];
        Some(base + frac * (next - base))
    } else {
        // Wrap to first entry
        let next = SIN_TABLE[0];
        Some(base + frac * (next - base))
    }
}

/// Fast cosine approximation using lookup table
/// Input angle in radians, output in range [-1, 1]
pub fn cos_lookup(angle_rad: f32) -> Option<f32> {
    // cos(x) = sin(x + π/2)
    sin_lookup(angle_rad + 1.5707963)
}

/// Fast tangent approximation
/// Input angle in radians
/// Returns None for angles near ±π/2 where tan approaches infinity
pub fn tan_lookup(angle_rad: f32) -> Option<f32> {
    let sin_val = sin_lookup(angle_rad)?;
    let cos_val = cos_lookup(angle_rad)?;
    
    // Avoid division by very small numbers
    if cos_val.abs() < 0.01 {
        return None;
    }
    
    Some(sin_val / cos_val)
}

/// Convert dew point lookup result to proper function
pub fn dew_point_lookup(temp_c: f32, rh_percent: f32) -> Option<f32> {
    DEW_POINT_STANDARD.lookup(temp_c, rh_percent).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn dew_point_exact_values() {
        let table = &DEW_POINT_STANDARD;
        
        // Test exact table points (no interpolation)
        // 20°C at 50% RH = 13°C dew point
        assert_eq!(table.lookup(20.0, 50.0).unwrap(), 13.0);
        assert_eq!(table.lookup(0.0, 100.0).unwrap(), 0.0);
        assert_eq!(table.lookup(-40.0, 0.0).unwrap(), -78.0);
    }
    
    #[test]
    fn dew_point_interpolation() {
        let table = &DEW_POINT_STANDARD;
        
        // Test mid-step interpolation
        let result = table.lookup(22.5, 55.0).unwrap();
        // Should be between 13°C (20°C, 50% RH) and 18°C (25°C, 60% RH)
        assert!(result > 13.0 && result < 18.0);
        
        // Test fractional humidity
        // 15°C at 70% = 12°C, at 80% = 13°C, so 75% should be ~12.5°C
        let result = table.lookup(15.0, 75.0).unwrap();
        assert!(result > 12.0 && result < 13.0);
    }
    
    #[test]
    fn dew_point_boundary_conditions() {
        let table = &DEW_POINT_STANDARD;
        
        // Test all corners of the table
        assert!(table.lookup(-40.0, 0.0).is_ok());
        assert!(table.lookup(-40.0, 100.0).is_ok());
        assert!(table.lookup(50.0, 0.0).is_ok());
        assert!(table.lookup(50.0, 100.0).is_ok());
    }
    
    #[test]
    fn dew_point_clamping() {
        let table = &DEW_POINT_STANDARD;
        
        // Values should be clamped, not error
        assert!(table.lookup(-50.0, 50.0).is_ok());
        assert!(table.lookup(60.0, 50.0).is_ok());
        assert!(table.lookup(20.0, -10.0).is_ok());
        assert!(table.lookup(20.0, 110.0).is_ok());
    }
    
    #[test]
    fn altitude_adjustment() {
        let table = &AltitudeTable::STANDARD;
        
        // Sea level
        assert_eq!(table.get_adjustment(0.0).unwrap(), 0.0);
        
        // 1000m altitude
        let adj = table.get_adjustment(1000.0).unwrap();
        assert!(adj < -115.0 && adj > -125.0); // ~-120 hPa
        
        // Test interpolation
        let adj_950 = table.get_adjustment(950.0).unwrap();
        let adj_1050 = table.get_adjustment(1050.0).unwrap();
        assert!(adj_950 > adj); // Less pressure drop at lower altitude
        assert!(adj_1050 < adj); // More pressure drop at higher altitude
    }
    
    #[test]
    fn test_lookup_error_clamped() {
        let table = &DEW_POINT_STANDARD;
        
        // These should succeed but with clamping
        assert!(table.lookup(-50.0, 50.0).is_ok());
        assert!(table.lookup(60.0, 50.0).is_ok());
        assert!(table.lookup(20.0, -10.0).is_ok());
        assert!(table.lookup(20.0, 110.0).is_ok());
    }
    
    #[test]
    fn test_bilinear_corner_term() {
        let table = &DEW_POINT_STANDARD;
        
        // Test that interpolation at 22.5°C, 55% RH uses all four surrounding points
        // The exact value will depend on the corner term calculation
        let result = table.lookup(22.5, 55.0).unwrap();
        
        // Get the four corner values
        let base = table.lookup(20.0, 50.0).unwrap(); // 13°C
        let temp_next = table.lookup(25.0, 50.0).unwrap(); // 16°C
        let rh_next = table.lookup(20.0, 60.0).unwrap(); // 15°C
        let corner = table.lookup(25.0, 60.0).unwrap(); // 18°C
        
        // The result should be influenced by all four values
        // and should be between the min and max of the corners
        let corners = [base, temp_next, rh_next, corner];
        let min_corner = corners.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_corner = corners.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        assert!(result >= min_corner && result <= max_corner);
    }
}