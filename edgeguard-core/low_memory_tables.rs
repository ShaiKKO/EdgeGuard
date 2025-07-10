// Auto-generated lookup tables for EdgeGuard
// Configuration: LOW_MEMORY
// Temperature: -40..50 step 10, Humidity: 0..100 step 20
// Generated on: (use --timestamp for current time)

/// Dew point table dimensions for LOW_MEMORY
pub const DEW_POINT_LOW_MEMORY_ROWS: usize = 10;
pub const DEW_POINT_LOW_MEMORY_COLS: usize = 6;

/// Dew point values for LOW_MEMORY configuration
pub const DEW_POINT_VALUES_LOW_MEMORY: [[i8; 6]; 10] = [
    // -40°C
    [-100, -54, -48, -45, -42, -40],
    // -30°C
    [-100, -46, -39, -35, -32, -30],
    // -20°C
    [-100, -37, -30, -26, -23, -20],
    // -10°C
    [-100, -29, -21, -16, -13, -10],
    // 0°C
    [-100, -20, -12, -7, -3, 0],
    // 10°C
    [-100, -12, -3, 3, 7, 10],
    // 20°C
    [-100, -4, 6, 12, 16, 20],
    // 30°C
    [-100, 5, 15, 21, 26, 30],
    // 40°C
    [-100, 13, 24, 31, 36, 40],
    // 50°C
    [-100, 21, 33, 40, 46, 50]
];

/// Altitude adjustment table size for LOW_MEMORY
pub const ALTITUDE_LOW_MEMORY_ENTRIES: usize = 28;

/// Altitude pressure adjustments for LOW_MEMORY configuration
pub const ALTITUDE_ADJUSTMENTS_LOW_MEMORY: [f32; 28] = [
    61.5, 36.6, 12.1, -12.0, -35.5,  // -500m to 300m
    -58.6, -81.3, -103.5, -125.4, -146.7,  // 500m to 1300m
    -167.7, -188.2, -208.4, -228.1, -247.5,  // 1500m to 2300m
    -266.4, -285.0, -303.2, -321.0, -338.5,  // 2500m to 3300m
    -355.6, -372.4, -388.8, -404.8, -420.6,  // 3500m to 4300m
    -436.0, -451.0, -465.8 // 4800m to 5000m
];
