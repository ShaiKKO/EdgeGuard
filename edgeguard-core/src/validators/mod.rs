//! Sensor validators with physics constraints
//!
//! Each validator enforces physical laws specific to its sensor type.
//! We use fixed limits based on what's physically possible, not just
//! what the sensor datasheet says.

mod temperature;
mod humidity;
mod pressure;
mod utils;

pub use temperature::TemperatureValidator;
pub use humidity::HumidityValidator;
pub use pressure::PressureValidator;