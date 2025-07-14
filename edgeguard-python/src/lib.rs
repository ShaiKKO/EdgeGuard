//! EdgeGuard Python Bindings
//! 
//! This module provides Python bindings for EdgeGuard's physics-aware IoT sensor validation library.

use pyo3::prelude::*;

/// EdgeGuard Python module
#[pymodule]
fn edgeguard(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}