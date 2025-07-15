"""
EdgeGuard Python Bindings

Physics-aware data validation and aggregation for IoT edge devices.

This package provides Python bindings for the EdgeGuard Rust library,
enabling high-performance sensor validation with physics-based constraints.

Example:
    Basic usage of EdgeGuard validators:

    >>> import edgeguard
    >>> validator = edgeguard.TemperatureValidator(min_temp=-20.0, max_temp=60.0)
    >>> result = validator.validate(25.5)
    >>> print(f"Valid: {result.is_valid}")
"""

from ._edgeguard import *

__version__ = "0.1.0"
__author__ = "EdgeGuard Contributors"

__all__ = [
    # Core version
    "__version__",
    
    # Will be populated as we add bindings
    # "TemperatureValidator",
    # "HumidityValidator", 
    # "PressureValidator",
    # "ValidationResult",
    # "Pipeline",
    # "Event",
]