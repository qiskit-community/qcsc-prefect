"""GB SQD Prefect workflow integration."""

__version__ = "0.1.0"

from .main import ext_sqd_flow, trim_sqd_flow, ext_sqd_simple_flow, trim_sqd_simple_flow

__all__ = [
    "ext_sqd_flow",
    "trim_sqd_flow",
    "ext_sqd_simple_flow",
    "trim_sqd_simple_flow",
]

# Made with Bob
