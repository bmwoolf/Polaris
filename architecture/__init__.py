"""
Neural Tropism World Model (NTWM) - Main package entry point.

This package provides a modular, safety-focused architecture for RNA/DNA delivery modeling.
For the main package, see ntwm/ subdirectory.
"""

# Import everything from the main package
from ntwm import *

__version__ = "0.1.0"
__all__ = [
    # Re-export everything from ntwm package
    "ntwm",
]
