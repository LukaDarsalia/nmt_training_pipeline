"""
Data Splitting Module

A flexible, YAML-configured data splitting system for creating train/valid/test sets
from various pipeline artifacts with contamination checking.
"""

from .contamination import ContaminationDetector
from .splitter import DataSplitter

__all__ = [
    'ContaminationDetector',
    'DataSplitter'
]
