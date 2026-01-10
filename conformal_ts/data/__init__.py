"""Data loaders for benchmark datasets."""

from .m5_real import M5RealDataLoader, M5Config, M5ExperimentRunner
from .gefcom2014 import GEFCom2014Loader, GEFComConfig, GEFComExperimentRunner

# Keep original simulation utilities available
try:
    from .m5_loader import create_m5_simulation_data, M5SpecificationSelector
    from .synthetic import SyntheticDataGenerator
except ImportError:
    pass

__all__ = [
    'M5RealDataLoader',
    'M5Config', 
    'M5ExperimentRunner',
    'GEFCom2014Loader',
    'GEFComConfig',
    'GEFComExperimentRunner',
]