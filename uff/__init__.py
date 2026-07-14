"""QSOL UFF: transparent galaxy and compact-object model laboratory."""

from .compact import (
    compact_object_report,
    kerr_characteristic_radii,
    lqg_area_gap_m2,
    smbh_velocity_kms,
)
from .data import GalaxyData, load_galaxy_csv
from .fitting import FitResult, fit_model, fit_models
from .models import ModelOptions, available_models, build_model
from .sampling import PosteriorResult, sample_posterior

__version__ = "4.0.0"

__all__ = [
    "FitResult",
    "GalaxyData",
    "ModelOptions",
    "PosteriorResult",
    "available_models",
    "build_model",
    "compact_object_report",
    "fit_model",
    "fit_models",
    "kerr_characteristic_radii",
    "load_galaxy_csv",
    "lqg_area_gap_m2",
    "sample_posterior",
    "smbh_velocity_kms",
]
