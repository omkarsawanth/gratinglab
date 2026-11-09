"""
GratingLab
==========
A Python simulation suite to study laser diffraction patterns from optical gratings
and their utility in advanced material characterization.
"""

from gratinglab import analytical
from gratinglab import characterize
from gratinglab import constants
from gratinglab import gratings
from gratinglab import propagation
from gratinglab import visualize

__version__ = "0.1.0"
__all__ = [
    "analytical",
    "characterize",
    "constants",
    "gratings",
    "propagation",
    "visualize",
]
