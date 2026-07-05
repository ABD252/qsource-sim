"""qsource — SPDC entanglement-source simulation toolkit.

Built by Abdullah Al-Baqami as an undergraduate research project at
King Abdulaziz University: how do pump-laser imperfections propagate
through SPDC and degrade entanglement quality?
"""

from .pump import GaussianPump
from .crystal import Crystal, GaussianCrystal, SINC_TO_GAUSSIAN_GAMMA
from .jsa import compute_jsa, JSAResult
from .metrics import schmidt_analysis, SchmidtResult

__all__ = [
    "GaussianPump", "Crystal", "GaussianCrystal", "SINC_TO_GAUSSIAN_GAMMA",
    "compute_jsa", "JSAResult",
    "schmidt_analysis", "SchmidtResult",
]
__version__ = "0.1.0"
