"""Index-preserving query-side calibration utilities.

The modules in this package intentionally consume arrays from the frozen
embedding cache. They never update document vectors or build a second index.
"""

from .linear import (
    LinearQueryCalibrator,
    fit_corpus_alignment,
    fit_gold_supervised,
)

__all__ = ["LinearQueryCalibrator", "fit_corpus_alignment", "fit_gold_supervised"]
