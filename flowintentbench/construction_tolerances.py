"""Explicit reporting-precision helpers for construction-time GT authoring.

These helpers are intentionally outside the runtime evaluator. A constructor
uses them to author a concrete tolerance into each ReferenceFinding after
choosing an appropriate reporting precision for that scientific quantity.
They do not infer a verification rule from a model prediction or Python type.
"""

from __future__ import annotations

import math


def absolute_tolerance_for_significant_figures(
    value: float, *, significant_figures: int
) -> float:
    """Return half a unit in the final displayed significant-figure place.

    For example, reporting ``0.804935...`` as ``0.80494`` (five significant
    figures) permits an absolute rounding error of ``0.000005``. The caller
    must select ``significant_figures`` from the scientific quantity, stored
    data precision, and intended reporting precision before construction.
    """

    if not isinstance(value, float) or not math.isfinite(value) or value == 0.0:
        raise ValueError("value must be a finite non-zero continuous scalar")
    if not isinstance(significant_figures, int) or significant_figures < 1:
        raise ValueError("significant_figures must be a positive integer")
    exponent = math.floor(math.log10(abs(value)))
    return 0.5 * 10.0 ** (exponent - significant_figures + 1)


__all__ = ["absolute_tolerance_for_significant_figures"]
