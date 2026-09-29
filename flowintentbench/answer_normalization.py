"""Explicit, dimension-preserving unit conversion."""
import math

_UNITS = {}
for dimension, scale, names in [
    ("ratio", 1.0, ("1", "dimensionless", "unitless", "fraction")),
    ("ratio", 0.01, ("%", "percent", "percentage")),
    ("length", 1.0, ("m", "meter", "meters", "metre", "metres")),
    ("length", 0.01, ("cm", "centimeter", "centimeters")),
    ("length", 0.001, ("mm", "millimeter", "millimeters")),
    ("time", 1.0, ("s", "second", "seconds")),
    ("time", 0.001, ("ms", "millisecond", "milliseconds")),
    ("pressure", 1.0, ("Pa", "pascal", "pascals")),
    ("pressure", 1000.0, ("kPa", "kilopascal", "kilopascals")),
    ("pressure", 100000.0, ("bar",)),
    ("speed", 1.0, ("m/s", "m s^-1")),
    ("speed", 0.01, ("cm/s", "cm s^-1")),
    ("speed", 0.001, ("mm/s", "mm s^-1")),
    ("rate", 1.0, ("1/s", "s^-1", "s^{-1}", "s\u207b\u00b9")),
    ("viscosity", 1.0, ("Pa s", "Pa*s", "Pa.s", "Pa\u00b7s")),
    ("viscosity", 0.001, ("mPa s", "mPa*s", "mPa\u00b7s", "cP")),
    ("angle", 1.0, ("rad", "radian", "radians")),
    ("angle", math.pi / 180.0, ("degree", "degrees", "deg", "\u00b0")),
]:
    for name in names:
        _UNITS[name] = (dimension, scale)


def convert_explicit_units(value, source, target):
    """Return a conversion record, or None when representation is not established."""
    if not isinstance(source, str) or not isinstance(target, str):
        return None
    left = _UNITS.get(" ".join(source.split()))
    right = _UNITS.get(" ".join(target.split()))
    if left is None or right is None or left[0] != right[0]:
        return None
    values = value if isinstance(value, list) else [value]
    if not values or any(isinstance(v, bool) or not isinstance(v, (int, float))
                         or not math.isfinite(v) for v in values):
        return None
    factor = left[1] / right[1]
    converted = [v * factor for v in values]
    if not all(math.isfinite(v) for v in converted):
        return None
    return {"rule": "N02", "source_unit": source, "target_unit": target,
            "original_value": value, "factor": factor,
            "converted_value": converted if isinstance(value, list) else converted[0]}

