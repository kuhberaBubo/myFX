"""
Parameter normalization utilities for multilib FX chain prediction.

Handles normalization/denormalization of multilib FX parameters to [0, 1] range
with support for both linear and log-scale parameters.

Auto-generates PARAM_RANGES from the multiafx effect registry.
"""

import math
from typing import Dict, Tuple, Optional

from multiafx import registry as _registry


def quantize_value(normalized: float, num_bins: int = 101) -> int:
    """Quantize a normalized value in [0, 1] to a discrete bin index."""
    normalized = max(0.0, min(1.0, normalized))
    return max(0, min(num_bins - 1, int(round(normalized * (num_bins - 1)))))


def dequantize_value(bin_idx: int, num_bins: int = 101) -> float:
    """Dequantize a bin index back to a normalized value in [0, 1]."""
    bin_idx = max(0, min(num_bins - 1, bin_idx))
    return bin_idx / (num_bins - 1)


# Auto-generate effect-specific parameter ranges from multiafx registry: (min_val, max_val, log_scale)
PARAM_RANGES: Dict[str, Tuple[float, float, bool]] = {}
for _name in _registry.list_effects():
    _eff = _registry.get(_name)
    for _param_name, _param_range in _eff.param_ranges.items():
        _key = f"{_name}_{_param_name}"
        PARAM_RANGES[_key] = (_param_range.min_val, _param_range.max_val, _param_range.log_scale)


def normalize_param(value: float, param_name: str, effect_type: str) -> float:
    """
    Normalize a parameter value to [0, 1] range.

    Args:
        value: Original parameter value
        param_name: Name of the parameter
        effect_type: Type of effect (e.g., 'am_gain', 'sox_reverb')

    Returns:
        Normalized value in [0, 1]

    Raises:
        ValueError: If parameter is unknown
    """
    key = f"{effect_type}_{param_name}"

    if key not in PARAM_RANGES:
        raise ValueError(f"Unknown parameter: {key} (effect={effect_type}, param={param_name})")

    min_val, max_val, log_scale = PARAM_RANGES[key]

    if log_scale:
        log_value = math.log(max(value, min_val))
        log_min = math.log(min_val)
        log_max = math.log(max_val)
        normalized = (log_value - log_min) / (log_max - log_min)
    else:
        normalized = (value - min_val) / (max_val - min_val)

    return max(0.0, min(1.0, normalized))


def denormalize_param(normalized: float, param_name: str, effect_type: str) -> float:
    """
    Denormalize a parameter value from [0, 1] back to original range.

    Args:
        normalized: Normalized value in [0, 1]
        param_name: Name of the parameter
        effect_type: Type of effect

    Returns:
        Original parameter value

    Raises:
        ValueError: If parameter is unknown
    """
    key = f"{effect_type}_{param_name}"

    if key not in PARAM_RANGES:
        raise ValueError(f"Unknown parameter: {key} (effect={effect_type}, param={param_name})")

    min_val, max_val, log_scale = PARAM_RANGES[key]

    normalized = max(0.0, min(1.0, normalized))

    if log_scale:
        log_min = math.log(min_val)
        log_max = math.log(max_val)
        log_value = log_min + normalized * (log_max - log_min)
        value = math.exp(log_value)
    else:
        value = min_val + normalized * (max_val - min_val)

    return value


def get_all_param_keys() -> list:
    """Get list of all effect-specific parameter keys."""
    return list(PARAM_RANGES.keys())


def is_log_scale(param_name: str, effect_type: str) -> bool:
    """Check if a parameter uses log-scale normalization."""
    key = f"{effect_type}_{param_name}"
    if key not in PARAM_RANGES:
        return False
    return PARAM_RANGES[key][2]


def get_param_range(param_name: str, effect_type: str) -> Optional[Tuple[float, float]]:
    """
    Get the min/max range for a parameter.

    Args:
        param_name: Name of the parameter
        effect_type: Type of effect

    Returns:
        Tuple of (min_val, max_val) or None if parameter unknown
    """
    key = f"{effect_type}_{param_name}"
    if key not in PARAM_RANGES:
        return None
    min_val, max_val, _ = PARAM_RANGES[key]
    return (min_val, max_val)
