"""
Value normalization for Guitar Rig parameters.

Every parameter in the dataset already has `raw` = plugin knob position in
[0, 1]. We tokenize raw directly into the 101 VALUE bins StemFX was trained
with (VALUE_0 .. VALUE_100), so no per-parameter range table is needed for
training.

    raw  --round(raw*100)-->  bin  --bin/100-->  raw'  --snap-->  grid index / value

For grid params with n <= 101 (bool, str with few options, small grids) the
round trip is exact after snapping. For n > 101 (1001-step knobs, jump_vac,
rack_input_gain, ...) the bin error is <= 0.005 in raw, plus up to half a grid
step from snapping (max ~0.009 for the n=122 knobs).

Spec dicts come from param_info/<preset>.json -> params[key]; specs for the
same key are identical across presets, so one global table is used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

NUM_VALUE_BINS = 101


def raw_to_bin(raw: float) -> int:
    return int(min(max(round(float(raw) * (NUM_VALUE_BINS - 1)), 0), NUM_VALUE_BINS - 1))


def bin_to_raw(b: int) -> float:
    return float(b) / (NUM_VALUE_BINS - 1)


def snap_raw(spec: Dict[str, Any], raw: float) -> float:
    """Snap a raw value onto the parameter's grid (no-op for curve params)."""
    if spec.get("kind") == "grid" and spec.get("n"):
        n = int(spec["n"])
        idx = int(min(max(round(raw * (n - 1)), 0), n - 1))
        return idx / (n - 1)
    return float(raw)


def raw_to_value(spec: Dict[str, Any], raw: float) -> Any:
    """Human-readable value (bool / option string / number). Curve -> raw."""
    if spec.get("kind") != "grid" or not spec.get("n"):
        return float(raw)
    n = int(spec["n"])
    idx = int(min(max(round(raw * (n - 1)), 0), n - 1))
    if spec.get("values"):
        return spec["values"][idx]
    if spec.get("min") is not None and spec.get("step") is not None:
        v = spec["min"] + idx * spec["step"]
        d = spec.get("decimals")
        return round(v, d) if d is not None else v
    return idx


def read_param_info(param_info_dir: str | Path) -> Dict[str, Any]:
    """{file stem: parsed json} for param_info/*.json (stored in checkpoints)."""
    files = sorted(Path(param_info_dir).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No param_info/*.json in {param_info_dir}")
    return {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in files}


def load_param_specs(param_info: str | Path | Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns:
        {
          "presets": {ps_id: {"chain_order": [...], "keys": [...]}},
          "specs":   {key: spec_dict},      # global, identical across presets
          "raw":     {file stem: json},
        }
    `keys` are the used (use=True) param keys of that preset in file order.
    `param_info` is a param_info/ folder or the dict from read_param_info().
    """
    if not isinstance(param_info, dict):
        param_info = read_param_info(param_info)
    presets, specs = {}, {}
    for name in sorted(param_info):
        d = param_info[name]
        ps = d["preset"]["id"]
        keys = [k for k, v in d["params"].items() if v.get("use", True)]
        presets[ps] = {"chain_order": list(d["chain_order"]), "keys": keys}
        for k, v in d["params"].items():
            specs.setdefault(k, v)
    return {"presets": presets, "specs": specs, "raw": param_info}
