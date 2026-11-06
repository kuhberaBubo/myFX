"""Minimal SCNet inference helpers.

This is an inference-only extraction of the chunked separation logic from
Music-Source-Separation-Training (MIT, see ``third_party/NOTICE``). The
upstream ``utils`` modules interleave training and inference code, which
drags in ``wandb``, ``matplotlib`` and ``loralib`` at import time; only the
pieces below are needed to run a model forward.

The numerics follow upstream exactly: same windowing, same overlap-add, same
reflect padding.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml


class Config(dict):
    """Attribute-accessible config, standing in for ``ml_collections.ConfigDict``.

    Only the surface used by SCNet inference is supported: attribute access,
    ``in``, and ``get``. Nested mappings are wrapped on construction.
    """

    def __init__(self, mapping: Dict[str, Any] | None = None):
        super().__init__()
        for key, value in (mapping or {}).items():
            self[key] = Config(value) if isinstance(value, dict) else value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:                       # pragma: no cover - defensive
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class _ConfigLoader(yaml.SafeLoader):
    """SafeLoader that also understands the ``!!python/tuple`` tag.

    Upstream configs use that tag for a few training-time augmentation
    settings. Upstream parses them with ``yaml.FullLoader``; sticking to
    SafeLoader plus this one constructor keeps arbitrary-object construction
    off the table for user-supplied configs.
    """


_ConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node)),
)


def load_config(config_path: str) -> Config:
    """Load a Music-Source-Separation-Training YAML config."""
    with open(config_path, "r") as fh:
        return Config(yaml.load(fh, Loader=_ConfigLoader))


def prefer_target_instrument(config: Config) -> List[str]:
    """Target instrument list, or all instruments when none is pinned."""
    if getattr(config.training, "target_instrument", None):
        return [config.training.target_instrument]
    return list(config.training.instruments)


def get_windowing_array(window_size: int, fade_size: int) -> torch.Tensor:
    """Linear fade-in/fade-out window used to cross-fade adjacent chunks."""
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] = fadeout
    window[:fade_size] = fadein
    return window


def normalize_audio(audio: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Normalize by the mean/std of the mono downmix."""
    mono = audio.mean(0)
    mean, std = mono.mean(), mono.std()
    return (audio - mean) / std, {"mean": mean, "std": std}


def denormalize_audio(audio: np.ndarray, norm_params: Dict[str, float]) -> np.ndarray:
    """Invert :func:`normalize_audio`."""
    return audio * norm_params["std"] + norm_params["mean"]


@torch.inference_mode()
def demix(
    config: Config,
    model: torch.nn.Module,
    mix: np.ndarray | torch.Tensor,
    device: torch.device | str,
) -> Dict[str, np.ndarray]:
    """Chunked overlap-add separation.

    Args:
        config: Loaded SCNet config.
        model: SCNet module in eval mode, already on ``device``.
        mix: Stereo mixture, shape ``(channels, time)``.
        device: Torch device to run inference on.

    Returns:
        ``{instrument_name: np.ndarray (channels, time)}``.
    """
    mix = torch.as_tensor(mix, dtype=torch.float32)

    chunk_size = (
        config.inference.chunk_size
        if "chunk_size" in config.inference
        else config.audio.chunk_size
    )
    instruments = prefer_target_instrument(config)
    num_instruments = len(instruments)
    num_overlap = config.inference.num_overlap
    batch_size = config.inference.batch_size

    fade_size = chunk_size // 10
    step = chunk_size // num_overlap
    border = chunk_size - step
    length_init = mix.shape[-1]
    windowing_array = get_windowing_array(chunk_size, fade_size)

    # Reflect-pad the edges so the first/last chunks are not attenuated.
    if length_init > 2 * border and border > 0:
        mix = nn.functional.pad(mix, (border, border), mode="reflect")

    # AMP is a CUDA-only path upstream; on CPU it would be an inert no-op.
    use_amp = getattr(config.training, "use_amp", True)
    dev = torch.device(device)
    amp_ctx = (
        torch.amp.autocast("cuda", enabled=use_amp)
        if dev.type == "cuda"
        else contextlib.nullcontext()
    )

    req_shape = (num_instruments,) + tuple(mix.shape)
    result = torch.zeros(req_shape, dtype=torch.float32)
    counter = torch.zeros(req_shape, dtype=torch.float32)

    with amp_ctx:
        i = 0
        batch_data: List[torch.Tensor] = []
        batch_locations: List[Tuple[int, int]] = []

        while i < mix.shape[1]:
            part = mix[:, i:i + chunk_size].to(dev)
            chunk_len = part.shape[-1]
            pad_mode = "reflect" if chunk_len > chunk_size // 2 else "constant"
            part = nn.functional.pad(
                part, (0, chunk_size - chunk_len), mode=pad_mode, value=0
            )

            batch_data.append(part)
            batch_locations.append((i, chunk_len))
            i += step

            if len(batch_data) >= batch_size or i >= mix.shape[1]:
                x = model(torch.stack(batch_data, dim=0))

                window = windowing_array.clone()
                if i - step == 0:            # first chunk: no fade-in
                    window[:fade_size] = 1
                elif i >= mix.shape[1]:      # last chunk: no fade-out
                    window[-fade_size:] = 1

                for j, (start, seg_len) in enumerate(batch_locations):
                    result[..., start:start + seg_len] += (
                        x[j, ..., :seg_len].float().cpu() * window[..., :seg_len]
                    )
                    counter[..., start:start + seg_len] += window[..., :seg_len]

                batch_data.clear()
                batch_locations.clear()

    estimated_sources = (result / counter).cpu().numpy()
    np.nan_to_num(estimated_sources, copy=False, nan=0.0)

    if length_init > 2 * border and border > 0:
        estimated_sources = estimated_sources[..., border:-border]

    return dict(zip(instruments, estimated_sources))
