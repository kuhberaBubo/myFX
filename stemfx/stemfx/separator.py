"""SCNet-based source separator.

Splits a stereo mixture into 4 stems: vocals, bass, drums, other.

The SCNet model source is vendored under ``stemfx/third_party/scnet`` (MIT,
see ``third_party/NOTICE``). The pretrained weights are not redistributed —
they are downloaded from the upstream project's GitHub release on first use
and cached locally.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from stemfx._scnet_infer import (
    Config,
    demix,
    denormalize_audio,
    load_config,
    normalize_audio,
)
from stemfx.weights import default_scnet_config_path, default_scnet_path


class SCNetSeparator:
    """Wrapper around SCNet for in-memory stereo→4-stem separation."""

    STEMS = ("vocals", "bass", "drums", "other")

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "cuda",
        scnet_dir: Optional[str] = None,   # accepted for backwards compatibility
    ):
        """Build a separator.

        Args:
            model_path: SCNet ``.ckpt``. If ``None``, the upstream checkpoint is
                downloaded and cached (~214 MB on first use).
            config_path: SCNet YAML config. Defaults to the bundled config that
                matches the default checkpoint.
            device: Torch device string.
            scnet_dir: Ignored. Retained so existing call sites keep working now
                that the model source is vendored.
        """
        from stemfx.third_party.scnet import SCNet

        if model_path is None:
            model_path = default_scnet_path()
        if config_path is None:
            config_path = default_scnet_config_path()

        config = load_config(config_path)
        model = SCNet(**config.model)

        state = torch.load(model_path, map_location="cpu", weights_only=False)
        # Upstream checkpoints are sometimes wrapped in a training envelope.
        if isinstance(state, dict):
            for key in ("state_dict", "state", "model_state_dict"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        model.load_state_dict(state)

        self.model = model.to(device).eval()
        self.config: Config = config
        self.device = device
        self.sample_rate = config.audio.get("sample_rate", 44100)

    @torch.no_grad()
    def separate(self, audio) -> Dict[str, torch.Tensor]:
        """Separate stereo audio into a dict of 4 stems.

        Args:
            audio: ``torch.Tensor`` or ``np.ndarray`` of shape (2, T) or (T,).

        Returns:
            ``{stem_name: torch.FloatTensor (2, T)}``
        """
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = np.asarray(audio)
        if audio_np.ndim == 1:
            audio_np = np.stack([audio_np, audio_np], axis=0)

        norm_params = None
        if self.config.inference.get("normalize", False):
            audio_np, norm_params = normalize_audio(audio_np)

        waveforms = demix(self.config, self.model, audio_np, self.device)

        if norm_params is not None:
            for stem in waveforms:
                waveforms[stem] = denormalize_audio(waveforms[stem], norm_params)

        return {
            stem: torch.from_numpy(waveforms[stem]).float()
            for stem in self.STEMS
        }
