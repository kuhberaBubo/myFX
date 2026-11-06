"""StemFX: Mixing style representation learning via autoregressive FX chain prediction.

Public API::

    import stemfx

    model = stemfx.load()                              # downloads default checkpoint
    emb   = model.embed("song.wav")                    # L2-normalized 512-d embedding
    chain = model.transfer(orig_emb, target_emb)       # multiafx-format per-stem chain
    res   = model.transfer_audio("a.wav", "b.wav")     # bundled audio + chain
"""

from typing import Optional as _Optional

import torch as _torch

from stemfx.api import StemFX, TransferResult
from stemfx.encoder import BSFiLMEncoder
from stemfx.generator import FXChainGenerator
from stemfx.losses import FXChainLoss, MultiResolutionSTFTLoss
from stemfx.model import StemFX as StemFXNet
from stemfx.tokenizer import FXChainTokenizer
from stemfx.weights import (
    resolve_checkpoint,
    default_checkpoint_path,
    default_scnet_path,
)

__version__ = "0.2.0"

__all__ = [
    "load",
    "StemFX",
    "TransferResult",
    "BSFiLMEncoder",
    "FXChainGenerator",
    "FXChainLoss",
    "FXChainTokenizer",
    "MultiResolutionSTFTLoss",
    "StemFXNet",
    "default_checkpoint_path",
    "default_scnet_path",
    "__version__",
]


def load(
    checkpoint: _Optional[str] = None,
    device: _Optional[str] = None,
    scnet_model: _Optional[str] = None,
    scnet_config: _Optional[str] = None,
    scnet_dir: _Optional[str] = None,
) -> StemFX:
    """Load the default paper StemFX model.

    Args:
        checkpoint: Local path to a StemFX ``.pt`` checkpoint.
            If ``None``, downloads the default paper checkpoint from
            HuggingFace Hub on first call (cached afterwards).
        device: Torch device string (``"cuda"``, ``"cuda:0"``, ``"cpu"``, …).
            Defaults to ``"cuda"`` if available else ``"cpu"``.
        scnet_model: Override the SCNet separator checkpoint. Optional — when
            omitted, the upstream checkpoint is downloaded and cached on first
            use, so ``model.embed(<wav path>)`` works out of the box.
        scnet_config: Override the SCNet YAML config. Defaults to the config
            bundled with StemFX, which matches the default checkpoint.
        scnet_dir: Deprecated and ignored; the SCNet source is now vendored.

    Returns:
        :class:`StemFX` ready for inference.
    """
    ckpt_path = resolve_checkpoint(checkpoint)
    if device is None:
        device = "cuda" if _torch.cuda.is_available() else "cpu"
    dev = _torch.device(device)

    ckpt = _torch.load(ckpt_path, map_location=dev)
    args = ckpt.get("args", {}) or {}

    tokenizer = FXChainTokenizer()

    net = StemFXNet(
        encoder_dim=args.get("encoder_dim", 512),
        feature_dim=args.get("feature_dim", 64),
        n_mels=args.get("n_mels", 80),
        n_fft=args.get("n_fft", 2048),
        hop_length=args.get("hop_length", 512),
        sample_rate=args.get("sample_rate", 44100),
        use_film=not args.get("no_film", False),
        encoder_backbone=args.get("encoder_backbone", "cnn") or "cnn",
        vocab_size=tokenizer.vocab_size,
        hidden_dim=args.get("hidden_dim", 512),
        num_layers=args.get("num_layers", 6),
        num_heads=args.get("num_heads", 8),
        feedforward_dim=args.get("feedforward_dim", 2048),
        dropout=0.0,
        max_seq_len=args.get("max_seq_len", 512),
        pad_token_idx=tokenizer.pad_token_idx,
        bos_token_idx=tokenizer.bos_token_idx,
        num_memory_tokens=args.get("num_memory_tokens", 4),
    ).to(dev)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    api = StemFX(net=net, tokenizer=tokenizer, device=dev, separator=None)

    if scnet_model or scnet_config:
        api._scnet_paths = {
            "model": scnet_model,
            "config": scnet_config,
        }

    return api
