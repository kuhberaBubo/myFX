"""
Build StemFX with the Rig vocabulary and warm-start from the paper checkpoint.

Loaded as-is (vocab-independent): encoder, mixing-feature FiLM path,
generator transformer, memory projection, memory pos-embedding, value head,
positional-encoding buffer.

Rebuilt (vocab-sized):
    generator.token_embedding.weight, generator.token_head.{weight,bias}
New rows keep FXChainGenerator's random init; rows whose meaning carries over
are copied from the checkpoint:
    [0-7]      [PAD] [BOS] [EOS] [SEP] vocals bass drums other  (same idx)
    VALUE_0..100  <- last 101 rows of the old vocab
The old layout is read off the checkpoint shapes, so the original
FXChainTokenizer / multiafx registry is not needed for the mapping.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

# Make `import stemfx` work from a sibling clone (myFX/stemfx) when the package
# is not pip-installed.
_SIBLING = Path(__file__).resolve().parent.parent / "stemfx"
try:
    import stemfx.model  # noqa: F401
except ImportError:
    if (_SIBLING / "stemfx" / "model.py").is_file():
        sys.path.insert(0, str(_SIBLING))
        for m in [m for m in sys.modules if m == "stemfx" or m.startswith("stemfx.")]:
            del sys.modules[m]       # drop a namespace-package stub, if any
    else:
        raise

from stemfx.model import StemFX as StemFXNet          # noqa: E402
from stemfx.weights import resolve_checkpoint         # noqa: E402

from rig_normalization import NUM_VALUE_BINS          # noqa: E402

VOCAB_KEYS = (
    "generator.token_embedding.weight",
    "generator.token_head.weight",
    "generator.token_head.bias",
)
N_SHARED_PREFIX = 8   # specials + 4 stems keep their StemFX index


def build_model(args: dict, vocab_size: int, pad_idx: int = 0, bos_idx: int = 1,
                dropout: float = 0.1) -> StemFXNet:
    """StemFXNet with the checkpoint's hyper-params and a new vocab size."""
    return StemFXNet(
        encoder_dim=args.get("encoder_dim", 512),
        feature_dim=args.get("feature_dim", 64),
        n_mels=args.get("n_mels", 80),
        n_fft=args.get("n_fft", 2048),
        hop_length=args.get("hop_length", 512),
        sample_rate=args.get("sample_rate", 44100),
        use_film=not args.get("no_film", False),
        encoder_backbone=args.get("encoder_backbone", "cnn") or "cnn",
        vocab_size=vocab_size,
        hidden_dim=args.get("hidden_dim", 512),
        num_layers=args.get("num_layers", 6),
        num_heads=args.get("num_heads", 8),
        feedforward_dim=args.get("feedforward_dim", 2048),
        dropout=dropout,
        max_seq_len=args.get("max_seq_len", 512),
        pad_token_idx=pad_idx,
        bos_token_idx=bos_idx,
        num_memory_tokens=args.get("num_memory_tokens", 4),
    )


def rig_row_map(old_vocab_size: int, new_value_start: int) -> Dict[int, int]:
    """{new_idx: old_idx} for tokens whose meaning carries over."""
    m = {i: i for i in range(N_SHARED_PREFIX)}
    old_value_start = old_vocab_size - NUM_VALUE_BINS
    for b in range(NUM_VALUE_BINS):
        m[new_value_start + b] = old_value_start + b
    return m


def load_pretrained_with_new_vocab(
    tokenizer,
    checkpoint: Optional[str] = None,
    dropout: float = 0.1,
) -> Tuple[StemFXNet, dict]:
    """
    Args:
        tokenizer: RigTokenizer (needs vocab_size, value_token_start, pad/bos idx).
        checkpoint: local StemFX .pt, or None -> download the paper checkpoint (HF Hub).
    Returns:
        (model on CPU, info dict incl. the checkpoint `args` to store on save)
    """
    ckpt = torch.load(resolve_checkpoint(checkpoint), map_location="cpu", weights_only=False)
    args = dict(ckpt.get("args", {}) or {})
    old_sd = ckpt["model_state_dict"]
    old_vocab = old_sd["generator.token_embedding.weight"].shape[0]

    V = tokenizer.vocab_size
    model = build_model(args, V, tokenizer.pad_token_idx, tokenizer.bos_token_idx, dropout)
    new_sd = model.state_dict()

    transfer = {k: v for k, v in old_sd.items() if k not in VOCAB_KEYS}
    bad = [k for k, v in transfer.items() if k in new_sd and new_sd[k].shape != v.shape]
    if bad:
        raise RuntimeError(f"Unexpected non-vocab shape mismatch: {bad}")
    missing, unexpected = model.load_state_dict(transfer, strict=False)
    missing = [k for k in missing if k not in VOCAB_KEYS]
    if missing or unexpected:
        raise RuntimeError(f"State-dict mismatch: missing={missing} unexpected={unexpected}")

    row_map = rig_row_map(old_vocab, tokenizer.value_token_start)
    g = model.generator
    with torch.no_grad():
        for new_i, old_i in row_map.items():
            g.token_embedding.weight[new_i] = old_sd["generator.token_embedding.weight"][old_i]
            g.token_head.weight[new_i] = old_sd["generator.token_head.weight"][old_i]
            g.token_head.bias[new_i] = old_sd["generator.token_head.bias"][old_i]
        g.token_embedding.weight[tokenizer.pad_token_idx].zero_()

    info = {
        "args": args,
        "old_vocab_size": old_vocab,
        "new_vocab_size": V,
        "rows_copied": len(row_map),
        "rows_random": V - len(row_map),
        "tensors_loaded": len(transfer),
    }
    return model, info


def set_encoder_trainable(model: StemFXNet, trainable: bool) -> None:
    """Stage 1: frozen encoder (and FiLM feature path). Stage 2: unfreeze."""
    for p in model.encoder.parameters():
        p.requires_grad = trainable
