"""High-level public API for StemFX.

Two user-facing operations:

* :py:meth:`StemFX.embed` — produce an L2-normalized 512-d mixing-style
  embedding from a mixture wav (auto-separated) or a stems dict.
* :py:meth:`StemFX.transfer` / :py:meth:`StemFX.transfer_audio` —
  predict a per-stem FX chain (multiafx-format) from a pair of embeddings,
  optionally rendering audio with the predicted chain.

Both run the default StemFX paper checkpoint loaded via :py:func:`load`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from stemfx.model import StemFX as _StemFXNet
from stemfx.tokenizer import FXChainTokenizer
from stemfx.normalization import quantize_value
from stemfx.weights import resolve_checkpoint


SAMPLE_RATE = 44100
SEGMENT_SECONDS = 10.0
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_SECONDS)
STEM_NAMES = ("vocals", "bass", "drums", "other")

# Type alias: anything we can embed.
EmbedSource = Union[str, os.PathLike, Dict[str, torch.Tensor]]


# --------------------------------------------------------------------------- #
#  TransferResult
# --------------------------------------------------------------------------- #
@dataclass
class TransferResult:
    """Bundle of audio + chain returned by :py:meth:`StemFX.transfer_audio`."""
    audio: Dict[str, torch.Tensor]                    # processed stems (2, T) at 44.1 kHz
    chain: Dict[str, List[Dict]]                      # multiafx-format per-stem chain
    mix: torch.Tensor                                 # downmix (2, T)
    sample_rate: int = SAMPLE_RATE

    def pretty(self) -> str:
        """One-line-per-effect, human-readable summary."""
        lines: List[str] = []
        for stem in STEM_NAMES:
            steps = self.chain.get(stem, [])
            if not steps:
                lines.append(f"  {stem}: (no FX)")
                continue
            chunks = []
            for step in steps:
                eff = step["effect"]
                params = ", ".join(
                    f"{k}={_fmt(v)}" for k, v in step.get("params", {}).items()
                )
                chunks.append(f"{eff}({params})" if params else eff)
            lines.append(f"  {stem}: " + " → ".join(chunks))
        return "\n".join(lines)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


# --------------------------------------------------------------------------- #
#  StemFX
# --------------------------------------------------------------------------- #
class StemFX:
    """High-level inference wrapper for the StemFX paper model.

    Construct via :py:func:`stemfx.load`.
    """

    def __init__(
        self,
        net: _StemFXNet,
        tokenizer: FXChainTokenizer,
        device: torch.device,
        separator=None,                # lazy: only built if a mixture path is embedded
    ):
        self._net = net.eval()
        self._tokenizer = tokenizer
        self._device = device
        self._separator = separator
        self._scnet_paths: Optional[Dict[str, str]] = None  # set by load()

    # ------------------------------------------------------------------ embed
    def embed(self, source: EmbedSource | Sequence[EmbedSource]) -> torch.Tensor:
        """Compute the L2-normalized mixing-style embedding.

        Args:
            source: One of
              - path to a mixture wav (str / Path) — separated internally;
              - dict with keys ``vocals``, ``bass``, ``drums``, ``other``
                (each a (2, T) float32 tensor at 44.1 kHz);
              - a list of the above for batched embedding.

        Returns:
            ``torch.Tensor`` of shape ``(D,)`` for a single source, or
            ``(B, D)`` for a list.
        """
        single = not isinstance(source, list)
        items = [source] if single else list(source)

        stems_list = [self._to_stems(s) for s in items]
        stems_batch = {
            k: torch.stack([d[k] for d in stems_list], dim=0).to(self._device)
            for k in STEM_NAMES
        }

        with torch.no_grad():
            emb = self._net._encode_stems(stems_batch)
            emb = F.normalize(emb, p=2, dim=-1)
        return emb[0] if single else emb

    # ---------------------------------------------------------------- transfer
    def transfer(
        self,
        orig_emb: torch.Tensor,
        target_emb: torch.Tensor,
        max_length: int = 512,
    ) -> Dict[str, List[Dict]]:
        """Predict a per-stem FX chain from a pair of embeddings.

        Both inputs must be L2-normalized (as produced by :py:meth:`embed`).
        Single embeddings are accepted; shape (D,) is auto-batched.

        Returns a dict matching the :py:class:`multiafx.FXChain` input format::

            {
                "vocals": [{"effect": "...", "params": {...}}, ...],
                "bass":   [...],
                "drums":  [...],
                "other":  [...],
            }
        """
        if orig_emb.dim() == 1:
            orig_emb = orig_emb.unsqueeze(0)
        if target_emb.dim() == 1:
            target_emb = target_emb.unsqueeze(0)
        assert orig_emb.shape == target_emb.shape, "embedding shapes must match"
        if orig_emb.shape[0] != 1:
            raise NotImplementedError("transfer() currently supports batch size 1")

        return self._predict_chain(
            orig_emb.to(self._device),
            target_emb.to(self._device),
            max_length=max_length,
        )

    # ----------------------------------------------------------- transfer_audio
    def transfer_audio(
        self,
        orig: EmbedSource,
        target: EmbedSource,
        *,
        normalize_loudness: bool = True,
        target_lufs: float = -23.0,
        max_length: int = 512,
    ) -> TransferResult:
        """End-to-end: embed both inputs, predict chain, render audio.

        The chain is applied to the **original** stems (separated internally
        from a mixture path if needed). When ``normalize_loudness`` is true
        each output stem and the downmix are normalized to ``target_lufs``.
        """
        orig_stems = self._to_stems(orig)
        target_stems = self._to_stems(target)

        with torch.no_grad():
            orig_batch = {k: v.unsqueeze(0).to(self._device) for k, v in orig_stems.items()}
            tgt_batch = {k: v.unsqueeze(0).to(self._device) for k, v in target_stems.items()}
            emb_orig = F.normalize(self._net._encode_stems(orig_batch), p=2, dim=-1)
            emb_tgt = F.normalize(self._net._encode_stems(tgt_batch), p=2, dim=-1)

        chain = self._predict_chain(emb_orig, emb_tgt, max_length=max_length)

        # Apply chain to full-length original stems via multiafx.
        from multiafx import FXChain
        processed: Dict[str, torch.Tensor] = {}
        for stem in STEM_NAMES:
            audio_np = orig_stems[stem].cpu().numpy().astype(np.float32)
            steps = chain.get(stem, [])
            if steps:
                fx = FXChain(steps)
                audio_np = fx(audio_np, SAMPLE_RATE)
            processed[stem] = torch.from_numpy(audio_np)

        if normalize_loudness:
            processed = {k: _loudness_normalize(v, target_lufs) for k, v in processed.items()}

        mix = sum(processed.values())
        peak = mix.abs().max()
        if peak > 0.95:
            mix = mix * (0.95 / peak)
        if normalize_loudness:
            mix = _loudness_normalize(mix, target_lufs)

        return TransferResult(audio=processed, chain=chain, mix=mix, sample_rate=SAMPLE_RATE)

    # ============================================================= internals
    def _to_stems(self, source: EmbedSource) -> Dict[str, torch.Tensor]:
        """Resolve a source to a stems dict cropped to 10 s @ 44.1 kHz."""
        if isinstance(source, dict):
            stems = {}
            for k in STEM_NAMES:
                if k not in source:
                    raise KeyError(f"stems dict missing required key {k!r}")
                stems[k] = _ensure_segment(source[k])
            return stems

        path = Path(source)
        audio = _load_wav(path)
        sep_stems = self._get_separator().separate(audio)
        return {k: _ensure_segment(v) for k, v in sep_stems.items()}

    def _get_separator(self):
        """Lazily build the SCNet separator.

        With no explicit paths, the upstream SCNet checkpoint is downloaded and
        cached on first use (~214 MB) so that ``embed("mix.wav")`` works from a
        plain ``pip install stemfx``.
        """
        if self._separator is not None:
            return self._separator

        from stemfx.separator import SCNetSeparator
        paths = self._scnet_paths or {}
        self._separator = SCNetSeparator(
            model_path=paths.get("model"),
            config_path=paths.get("config"),
            device=str(self._device),
        )
        return self._separator

    @torch.no_grad()
    def _predict_chain(
        self,
        emb_orig: torch.Tensor,        # (1, D), L2-normalized
        emb_tgt: torch.Tensor,         # (1, D), L2-normalized
        max_length: int,
    ) -> Dict[str, List[Dict]]:
        gen = self._net.generator
        encoder_dim = gen.encoder_dim
        embedding_pairs = torch.cat([emb_orig, emb_tgt], dim=1)

        emb_o = embedding_pairs[:, :encoder_dim]
        emb_a = embedding_pairs[:, encoder_dim:]
        memory = torch.cat(
            [gen.embedding_proj(emb_o).unsqueeze(1), gen.embedding_proj(emb_a).unsqueeze(1)],
            dim=1,
        )
        if hasattr(gen, "memory_pos_embedding"):
            memory = memory + gen.memory_pos_embedding.unsqueeze(0)

        bos = self._tokenizer.bos_token_idx
        eos = self._tokenizer.eos_token_idx
        tokens: List[int] = [bos]
        values: List[float] = [0.0]

        for _ in range(max_length):
            toks = torch.tensor([tokens], dtype=torch.long, device=self._device)
            tok_emb = gen.pos_encoding(gen.token_embedding(toks))
            mask = gen._generate_square_subsequent_mask(toks.size(1)).to(self._device)
            dec_out = gen.transformer_decoder(tgt=tok_emb, memory=memory, tgt_mask=mask)
            logits = gen.token_head(dec_out[:, -1, :])
            next_tok = int(logits.argmax(dim=-1).item())

            if next_tok == eos:
                tokens.append(next_tok); values.append(0.0); break

            if self._tokenizer.is_value_token(next_tok):
                bin_idx = next_tok - self._tokenizer.value_token_start
                tokens.append(next_tok)
                values.append(bin_idx / 100.0)
            else:
                tokens.append(next_tok)
                values.append(0.0)

        meta = self._tokenizer.decode_fx_chain(tokens, values)
        out: Dict[str, List[Dict]] = {s: [] for s in STEM_NAMES}
        for stem, data in (meta.get("stems") or {}).items():
            if stem in out:
                out[stem] = list(data.get("fx_chain", []))
        return out


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _load_wav(path: Path) -> torch.Tensor:
    # soundfile rather than torchaudio.load: torchaudio >= 2.9 delegates
    # decoding to torchcodec, which is not a dependency of StemFX.
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.T.copy())          # (frames, ch) -> (ch, frames)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    return audio.float()


def _ensure_segment(audio: torch.Tensor) -> torch.Tensor:
    """Crop/pad to exactly 10 s, ensure stereo float32."""
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).repeat(2, 1)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    audio = audio.float()
    T = audio.shape[1]
    if T >= SEGMENT_SAMPLES:
        return audio[:, :SEGMENT_SAMPLES]
    return torch.nn.functional.pad(audio, (0, SEGMENT_SAMPLES - T))


def _loudness_normalize(audio: torch.Tensor, target_lufs: float = -23.0) -> torch.Tensor:
    import pyloudnorm as pyln
    meter = pyln.Meter(SAMPLE_RATE)
    audio_np = audio.cpu().numpy().astype(np.float32)
    integrated = meter.integrated_loudness(audio_np.T)
    if not np.isfinite(integrated) or integrated < -70:
        return audio
    out = pyln.normalize.loudness(audio_np.T, integrated, target_lufs).T
    return torch.from_numpy(out.astype(np.float32))
