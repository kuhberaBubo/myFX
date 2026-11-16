"""
Swappable encoders for testFX. Each one maps a (dry, wet) pair of mono 44.1 kHz
waveforms (B, N) to a decoder memory (B, M, hidden) of M tokens, so the decoder
(StemFX's FXChainGenerator, see tfx_model.py) stays the same.

GuitarEncoder ("g")
    spectral branch : log-mel [dry, wet, wet - dry] (fixed dB scale, no per-clip
                      norm -> level survives) -> 11 overlapping mel bands
                      -> shared band projection -> axial transformer
                      (time-axis / band-axis attention, StemFX blocks), with a
                      time-varying FiLM from the frame-level pedal features
    memory          : [5 global tokens] + [frame-feature tokens over time]
                      + [band x time tokens]   (time is pooled, not removed)

MERTEncoder ("mert")
    m-a-p/MERT-v1-95M on 5 s chunks at 24 kHz, weighted sum of the hidden
    layers, only the top `trainable_layers` transformer layers are trained.
    memory          : [dry, wet, wet - dry] token streams over time
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from stemfx.encoder import AxialTransformerBlock, BandFeatureProjection
from tfx_features import FRAME_DIM, GLOBAL_GROUPS, HOP, N_FFT, PedalFeatures, global_dims


def sinusoid(pos: torch.Tensor, dim: int) -> torch.Tensor:
    """pos (T,) in seconds -> (T, dim)."""
    i = torch.arange(dim // 2, device=pos.device, dtype=torch.float32)
    freq = torch.exp(-math.log(10000.0) * 2 * i / dim) * 50.0        # 50x: sub-second resolution
    ang = pos[:, None].float() * freq[None, :]
    return torch.cat([ang.sin(), ang.cos()], -1)


def pool_time(x: torch.Tensor, k: int, dim: int) -> torch.Tensor:
    """Average-pool dimension `dim` by factor k (floor)."""
    if k <= 1:
        return x
    x = x.movedim(dim, -1)
    n = x.size(-1) // k
    x = x[..., :n * k].reshape(*x.shape[:-1], n, k).mean(-1)
    return x.movedim(-1, dim)


class SDPAAxialBlock(AxialTransformerBlock):
    """StemFX axial block with memory-efficient attention (same weights/math)
    and a time-varying FiLM (gamma/beta per frame instead of per clip)."""

    def _attention(self, x, q_proj, k_proj, v_proj, out_proj, attn_dropout, rope):
        N, S, D = x.shape
        H, HD = self.num_heads, self.head_dim
        q = rope.apply_rotary(q_proj(x).view(N, S, H, HD)).transpose(1, 2)
        k = rope.apply_rotary(k_proj(x).view(N, S, H, HD)).transpose(1, 2)
        v = v_proj(x).view(N, S, H, HD).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=attn_dropout.p if self.training else 0.0)
        return out_proj(o.transpose(1, 2).reshape(N, S, D))

    def forward(self, x, gamma_t=None, beta_t=None):
        x = super().forward(x)                                  # (B, K, T, D)
        if gamma_t is not None:                                 # (B, T, D)
            x = x * (1 + gamma_t[:, None]) + beta_t[:, None]
        return x


class _MemoryMixin:
    """Common helpers: type / time embeddings for the decoder memory."""

    def _time_emb(self, n: int, sec_per_step: float, device) -> torch.Tensor:
        return self.time_proj(sinusoid(torch.arange(n, device=device) * sec_per_step,
                                       self.hidden))


class GuitarEncoder(nn.Module, _MemoryMixin):
    def __init__(self, hidden: int = 512, d_model: int = 192, n_blocks: int = 4,
                 n_heads: int = 6, ffn: int = 768, dropout: float = 0.1, sr: int = 44100,
                 n_mels: int = 128, split: int = 20, overlap: int = 10,
                 band_time_pool: int = 2, memory_time_pool: int = 4, play_sec: float = 10.0):
        super().__init__()
        import torchaudio
        self.hidden, self.d = hidden, d_model
        self.split, self.overlap = split, overlap
        self.n_bands = (n_mels - split) // overlap + 1
        self.tp1, self.tp2 = band_time_pool, memory_time_pool
        self.sec_per_tok = HOP / sr * band_time_pool * memory_time_pool
        self.mel = torchaudio.transforms.MelSpectrogram(sr, n_fft=N_FFT, hop_length=HOP,
                                                        n_mels=n_mels, power=2.0)
        self.feats = PedalFeatures(sr, play_sec)

        self.band_proj = BandFeatureProjection(3, d_model, split)
        self.band_emb = nn.Parameter(torch.randn(self.n_bands, d_model) * 0.02)
        self.blocks = nn.ModuleList(SDPAAxialBlock(d_model, n_heads, ffn, dropout)
                                    for _ in range(n_blocks))
        # time-varying FiLM from frame features: conv over time -> gamma/beta per block
        self.film = nn.Sequential(nn.Conv1d(FRAME_DIM, 128, 5, padding=2), nn.GELU(),
                                  nn.Conv1d(128, 128, 5, padding=2), nn.GELU(),
                                  nn.Conv1d(128, n_blocks * 2 * d_model, 1))
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)                      # starts as identity

        # memory
        self.band_out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden))
        self.band_type = nn.Parameter(torch.randn(self.n_bands, hidden) * 0.02)
        self.frame_out = nn.Sequential(nn.Linear(FRAME_DIM, hidden), nn.GELU(),
                                       nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        gd = global_dims(sr)
        self.global_out = nn.ModuleDict({g: nn.Sequential(nn.Linear(gd[g], hidden), nn.GELU(),
                                                          nn.Linear(hidden, hidden),
                                                          nn.LayerNorm(hidden))
                                         for g in GLOBAL_GROUPS})
        self.type_emb = nn.Parameter(torch.randn(2 + len(GLOBAL_GROUPS), hidden) * 0.02)
        self.time_proj = nn.Linear(hidden, hidden)

    def pretrained_parameters(self):
        return []                                               # trained from scratch

    def forward(self, dry, wet):
        with torch.autocast("cuda", enabled=False):
            frames, glob = self.feats(dry, wet)                 # (B, F, 39), fp32
            md = 10 * torch.log10(self.mel(dry.float()) + 1e-10)
            mw = 10 * torch.log10(self.mel(wet.float()) + 1e-10)
        x = torch.stack([(md + 40) / 20, (mw + 40) / 20, (mw - md) / 20], 1)   # (B, 3, mels, F)
        nF = min(x.size(-1), frames.size(1))
        x, frames = x[..., :nF], frames[:, :nF]

        # band x time grid
        bands = [self.band_proj(x[:, :, i * self.overlap:i * self.overlap + self.split])
                 for i in range(self.n_bands)]                  # each (B, D, F)
        h = torch.stack(bands, 1).transpose(2, 3)               # (B, K, F, D)
        h = pool_time(h, self.tp1, 2) + self.band_emb[None, :, None]
        T = h.size(2)
        fr = pool_time(frames, self.tp1, 1)[:, :T]              # (B, T, 39)
        film = self.film(fr.transpose(1, 2)).transpose(1, 2)   # (B, T, n_blocks*2*D)
        film = film.view(*film.shape[:2], len(self.blocks), 2, self.d)
        for j, blk in enumerate(self.blocks):
            h = blk(h, film[:, :, j, 0], film[:, :, j, 1])

        # memory tokens
        B = h.size(0)
        hb = pool_time(h, self.tp2, 2)                          # (B, K, T', D)
        T2 = hb.size(2)
        te = self._time_emb(T2, self.sec_per_tok, h.device)     # (T', H)
        band_tok = self.band_out(hb) + self.band_type[None, :, None] + te[None, None] \
            + self.type_emb[0]
        frame_tok = self.frame_out(pool_time(fr, self.tp2, 1)[:, :T2]) + te[None] + self.type_emb[1]
        glob_tok = torch.stack([self.global_out[g](glob[g].to(h.dtype)) + self.type_emb[2 + i]
                                for i, g in enumerate(GLOBAL_GROUPS)], 1)
        return torch.cat([glob_tok, frame_tok, band_tok.reshape(B, -1, self.hidden)], 1)


class MERTEncoder(nn.Module, _MemoryMixin):
    MODEL = "m-a-p/MERT-v1-95M"

    def __init__(self, hidden: int = 512, trainable_layers: int = 4, chunk_sec: float = 5.0,
                 memory_time_pool: int = 15, sr: int = 44100, model_name: str | None = None):
        super().__init__()
        from transformers import AutoModel
        self.hidden, self.sr, self.chunk_sec, self.pool = hidden, sr, chunk_sec, memory_time_pool
        self.mert = AutoModel.from_pretrained(model_name or self.MODEL, trust_remote_code=True)
        self.mert_sr = 24000
        n_layers = self.mert.config.num_hidden_layers
        dim = self.mert.config.hidden_size
        for p in self.mert.parameters():
            p.requires_grad = False
        self.trainable = list(self.mert.encoder.layers[n_layers - trainable_layers:]) \
            if trainable_layers > 0 else []
        for layer in self.trainable:
            for p in layer.parameters():
                p.requires_grad = True
        self.first_trainable = n_layers - trainable_layers
        self.layer_w = nn.Parameter(torch.zeros(n_layers + 1))
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden))
        self.type_emb = nn.Parameter(torch.randn(3, hidden) * 0.02)
        self.time_proj = nn.Linear(hidden, hidden)
        self.sec_per_tok = memory_time_pool / 75.0              # MERT frames at 75 Hz

    def pretrained_parameters(self):
        return [p for layer in self.trainable for p in layer.parameters()]

    def train(self, mode: bool = True):
        super().train(mode)
        self.mert.eval()                                        # frozen part: no dropout drift
        for layer in self.trainable:
            layer.train(mode)
        return self

    def _hidden_states(self, chunks):
        """Input to the first transformer layer + every layer's output (L+1 tensors).
        MERT's remote code returns hidden_states=None under transformers 5.x, so the
        layers are tapped with hooks when the output does not carry them."""
        layers, taps = self.mert.encoder.layers, []

        def pre(_m, args, kwargs):
            taps.append(args[0] if args else kwargs["hidden_states"])

        def post(_m, _a, o):
            taps.append(o[0] if isinstance(o, (tuple, list)) else o)

        hooks = [layers[0].register_forward_pre_hook(pre, with_kwargs=True)]
        hooks += [layer.register_forward_hook(post) for layer in layers]
        try:
            out = self.mert(chunks, output_hidden_states=True)
        finally:
            for h in hooks:
                h.remove()
        hs = getattr(out, "hidden_states", None)
        if hs is not None:
            return list(hs)
        if len(taps) != len(layers) + 1:
            raise RuntimeError(f"MERT hooks caught {len(taps)} states, expected {len(layers) + 1}")
        return taps

    def _embed(self, x):
        """x (B, N) at 24 kHz, already scaled -> (B, frames, dim)"""
        B, N = x.shape
        c = int(self.chunk_sec * self.mert_sr)
        n = max(1, N // c)
        chunks = x[:, :n * c].reshape(B * n, c)
        hs = torch.stack(self._hidden_states(chunks), 0)         # (L+1, B*n, T, dim)
        w = torch.softmax(self.layer_w, 0).to(hs.dtype)
        h = (w[:, None, None, None] * hs).sum(0)
        return h.reshape(B, -1, h.size(-1))

    def forward(self, dry, wet):
        import torchaudio.functional as AF
        with torch.autocast("cuda", enabled=False):
            d = AF.resample(dry.float(), self.sr, self.mert_sr)
            w = AF.resample(wet.float(), self.sr, self.mert_sr)
            mu = d.mean(1, keepdim=True)
            sd = d.std(1, keepdim=True) + 1e-7                  # dry stats for both ->
            d, w = (d - mu) / sd, (w - mu) / sd                 # wet/dry level ratio kept
        hd = pool_time(self._embed(d), self.pool, 1)
        hw = pool_time(self._embed(w), self.pool, 1)
        T = min(hd.size(1), hw.size(1))
        hd, hw = hd[:, :T], hw[:, :T]
        te = self._time_emb(T, self.sec_per_tok, hd.device)[None]
        toks = [self.proj(hd) + te + self.type_emb[0],
                self.proj(hw) + te + self.type_emb[1],
                self.proj(hw - hd) + te + self.type_emb[2]]
        return torch.cat(toks, 1)


def build_encoder(cfg: dict, hidden: int) -> nn.Module:
    kind = cfg["kind"]
    kw = {k: v for k, v in cfg.items() if k != "kind"}
    if kind == "g":
        return GuitarEncoder(hidden=hidden, **kw)
    if kind == "mert":
        return MERTEncoder(hidden=hidden, **kw)
    raise ValueError(f"unknown encoder kind {kind!r}")
