"""
Hand-made "pedal" descriptors of a (dry, wet) guitar pair, computed on the GPU
from the waveforms (no preprocessing step).

Frame features  (B, F, 39), F = frames at hop 1024 (23 ms @ 44.1 kHz)
    per signal (13):  rms dB, crest factor dB, spectral centroid, flatness,
                      tilt (dB / octave), 8 octave-band energies (63 Hz .. 8 kHz)
    stacked as [dry 13, wet 13, wet - dry 13]; used as time-varying FiLM and as
    one token stream of the decoder memory.

Global features (clip level), grouped into 5 tokens, each aimed at a pedal type
    tone   wet/dry long-term spectrum ratio in 1/3-octave bands  (amp / EQ curve)
    dyn    fit of wet dB vs dry dB over active frames (slope ~ 1/ratio,
           offset ~ makeup), dynamic ranges, crest                (compressor)
    dist   mean change of flatness / centroid / crest / tilt       (drive, fuzz)
    echo   dry->wet envelope cross-correlation and wet envelope
           autocorrelation over 0..2 s lags                       (delay time / feedback)
    tail   wet energy after the 10 s playing part vs before, per octave,
           decay slope                                            (reverb / delay tail)
All values are on fixed scales (dB / 20, ...), not per-clip normalized, so the
level relation between dry and wet survives.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

N_FFT = 2048
HOP = 1024
EPS = 1e-10
OCT_CENTERS = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
THIRD_OCT = [1000 * 2 ** (i / 3) for i in range(-13, 13)]         # ~50 Hz .. 16 kHz
FRAME_DIM = 39
ECHO_MAX_SEC = 2.0
GLOBAL_GROUPS = ("tone", "dyn", "dist", "echo", "tail")


def global_dims(sr: int) -> dict:
    lags = int(ECHO_MAX_SEC * sr / HOP)
    return {"tone": len(THIRD_OCT), "dyn": 6, "dist": 4,
            "echo": (lags + 1) + lags, "tail": 2 + len(OCT_CENTERS)}


def _db(p):            # power -> dB
    return 10 * torch.log10(p + EPS)


def _band_masks(freqs, centers, half_width_oct):
    lo = torch.tensor(centers, device=freqs.device) * 2 ** -half_width_oct
    hi = torch.tensor(centers, device=freqs.device) * 2 ** half_width_oct
    m = (freqs[None, :] >= lo[:, None]) & (freqs[None, :] < hi[:, None])
    return m.float() / m.float().sum(1, keepdim=True).clamp(min=1)       # (nb, nfreq) mean


class PedalFeatures(torch.nn.Module):
    def __init__(self, sr: int = 44100, play_sec: float = 10.0):
        super().__init__()
        self.sr, self.play_sec = sr, play_sec
        freqs = torch.fft.rfftfreq(N_FFT, 1 / sr)
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)
        self.register_buffer("freqs", freqs, persistent=False)
        self.register_buffer("oct", _band_masks(freqs, OCT_CENTERS, 0.5), persistent=False)
        self.register_buffer("third", _band_masks(freqs, THIRD_OCT, 1 / 6), persistent=False)
        lf = torch.log2(freqs.clamp(min=1.0))
        band = (freqs >= 50) & (freqs <= 16000)
        x = torch.where(band, lf - lf[band].mean(), torch.zeros_like(lf))
        self.register_buffer("tilt_x", x / (x.pow(2).sum()), persistent=False)
        self.register_buffer("tilt_band", band.float(), persistent=False)
        self.lags = int(ECHO_MAX_SEC * sr / HOP)

    # ------------------------------------------------------------- per signal
    def _analyze(self, x):
        """x (B, N) -> frame features (B, F, 13), power spectrum (B, F, K), rms dB (B, F)"""
        spec = torch.stft(x, N_FFT, HOP, window=self.window, center=True, return_complex=True)
        P = spec.abs().pow(2).transpose(1, 2)                            # (B, F, K)
        xf = F.pad(x, (N_FFT // 2, N_FFT // 2)).unfold(-1, N_FFT, HOP)     # (B, F, N_FFT)
        nF = min(P.size(1), xf.size(1))
        P, xf = P[:, :nF], xf[:, :nF]
        ms = xf.pow(2).mean(-1)
        rms_db = _db(ms)
        crest_db = _db(xf.abs().amax(-1).pow(2)) - rms_db
        tot = P.sum(-1) + EPS
        centroid = torch.log2((P * self.freqs).sum(-1) / tot / 1000 + 1e-3)
        flat_db = 10 * (torch.log(P + EPS).mean(-1) - torch.log(P.mean(-1) + EPS)) / math.log(10)
        logp = _db(P)
        tilt = ((logp * self.tilt_band) * self.tilt_x).sum(-1)              # dB / octave
        oct_db = _db(P @ self.oct.T)                                       # (B, F, 8)
        feat = torch.cat([torch.stack([rms_db / 20, crest_db / 20, centroid, flat_db / 10,
                                       tilt / 10], -1), oct_db / 20], -1)
        return feat, P, rms_db

    # ---------------------------------------------------------------- forward
    def forward(self, dry, wet):
        """dry, wet (B, N) float32 -> frames (B, F, 39), globals {group: (B, d)}"""
        dry, wet = dry.float(), wet.float()
        fd, Pd, rd = self._analyze(dry)
        fw, Pw, rw = self._analyze(wet)
        frames = torch.cat([fd, fw, fw - fd], -1)

        B, nF = rd.shape
        play = int(self.play_sec * self.sr / HOP)
        act = (rd > rd.amax(1, keepdim=True) - 40).float()                  # dry frames with playing
        na = act.sum(1).clamp(min=1)
        amean = lambda v: (v * act).sum(1) / na                             # noqa: E731

        # tone: long-term spectrum ratio (1/3 octave)
        ltd, ltw = _db(Pd[:, :play].mean(1) @ self.third.T), _db(Pw[:, :play].mean(1) @ self.third.T)
        tone = (ltw - ltd) / 20

        # dyn: wet dB = a * dry dB + b over active frames
        md, mw = amean(rd), amean(rw)
        cov = amean((rd - md[:, None]) * (rw - mw[:, None]))
        vd = amean((rd - md[:, None]) ** 2).clamp(min=1e-3)
        vw = amean((rw - mw[:, None]) ** 2)
        slope = cov / vd
        dyn = torch.stack([slope, (mw - md) / 20, vd.sqrt() / 20, vw.sqrt() / 20,
                           amean(fd[..., 1]), amean(fw[..., 1])], -1)

        # dist: mean change of flatness / centroid / crest / tilt
        dd = fw - fd
        dist = torch.stack([amean(dd[..., 3]), amean(dd[..., 2]), amean(dd[..., 1]),
                            amean(dd[..., 4])], -1)

        # echo: envelope correlations (linear rms, z-normalized), lags 0..2 s
        def z(e):
            e = e - e.mean(1, keepdim=True)
            return e / (e.pow(2).mean(1, keepdim=True).sqrt() + 1e-8)
        ed, ew = z(10 ** (rd / 20)), z(10 ** (rw / 20))
        L = self.lags
        xc = torch.stack([(ed[:, :nF - k] * ew[:, k:]).mean(1) for k in range(L + 1)], -1)
        ac = torch.stack([(ew[:, :nF - k] * ew[:, k:]).mean(1) for k in range(1, L + 1)], -1)
        echo = torch.cat([xc, ac], -1)

        # tail: after the playing part
        body_w = _db(Pw[:, :play].mean((1, 2)))
        tail_w = _db(Pw[:, play:].mean((1, 2)))
        oct_body = _db(Pw[:, :play].mean(1) @ self.oct.T)
        oct_tail = _db(Pw[:, play:].mean(1) @ self.oct.T)
        t2 = min(nF, play + int(2.0 * self.sr / HOP))
        seg = rw[:, play:t2]
        tt = torch.arange(seg.size(1), device=seg.device, dtype=seg.dtype) * HOP / self.sr
        tt = tt - tt.mean()
        decay = (seg * tt).sum(1) / (tt.pow(2).sum() + 1e-8)                 # dB / s
        tail = torch.cat([((tail_w - body_w) / 20)[:, None], (decay / 60)[:, None],
                          (oct_tail - oct_body) / 20], -1)

        g = {"tone": tone, "dyn": dyn, "dist": dist, "echo": echo, "tail": tail}
        g = {k: torch.nan_to_num(v, 0.0, 5.0, -5.0).clamp(-5, 5) for k, v in g.items()}
        return torch.nan_to_num(frames, 0.0, 5.0, -5.0).clamp(-6, 6), g
