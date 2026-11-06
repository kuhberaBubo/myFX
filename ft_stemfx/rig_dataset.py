"""
Guitar Rig pair dataset -> StemFX batch format.

Per sample
    input  (DI, -32 LUFS)    -> original_stems["other"]
    target (rendered chain)  -> augmented_stems["other"]
    vocals / bass / drums    -> quiet dummy noise, IDENTICAL in original and
                                augmented (so the label has no chain for them)

Audio: mono 15 s FLAC -> first `clip_duration` s (default 10 s, the length the
StemFX encoder was trained on; delay/reverb tails after 10 s are dropped)
-> duplicated to stereo (2, T). Levels are NOT normalized: the target level
carries information about the chain (gain, volume knobs).

Dummy stems must not be digital silence: stemfx.mixing_features computes
log10(peak / rms) and silence gives -inf -> NaN in the FiLM input.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from rig_tokenizer import STEMS, GUITAR_STEM, RigTokenizer

DUMMY_STEMS = [s for s in STEMS if s != GUITAR_STEM]


def read_jsonl(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise, unit RMS."""
    spec = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.arange(n // 2 + 1, dtype=np.float64)
    f[0] = 1.0
    x = np.fft.irfft(spec / np.sqrt(f), n=n)
    return (x / (np.sqrt(np.mean(x ** 2)) + 1e-12)).astype(np.float32)


def make_dummy_bank(length: int, level_db: float = -60.0, size: int = 16,
                    seed: int = 0) -> np.ndarray:
    """(size, length) pink-noise buffers at `level_db` RMS. Same args -> same bank."""
    rng = np.random.default_rng(seed)
    gain = 10 ** (level_db / 20)
    return np.stack([_pink_noise(length, rng) * gain for _ in range(size)])


def build_stems(di: torch.Tensor, wet: torch.Tensor, bank: np.ndarray, key: int):
    """(2,T) DI / wet -> (original_stems, augmented_stems) dicts in StemFX order.
    Dummy stems are identical in both, picked from `bank` by `key`."""
    orig, aug = {}, {}
    for j, s in enumerate(DUMMY_STEMS):
        d = torch.from_numpy(bank[(key + 7 * j) % len(bank)]).unsqueeze(0).expand(2, -1).contiguous()
        orig[s] = d
        aug[s] = d                          # identical -> "no FX" on dummy stems
    orig[GUITAR_STEM] = di
    aug[GUITAR_STEM] = wet
    return {s: orig[s] for s in STEMS}, {s: aug[s] for s in STEMS}


class RigPairDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split_file: str,
        tokenizer: RigTokenizer,
        sample_rate: int = 44100,
        clip_start_sec: float = 0.0,
        clip_duration: float = 10.0,
        dummy_level_db: float = -60.0,
        dummy_bank_size: int = 16,
        max_seq_len: int = 512,
        max_samples: Optional[int] = None,
        skip_missing_audio: bool = False,
        seed: int = 0,
    ):
        self.root = Path(root)
        self.tok = tokenizer
        self.sr = sample_rate
        self.start = int(round(clip_start_sec * sample_rate))
        self.length = int(round(clip_duration * sample_rate))
        self.max_seq_len = max_seq_len

        rows = read_jsonl(self.root / split_file)
        if skip_missing_audio:   # local smoke tests with a partial copy of the data
            rows = [r for r in rows
                    if (self.root / r["input"]).is_file() and (self.root / r["target"]).is_file()]
        if max_samples:
            rows = rows[:max_samples]
        if not rows:
            raise RuntimeError(f"No usable samples in {self.root / split_file}")
        self.rows = rows

        # Bank of dummy noise buffers (one per stem slot), picked per sample
        # deterministically from the sample id -> same dummy every epoch,
        # so encoder outputs are cacheable later.
        self.bank = make_dummy_bank(self.length, dummy_level_db, dummy_bank_size, seed)

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, rel: str) -> torch.Tensor:
        x, sr = sf.read(self.root / rel, dtype="float32", frames=self.start + self.length)
        if sr != self.sr:
            raise ValueError(f"{rel}: sample rate {sr} != {self.sr}")
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = x[self.start:self.start + self.length]
        if len(x) < self.length:
            x = np.pad(x, (0, self.length - len(x)))
        x = torch.from_numpy(np.ascontiguousarray(x))
        return x.unsqueeze(0).expand(2, -1).contiguous()     # (2, T)

    def __getitem__(self, i: int) -> Dict:
        row = self.rows[i]
        di = self._load(row["input"])
        wet = self._load(row["target"])

        h = zlib.crc32(row["id"].encode())   # stable across processes (hash() is salted)
        orig, aug = build_stems(di, wet, self.bank, h)

        tokens, values, vmask = self.tok.encode(row)
        L = self.max_seq_len
        return {
            "original_stems": orig,
            "augmented_stems": aug,
            "tokens": torch.tensor(tokens[:L], dtype=torch.long),
            "values": torch.tensor(values[:L], dtype=torch.float32),
            "value_mask": torch.tensor(vmask[:L], dtype=torch.bool),
            "id": row["id"],
        }


def rig_collate(batch: List[Dict]) -> Dict:
    """Same output as stemfx data.sepaug_dataset.sepaug_collate."""
    B = len(batch)
    out_o = {s: torch.stack([b["original_stems"][s] for b in batch]) for s in STEMS}
    out_a = {s: torch.stack([b["augmented_stems"][s] for b in batch]) for s in STEMS}
    L = max(b["tokens"].size(0) for b in batch)
    tokens = torch.zeros(B, L, dtype=torch.long)
    values = torch.zeros(B, L, dtype=torch.float32)
    mask = torch.zeros(B, L, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["tokens"].size(0)
        tokens[i, :n] = b["tokens"]
        values[i, :n] = b["values"]
        mask[i, :n] = b["value_mask"]
    return {"original_stems": out_o, "augmented_stems": out_a,
            "tokens": tokens, "values": values, "value_mask": mask,
            "ids": [b["id"] for b in batch]}


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=rig_collate, pin_memory=torch.cuda.is_available(),
                      drop_last=shuffle, persistent_workers=num_workers > 0)
