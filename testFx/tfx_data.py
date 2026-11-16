"""
(dry, wet) pairs for testFX: mono, full 15 s clip by default (10 s playing +
5 s tail, so delay / reverb tails are visible). Same jsonl rows, tokenizer and
target sequence as ft_stemfx; no 4-stem packing, no dummy stems.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from rig_dataset import read_jsonl


class PairDataset(Dataset):
    def __init__(self, root, split_file, tokenizer, sample_rate=44100, clip_start_sec=0.0,
                 clip_duration=15.0, max_seq_len=512, max_samples=None,
                 skip_missing_audio=False):
        self.root, self.tok = Path(root), tokenizer
        self.sr, self.max_seq_len = sample_rate, max_seq_len
        self.start = int(round(clip_start_sec * sample_rate))
        self.length = int(round(clip_duration * sample_rate))
        rows = read_jsonl(self.root / split_file)
        if skip_missing_audio:
            rows = [r for r in rows if (self.root / r["input"]).is_file()
                    and (self.root / r["target"]).is_file()]
        if max_samples:
            rows = rows[:max_samples]
        if not rows:
            raise RuntimeError(f"No usable samples in {self.root / split_file}")
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def _load(self, rel):
        x, sr = sf.read(self.root / rel, dtype="float32", frames=self.start + self.length,
                        always_2d=True)
        if sr != self.sr:
            raise ValueError(f"{rel}: sample rate {sr} != {self.sr}")
        x = x.mean(1)[self.start:self.start + self.length]
        if len(x) < self.length:
            x = np.pad(x, (0, self.length - len(x)))
        return torch.from_numpy(np.ascontiguousarray(x))

    def __getitem__(self, i):
        row = self.rows[i]
        tokens, values, vmask = self.tok.encode(row)
        L = self.max_seq_len
        return {"dry": self._load(row["input"]), "wet": self._load(row["target"]),
                "tokens": torch.tensor(tokens[:L]), "values": torch.tensor(values[:L]),
                "value_mask": torch.tensor(vmask[:L]), "id": row["id"]}


def pair_collate(batch):
    B, L = len(batch), max(b["tokens"].size(0) for b in batch)
    tokens = torch.zeros(B, L, dtype=torch.long)
    values = torch.zeros(B, L)
    mask = torch.zeros(B, L, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["tokens"].size(0)
        tokens[i, :n], values[i, :n], mask[i, :n] = b["tokens"], b["values"], b["value_mask"]
    return {"dry": torch.stack([b["dry"] for b in batch]),
            "wet": torch.stack([b["wet"] for b in batch]),
            "tokens": tokens, "values": values, "value_mask": mask,
            "ids": [b["id"] for b in batch]}


def make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=pair_collate, pin_memory=torch.cuda.is_available(),
                      drop_last=shuffle, persistent_workers=num_workers > 0)
