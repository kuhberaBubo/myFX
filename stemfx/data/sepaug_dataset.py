"""
SepAug paired dataset for StemFX training.

Loads (original_stems, augmented_stems, fx_chain) triples produced by the
Sep-Aug Pipeline (separation + cross-song stem mixing + multilib FX
augmentation). The augmented version's FX chain is the training target;
the model learns to recover the chain from the (original, augmented)
embedding pair.
"""

import json
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from stemfx import FXChainTokenizer

STEM_NAMES = ("vocals", "bass", "drums", "other")


def _load_audio_soundfile(path: str, target_sr: int = 44100) -> Tuple[torch.Tensor, int]:
    audio_np, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio_np.T.copy())
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio, target_sr


def _load_audio_ffmpeg(path: str, target_sr: int = 44100, channels: int = 2) -> Tuple[torch.Tensor, int]:
    cmd = [
        "ffmpeg", "-v", "quiet", "-i", path,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", str(channels), "-ar", str(target_sr), "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    audio_np = np.frombuffer(result.stdout, dtype=np.float32)
    if len(audio_np) == 0:
        raise RuntimeError(f"ffmpeg failed to decode {path}")
    audio_np = audio_np.reshape(-1, channels).T
    return torch.from_numpy(audio_np.copy()), target_sr


def _load_stereo(path: str, target_sr: int = 44100, use_ffmpeg: bool = False) -> torch.Tensor:
    if use_ffmpeg:
        audio, _ = _load_audio_ffmpeg(path, target_sr=target_sr)
    else:
        try:
            audio, sr = torchaudio.load(path)
            if sr != target_sr:
                audio = torchaudio.functional.resample(audio, sr, target_sr)
        except Exception:
            audio, _ = _load_audio_soundfile(path, target_sr=target_sr)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    return audio


class SepAugDataset(Dataset):
    """Paired (original, augmented + chain) samples for FX chain training.

    Each example yields:
      - ``original_stems``:  Dict[str, (2, T)] cropped to ``clip_duration``
      - ``augmented_stems``: Dict[str, (2, T)] cropped to ``clip_duration``
      - ``tokens``, ``values``, ``value_mask``: tokenized FX chain target
    """

    def __init__(
        self,
        original_path: str,
        augmented_path: str,
        tokenizer: FXChainTokenizer,
        clip_duration: float = 10.0,
        sample_rate: int = 44100,
        max_seq_len: int = 512,
        scroll_prob: float = 0.5,
        same_segment: bool = False,
        use_ffmpeg: bool = False,
        mix_stems: bool = False,
        valid_tracks_path: Optional[str] = None,
    ):
        self.original_path = Path(original_path)
        self.augmented_path = Path(augmented_path)
        self.tokenizer = tokenizer
        self.clip_duration = clip_duration
        self.sample_rate = sample_rate
        self.max_seq_len = max_seq_len
        self.scroll_prob = scroll_prob
        self.same_segment = same_segment
        self.use_ffmpeg = use_ffmpeg
        self.mix_stems = mix_stems

        if self.mix_stems:
            if valid_tracks_path is None:
                valid_tracks_path = str(self.augmented_path / "valid_tracks.json")
            with open(valid_tracks_path) as f:
                self.valid_tracks = json.load(f)

        self.song_ids = self._find_valid_song_ids()
        if len(self.song_ids) == 0:
            raise RuntimeError(
                f"No valid (original, augmented) song pairs found "
                f"under {self.original_path} / {self.augmented_path}"
            )

    def _find_valid_song_ids(self) -> List[str]:
        augmented_ids = set()
        for d in self.augmented_path.iterdir():
            if d.is_dir() and (d / "fx_metadata.json").exists():
                augmented_ids.add(d.name)
        valid: List[str] = []
        for sid in sorted(augmented_ids):
            orig = self.original_path / sid
            if orig.exists() and all((orig / f"{s}.mp3").exists() for s in STEM_NAMES):
                valid.append(sid)
        return valid

    def __len__(self) -> int:
        return len(self.song_ids)

    def _load_stems(self, song_dir: Path) -> Dict[str, torch.Tensor]:
        return {
            s: _load_stereo(str(song_dir / f"{s}.mp3"), self.sample_rate, self.use_ffmpeg)
            for s in STEM_NAMES
        }

    def _crop_at(self, stems: Dict[str, torch.Tensor], start: int, n: int) -> Dict[str, torch.Tensor]:
        out = {}
        for s, a in stems.items():
            clip = a[:, start:start + n]
            if clip.shape[1] < n:
                clip = torch.nn.functional.pad(clip, (0, n - clip.shape[1]))
            out[s] = clip
        return out

    def _scroll(self, stems: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        n = next(iter(stems.values())).shape[1]
        out = {}
        for s, a in stems.items():
            if random.random() < self.scroll_prob:
                shift = random.randint(-n // 2, n // 2)
                a = torch.roll(a, shifts=shift, dims=1)
            out[s] = a
        return out

    def _getitem_single(self, idx: int) -> Dict:
        sid = self.song_ids[idx]
        orig = self._load_stems(self.original_path / sid)
        aug = self._load_stems(self.augmented_path / sid)

        n = int(self.clip_duration * self.sample_rate)
        total = next(iter(orig.values())).shape[1]

        if self.same_segment:
            orig_start = aug_start = random.randint(0, max(0, total - n))
        elif total >= 2 * n:
            first = random.randint(0, total - 2 * n)
            second = first + n + random.randint(0, total - first - 2 * n)
            orig_start, aug_start = (first, second) if random.random() < 0.5 else (second, first)
        elif total > n:
            orig_start = random.randint(0, total - n)
            aug_start = random.randint(0, total - n)
        else:
            orig_start = aug_start = 0

        orig_clip = self._scroll(self._crop_at(orig, orig_start, n))
        aug_clip = self._crop_at(aug, aug_start, n)

        with open(self.augmented_path / sid / "fx_metadata.json") as f:
            chain = json.load(f)
        tokens, values, mask = self.tokenizer.encode_fx_chain(chain)
        return {
            "original_stems": orig_clip,
            "augmented_stems": aug_clip,
            "tokens": torch.tensor(tokens[:self.max_seq_len], dtype=torch.long),
            "values": torch.tensor(values[:self.max_seq_len], dtype=torch.float32),
            "value_mask": torch.tensor(mask[:self.max_seq_len], dtype=torch.bool),
            "song_id": sid,
        }

    def _getitem_mixed(self, idx: int) -> Dict:
        n = int(self.clip_duration * self.sample_rate)
        orig_clip: Dict[str, torch.Tensor] = {}
        aug_clip: Dict[str, torch.Tensor] = {}
        chain = {"stems": {}}
        for stem in STEM_NAMES:
            pool = self.valid_tracks.get(stem, []) or self.song_ids
            sid = random.choice(pool)
            orig_dir = self.original_path / sid
            aug_dir = self.augmented_path / sid
            orig_audio = _load_stereo(str(orig_dir / f"{stem}.mp3"), self.sample_rate, self.use_ffmpeg)
            aug_audio = _load_stereo(str(aug_dir / f"{stem}.mp3"), self.sample_rate, self.use_ffmpeg)
            if self.same_segment:
                start = random.randint(0, max(0, min(orig_audio.shape[1], aug_audio.shape[1]) - n))
                o_start = a_start = start
            else:
                o_start = random.randint(0, max(0, orig_audio.shape[1] - n))
                a_start = random.randint(0, max(0, aug_audio.shape[1] - n))
            o = orig_audio[:, o_start:o_start + n]
            a = aug_audio[:, a_start:a_start + n]
            if o.shape[1] < n:
                o = torch.nn.functional.pad(o, (0, n - o.shape[1]))
            if a.shape[1] < n:
                a = torch.nn.functional.pad(a, (0, n - a.shape[1]))
            orig_clip[stem] = o
            aug_clip[stem] = a
            with open(aug_dir / "fx_metadata.json") as f:
                meta = json.load(f)
            if stem in meta.get("stems", {}):
                chain["stems"][stem] = meta["stems"][stem]
        orig_clip = self._scroll(orig_clip)
        tokens, values, mask = self.tokenizer.encode_fx_chain(chain)
        return {
            "original_stems": orig_clip,
            "augmented_stems": aug_clip,
            "tokens": torch.tensor(tokens[:self.max_seq_len], dtype=torch.long),
            "values": torch.tensor(values[:self.max_seq_len], dtype=torch.float32),
            "value_mask": torch.tensor(mask[:self.max_seq_len], dtype=torch.bool),
            "song_id": "mixed",
        }

    def __getitem__(self, idx: int) -> Dict:
        try:
            return self._getitem_mixed(idx) if self.mix_stems else self._getitem_single(idx)
        except Exception:
            j = random.randint(0, len(self.song_ids) - 1)
            return self._getitem_mixed(j) if self.mix_stems else self._getitem_single(j)


def sepaug_collate(batch: List[Dict]) -> Dict:
    """Collate function: stacks stems, dynamically pads token sequences."""
    B = len(batch)
    out_orig: Dict[str, torch.Tensor] = {}
    out_aug: Dict[str, torch.Tensor] = {}
    for s in STEM_NAMES:
        out_orig[s] = torch.stack([b["original_stems"][s] for b in batch])
        out_aug[s] = torch.stack([b["augmented_stems"][s] for b in batch])

    L = max(b["tokens"].size(0) for b in batch)
    tokens = torch.zeros(B, L, dtype=torch.long)
    values = torch.zeros(B, L, dtype=torch.float32)
    mask = torch.zeros(B, L, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["tokens"].size(0)
        tokens[i, :n] = b["tokens"]
        values[i, :n] = b["values"]
        mask[i, :n] = b["value_mask"]
    return {
        "original_stems": out_orig,
        "augmented_stems": out_aug,
        "tokens": tokens,
        "values": values,
        "value_mask": mask,
        "song_ids": [b["song_id"] for b in batch],
    }


def create_dataloaders(
    original_path: str,
    augmented_path: str,
    batch_size: int = 32,
    num_workers: int = 8,
    clip_duration: float = 10.0,
    max_seq_len: int = 512,
    val_split: float = 0.05,
    scroll_prob: float = 0.5,
    same_segment: bool = False,
    use_ffmpeg: bool = False,
    mix_stems: bool = False,
    valid_tracks_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, FXChainTokenizer]:
    """Build train/val DataLoaders + tokenizer for SepAug data."""
    tokenizer = FXChainTokenizer()
    full = SepAugDataset(
        original_path=original_path,
        augmented_path=augmented_path,
        tokenizer=tokenizer,
        clip_duration=clip_duration,
        max_seq_len=max_seq_len,
        scroll_prob=scroll_prob,
        same_segment=same_segment,
        use_ffmpeg=use_ffmpeg,
        mix_stems=mix_stems,
        valid_tracks_path=valid_tracks_path,
    )

    total = len(full)
    if max_samples is not None and max_samples < total:
        full, _ = torch.utils.data.random_split(
            full, [max_samples, total - max_samples],
            generator=torch.Generator().manual_seed(seed),
        )
        total = max_samples

    val_size = int(total * val_split)
    train_size = total - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        full, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed + 1),
    )

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=sepaug_collate,
        pin_memory=True,
    )
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
        tokenizer,
    )
