"""
Preprocess an audio dataset (default: FMA) by separating each track into
4 stems with SCNet, then encoding the stems as MP3.

The separated stems are written to ``<output_dir>/<track_id>/{vocals,bass,drums,other}.mp3``
and become the input to the SepAug pipeline.

Usage:
    python pipeline/preprocess_fma.py \\
        --input_dir /path/to/fma_full/ \\
        --output_dir /path/to/fma_separated/ \\
        --scnet_model third_party/Music-Source-Separation-Training/model_scnet_masked_ep_111_sdr_9.8286.ckpt \\
        --scnet_config third_party/Music-Source-Separation-Training/configs/config_musdb18_scnet_xl_ihf.yaml \\
        --device cuda:0 \\
        --skip_existing
"""

import argparse
import glob
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from stemfx.separator import SCNetSeparator

STEMS = ("vocals", "bass", "drums", "other")


def _is_done(output_dir: str, track_id: str) -> bool:
    return all(
        os.path.exists(os.path.join(output_dir, track_id, f"{s}.mp3"))
        for s in STEMS
    )


def _encode_mp3(audio: torch.Tensor, output_path: str, sample_rate: int, bitrate: str) -> bool:
    tmp_wav = output_path.replace(".mp3", "_tmp.wav")
    sf.write(tmp_wav, audio.cpu().numpy().T, sample_rate)
    cmd = ["ffmpeg", "-y", "-i", tmp_wav, "-codec:a", "libmp3lame",
           "-b:a", bitrate, "-ar", str(sample_rate), output_path]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        os.remove(tmp_wav)
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate audio dataset into 4-stem MP3s.")
    parser.add_argument("--input_dir", required=True, help="Source dataset directory (recursively scanned).")
    parser.add_argument("--output_dir", required=True, help="Where stems are written.")
    parser.add_argument("--scnet_model", required=True, help="Path to SCNet .ckpt checkpoint.")
    parser.add_argument("--scnet_config", required=True, help="Path to SCNet YAML config.")
    parser.add_argument("--scnet_dir", default=None,
                        help="Path to Music-Source-Separation-Training source tree "
                             "(defaults to third_party/Music-Source-Separation-Training).")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bitrate", default="192k")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mp3_workers", type=int, default=8)
    args = parser.parse_args()

    audio_files = []
    for ext in ("*.mp3", "*.wav", "*.flac"):
        audio_files.extend(glob.glob(os.path.join(args.input_dir, "**", ext), recursive=True))
    audio_files.sort()
    if args.limit is not None:
        audio_files = audio_files[: args.limit]

    if args.skip_existing:
        audio_files = [f for f in audio_files if not _is_done(args.output_dir, Path(f).stem)]

    print(f"Tracks to process: {len(audio_files)}")
    if not audio_files:
        return

    os.makedirs(args.output_dir, exist_ok=True)

    sep = SCNetSeparator(
        model_path=args.scnet_model,
        config_path=args.scnet_config,
        device=args.device,
        scnet_dir=args.scnet_dir,
    )
    sr = sep.sample_rate

    pool = ThreadPoolExecutor(max_workers=args.mp3_workers)
    futures: list = []

    for path in tqdm(audio_files, desc="Separating"):
        tid = Path(path).stem
        out = os.path.join(args.output_dir, tid)
        os.makedirs(out, exist_ok=True)
        try:
            audio, _ = librosa.load(path, sr=sr, mono=False)
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=0)
            stems = sep.separate(torch.from_numpy(audio).float())
        except Exception as e:
            print(f"\n  ! skipped {path}: {e}")
            continue
        for stem in STEMS:
            futures.append(pool.submit(
                _encode_mp3, stems[stem], os.path.join(out, f"{stem}.mp3"), sr, args.bitrate
            ))

    for f in as_completed(futures):
        f.result()
    pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
