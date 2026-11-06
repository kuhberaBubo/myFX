"""
Compute integrated LUFS for every stem in an augmented dataset and emit
``valid_tracks.json``: the set of song IDs whose stems exceed the loudness
threshold for each role. ``valid_tracks.json`` is consumed by the SepAug
``mix_stems`` mode, which samples one stem from a different song per role.

For every song folder it writes ``stem_lufs.json``; at the dataset root it
writes ``valid_tracks.json`` and ``lufs_summary.json``.

Usage:
    python pipeline/compute_stem_lufs.py \\
        --dataset_dir /path/to/fma_separated_augmented/ \\
        --threshold -50.0 \\
        --num_workers 32
"""

import argparse
import json
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

STEM_NAMES = ("vocals", "bass", "drums", "other")


def compute_lufs_for_song(song_dir: str, sample_rate: int = 44100) -> dict:
    """Integrated LUFS for each stem in ``song_dir``."""
    meter = pyln.Meter(sample_rate)
    stem_lufs = {}
    for stem in STEM_NAMES:
        path = os.path.join(song_dir, f"{stem}.mp3")
        if not os.path.exists(path):
            stem_lufs[stem] = float("-inf")
            continue
        try:
            audio, sr = sf.read(path, dtype="float32", always_2d=True)
            m = meter if sr == sample_rate else pyln.Meter(sr)
            loudness = m.integrated_loudness(audio)
            stem_lufs[stem] = float("-inf") if not np.isfinite(loudness) else round(float(loudness), 2)
        except Exception:
            stem_lufs[stem] = float("-inf")
    return {"song_id": os.path.basename(song_dir), "stems": stem_lufs}


def _process_song(song_dir: str, sample_rate: int) -> dict:
    result = compute_lufs_for_song(song_dir, sample_rate)
    try:
        with open(os.path.join(song_dir, "stem_lufs.json"), "w") as f:
            json.dump(result, f, indent=2)
    except Exception:
        pass
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-stem LUFS for an augmented dataset.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--threshold", type=float, default=-50.0,
                        help="LUFS threshold (default: -50). Stems below this are excluded from valid_tracks.json.")
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--sample_rate", type=int, default=44100)
    args = parser.parse_args()

    song_dirs = sorted(
        os.path.join(args.dataset_dir, d)
        for d in os.listdir(args.dataset_dir)
        if os.path.isdir(os.path.join(args.dataset_dir, d))
    )
    total = len(song_dirs)
    print(f"Found {total} song directories under {args.dataset_dir}")

    fn = partial(_process_song, sample_rate=args.sample_rate)
    results = []
    with Pool(processes=args.num_workers) as pool:
        for r in pool.imap_unordered(fn, song_dirs, chunksize=64):
            results.append(r)
            if len(results) % 5000 == 0:
                print(f"  {len(results)}/{total}")

    valid: dict = {s: [] for s in STEM_NAMES}
    vals: dict = {s: [] for s in STEM_NAMES}
    for r in results:
        for stem in STEM_NAMES:
            lufs = r["stems"].get(stem, float("-inf"))
            if np.isfinite(lufs):
                vals[stem].append(lufs)
            if lufs >= args.threshold:
                valid[stem].append(r["song_id"])
    for s in STEM_NAMES:
        valid[s].sort()

    valid_path = os.path.join(args.dataset_dir, "valid_tracks.json")
    with open(valid_path, "w") as f:
        json.dump(valid, f, indent=2)
    print(f"\nSaved {valid_path}")

    print(f"\nLUFS summary (threshold={args.threshold} dB)")
    print(f"{'stem':<10} {'valid':>8} / {'total':>8}  {'mean':>8} {'median':>8}")
    for s in STEM_NAMES:
        v = np.asarray(vals[s], dtype=float) if vals[s] else None
        mean_v = f"{v.mean():.1f}" if v is not None else "n/a"
        med_v = f"{np.median(v):.1f}" if v is not None else "n/a"
        print(f"{s:<10} {len(valid[s]):>8} / {total:>8}  {mean_v:>8} {med_v:>8}")

    summary = {
        "threshold_lufs": args.threshold,
        "total_songs": total,
        "per_stem": {
            s: {
                "valid": len(valid[s]),
                "mean_lufs": round(float(np.mean(vals[s])), 2) if vals[s] else None,
                "median_lufs": round(float(np.median(vals[s])), 2) if vals[s] else None,
            }
            for s in STEM_NAMES
        },
    }
    with open(os.path.join(args.dataset_dir, "lufs_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
