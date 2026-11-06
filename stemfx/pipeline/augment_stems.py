"""
Apply random multilib FX chains to separated stems and write the
augmented stems plus ``fx_metadata.json`` (the StemFX training target).

For each input track folder containing ``{vocals,bass,drums,other}.mp3``,
a fresh per-stem random chain is generated via ``multiafx.generate_random_chain``,
applied with ``multiafx.FXChain``, and saved next to the rendered MP3s.

Usage:
    python pipeline/augment_stems.py \\
        --source /path/to/fma_separated/ \\
        --output /path/to/fma_separated_augmented/ \\
        --min_fx 1 --max_fx 10 \\
        --workers 32
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

from multiafx import FXChain, generate_random_chain

STEMS = ("vocals", "drums", "bass", "other")


def _load_stem(path: str) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return audio.T, sr


def _save_stem(path: str, audio: np.ndarray, sr: int) -> None:
    sf.write(path, audio.T, sr, format="mp3")


def _process_folder(args: tuple) -> Tuple[str, bool, str]:
    folder_name, source_dir, output_dir, min_fx, max_fx = args
    src = Path(source_dir) / folder_name
    out = Path(output_dir) / folder_name

    try:
        seed = hash(folder_name) % (2 ** 31)
        out.mkdir(parents=True, exist_ok=True)

        chains = {s: generate_random_chain((min_fx, max_fx), seed=seed + i)
                  for i, s in enumerate(STEMS)}

        stems_meta = {}
        for stem in STEMS:
            audio, sr = _load_stem(str(src / f"{stem}.mp3"))
            chain = chains[stem]
            processed = chain(audio, sr)
            _save_stem(str(out / f"{stem}.mp3"), processed, sr)

            chain_meta = []
            for i, step in enumerate(chain):
                chain_meta.append({
                    "index": i,
                    "effect": step["effect"],
                    "params": {k: round(v, 6) if isinstance(v, float) else v
                               for k, v in step["params"].items()},
                })
            stems_meta[stem] = {
                "source_file": f"{stem}.mp3",
                "output_file": f"{stem}.mp3",
                "sample_rate": sr,
                "duration_seconds": round(audio.shape[1] / sr, 2),
                "fx_chain": chain_meta,
                "num_effects": len(chain_meta),
            }

        metadata = {
            "version": "1.0",
            "backend": "multiafx",
            "source_folder": folder_name,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "random_seed": seed,
            "stems": stems_meta,
        }
        with open(out / "fx_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        return (folder_name, True, "")
    except Exception as e:
        return (folder_name, False, str(e))


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply random FX chains to separated stems.")
    parser.add_argument("--source", required=True, help="Directory of separated stems (one folder per song).")
    parser.add_argument("--output", required=True, help="Directory to write augmented stems + fx_metadata.json.")
    parser.add_argument("--min_fx", type=int, default=1)
    parser.add_argument("--max_fx", type=int, default=10)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    src = Path(args.source)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    folders = sorted(p.name for p in src.iterdir() if p.is_dir() and not p.name.startswith("."))
    if args.skip_existing:
        folders = [f for f in folders if not (out / f / "fx_metadata.json").exists()]
    if args.limit is not None:
        folders = folders[: args.limit]

    print(f"Augmenting {len(folders)} folders, FX per stem: [{args.min_fx}, {args.max_fx}]")
    if not folders:
        return

    work = [(f, str(src), str(out), args.min_fx, args.max_fx) for f in folders]
    t0 = time.time()
    completed = failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_folder, w): w[0] for w in work}
        for fut in as_completed(futures):
            name, ok, err = fut.result()
            if ok:
                completed += 1
            else:
                failed += 1
                print(f"  ! {name}: {err[:120]}", file=sys.stderr)
            done = completed + failed
            if done % 100 == 0 or done == len(work):
                rate = done / max(time.time() - t0, 1e-6)
                eta = (len(work) - done) / max(rate, 1e-6) / 60
                print(f"  {done}/{len(work)}  ok={completed}  failed={failed}  "
                      f"{rate:.1f}/s  ETA={eta:.1f}min")

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min  ok={completed} failed={failed}")


if __name__ == "__main__":
    main()
