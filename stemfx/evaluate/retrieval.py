"""
Paired FX retrieval evaluation for the StemFX BSFiLM Encoder.

Dataset layout (paired-FX format used in the paper):

    <dataset>/
        query/
            <variant_id>/{vocals,bass,drums,other}.wav   # 10 s segments
        pool/
            <variant_id>/{vocals,bass,drums,other}.wav

For every variant we extract one query and one pool embedding (L2-normalized
mixing-style embedding from the BSFiLM Encoder). Top-k accuracy is computed
by ranking pool embeddings against each query under cosine similarity.

Usage:
    python evaluate/retrieval.py \\
        --dataset /path/to/MUSDB18_paired_fx_corrected_5/ \\
        --checkpoint /path/to/best_checkpoint.pt \\
        --device cuda:1
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import stemfx  # noqa: E402

STEMS = ("vocals", "bass", "drums", "other")
SAMPLE_RATE = 44100
SEG_SAMPLES = SAMPLE_RATE * 10


def load_segment(path: str) -> torch.Tensor:
    audio, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    audio = audio[:, :SEG_SAMPLES]
    if audio.shape[1] < SEG_SAMPLES:
        audio = F.pad(audio, (0, SEG_SAMPLES - audio.shape[1]))
    return audio.float()


def load_stems(dir_: str):
    stems = {}
    for s in STEMS:
        path = os.path.join(dir_, f"{s}.wav")
        if not os.path.exists(path):
            return None
        stems[s] = load_segment(path)
    return stems


def topk_metrics(queries: np.ndarray, pool: np.ndarray, ks=(1, 5, 10)) -> dict:
    q = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
    p = pool / (np.linalg.norm(pool, axis=1, keepdims=True) + 1e-8)
    sims = q @ p.T
    n = len(queries)
    order = np.argsort(-sims, axis=1)
    out = {}
    for k in ks:
        kk = min(k, p.shape[0])
        out[f"top{k}"] = float(np.mean([i in order[i, :kk] for i in range(n)]) * 100)
    rr = []
    for i in range(n):
        rank = np.where(order[i] == i)[0]
        rr.append(1.0 / (rank[0] + 1) if rank.size else 0.0)
    out["mrr"] = float(np.mean(rr))
    out["n"] = n
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="Directory with query/ and pool/ subdirs.")
    parser.add_argument("--checkpoint", default=None,
                        help="StemFX checkpoint (defaults to HF Hub download).")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading StemFX from: {args.checkpoint or 'HuggingFace Hub'}")
    model = stemfx.load(checkpoint=args.checkpoint, device=args.device)

    qroot = os.path.join(args.dataset, "query")
    proot = os.path.join(args.dataset, "pool")
    variants = sorted(d for d in os.listdir(qroot)
                      if os.path.isdir(os.path.join(qroot, d))
                      and os.path.isdir(os.path.join(proot, d)))
    if args.limit:
        variants = variants[: args.limit]

    q_embs, p_embs = [], []
    for v in tqdm(variants, desc="embedding"):
        q = load_stems(os.path.join(qroot, v))
        p = load_stems(os.path.join(proot, v))
        if q is None or p is None:
            continue
        q_embs.append(model.embed(q).cpu().numpy())
        p_embs.append(model.embed(p).cpu().numpy())

    q_arr = np.stack(q_embs)
    p_arr = np.stack(p_embs)
    metrics = topk_metrics(q_arr, p_arr)
    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k:<6} {v:.3f}" if isinstance(v, float) else f"  {k:<6} {v}")


if __name__ == "__main__":
    main()
