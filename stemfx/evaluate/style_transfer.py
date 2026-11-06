"""
Style-transfer evaluation for StemFX.

For each (original, target) stem pair the script:
  1. Embeds both via the BSFiLM Encoder.
  2. Predicts a per-stem FX chain (default: free autoregressive generation).
  3. Optionally evaluates a "forced" mode that constrains the output to a
     fixed 15-effect universal chain — useful for sanity-checking value
     prediction on a fixed structure.
  4. Applies the predicted chain via multiafx and computes MRSTFT against
     the target downmix as the audio-quality score.

Dataset layout (same track names in both directories):

    <original_dir>/<track_id>/{vocals,bass,drums,other}.wav
    <target_dir>/<track_id>/{vocals,bass,drums,other}.wav

Usage:
    python evaluate/style_transfer.py \\
        --original_dir /path/to/MUSDB18_normalized/test \\
        --target_dir /path/to/MUSDB18_styled/test \\
        --checkpoint /path/to/best_checkpoint.pt \\
        --output_dir outputs/eval_st \\
        --num_examples 20 \\
        --device cuda:1
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import stemfx  # noqa: E402
from stemfx import MultiResolutionSTFTLoss  # noqa: E402

STEMS = ("vocals", "bass", "drums", "other")
SR = 44100
SEG_S = 10.0

# 15-effect universal chain used for the "forced" baseline.
UNIVERSAL_CHAIN = [
    ("sox_gain", ["gain_db"]),
    ("sox_highpass", ["frequency", "width_q"]),
    ("sox_lowpass", ["frequency", "width_q"]),
    ("am_low_shelf_filter", ["center_freq", "gain_db", "q"]),
    ("am_peaking_filter", ["center_freq", "gain_db", "q"]),
    ("am_peaking_filter", ["center_freq", "gain_db", "q"]),
    ("am_high_shelf_filter", ["center_freq", "gain_db", "q"]),
    ("sox_compand", ["attack_time", "decay_time", "soft_knee_db"]),
    ("am_peaking_filter", ["center_freq", "gain_db", "q"]),
    ("am_peaking_filter", ["center_freq", "gain_db", "q"]),
    ("sox_overdrive", ["gain_db", "colour"]),
    ("sox_chorus", ["gain_in", "gain_out", "delay_ms", "decay", "speed", "depth"]),
    ("sox_echo", ["gain_in", "gain_out", "delay_ms", "decay"]),
    ("sox_reverb", ["reverberance", "high_freq_damping", "room_scale", "stereo_depth", "pre_delay", "wet_gain"]),
    ("sox_gain", ["gain_db"]),
]


# --------------------------------------------------------------------------- #
#  Audio helpers
# --------------------------------------------------------------------------- #
def load_stem(d: str, name: str) -> torch.Tensor:
    path = os.path.join(d, f"{name}.wav")
    audio, sr = torchaudio.load(path)
    if sr != SR:
        audio = torchaudio.functional.resample(audio, sr, SR)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    return audio.float()


def load_all_stems(d: str) -> Dict[str, torch.Tensor]:
    return {s: load_stem(d, s) for s in STEMS}


def crop_to_seg(stems: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    n = int(SEG_S * SR)
    out = {}
    for s, a in stems.items():
        clip = a[:, :n]
        if clip.shape[1] < n:
            clip = F.pad(clip, (0, n - clip.shape[1]))
        out[s] = clip
    return out


def downmix(stems: Dict[str, torch.Tensor]) -> torch.Tensor:
    mix = sum(stems.values())
    peak = mix.abs().max()
    if peak > 0.95:
        mix = mix * (0.95 / peak)
    return mix


def peak_norm(audio: torch.Tensor) -> torch.Tensor:
    p = audio.abs().max()
    return audio * (0.95 / p) if p > 1e-8 else audio


# --------------------------------------------------------------------------- #
#  Forced generation (constrain structure to UNIVERSAL_CHAIN)
# --------------------------------------------------------------------------- #
def _build_forced_seq(tokenizer) -> List:
    seq = []
    for stem in STEMS:
        seq.append(tokenizer.token_to_idx[stem])
        for effect, params in UNIVERSAL_CHAIN:
            seq.append(tokenizer.token_to_idx[effect])
            for pname in params:
                seq.append(tokenizer.token_to_idx[f"{effect}_{pname}"])
                seq.append(None)
        seq.append(tokenizer.sep_token_idx)
    seq.append(tokenizer.eos_token_idx)
    return seq


@torch.no_grad()
def forced_chain(model_api, orig_stems, tgt_stems) -> Dict[str, list]:
    net = model_api._net
    tokenizer = model_api._tokenizer
    device = model_api._device

    orig_b = {k: v.unsqueeze(0).to(device) for k, v in orig_stems.items()}
    tgt_b = {k: v.unsqueeze(0).to(device) for k, v in tgt_stems.items()}
    e_o = F.normalize(net._encode_stems(orig_b), p=2, dim=-1)
    e_t = F.normalize(net._encode_stems(tgt_b), p=2, dim=-1)

    gen = net.generator
    encoder_dim = gen.encoder_dim
    pair = torch.cat([e_o, e_t], dim=1)
    memory = torch.cat([
        gen.embedding_proj(pair[:, :encoder_dim]).unsqueeze(1),
        gen.embedding_proj(pair[:, encoder_dim:]).unsqueeze(1),
    ], dim=1)
    if hasattr(gen, "memory_pos_embedding"):
        memory = memory + gen.memory_pos_embedding.unsqueeze(0)

    forced = _build_forced_seq(tokenizer)
    tokens = [tokenizer.bos_token_idx]
    values = [0.0]
    for tok in forced:
        toks_t = torch.tensor([tokens], dtype=torch.long, device=device)
        emb = gen.pos_encoding(gen.token_embedding(toks_t))
        mask = gen._generate_square_subsequent_mask(toks_t.size(1)).to(device)
        h = gen.transformer_decoder(tgt=emb, memory=memory, tgt_mask=mask)[:, -1, :]
        if tok is not None:
            tokens.append(tok)
            values.append(0.0)
        else:
            logits = gen.token_head(h)
            block = torch.full_like(logits, -1e9)
            block[:, tokenizer.value_token_start:tokenizer.value_token_end + 1] = 0.0
            nxt = (logits + block).argmax(dim=-1).item()
            tokens.append(nxt)
            values.append((nxt - tokenizer.value_token_start) / 100.0)

    meta = tokenizer.decode_fx_chain(tokens, values)
    out = {s: [] for s in STEMS}
    for s, data in (meta.get("stems") or {}).items():
        if s in out:
            out[s] = list(data.get("fx_chain", []))
    return out


def apply_chain_to_full(orig_full: Dict[str, torch.Tensor], chain: Dict[str, list]) -> Dict[str, torch.Tensor]:
    from multiafx import FXChain
    out = {}
    for s in STEMS:
        steps = chain.get(s, [])
        audio_np = orig_full[s].cpu().numpy().astype(np.float32)
        if steps:
            audio_np = FXChain(steps)(audio_np, SR)
        out[s] = torch.from_numpy(audio_np)
    return out


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dir", required=True)
    parser.add_argument("--target_dir", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="StemFX checkpoint (defaults to HF Hub).")
    parser.add_argument("--output_dir", default="outputs/eval_st")
    parser.add_argument("--num_examples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--mode", choices=("free", "forced", "both"), default="both")
    parser.add_argument("--save_audio", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)

    common = sorted(set(os.listdir(args.original_dir)) & set(os.listdir(args.target_dir)))
    common = [t for t in common
              if os.path.isfile(os.path.join(args.original_dir, t, "vocals.wav"))
              and os.path.isfile(os.path.join(args.target_dir, t, "vocals.wav"))]
    selected = sorted(rng.sample(common, min(args.num_examples, len(common))))
    print(f"Evaluating on {len(selected)} of {len(common)} paired tracks")

    print(f"Loading StemFX from: {args.checkpoint or 'HuggingFace Hub'}")
    model = stemfx.load(checkpoint=args.checkpoint, device=args.device)
    mrstft = MultiResolutionSTFTLoss().to(args.device)

    rows = []
    for idx, name in enumerate(tqdm(selected, desc="tracks")):
        orig_full = load_all_stems(os.path.join(args.original_dir, name))
        tgt_full = load_all_stems(os.path.join(args.target_dir, name))
        orig_seg = crop_to_seg(orig_full)
        tgt_seg = crop_to_seg(tgt_full)
        target_mix = downmix(tgt_full)

        ex_dir = os.path.join(args.output_dir, name.replace(" ", "_"))
        if args.save_audio:
            os.makedirs(ex_dir, exist_ok=True)
            sf.write(os.path.join(ex_dir, "original.wav"), downmix(orig_full).T.numpy(), SR)
            sf.write(os.path.join(ex_dir, "target.wav"), target_mix.T.numpy(), SR)

        row: dict = {"track": name}

        if args.mode in ("free", "both"):
            chain_free = model.transfer(model.embed(orig_seg), model.embed(tgt_seg))
            pred_stems = apply_chain_to_full(orig_full, chain_free)
            pred_mix = downmix(pred_stems)
            n = min(pred_mix.shape[-1], target_mix.shape[-1])
            with torch.no_grad():
                a = pred_mix[:, :n].unsqueeze(0).to(args.device)
                b = target_mix[:, :n].unsqueeze(0).to(args.device)
                row["free_mrstft"] = round(mrstft(a, b).item(), 4)
                a_n = peak_norm(pred_mix)[:, :n].unsqueeze(0).to(args.device)
                b_n = peak_norm(target_mix)[:, :n].unsqueeze(0).to(args.device)
                row["free_mrstft_norm"] = round(mrstft(a_n, b_n).item(), 4)
            if args.save_audio:
                sf.write(os.path.join(ex_dir, "predicted_free.wav"), pred_mix.T.numpy(), SR)
                with open(os.path.join(ex_dir, "chain_free.json"), "w") as f:
                    json.dump(chain_free, f, indent=2)

        if args.mode in ("forced", "both"):
            chain_forced = forced_chain(model, orig_seg, tgt_seg)
            pred_stems = apply_chain_to_full(orig_full, chain_forced)
            pred_mix = downmix(pred_stems)
            n = min(pred_mix.shape[-1], target_mix.shape[-1])
            with torch.no_grad():
                a = pred_mix[:, :n].unsqueeze(0).to(args.device)
                b = target_mix[:, :n].unsqueeze(0).to(args.device)
                row["forced_mrstft"] = round(mrstft(a, b).item(), 4)
                a_n = peak_norm(pred_mix)[:, :n].unsqueeze(0).to(args.device)
                b_n = peak_norm(target_mix)[:, :n].unsqueeze(0).to(args.device)
                row["forced_mrstft_norm"] = round(mrstft(a_n, b_n).item(), 4)
            if args.save_audio:
                sf.write(os.path.join(ex_dir, "predicted_forced.wav"), pred_mix.T.numpy(), SR)
                with open(os.path.join(ex_dir, "chain_forced.json"), "w") as f:
                    json.dump(chain_forced, f, indent=2)

        rows.append(row)
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(rows, f, indent=2)
        if rows:
            with open(os.path.join(args.output_dir, "summary.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys()}))
                w.writeheader()
                w.writerows(rows)

    print("\n=== Summary ===")
    for key in ("free_mrstft", "free_mrstft_norm", "forced_mrstft", "forced_mrstft_norm"):
        vals = [r[key] for r in rows if key in r]
        if vals:
            arr = np.asarray(vals, dtype=float)
            print(f"  {key:<20} mean={arr.mean():.4f}  std={arr.std():.4f}  n={len(arr)}")


if __name__ == "__main__":
    main()
