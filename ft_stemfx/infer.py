"""
Predict the Guitar Rig FX chain (which effects, in what order, with what
values) from a (DI, processed) audio pair. Runs on CPU.

    python ft_stemfx/infer.py --ckpt best_checkpoint.pt --di di.wav --wet wet.wav \
        --offset 0 --out pred.json

Preprocessing matches training:
  * any sample rate -> 44.1 kHz, mono, `clip_duration` s (10 s) from --offset
    (DI and wet must be time-aligned)
  * the DI segment is gain-matched to -32 LUFS (how the training inputs were
    normalized) and the SAME gain is applied to the wet signal, so the
    wet/dry level relation (gain / volume knobs) is preserved
  * dummy vocals/bass/drums identical in both, as in training

Output JSON
  chain    ordered list of effects, each with its predicted params
  values   flat dict in the dataset/rigkit `values` format
           ({key: {"value", "raw", "source"}}), incl. <comp>_on_off for the
           first renderable preset; params the model did not emit are filled
           with the preset default (source="default")
  renderable_presets  rigkit presets that can reproduce the predicted chain
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from load_pretrained import build_model  # noqa: E402
from rig_dataset import build_stems, make_dummy_bank  # noqa: E402
from rig_normalization import raw_to_value  # noqa: E402
from rig_tokenizer import RACK, RigTokenizer  # noqa: E402

TRAIN_INPUT_LUFS = -32.0


def load_audio(path: str, sr: int, offset: float, length: int) -> np.ndarray:
    x, file_sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if file_sr != sr:
        import torchaudio.functional as AF
        x = AF.resample(torch.from_numpy(x), file_sr, sr).numpy()
    start = int(round(offset * sr))
    x = x[start:start + length]
    if len(x) < length:
        x = np.pad(x, (0, length - len(x)))
    return x


def integrated_lufs(x: np.ndarray, sr: int) -> float:
    try:
        import pyloudnorm as pyln
        return float(pyln.Meter(sr).integrated_loudness(x))
    except ImportError:
        print("[warn] pyloudnorm not installed -> RMS approximation of LUFS "
              "(pip install pyloudnorm for BS.1770)")
        return float(-0.691 + 10 * np.log10(np.mean(x ** 2) + 1e-12))


def to_rigkit_values(tok: RigTokenizer, pred: dict, preset: str | None) -> dict:
    out = {}
    if preset is not None:
        used = set(pred["effects"])
        for comp, keys, onoff in tok.preset_layout[preset]:
            if onoff is not None:
                on = comp in used
                out[onoff] = {"value": on, "raw": float(on), "source": "predicted"}
            if comp in used:
                for k in keys:                       # defaults for params not emitted
                    if k not in pred["params"]:
                        raw = float(tok.specs[k]["default"]["raw"])
                        out[k] = {"value": raw_to_value(tok.specs[k], raw), "raw": raw,
                                  "source": "default"}
    for k, raw in pred["params"].items():
        out[k] = {"value": pred["values"][k], "raw": raw, "source": "predicted"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--di", required=True, help="dry DI audio")
    ap.add_argument("--wet", required=True, help="processed audio, time-aligned with --di")
    ap.add_argument("--offset", type=float, default=0.0, help="segment start (s)")
    ap.add_argument("--no-gain-match", action="store_true")
    ap.add_argument("--param-info", default=None, help="only for checkpoints without param_info")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    t0 = time.time()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    tok = (RigTokenizer(a.param_info) if a.param_info
           else RigTokenizer.from_checkpoint(ck["tokenizer"]))
    tok.check_compatible(ck["tokenizer"])
    model = build_model(ck["args"], tok.vocab_size, dropout=0.0)
    model.load_state_dict(ck["model_state_dict"])
    model.to(a.device).eval()
    dc, seed = ck["config"]["data"], ck["config"]["train"]["seed"]
    sr = dc["sample_rate"]
    length = int(round(dc["clip_duration"] * sr))
    t_load = time.time() - t0

    di = load_audio(a.di, sr, a.offset, length)
    wet = load_audio(a.wet, sr, a.offset, length)
    gain_db = 0.0
    if not a.no_gain_match:
        lufs = integrated_lufs(di, sr)
        if np.isfinite(lufs):
            gain_db = TRAIN_INPUT_LUFS - lufs
            g = np.float32(10 ** (gain_db / 20))
            di, wet = di * g, wet * g

    to2 = lambda x: torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).expand(2, -1).contiguous()  # noqa: E731
    bank = make_dummy_bank(length, dc["dummy_level_db"], 16, seed)
    orig, aug = build_stems(to2(di), to2(wet), bank, key=0)
    orig = {k: v.unsqueeze(0).to(a.device) for k, v in orig.items()}
    aug = {k: v.unsqueeze(0).to(a.device) for k, v in aug.items()}

    t1 = time.time()
    with torch.no_grad():
        seq, _ = model.generate(orig, aug, max_length=dc["max_seq_len"],
                                bos_token_idx=tok.bos_token_idx,
                                eos_token_idx=tok.eos_token_idx, top_k=1, top_p=1.0)
    t_gen = time.time() - t1

    pred = tok.decode(seq[0].tolist())
    presets = tok.renderable_presets(pred["effects"]) if pred["effects"] else []
    chain = [{"effect": e,
              "params": {k: {"value": pred["values"][k], "raw": round(r, 4)}
                         for k, r in pred["params"].items()
                         if tok.specs[k]["component"] == e}}
             for e in pred["effects"]]
    result = {
        "chain": chain,
        "values": to_rigkit_values(tok, pred, presets[0] if presets else None),
        "renderable_presets": presets,
        "gain_match_db": round(gain_db, 2),
        "tokens": [tok.idx_to_token[i] for i in seq[0].tolist()],
        "timing_s": {"load": round(t_load, 2), "generate": round(t_gen, 2)},
    }

    print(" -> ".join(e for e in pred["effects"] if e != RACK) or "(no effects)")
    for c in chain:
        print(f"  {c['effect']}: " + ", ".join(f"{k.split(c['effect'] + '_', 1)[-1]}={v['value']}"
                                                for k, v in c["params"].items()))
    print(f"load {t_load:.1f}s, generate {t_gen:.1f}s on {a.device}")
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str),
                               encoding="utf-8")


if __name__ == "__main__":
    main()
