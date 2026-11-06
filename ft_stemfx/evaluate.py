"""
Free-running (greedy) evaluation of a fine-tuned checkpoint on val.jsonl.

Metrics (prediction vs ground truth after bin quantization)
  chain_exact    predicted effect list (incl. order) == true effect list
  effect_acc     per-effect used/unused correctness over all effect types
  effect_f1      F1 of the predicted effect set vs the true set
  order_ok       for effects in both, relative order is the same
  param_mae      mean |raw_pred - raw_true| over params present in both
  discrete_acc   exact match for bool / str params present in both
  param_recall   fraction of true params that were predicted at all

    python ft_stemfx/evaluate.py --ckpt /content/drive/.../best_checkpoint.pt \
        --root /content/dataset/run_20k --max-samples 300 --out eval_val.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from load_pretrained import build_model  # noqa: E402
from rig_dataset import RigPairDataset, make_loader  # noqa: E402
from rig_tokenizer import RACK, RigTokenizer  # noqa: E402


def compare(tok: RigTokenizer, pred: dict, true: dict) -> dict:
    pe, te = set(pred["effects"]), set(true["effects"])
    types = [c for c in tok.effect_tokens if c != RACK]
    eff = [(c in pe) == (c in te) for c in types]
    tp = len(pe & te)
    f1 = 2 * tp / max(len(pe) + len(te), 1)
    both_p = [e for e in pred["effects"] if e in te]
    both_t = [e for e in true["effects"] if e in pe]
    common = [k for k in true["params"] if k in pred["params"]]
    disc = [k for k in common if tok.specs[k].get("element") in ("bool", "str")]
    return {
        "chain_exact": float(pred["effects"] == true["effects"]),
        "effect_acc": float(np.mean(eff)),
        "effect_f1": f1,
        "order_ok": float(both_p == both_t),
        "abs_err": [abs(pred["params"][k] - true["params"][k]) for k in common],
        "disc_ok": [pred["values"][k] == true["values"][k] for k in disc],
        "recall": len(common) / max(len(true["params"]), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split-file", default="val.jsonl")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-missing-audio", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    root = Path(a.root)
    tok = RigTokenizer(root / "param_info")
    tok.check_compatible(ck["tokenizer"])
    model = build_model(ck["args"], tok.vocab_size, dropout=0.0)
    model.load_state_dict(ck["model_state_dict"])
    model.to(a.device).eval()

    dc = ck["config"]["data"]
    ds = RigPairDataset(root, a.split_file, tok, sample_rate=dc["sample_rate"],
                        clip_start_sec=dc["clip_start_sec"], clip_duration=dc["clip_duration"],
                        dummy_level_db=dc["dummy_level_db"], max_samples=a.max_samples,
                        skip_missing_audio=a.skip_missing_audio,
                        seed=ck["config"]["train"]["seed"])
    rows = {r["id"]: r for r in ds.rows}
    loader = make_loader(ds, a.batch_size, False, 0)

    per = []
    for batch in loader:
        orig = {k: v.to(a.device) for k, v in batch["original_stems"].items()}
        aug = {k: v.to(a.device) for k, v in batch["augmented_stems"].items()}
        gen, _ = model.generate(orig, aug, max_length=dc["max_seq_len"],
                                bos_token_idx=tok.bos_token_idx, eos_token_idx=tok.eos_token_idx,
                                top_k=1, top_p=1.0)
        for sid, seq in zip(batch["ids"], gen.cpu().tolist()):
            row = rows[sid]
            pred = tok.decode(seq)
            true = tok.target_from_row(row)
            per.append({"id": sid, **compare(tok, pred, true),
                        "pred": pred, "true": true})
        print(f"\r{len(per)}/{len(ds)}", end="", flush=True)
    print()

    errs = [e for p in per for e in p["abs_err"]]
    disc = [d for p in per for d in p["disc_ok"]]
    summary = {
        "n": len(per),
        "chain_exact": float(np.mean([p["chain_exact"] for p in per])),
        "effect_acc": float(np.mean([p["effect_acc"] for p in per])),
        "effect_f1": float(np.mean([p["effect_f1"] for p in per])),
        "order_ok": float(np.mean([p["order_ok"] for p in per])),
        "param_mae": float(np.mean(errs)) if errs else None,
        "discrete_acc": float(np.mean(disc)) if disc else None,
        "param_recall": float(np.mean([p["recall"] for p in per])),
    }
    print(json.dumps(summary, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summary, "samples": per}, indent=1,
                                          ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
