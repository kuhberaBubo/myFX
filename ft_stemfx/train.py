"""
Fine-tune the pretrained StemFX checkpoint on Guitar Rig (DI -> rendered) pairs.

Same training recipe as stemfx/train.py (AdamW, warmup + cosine, FXChainLoss,
teacher forcing); differences:
  * RigTokenizer / RigPairDataset instead of the Sep-Aug pipeline
  * model warm-started from the paper checkpoint with a new vocabulary
  * optional frozen encoder stage (train.freeze_encoder / unfreeze_at_epoch);
    while frozen, the encoder runs under no_grad and in eval mode
  * train.resume=auto picks up <checkpoint_dir>/latest_checkpoint.pt
    (for Colab disconnects); the tokenizer vocab is stored in every checkpoint

    python ft_stemfx/train.py --config ft_stemfx/configs/ft_default.yaml \
        data.root=/content/dataset/run_20k \
        train.checkpoint_dir=/content/drive/MyDrive/myFX/ckpt/ft_v1
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from load_pretrained import (  # noqa: E402  (also puts stemfx on sys.path)
    build_model, load_pretrained_with_new_vocab, set_encoder_trainable,
)
from stemfx.losses import FXChainLoss  # noqa: E402
from rig_dataset import RigPairDataset, make_loader  # noqa: E402
from rig_tokenizer import RigTokenizer  # noqa: E402

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard not installed -> no-op writer
    class SummaryWriter:  # type: ignore
        def __init__(self, *a, **k): pass
        def add_scalar(self, *a, **k): pass
        def close(self): pass


# --------------------------------------------------------------------------- config
def _set_dotted(cfg: dict, dotted: str, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    if isinstance(value, str):
        try:
            value = yaml.safe_load(value)
        except yaml.YAMLError:
            pass
    d[keys[-1]] = value


def load_config(argv) -> dict:
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--config", default=str(HERE / "configs" / "ft_default.yaml"))
    a, overrides = p.parse_known_args(argv)
    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be key.path=value, got {ov!r}")
        k, v = ov.split("=", 1)
        _set_dotted(cfg, k, v)
    return cfg


# --------------------------------------------------------------------------- model io
def forward_batch(model, batch, device, encoder_frozen: bool):
    """Same computation as stemfx.model.StemFX.forward, but the encoder can run
    under no_grad when frozen (much less memory, faster)."""
    orig = {k: v.to(device, non_blocking=True) for k, v in batch["original_stems"].items()}
    aug = {k: v.to(device, non_blocking=True) for k, v in batch["augmented_stems"].items()}
    tokens = batch["tokens"].to(device, non_blocking=True)
    with torch.no_grad() if encoder_frozen else nullcontext():
        e_o = F.normalize(model._encode_stems(orig), p=2, dim=-1)
        e_a = F.normalize(model._encode_stems(aug), p=2, dim=-1)
    logits, vpred = model.generator(torch.cat([e_o, e_a], dim=1), tokens)
    return logits, vpred, tokens


@torch.no_grad()
def value_bin_mae(logits, tokens, tok: RigTokenizer) -> float:
    """Teacher-forced |pred_bin - true_bin| / 100 at VALUE positions, with the
    prediction restricted to the VALUE range. ~ mean knob error in raw [0,1]."""
    vs, ve = tok.value_token_start, tok.value_token_end
    is_v = (tokens >= vs) & (tokens <= ve)
    if not is_v.any():
        return 0.0
    pred = logits[..., vs:ve + 1].argmax(-1)
    return ((pred[is_v] - (tokens[is_v] - vs)).abs().float().mean() / 100.0).item()


def save_ckpt(path_dir: Path, name: str, model, optim_, scheduler, epoch, step,
              best_val, cfg, tok: RigTokenizer, model_args: dict, slim: bool = False):
    """slim=True -> model + tokenizer + config only (~1/3 size), for inference."""
    path_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch, "global_step": step, "best_val": best_val,
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "args": model_args,                 # StemFX hyper-params (build_model)
        "tokenizer": tok.state_dict(),      # incl. param_info -> infer.py needs no dataset
    }
    if not slim:
        payload["optimizer_state_dict"] = optim_.state_dict()
        payload["scheduler_state_dict"] = scheduler.state_dict()
    tmp = path_dir / (name + ".tmp")
    torch.save(payload, tmp)                # write-then-rename: safe on disconnect
    tmp.replace(path_dir / name)


# --------------------------------------------------------------------------- loops
def run_epoch(model, loader, criterion, tok, device, cfg, train: bool, encoder_frozen: bool,
              optim_=None, scheduler=None, writer=None, step=0, epoch=0):
    model.train(train)
    if encoder_frozen:
        model.encoder.eval()               # keep BN/IN stats from the checkpoint
    accum = cfg["train"]["gradient_accumulation_steps"]
    clip = cfg["train"]["clip_grad_norm"]
    sums = {k: 0.0 for k in ("loss", "ce", "acc_cat", "acc_val", "mae_val")}
    n = 0
    pbar = tqdm(loader, desc=f"{'train' if train else 'val'} {epoch}", leave=False)
    if train:
        optim_.zero_grad()
    for i, batch in enumerate(pbar):
        with nullcontext() if train else torch.no_grad():
            logits, vpred, tokens = forward_batch(model, batch, device, encoder_frozen)
            values = batch["values"].to(device)
            vmask = batch["value_mask"].to(device)
            loss, parts = criterion(logits, vpred, tokens, values, vmask)
        if train:
            (loss / accum).backward()
            if (i + 1) % accum == 0:
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], clip)
                optim_.step()
                optim_.zero_grad()
                scheduler.step()
        acc = criterion.compute_accuracy(logits, tokens, vmask)
        mae = value_bin_mae(logits, tokens, tok)
        sums["loss"] += loss.item()
        sums["ce"] += parts["ce_categorical"].item()
        sums["acc_cat"] += acc["categorical"]
        sums["acc_val"] += acc["value"]
        sums["mae_val"] += mae
        n += 1
        if train:
            if i % cfg["train"]["log_interval"] == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/acc_cat", acc["categorical"], step)
                writer.add_scalar("train/mae_val", mae, step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[-1], step)
            step += 1
        pbar.set_postfix(loss=f"{loss.item():.3f}", cat=f"{acc['categorical']:.3f}",
                         mae=f"{mae:.3f}")
    return {k: v / max(n, 1) for k, v in sums.items()}, step


# --------------------------------------------------------------------------- main
def main(argv=None):
    cfg = load_config(argv if argv is not None else sys.argv[1:])
    dc, tc = cfg["data"], cfg["train"]
    torch.manual_seed(tc["seed"])
    device = torch.device(tc["device"] if torch.cuda.is_available() or
                          not str(tc["device"]).startswith("cuda") else "cpu")
    print(f"device = {device}")

    root = Path(dc["root"])
    tok = RigTokenizer(root / "param_info")
    print(f"vocab = {tok.vocab_size}  (effects {len(tok.effect_tokens)}, "
          f"params {len(tok.param_tokens)}, VALUE {tok.value_token_start}..{tok.value_token_end})")

    ds_kw = dict(tokenizer=tok, sample_rate=dc["sample_rate"],
                 clip_start_sec=dc["clip_start_sec"], clip_duration=dc["clip_duration"],
                 dummy_level_db=dc["dummy_level_db"], max_seq_len=dc["max_seq_len"],
                 skip_missing_audio=dc["skip_missing_audio"], seed=tc["seed"])
    train_ds = RigPairDataset(root, dc["train_file"], max_samples=dc["max_train_samples"], **ds_kw)
    val_ds = RigPairDataset(root, dc["val_file"], max_samples=dc["max_val_samples"], **ds_kw)
    print(f"train = {len(train_ds)}  val = {len(val_ds)}")
    train_loader = make_loader(train_ds, tc["batch_size"], True, dc["num_workers"])
    val_loader = make_loader(val_ds, tc["batch_size"], False, dc["num_workers"])

    # ---- resume target
    ckpt_dir = Path(tc["checkpoint_dir"])
    resume = tc.get("resume")
    if resume == "auto":
        cand = ckpt_dir / "latest_checkpoint.pt"
        resume = str(cand) if cand.is_file() else None
    ck = torch.load(resume, map_location="cpu", weights_only=False) if resume else None

    # ---- model
    if ck is not None:
        tok.check_compatible(ck["tokenizer"])
        model_args = ck["args"]
        model = build_model(model_args, tok.vocab_size, tok.pad_token_idx,
                            tok.bos_token_idx, cfg["model"]["dropout"])
        model.load_state_dict(ck["model_state_dict"])
        print(f"resuming from {resume} (epoch {ck['epoch']})")
    else:
        model, info = load_pretrained_with_new_vocab(
            tok, cfg["model"]["pretrained_checkpoint"], cfg["model"]["dropout"])
        model_args = info["args"]
        print(f"warm start: {info['tensors_loaded']} tensors loaded, vocab "
              f"{info['old_vocab_size']} -> {info['new_vocab_size']} "
              f"({info['rows_copied']} rows copied, {info['rows_random']} new)")
    model.to(device)

    criterion = FXChainLoss(
        pad_token_idx=tok.pad_token_idx,
        value_token_start=tok.value_token_start,
        value_token_end=tok.value_token_end,
        lambda_mse=tc["lambda_mse"],
        no_value_head=tc["no_value_head"],
    )

    optim_ = torch.optim.AdamW(
        [{"params": list(model.encoder.parameters()),
          "lr": tc["learning_rate"] * tc["encoder_lr_multiplier"]},
         {"params": list(model.generator.parameters()), "lr": tc["learning_rate"]}],
        weight_decay=tc["weight_decay"])
    steps_per_epoch = math.ceil(len(train_loader) / tc["gradient_accumulation_steps"])
    total = steps_per_epoch * tc["num_epochs"]
    warm = tc["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim_, lambda s: s / max(warm, 1) if s < warm
        else 0.5 * (1 + math.cos(math.pi * min((s - warm) / max(total - warm, 1), 1.0))))

    start_epoch, step, best_val = 1, 0, float("inf")
    if ck is not None:
        optim_.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch, step = ck["epoch"] + 1, ck["global_step"]
        best_val = ck.get("best_val", best_val)
        del ck

    log_dir = Path(tc["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    tok.save(log_dir / "tokenizer.json")
    writer = SummaryWriter(str(log_dir))

    unfreeze_at = tc.get("unfreeze_at_epoch")
    for epoch in range(start_epoch, tc["num_epochs"] + 1):
        frozen = bool(tc["freeze_encoder"]) and not (unfreeze_at and epoch >= unfreeze_at)
        set_encoder_trainable(model, not frozen)
        t0 = time.time()
        tr, step = run_epoch(model, train_loader, criterion, tok, device, cfg, True, frozen,
                             optim_, scheduler, writer, step, epoch)
        va, _ = run_epoch(model, val_loader, criterion, tok, device, cfg, False, True,
                          epoch=epoch)
        fmt = lambda d: " ".join(f"{k}={v:.4f}" for k, v in d.items())  # noqa: E731
        print(f"[epoch {epoch}{' (enc frozen)' if frozen else ''}] {time.time() - t0:.0f}s\n"
              f"  train: {fmt(tr)}\n  val:   {fmt(va)}")
        for k, v in tr.items():
            writer.add_scalar(f"epoch_train/{k}", v, epoch)
        for k, v in va.items():
            writer.add_scalar(f"epoch_val/{k}", v, epoch)

        is_best = va["loss"] < best_val
        best_val = min(best_val, va["loss"])
        args = (ckpt_dir, model, optim_, scheduler, epoch, step, best_val, cfg, tok, model_args)
        save_ckpt(args[0], "latest_checkpoint.pt", *args[1:])
        if is_best:
            save_ckpt(args[0], "best_checkpoint.pt", *args[1:])
            save_ckpt(args[0], "best_model.pt", *args[1:], slim=True)   # for infer.py
        if epoch % tc["save_interval"] == 0:
            save_ckpt(args[0], f"checkpoint_epoch{epoch}.pt", *args[1:])
    writer.close()


if __name__ == "__main__":
    main()
