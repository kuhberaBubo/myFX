"""
Train testFX (swappable encoder + StemFX decoder) on Guitar Rig pairs.

Same recipe as ft_stemfx/train.py (AdamW, warmup + cosine, FXChainLoss with CE
over all tokens, teacher forcing, early stopping, resume=auto, bf16 autocast,
audio-free baseline printed every epoch), with three lr groups:
    decoder (warm-started from StemFX)           learning_rate
    new encoder parts / memory projections       learning_rate * new_lr_multiplier
    pretrained encoder parts (MERT top layers)    learning_rate * pretrained_encoder_lr_multiplier

    python testFx/train_tfx.py --config testFx/configs/tfx_g.yaml \
        data.root=/content/run_20k \
        train.checkpoint_dir=/content/drive/MyDrive/myFX/ckpt/tfx_g \
        train.log_dir=/content/drive/MyDrive/myFX/logs/tfx_g
"""

from __future__ import annotations

import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tfx_model import FT as FT_DIR, build_testfx  # noqa: E402  (puts ft_stemfx + stemfx on sys.path)
from rig_tokenizer import RigTokenizer  # noqa: E402


def _load_ft_train():
    """ft_stemfx/train.py by path (`import train` would hit stemfx/train.py)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ft_stemfx_train", FT_DIR / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FT = _load_ft_train()   # helpers: load_config, save_ckpt, amp_context, baselines
from stemfx.losses import FXChainLoss    # noqa: E402
from tfx_data import PairDataset, make_loader  # noqa: E402


def n_params(ps):
    return sum(p.numel() for p in ps) / 1e6


def run_epoch(model, loader, criterion, tok, device, cfg, train, optim_=None, scheduler=None,
              writer=None, step=0, epoch=0):
    model.train(train)
    tc = cfg["train"]
    accum, clip = tc["gradient_accumulation_steps"], tc["clip_grad_norm"]
    autocast = FT.amp_context(tc.get("amp", "auto"), device)
    sums = {k: 0.0 for k in ("loss", "acc_cat", "acc_val", "mae_val")}
    n = 0
    pbar = tqdm(loader, desc=f"{'train' if train else 'val'} {epoch}", leave=False)
    if train:
        optim_.zero_grad()
    for i, b in enumerate(pbar):
        dry, wet = b["dry"].to(device, non_blocking=True), b["wet"].to(device, non_blocking=True)
        tokens = b["tokens"].to(device)
        with nullcontext() if train else torch.no_grad():
            with autocast():
                logits, vpred = model(dry, wet, tokens)
            logits, vpred = logits.float(), vpred.float()
            vmask = b["value_mask"].to(device)
            loss, _ = criterion(logits, vpred, tokens, b["values"].to(device), vmask)
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
        mae = FT.value_bin_mae(logits, tokens, tok)
        sums["loss"] += loss.item()
        sums["acc_cat"] += acc["categorical"]
        sums["acc_val"] += acc["value"]
        sums["mae_val"] += mae
        n += 1
        if train:
            if i % tc["log_interval"] == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/mae_val", mae, step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
            step += 1
        pbar.set_postfix(loss=f"{loss.item():.3f}", cat=f"{acc['categorical']:.3f}",
                         mae=f"{mae:.3f}")
    return {k: v / max(n, 1) for k, v in sums.items()}, step


def main(argv=None):
    cfg = FT.load_config(argv if argv is not None else sys.argv[1:])
    dc, tc, mc = cfg["data"], cfg["train"], cfg["model"]
    torch.manual_seed(tc["seed"])
    device = torch.device(tc["device"] if torch.cuda.is_available() or
                          not str(tc["device"]).startswith("cuda") else "cpu")
    print(f"device = {device} | encoder = {mc['encoder']['kind']}")

    root = Path(dc["root"])
    tok = RigTokenizer(root / "param_info")
    kw = dict(sample_rate=dc["sample_rate"], clip_start_sec=dc["clip_start_sec"],
              clip_duration=dc["clip_duration"], max_seq_len=dc["max_seq_len"],
              skip_missing_audio=dc["skip_missing_audio"])
    train_ds = PairDataset(root, dc["train_file"], tok, max_samples=dc["max_train_samples"], **kw)
    val_ds = PairDataset(root, dc["val_file"], tok, max_samples=dc["max_val_samples"], **kw)
    base_mae = FT.median_baseline_mae(tok, train_ds.rows, val_ds.rows)
    print(f"train = {len(train_ds)}  val = {len(val_ds)}  clip {dc['clip_duration']} s")
    print(f"audio-free baseline: val mae_val = {base_mae:.4f} (per-param train median)")
    train_loader = make_loader(train_ds, tc["batch_size"], True, dc["num_workers"])
    val_loader = make_loader(val_ds, tc["batch_size"], False, dc["num_workers"])

    ckpt_dir = Path(tc["checkpoint_dir"])
    resume = tc.get("resume")
    if resume == "auto":
        cand = ckpt_dir / "latest_checkpoint.pt"
        resume = str(cand) if cand.is_file() else None
    ck = torch.load(resume, map_location="cpu", weights_only=False) if resume else None

    model, info = build_testfx(mc, tok, warm_start=ck is None)
    if ck is not None:
        tok.check_compatible(ck["tokenizer"])
        model.load_state_dict(ck["model_state_dict"])
        print(f"resuming from {resume} (epoch {ck['epoch']})")
    elif info:
        print(f"decoder warm start from StemFX: {info['tensors_loaded']} tensors, "
              f"{info['rows_copied']} token rows copied, {info['rows_random']} new")
    model.to(device)

    groups = model.param_groups(tc["learning_rate"], tc["new_lr_multiplier"],
                                tc["pretrained_encoder_lr_multiplier"], tc["weight_decay"])
    for g in groups:
        print(f"  {g['name']:20s} {n_params(g['params']):7.2f} M params  lr {g['lr']:.1e}")
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad) / 1e6
    if frozen:
        print(f"  {'frozen':20s} {frozen:7.2f} M params")
    optim_ = torch.optim.AdamW(groups)
    steps_per_epoch = math.ceil(len(train_loader) / tc["gradient_accumulation_steps"])
    total, warm = steps_per_epoch * tc["num_epochs"], tc["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim_, lambda s: s / max(warm, 1) if s < warm
        else 0.5 * (1 + math.cos(math.pi * min((s - warm) / max(total - warm, 1), 1.0))))

    criterion = FXChainLoss(pad_token_idx=tok.pad_token_idx,
                            value_token_start=tok.value_token_start,
                            value_token_end=tok.value_token_end,
                            lambda_mse=tc["lambda_mse"], no_value_head=tc["no_value_head"])

    start_epoch, step, best_val, bad_epochs = 1, 0, float("inf"), 0
    if ck is not None:
        optim_.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch, step = ck["epoch"] + 1, ck["global_step"]
        best_val, bad_epochs = ck.get("best_val", best_val), ck.get("bad_epochs", 0)
        del ck
    best_metric = tc.get("best_metric", "loss")
    patience = tc.get("early_stop_patience")
    min_delta = float(tc.get("early_stop_min_delta", 0.0))

    log_dir = Path(tc["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    writer = FT.SummaryWriter(str(log_dir))
    writer.add_scalar("baseline/val_mae_median", base_mae, 0)
    if patience and bad_epochs >= patience:
        print(f"early stop already reached ({bad_epochs} epochs without improvement)")
        start_epoch = tc["num_epochs"] + 1

    for epoch in range(start_epoch, tc["num_epochs"] + 1):
        t0 = time.time()
        tr, step = run_epoch(model, train_loader, criterion, tok, device, cfg, True,
                             optim_, scheduler, writer, step, epoch)
        va, _ = run_epoch(model, val_loader, criterion, tok, device, cfg, False, epoch=epoch)
        for k, v in tr.items():
            writer.add_scalar(f"epoch_train/{k}", v, epoch)
        for k, v in va.items():
            writer.add_scalar(f"epoch_val/{k}", v, epoch)
        score = va[best_metric]
        is_best = score < best_val - min_delta
        if is_best:
            best_val, bad_epochs = score, 0
        else:
            bad_epochs += 1
        fmt = lambda d: " ".join(f"{k}={v:.4f}" for k, v in d.items())  # noqa: E731
        print(f"[epoch {epoch}] {time.time() - t0:.0f}s  lr {scheduler.get_last_lr()[0]:.2e}\n"
              f"  train: {fmt(tr)}\n  val:   {fmt(va)}  (baseline mae {base_mae:.4f})\n"
              f"  best val {best_metric} {best_val:.4f}"
              + ("  <- new best" if is_best else f"  ({bad_epochs}/{patience or '-'} without improvement)"))

        extra = {"model_kind": "testfx", "bad_epochs": bad_epochs, "best_metric": best_metric,
                 "val": va, "baseline_mae": base_mae}
        args = (ckpt_dir, model, optim_, scheduler, epoch, step, best_val, cfg, tok, mc)
        FT.save_ckpt(args[0], "latest_checkpoint.pt", *args[1:], extra=extra)
        if is_best:
            FT.save_ckpt(args[0], "best_checkpoint.pt", *args[1:], extra=extra)
            FT.save_ckpt(args[0], "best_model.pt", *args[1:], slim=True, extra=extra)
        if epoch % tc["save_interval"] == 0:
            FT.save_ckpt(args[0], f"checkpoint_epoch{epoch}.pt", *args[1:], extra=extra)
        stop = bool(patience and bad_epochs >= patience)
        FT.write_status(log_dir, epoch=epoch, num_epochs=tc["num_epochs"], best_metric=best_metric,
                        best_val=best_val, bad_epochs=bad_epochs, early_stopped=stop,
                        finished=stop or epoch == tc["num_epochs"], val=va, train=tr,
                        baseline_mae=base_mae, epoch_sec=time.time() - t0)
        if stop:
            print(f"early stop: no {best_metric} improvement for {patience} epochs (best {best_val:.4f})")
            break
    writer.close()


if __name__ == "__main__":
    main()
