"""
Train StemFX (BSFiLM Encoder + FX Chain Generator) end-to-end on
Sep-Aug paired data. Drop-in entry point for reproducing the paper.

Reads ``configs/default.yaml`` and accepts dotted CLI overrides:

    python train.py \\
        --config configs/default.yaml \\
        data.original_path=/path/to/fma_separated \\
        data.augmented_path=/path/to/fma_separated_augmented \\
        train.device=cuda:1 \\
        train.checkpoint_dir=checkpoints/stemfx_main

Resume:
    python train.py --config configs/default.yaml \\
        train.resume=checkpoints/stemfx_main/latest_checkpoint.pt
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from data.sepaug_dataset import create_dataloaders  # noqa: E402
from stemfx import FXChainLoss, FXChainTokenizer, StemFXNet  # noqa: E402


# --------------------------------------------------------------------------- #
#  Config loading
# --------------------------------------------------------------------------- #
def _set_dotted(cfg: dict, dotted_key: str, value):
    keys = dotted_key.split(".")
    d = cfg
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    # Try to coerce literal types from CLI strings.
    if isinstance(value, str):
        try:
            value = yaml.safe_load(value)
        except yaml.YAMLError:
            pass
    d[keys[-1]] = value


def load_config(argv) -> dict:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", default="configs/default.yaml")
    args, overrides = parser.parse_known_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be 'key.path=value', got {ov!r}")
        k, v = ov.split("=", 1)
        _set_dotted(cfg, k, v)
    return cfg


# --------------------------------------------------------------------------- #
#  Train / validate
# --------------------------------------------------------------------------- #
def train_epoch(model, loader, criterion, optim_, scheduler, device, epoch, writer,
                step: int, cfg) -> tuple:
    model.train()
    optim_.zero_grad()
    accum = cfg["train"]["gradient_accumulation_steps"]
    log_interval = cfg["train"]["log_interval"]
    clip = cfg["train"]["clip_grad_norm"]

    sums = {"loss": 0.0, "ce": 0.0, "mse": 0.0, "acc_cat": 0.0, "acc_val": 0.0}
    pbar = tqdm(loader, desc=f"epoch {epoch}")
    for i, batch in enumerate(pbar):
        orig = {k: v.to(device, non_blocking=True) for k, v in batch["original_stems"].items()}
        aug = {k: v.to(device, non_blocking=True) for k, v in batch["augmented_stems"].items()}
        tokens = batch["tokens"].to(device, non_blocking=True)
        values = batch["values"].to(device, non_blocking=True)
        vmask = batch["value_mask"].to(device, non_blocking=True)

        token_logits, value_preds = model(orig, aug, tokens)
        loss, parts = criterion(token_logits, value_preds, tokens, values, vmask)
        (loss / accum).backward()

        if (i + 1) % accum == 0:
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optim_.step()
            optim_.zero_grad()
            scheduler.step()

        acc = criterion.compute_accuracy(token_logits, tokens, vmask)
        sums["loss"] += loss.item()
        sums["ce"] += parts["ce_categorical"].item()
        sums["mse"] += parts["mse_value"].item()
        sums["acc_cat"] += acc["categorical"]
        sums["acc_val"] += acc["value"]

        if i % log_interval == 0:
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/ce", parts["ce_categorical"].item(), step)
            writer.add_scalar("train/mse", parts["mse_value"].item(), step)
            writer.add_scalar("train/acc_cat", acc["categorical"], step)
            writer.add_scalar("train/acc_val", acc["value"], step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{acc['categorical']:.3f}")
        step += 1

    n = len(loader)
    return {k: v / n for k, v in sums.items()}, step


@torch.no_grad()
def validate(model, loader, criterion, device) -> dict:
    model.eval()
    sums = {"loss": 0.0, "ce": 0.0, "mse": 0.0, "acc_cat": 0.0, "acc_val": 0.0}
    for batch in tqdm(loader, desc="val"):
        orig = {k: v.to(device) for k, v in batch["original_stems"].items()}
        aug = {k: v.to(device) for k, v in batch["augmented_stems"].items()}
        tokens = batch["tokens"].to(device)
        values = batch["values"].to(device)
        vmask = batch["value_mask"].to(device)
        token_logits, value_preds = model(orig, aug, tokens)
        loss, parts = criterion(token_logits, value_preds, tokens, values, vmask)
        acc = criterion.compute_accuracy(token_logits, tokens, vmask)
        sums["loss"] += loss.item()
        sums["ce"] += parts["ce_categorical"].item()
        sums["mse"] += parts["mse_value"].item()
        sums["acc_cat"] += acc["categorical"]
        sums["acc_val"] += acc["value"]
    n = len(loader)
    return {k: v / n for k, v in sums.items()}


def save_ckpt(model, optim_, scheduler, epoch, step, ckpt_dir, cfg, is_best=False):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "global_step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim_.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": cfg,
        # Back-compat with the inference loader (stemfx.load):
        "args": {
            **cfg["model"],
            "no_film": not cfg["model"]["use_film"],
            "encoder_backbone": cfg["model"]["encoder_backbone"],
            "max_seq_len": cfg["data"]["max_seq_len"],
            "sample_rate": cfg["data"]["sample_rate"],
        },
    }
    torch.save(payload, ckpt_dir / f"checkpoint_epoch{epoch}.pt")
    torch.save(payload, ckpt_dir / "latest_checkpoint.pt")
    if is_best:
        torch.save(payload, ckpt_dir / "best_checkpoint.pt")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    cfg = load_config(argv if argv is not None else sys.argv[1:])

    if not cfg["data"].get("original_path") or not cfg["data"].get("augmented_path"):
        raise SystemExit(
            "data.original_path and data.augmented_path must be set "
            "(via YAML or CLI override)."
        )

    torch.manual_seed(cfg["train"]["seed"])
    device = torch.device(cfg["train"]["device"])
    print(f"device = {device}")

    train_loader, val_loader, tokenizer = create_dataloaders(
        original_path=cfg["data"]["original_path"],
        augmented_path=cfg["data"]["augmented_path"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        clip_duration=cfg["data"]["clip_duration"],
        max_seq_len=cfg["data"]["max_seq_len"],
        val_split=cfg["data"]["val_split"],
        scroll_prob=cfg["data"]["scroll_prob"],
        same_segment=cfg["data"]["same_segment"],
        use_ffmpeg=cfg["data"]["use_ffmpeg"],
        mix_stems=cfg["data"]["mix_stems"],
        valid_tracks_path=cfg["data"]["valid_tracks_path"],
        max_samples=cfg["data"]["max_samples"],
        seed=cfg["train"]["seed"],
    )

    m = cfg["model"]
    model = StemFXNet(
        encoder_dim=m["encoder_dim"],
        feature_dim=m["feature_dim"],
        n_mels=m["n_mels"],
        n_fft=m["n_fft"],
        hop_length=m["hop_length"],
        sample_rate=cfg["data"]["sample_rate"],
        use_film=m["use_film"],
        encoder_backbone=m["encoder_backbone"],
        vocab_size=tokenizer.vocab_size,
        hidden_dim=m["hidden_dim"],
        num_layers=m["num_layers"],
        num_heads=m["num_heads"],
        feedforward_dim=m["feedforward_dim"],
        dropout=m["dropout"],
        max_seq_len=cfg["data"]["max_seq_len"],
        pad_token_idx=tokenizer.pad_token_idx,
        bos_token_idx=tokenizer.bos_token_idx,
        num_memory_tokens=m["num_memory_tokens"],
    ).to(device)

    counts = model.count_parameters()
    print(f"params: encoder={counts['encoder']/1e6:.2f}M  "
          f"generator={counts['generator']/1e6:.2f}M  "
          f"total={counts['total']/1e6:.2f}M")

    criterion = FXChainLoss(
        pad_token_idx=tokenizer.pad_token_idx,
        value_token_start=tokenizer.value_token_start,
        value_token_end=tokenizer.value_token_end,
        lambda_mse=cfg["train"]["lambda_mse"],
        no_value_head=cfg["train"]["no_value_head"],
    )

    enc_params = list(model.encoder.parameters())
    gen_params = list(model.generator.parameters())
    optim_ = optim.AdamW(
        [
            {"params": enc_params, "lr": cfg["train"]["learning_rate"] * cfg["train"]["encoder_lr_multiplier"]},
            {"params": gen_params, "lr": cfg["train"]["learning_rate"]},
        ],
        weight_decay=cfg["train"]["weight_decay"],
    )

    steps_per_epoch = math.ceil(len(train_loader) / cfg["train"]["gradient_accumulation_steps"])
    total_steps = steps_per_epoch * cfg["train"]["num_epochs"]
    warmup = cfg["train"]["warmup_steps"]

    def lr_lambda(s):
        if s < warmup:
            return s / max(warmup, 1)
        progress = (s - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(progress * math.pi))

    scheduler = optim.lr_scheduler.LambdaLR(optim_, lr_lambda)

    start_epoch = 1
    step = 0
    best_val = float("inf")
    if cfg["train"]["resume"]:
        ck = torch.load(cfg["train"]["resume"], map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        optim_.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch = ck["epoch"] + 1
        step = ck["global_step"]
        print(f"resumed at epoch {ck['epoch']}, step {step}")

    Path(cfg["train"]["log_dir"]).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(cfg["train"]["log_dir"])
    with open(Path(cfg["train"]["log_dir"]) / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    for epoch in range(start_epoch, cfg["train"]["num_epochs"] + 1):
        train_metrics, step = train_epoch(
            model, train_loader, criterion, optim_, scheduler,
            device, epoch, writer, step, cfg,
        )
        print(f"[epoch {epoch}] train: " +
              " ".join(f"{k}={v:.4f}" for k, v in train_metrics.items()))
        val_metrics = validate(model, val_loader, criterion, device)
        print(f"[epoch {epoch}] val:   " +
              " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
        for k, v in train_metrics.items():
            writer.add_scalar(f"epoch_train/{k}", v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"epoch_val/{k}", v, epoch)

        is_best = val_metrics["loss"] < best_val
        if is_best:
            best_val = val_metrics["loss"]
        if epoch % cfg["train"]["save_interval"] == 0 or is_best:
            save_ckpt(model, optim_, scheduler, epoch, step,
                      cfg["train"]["checkpoint_dir"], cfg, is_best=is_best)

    writer.close()


if __name__ == "__main__":
    main()
