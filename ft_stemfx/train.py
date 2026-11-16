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
  * train.cache_embeddings (encoder frozen for the whole run): the encoder is
    deterministic then (eval mode, fixed dummy stems), so its two 512-d outputs
    are computed once, saved next to the checkpoints and the decoder trains on
    them -> an epoch takes seconds instead of ~20 min
  * train.amp: bf16 autocast when the GPU supports it ("auto"), off on T4
  * train.early_stop_patience: stop after N epochs without improvement of
    train.best_metric (val loss by default); the counter survives resume
  * every epoch also prints the audio-free baseline for mae_val (per-param
    median of train), so "is it learning anything" is visible right away

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
        if isinstance(value, str):          # YAML 1.1 reads "3e-4" as a string
            try:
                value = float(value)
            except ValueError:
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
def encode_pair(model, batch, device):
    """(B, 1024) = [normalize(enc(orig)), normalize(enc(aug))], as StemFX.forward."""
    orig = {k: v.to(device, non_blocking=True) for k, v in batch["original_stems"].items()}
    aug = {k: v.to(device, non_blocking=True) for k, v in batch["augmented_stems"].items()}
    e_o = F.normalize(model._encode_stems(orig), p=2, dim=-1)
    e_a = F.normalize(model._encode_stems(aug), p=2, dim=-1)
    return torch.cat([e_o, e_a], dim=1)


def forward_batch(model, batch, device, encoder_frozen: bool):
    """Same computation as stemfx.model.StemFX.forward, but the encoder can run
    under no_grad when frozen (much less memory, faster), or be skipped when the
    batch carries cached embeddings."""
    tokens = batch["tokens"].to(device, non_blocking=True)
    if "emb" in batch:
        emb = batch["emb"].to(device, non_blocking=True)
    else:
        with torch.no_grad() if encoder_frozen else nullcontext():
            emb = encode_pair(model, batch, device)
    logits, vpred = model.generator(emb, tokens)
    return logits, vpred, tokens


# --------------------------------------------------------------------- embedding cache
class EmbeddingDataset(torch.utils.data.Dataset):
    """Rows of a RigPairDataset with the frozen encoder output instead of audio."""

    def __init__(self, ds: RigPairDataset, emb: torch.Tensor):
        self.ds, self.emb = ds, emb

    def __len__(self):
        return len(self.ds.rows)

    def __getitem__(self, i):
        row = self.ds.rows[i]
        tokens, values, vmask = self.ds.tok.encode(row)
        L = self.ds.max_seq_len
        return {"emb": self.emb[i], "tokens": torch.tensor(tokens[:L]),
                "values": torch.tensor(values[:L], dtype=torch.float32),
                "value_mask": torch.tensor(vmask[:L]), "id": row["id"]}


def emb_collate(batch):
    B, L = len(batch), max(b["tokens"].size(0) for b in batch)
    tokens = torch.zeros(B, L, dtype=torch.long)
    values = torch.zeros(B, L)
    mask = torch.zeros(B, L, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["tokens"].size(0)
        tokens[i, :n], values[i, :n], mask[i, :n] = b["tokens"], b["values"], b["value_mask"]
    return {"emb": torch.stack([b["emb"] for b in batch]), "tokens": tokens,
            "values": values, "value_mask": mask, "ids": [b["id"] for b in batch]}


@torch.no_grad()
def build_embedding_cache(model, ds: RigPairDataset, device, path: Path, batch_size: int,
                          num_workers: int, signature: dict) -> torch.Tensor:
    """Frozen-encoder outputs for every row of ds, cached on disk (Drive) so a
    reconnect does not pay for it again."""
    if path.is_file():
        c = torch.load(path, map_location="cpu", weights_only=False)
        if c.get("signature") == signature and c["ids"] == [r["id"] for r in ds.rows]:
            print(f"embedding cache: {path} ({len(c['ids'])} rows)")
            return c["emb"]
        print(f"embedding cache {path} is stale -> rebuilding")
    model.encoder.eval()
    out = []
    loader = make_loader(ds, batch_size, False, num_workers)
    for batch in tqdm(loader, desc=f"encode {path.stem}", leave=False):
        out.append(encode_pair(model, batch, device).float().cpu())
    emb = torch.cat(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"signature": signature, "ids": [r["id"] for r in ds.rows], "emb": emb}, tmp)
    tmp.replace(path)
    return emb


# --------------------------------------------------------------------------- baselines
def median_baseline_mae(tok: RigTokenizer, train_rows, val_rows) -> float:
    """mae_val of always predicting each param's train median bin (no audio)."""
    import statistics
    def pairs(rows):
        for r in rows:
            t, _, m = tok.encode(r)
            for i, is_v in enumerate(m):
                if is_v:
                    yield t[i - 1], t[i] - tok.value_token_start
    by = {}
    for k, b in pairs(train_rows):
        by.setdefault(k, []).append(b)
    med = {k: statistics.median_low(v) for k, v in by.items()}
    errs = [abs(med.get(k, 50) - b) for k, b in pairs(val_rows)]
    return sum(errs) / max(len(errs), 1) / 100.0


def use_sdpa_attention() -> None:
    """Swap the StemFX axial block's explicit softmax attention for
    F.scaled_dot_product_attention (same weights and math, verified identical
    output). Only matters for encoder_backbone='roformer'; the paper checkpoint
    (barry-mir/stemfx-bsfilm) uses the 'cnn' backbone, where this is a no-op."""
    from stemfx.encoder import AxialTransformerBlock

    def _attention(self, x, q_proj, k_proj, v_proj, out_proj, attn_dropout, rope):
        N, S, D = x.shape
        H, HD = self.num_heads, self.head_dim
        q = rope.apply_rotary(q_proj(x).view(N, S, H, HD)).transpose(1, 2)
        k = rope.apply_rotary(k_proj(x).view(N, S, H, HD)).transpose(1, 2)
        v = v_proj(x).view(N, S, H, HD).transpose(1, 2)
        o = F.scaled_dot_product_attention(
            q, k, v, dropout_p=attn_dropout.p if self.training else 0.0)
        return out_proj(o.transpose(1, 2).reshape(N, S, D))

    AxialTransformerBlock._attention = _attention


def write_status(log_dir: Path, **kw) -> None:
    """Small status.json next to the logs (Colab status cell reads it)."""
    import json
    (log_dir / "status.json").write_text(json.dumps(kw, indent=1, default=float), encoding="utf-8")


def amp_context(mode: str, device):
    """bf16 autocast (same range as fp32 -> safe for the mel / log features)."""
    if device.type != "cuda" or mode in (None, False, "none", "off"):
        return nullcontext
    if mode == "auto" and not torch.cuda.is_bf16_supported():
        return nullcontext
    return lambda: torch.autocast("cuda", dtype=torch.bfloat16)


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
              best_val, cfg, tok: RigTokenizer, model_args: dict, slim: bool = False,
              extra: dict | None = None):
    """slim=True -> model + tokenizer + config only (~1/3 size), for inference."""
    path_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **(extra or {}),
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
    autocast = amp_context(cfg["train"].get("amp", "auto"), device)
    sums = {k: 0.0 for k in ("loss", "ce", "acc_cat", "acc_val", "mae_val")}
    n = 0
    pbar = tqdm(loader, desc=f"{'train' if train else 'val'} {epoch}", leave=False)
    if train:
        optim_.zero_grad()
    for i, batch in enumerate(pbar):
        with nullcontext() if train else torch.no_grad():
            with autocast():
                logits, vpred, tokens = forward_batch(model, batch, device, encoder_frozen)
            logits, vpred = logits.float(), vpred.float()
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
    if tc.get("sdpa_attention", True):
        use_sdpa_attention()

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
    base_mae = median_baseline_mae(tok, train_ds.rows, val_ds.rows)
    print(f"audio-free baseline: val mae_val = {base_mae:.4f} (per-param train median)")

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

    # ---- data: audio, or cached frozen-encoder outputs (encoder never changes in
    # such a run, so a cache built before a disconnect stays valid after resume)
    use_cache = (bool(tc.get("cache_embeddings")) and bool(tc["freeze_encoder"])
                 and not tc.get("unfreeze_at_epoch"))
    if use_cache:
        sig = {"pretrained": str(cfg["model"]["pretrained_checkpoint"]),
               "clip": [dc["clip_start_sec"], dc["clip_duration"]],
               "dummy_db": dc["dummy_level_db"], "seed": tc["seed"]}
        embs = {name: build_embedding_cache(model, d, device, ckpt_dir / f"emb_cache_{name}.pt",
                                            tc["batch_size"], dc["num_workers"], sig)
                for name, d in (("train", train_ds), ("val", val_ds))}
        mk = lambda d, e, sh: torch.utils.data.DataLoader(  # noqa: E731
            EmbeddingDataset(d, e), batch_size=tc["batch_size"], shuffle=sh,
            collate_fn=emb_collate, drop_last=sh)
        train_loader, val_loader = mk(train_ds, embs["train"], True), mk(val_ds, embs["val"], False)
    else:
        train_loader = make_loader(train_ds, tc["batch_size"], True, dc["num_workers"])
        val_loader = make_loader(val_ds, tc["batch_size"], False, dc["num_workers"])
    amp_on = amp_context(tc.get("amp", "auto"), device) is not nullcontext
    print(f"embedding cache: {'on' if use_cache else 'off'} | bf16 autocast: {'on' if amp_on else 'off'}")

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

    start_epoch, step, best_val, bad_epochs = 1, 0, float("inf"), 0
    if ck is not None:
        optim_.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch, step = ck["epoch"] + 1, ck["global_step"]
        best_val = ck.get("best_val", best_val)
        bad_epochs = ck.get("bad_epochs", 0)
        del ck
    best_metric = tc.get("best_metric", "loss")          # val metric, lower is better
    patience = tc.get("early_stop_patience")             # None/0 -> off
    min_delta = float(tc.get("early_stop_min_delta", 0.0))

    log_dir = Path(tc["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    tok.save(log_dir / "tokenizer.json")
    writer = SummaryWriter(str(log_dir))

    unfreeze_at = tc.get("unfreeze_at_epoch")
    writer.add_scalar("baseline/val_mae_median", base_mae, 0)
    if patience and bad_epochs >= patience:
        print(f"early stop already reached ({bad_epochs} epochs without improvement)")
        start_epoch = tc["num_epochs"] + 1
    for epoch in range(start_epoch, tc["num_epochs"] + 1):
        frozen = bool(tc["freeze_encoder"]) and not (unfreeze_at and epoch >= unfreeze_at)
        set_encoder_trainable(model, not frozen and not use_cache)
        t0 = time.time()
        tr, step = run_epoch(model, train_loader, criterion, tok, device, cfg, True, frozen,
                             optim_, scheduler, writer, step, epoch)
        va, _ = run_epoch(model, val_loader, criterion, tok, device, cfg, False, True,
                          epoch=epoch)
        fmt = lambda d: " ".join(f"{k}={v:.4f}" for k, v in d.items())  # noqa: E731
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
        print(f"[epoch {epoch}{' (enc frozen)' if frozen else ''}] {time.time() - t0:.0f}s"
              f"  lr {scheduler.get_last_lr()[-1]:.2e}\n"
              f"  train: {fmt(tr)}\n  val:   {fmt(va)}  (baseline mae {base_mae:.4f})\n"
              f"  best val {best_metric} {best_val:.4f}"
              + ("  <- new best" if is_best else f"  ({bad_epochs}/{patience or '-'} without improvement)"))

        extra = {"bad_epochs": bad_epochs, "best_metric": best_metric, "val": va,
                 "baseline_mae": base_mae}
        args = (ckpt_dir, model, optim_, scheduler, epoch, step, best_val, cfg, tok, model_args)
        save_ckpt(args[0], "latest_checkpoint.pt", *args[1:], extra=extra)
        if is_best:
            save_ckpt(args[0], "best_checkpoint.pt", *args[1:], extra=extra)
            save_ckpt(args[0], "best_model.pt", *args[1:], slim=True, extra=extra)  # infer.py
        if epoch % tc["save_interval"] == 0:
            save_ckpt(args[0], f"checkpoint_epoch{epoch}.pt", *args[1:], extra=extra)
        stop = bool(patience and bad_epochs >= patience)
        write_status(log_dir, epoch=epoch, num_epochs=tc["num_epochs"], best_metric=best_metric,
                     best_val=best_val, bad_epochs=bad_epochs, early_stopped=stop,
                     finished=stop or epoch == tc["num_epochs"], val=va, train=tr,
                     baseline_mae=base_mae, epoch_sec=time.time() - t0)
        if stop:
            print(f"early stop: no {best_metric} improvement for {patience} epochs "
                  f"(best {best_val:.4f})")
            break
    writer.close()


if __name__ == "__main__":
    main()
