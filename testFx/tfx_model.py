"""
testFX model = swappable encoder (tfx_encoders) + StemFX decoder.

The decoder is StemFX's FXChainGenerator unchanged (6 layers, 512, 8 heads),
except that cross-attention reads a variable-length memory from the encoder
instead of the two projected 512-d embeddings. Its weights are warm-started
from the StemFX paper checkpoint exactly as in ft_stemfx (transformer layers,
value head, positional encoding; special / stem / VALUE token rows copied, effect
and param rows new), so every experiment starts from the same decoder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
FT = HERE.parent / "ft_stemfx"
for p in (str(HERE), str(FT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import load_pretrained as LP                     # noqa: E402  (puts stemfx on sys.path)
from stemfx.generator import FXChainGenerator    # noqa: E402
from tfx_encoders import build_encoder           # noqa: E402

DECODER_DEFAULTS = dict(hidden_dim=512, num_layers=6, num_heads=8, feedforward_dim=2048,
                        max_seq_len=512)


class MemoryGenerator(FXChainGenerator):
    """FXChainGenerator whose cross-attention memory is given directly."""

    def forward(self, memory, target_tokens):
        B, L = target_tokens.shape
        bos = torch.full((B, 1), self.bos_token_idx, dtype=torch.long, device=target_tokens.device)
        inp = torch.cat([bos, target_tokens[:, :-1]], 1)
        x = self.pos_encoding(self.token_embedding(inp))
        mask = self._generate_square_subsequent_mask(L).to(x.device)
        h = self.transformer_decoder(tgt=x, memory=memory, tgt_mask=mask,
                                     tgt_key_padding_mask=(inp == self.pad_token_idx))
        logits = self.token_head(h)
        vpred = self.value_head(torch.cat([h, self.token_embedding(inp)], -1)).squeeze(-1)
        return logits, vpred.clamp(0.0, 1.0)

    @torch.no_grad()
    def generate(self, memory, max_length=512, bos_token_idx=1, eos_token_idx=2):
        """Greedy decoding."""
        B = memory.size(0)
        seq = torch.full((B, 1), bos_token_idx, dtype=torch.long, device=memory.device)
        done = torch.zeros(B, dtype=torch.bool, device=memory.device)
        for _ in range(max_length - 1):
            x = self.pos_encoding(self.token_embedding(seq))
            mask = self._generate_square_subsequent_mask(seq.size(1)).to(x.device)
            h = self.transformer_decoder(tgt=x, memory=memory, tgt_mask=mask)
            nxt = self.token_head(h[:, -1]).argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, eos_token_idx), nxt)
            seq = torch.cat([seq, nxt[:, None]], 1)
            done |= nxt == eos_token_idx
            if done.all():
                break
        return seq


class TestFX(nn.Module):
    """memory = gate * LayerNorm(encoder(dry, wet)).

    The gate starts small (memory_gate_init, 0.1): a freshly initialised encoder
    emits ~1000 noise-like tokens, and fed at full scale they corrupt every
    cross-attention layer of the pretrained decoder (measured: 3-sample
    memorisation reaches loss 1.9 after 200 steps vs 0.05 with an empty memory).
    With a small gate the decoder first learns the chain grammar, and the
    learnable gate grows as the encoder output becomes useful."""

    def __init__(self, encoder: nn.Module, decoder: MemoryGenerator, gate_init: float = 0.1):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        hd = decoder.hidden_dim
        self.mem_norm = nn.LayerNorm(hd)
        self.mem_gate = nn.Parameter(torch.tensor(float(gate_init)))

    def memory(self, dry, wet):
        return self.mem_gate * self.mem_norm(self.encoder(dry, wet))

    def forward(self, dry, wet, tokens):
        return self.decoder(self.memory(dry, wet), tokens)

    @torch.no_grad()
    def generate(self, dry, wet, max_length=512, bos_token_idx=1, eos_token_idx=2):
        return self.decoder.generate(self.memory(dry, wet), max_length, bos_token_idx,
                                     eos_token_idx)

    def param_groups(self, lr: float, new_mult: float, pretrained_enc_mult: float, wd: float):
        """decoder: lr | new encoder parts + memory norm: lr*new_mult |
        pretrained encoder parts (MERT top layers): lr*pretrained_enc_mult"""
        pre = {id(p) for p in self.encoder.pretrained_parameters()}
        enc_new = [p for p in self.encoder.parameters() if p.requires_grad and id(p) not in pre]
        enc_pre = [p for p in self.encoder.parameters() if p.requires_grad and id(p) in pre]
        groups = [{"params": list(self.decoder.parameters()), "lr": lr, "name": "decoder"},
                  {"params": enc_new + list(self.mem_norm.parameters()) + [self.mem_gate],
                   "lr": lr * new_mult,
                   "name": "encoder_new"}]
        if enc_pre:
            groups.append({"params": enc_pre, "lr": lr * pretrained_enc_mult,
                           "name": "encoder_pretrained"})
        for g in groups:
            g["weight_decay"] = wd
        return groups


def build_decoder(tok, dropout: float = 0.1) -> MemoryGenerator:
    d = DECODER_DEFAULTS
    return MemoryGenerator(vocab_size=tok.vocab_size, encoder_dim=d["hidden_dim"],
                           hidden_dim=d["hidden_dim"], num_layers=d["num_layers"],
                           num_heads=d["num_heads"], feedforward_dim=d["feedforward_dim"],
                           dropout=dropout, max_seq_len=d["max_seq_len"],
                           pad_token_idx=tok.pad_token_idx, bos_token_idx=tok.bos_token_idx)


def warm_start_decoder(dec: MemoryGenerator, tok, checkpoint: str | None = None) -> dict:
    """Load StemFX generator weights (same recipe as ft_stemfx.load_pretrained)."""
    ck = torch.load(LP.resolve_checkpoint(checkpoint), map_location="cpu", weights_only=False)
    old = {k[len("generator."):]: v for k, v in ck["model_state_dict"].items()
           if k.startswith("generator.")}
    vocab_keys = {k[len("generator."):] for k in LP.VOCAB_KEYS}
    skip = vocab_keys | {k for k in old if k.startswith("embedding_proj.")} | {"memory_pos_embedding"}
    transfer = {k: v for k, v in old.items() if k not in skip}
    own = dec.state_dict()
    bad = [k for k, v in transfer.items() if k not in own or own[k].shape != v.shape]
    if bad:
        raise RuntimeError(f"decoder shape mismatch vs StemFX checkpoint: {bad[:5]}")
    dec.load_state_dict(transfer, strict=False)
    old_vocab = old["token_embedding.weight"].shape[0]
    rows = LP.rig_row_map(old_vocab, tok.value_token_start)
    with torch.no_grad():
        for n, o in rows.items():
            dec.token_embedding.weight[n] = old["token_embedding.weight"][o]
            dec.token_head.weight[n] = old["token_head.weight"][o]
            dec.token_head.bias[n] = old["token_head.bias"][o]
        dec.token_embedding.weight[tok.pad_token_idx].zero_()
    return {"tensors_loaded": len(transfer), "rows_copied": len(rows),
            "rows_random": tok.vocab_size - len(rows)}


def build_testfx(model_cfg: dict, tok, warm_start: bool = True) -> tuple[TestFX, dict]:
    dec = build_decoder(tok, model_cfg.get("dropout", 0.1))
    info = {}
    if warm_start and model_cfg.get("decoder_init", "stemfx") == "stemfx":
        info = warm_start_decoder(dec, tok, model_cfg.get("pretrained_checkpoint"))
    enc = build_encoder(dict(model_cfg["encoder"]), dec.hidden_dim)
    return TestFX(enc, dec, model_cfg.get("memory_gate_init", 0.1)), info


def load_testfx(ck: dict, device="cpu"):
    """Rebuild from a testFX checkpoint (render_eval / infer)."""
    from rig_tokenizer import RigTokenizer
    tok = RigTokenizer.from_checkpoint(ck["tokenizer"])
    model, _ = build_testfx(ck["config"]["model"], tok, warm_start=False)
    model.load_state_dict(ck["model_state_dict"])
    return model.to(device).eval(), tok
