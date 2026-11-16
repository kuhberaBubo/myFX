"""
Greedy chain decoding with a choice of how knob VALUE tokens are picked.

    argmax  most likely VALUE bin (what StemFX's generate(top_k=1) does)
    median  median of the predicted distribution over the 101 VALUE bins

The model is trained with cross-entropy over 101 unordered bins, so the argmax bin
is the *mode* of what it believes; under an absolute-error metric the median is the
better point estimate (and is what the audio-free baseline uses). Only continuous
knobs (number / curve) get the median; on/off and option switches (bool / str) keep
argmax. Structure tokens (effects, param names, SEP/EOS) are always argmax, so with
value_mode="argmax" this reproduces the original greedy decoding exactly.

Works for both decoders (StemFX FXChainGenerator and testFX MemoryGenerator): they
share token_embedding / pos_encoding / transformer_decoder / token_head.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

VALUE_MODES = ("argmax", "median")


def continuous_param_mask(tok, device) -> torch.Tensor:
    """(vocab,) bool: True for param tokens whose next VALUE is a continuous knob."""
    m = torch.zeros(tok.vocab_size, dtype=torch.bool)
    for k in tok.param_tokens:
        if tok.specs[k].get("element") not in ("bool", "str"):
            m[tok.token_to_idx[k]] = True
    return m.to(device)


def stemfx_memory(model, original_stems, augmented_stems) -> torch.Tensor:
    """StemFX encoder -> the 2-token decoder memory (same as FXChainGenerator.generate)."""
    gen = model.generator
    e_o = F.normalize(model._encode_stems(original_stems), p=2, dim=-1)
    e_a = F.normalize(model._encode_stems(augmented_stems), p=2, dim=-1)
    mem = torch.stack([gen.embedding_proj(e_o), gen.embedding_proj(e_a)], 1)
    return mem + gen.memory_pos_embedding.unsqueeze(0)


@torch.no_grad()
def greedy_decode(gen, memory: torch.Tensor, tok, max_length: int = 512,
                  value_mode: str = "argmax") -> torch.Tensor:
    if value_mode not in VALUE_MODES:
        raise ValueError(f"value_mode must be one of {VALUE_MODES}")
    B, dev = memory.size(0), memory.device
    vs, ve = tok.value_token_start, tok.value_token_end
    cont = continuous_param_mask(tok, dev)
    seq = torch.full((B, 1), tok.bos_token_idx, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    for _ in range(max_length - 1):
        x = gen.pos_encoding(gen.token_embedding(seq))
        mask = gen._generate_square_subsequent_mask(seq.size(1)).to(dev)
        h = gen.transformer_decoder(tgt=x, memory=memory, tgt_mask=mask)
        logits = gen.token_head(h[:, -1]).float()
        nxt = logits.argmax(-1)
        if value_mode == "median":
            is_knob = (nxt >= vs) & (nxt <= ve) & cont[seq[:, -1]]
            if is_knob.any():
                cdf = logits[:, vs:ve + 1].softmax(-1).cumsum(-1)
                med = (cdf < 0.5).sum(-1).clamp(max=ve - vs)
                nxt = torch.where(is_knob, vs + med, nxt)
        nxt = torch.where(done, torch.full_like(nxt, tok.eos_token_idx), nxt)
        seq = torch.cat([seq, nxt[:, None]], 1)
        done |= nxt == tok.eos_token_idx
        if done.all():
            break
    return seq
