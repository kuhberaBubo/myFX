"""
Knob-aware chain loss: the 101 VALUE bins are ordered, so a near miss should cost
less than a far one. Plain CE (FXChainLoss with no_value_head) treats bin 49 and
bin 0 as equally wrong when the answer is 50.

    ce      cross-entropy over the whole vocab, as before, except that a continuous
            knob's VALUE target is a Gaussian over the bins (std = value_sigma bins)
            instead of one-hot -> neighbours of the true bin get partial credit.
            Structure tokens and bool / str params keep one-hot targets.
    emd     value_emd_weight * W1 distance between the predicted distribution over
            the 101 bins (softmax restricted to the VALUE range) and the true bin,
            = sum_k |CDF_pred(k) - CDF_true(k)| / 100, i.e. the expected |error| in
            raw 0-1 units of a knob drawn from the prediction. Continuous knobs only.

value_sigma=0 and value_emd_weight=0 reproduce FXChainLoss(no_value_head=True).
The returned parts include `ce_hard` (plain one-hot CE) so runs with different
settings stay comparable on the same number.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KnobAwareChainLoss(nn.Module):
    def __init__(self, tok, value_sigma: float = 0.0, value_emd_weight: float = 0.0):
        super().__init__()
        self.pad = tok.pad_token_idx
        self.bos = tok.bos_token_idx
        self.vs, self.ve = tok.value_token_start, tok.value_token_end
        self.sigma, self.emd_w = float(value_sigma), float(value_emd_weight)
        cont = torch.zeros(tok.vocab_size, dtype=torch.bool)
        for k in tok.param_tokens:
            if tok.specs[k].get("element") not in ("bool", "str"):
                cont[tok.token_to_idx[k]] = True
        self.register_buffer("cont", cont, persistent=False)     # param tokens of continuous knobs
        n = self.ve - self.vs + 1
        bins = torch.arange(n, dtype=torch.float32)
        if self.sigma > 0:
            k = torch.exp(-0.5 * ((bins[None] - bins[:, None]) / self.sigma) ** 2)
            k = k / k.sum(1, keepdim=True)                       # row i = soft target for bin i
        else:
            k = torch.eye(n)
        self.register_buffer("kernel", k, persistent=False)

    def forward(self, logits, value_preds, target_tokens, target_values, value_mask
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, V = logits.shape
        tgt = target_tokens
        non_pad = tgt != self.pad
        prev = torch.cat([torch.full_like(tgt[:, :1], self.bos), tgt[:, :-1]], 1)
        knob = non_pad & (tgt >= self.vs) & (tgt <= self.ve) & self.cont.to(tgt.device)[prev]

        logp = logits.float().log_softmax(-1)
        nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # one-hot CE per position
        denom = non_pad.sum().clamp(min=1)
        ce_hard = (nll * non_pad).sum() / denom

        ce_pos = nll
        if self.sigma > 0 and knob.any():
            q = self.kernel.to(logits.device)[tgt[knob] - self.vs]          # (K, 101)
            soft = -(q * logp[knob][:, self.vs:self.ve + 1]).sum(-1)
            ce_pos = nll.masked_scatter(knob, soft)
        ce = (ce_pos * non_pad).sum() / denom

        emd = logits.new_zeros(())
        if self.emd_w > 0 and knob.any():
            p = logits[knob][:, self.vs:self.ve + 1].float().softmax(-1)
            cdf_p = p.cumsum(-1)
            idx = torch.arange(p.size(-1), device=p.device)
            cdf_t = (idx[None] >= (tgt[knob] - self.vs)[:, None]).float()
            emd = (cdf_p - cdf_t).abs().sum(-1).mean() / 100.0
        total = ce + self.emd_w * emd
        return total, {"total": total, "ce": ce, "ce_hard": ce_hard, "emd": emd}

    @torch.no_grad()
    def compute_accuracy(self, logits, target_tokens, value_mask) -> Dict[str, float]:
        pred = logits.argmax(-1)
        non_pad = target_tokens != self.pad
        is_v = (target_tokens >= self.vs) & (target_tokens <= self.ve)
        cat = non_pad & ~is_v
        f = lambda m: (pred[m] == target_tokens[m]).float().mean().item() if m.any() else 0.0  # noqa: E731
        return {"overall": f(non_pad), "categorical": f(cat), "value": f(is_v)}
