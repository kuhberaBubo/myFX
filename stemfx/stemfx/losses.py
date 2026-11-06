"""
Losses used by StemFX:

* :class:`FXChainLoss` — joint cross-entropy + value-MSE loss for the
  autoregressive FX-chain head (the main training objective).
* :class:`MultiResolutionSTFTLoss` — multi-resolution STFT loss used by the
  style-transfer evaluation script for audio-quality scoring.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FXChainLoss(nn.Module):
    """Mixed cross-entropy + MSE loss for autoregressive FX-chain prediction.

    The vocabulary contains a contiguous range of ``VALUE_*`` tokens
    representing quantized parameter bins (101 bins). The categorical loss
    targets the *non*-VALUE tokens (stems / effects / param names);
    the regression loss targets the continuous bin values for VALUE tokens.

    When ``no_value_head=True`` the model has no separate value head and we
    apply cross-entropy across the full vocabulary (treats the value bin
    itself as a categorical target).
    """

    def __init__(
        self,
        pad_token_idx: int = 0,
        value_token_start: int = 75,
        value_token_end: int = 175,
        lambda_mse: float = 20.0,
        no_value_head: bool = False,
    ):
        super().__init__()
        self.pad_token_idx = pad_token_idx
        self.value_token_start = value_token_start
        self.value_token_end = value_token_end
        self.lambda_mse = lambda_mse
        self.no_value_head = no_value_head
        self.mse_loss = nn.MSELoss(reduction="mean")

    def forward(
        self,
        token_logits: torch.Tensor,    # (B, T, V)
        value_preds: torch.Tensor,     # (B, T)
        target_tokens: torch.Tensor,   # (B, T)
        target_values: torch.Tensor,   # (B, T)
        value_mask: torch.Tensor,      # (B, T) bool
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, V = token_logits.shape
        non_pad = target_tokens != self.pad_token_idx

        if self.no_value_head:
            ce = F.cross_entropy(
                token_logits.reshape(-1, V), target_tokens.reshape(-1),
                ignore_index=self.pad_token_idx, reduction="mean",
            )
            mse = torch.tensor(0.0, device=token_logits.device)
            total = ce
        else:
            is_value = (
                (target_tokens >= self.value_token_start)
                & (target_tokens <= self.value_token_end)
            )
            cat_mask = non_pad & ~is_value
            if cat_mask.any():
                ce = F.cross_entropy(
                    token_logits[cat_mask], target_tokens[cat_mask], reduction="mean"
                )
            else:
                ce = torch.tensor(0.0, device=token_logits.device)
            if value_mask.any():
                mse = self.mse_loss(value_preds[value_mask], target_values[value_mask])
            else:
                mse = torch.tensor(0.0, device=value_preds.device)
            total = ce + self.lambda_mse * mse

        return total, {
            "total": total,
            "ce_categorical": ce,
            "mse_value": mse,
            "mse_value_weighted": self.lambda_mse * mse,
        }

    @torch.no_grad()
    def compute_accuracy(
        self,
        token_logits: torch.Tensor,
        target_tokens: torch.Tensor,
        value_mask: torch.Tensor,
    ) -> Dict[str, float]:
        pred = token_logits.argmax(dim=-1)
        non_pad = target_tokens != self.pad_token_idx
        is_value = (
            (target_tokens >= self.value_token_start)
            & (target_tokens <= self.value_token_end)
        )
        cat_mask = non_pad & ~is_value
        return {
            "overall": (pred[non_pad] == target_tokens[non_pad]).float().mean().item() if non_pad.any() else 0.0,
            "categorical": (pred[cat_mask] == target_tokens[cat_mask]).float().mean().item() if cat_mask.any() else 0.0,
            "value": (pred[is_value] == target_tokens[is_value]).float().mean().item() if is_value.any() else 0.0,
        }


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss (spectral convergence + log-magnitude L1).

    Used as the audio-quality metric in the style-transfer evaluation.
    """

    def __init__(
        self,
        fft_sizes=(1024, 2048, 512),
        hop_sizes=(256, 512, 128),
        win_sizes=(1024, 2048, 512),
    ):
        super().__init__()
        self.fft_sizes = list(fft_sizes)
        self.hop_sizes = list(hop_sizes)
        self.win_sizes = list(win_sizes)

    def _stft(self, x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        B, C, T = x.shape
        window = torch.hann_window(win, device=x.device)
        return torch.stft(
            x.reshape(B * C, T),
            n_fft=n_fft, hop_length=hop, win_length=win,
            window=window, return_complex=True,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        total = 0.0
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            x_mag = self._stft(x, n_fft, hop, win).abs()
            y_mag = self._stft(y, n_fft, hop, win).abs()
            sc = torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + 1e-8)
            log_mag = F.l1_loss(torch.log(x_mag + 1e-5), torch.log(y_mag + 1e-5))
            total = total + sc + log_mag
        return total / len(self.fft_sizes)
