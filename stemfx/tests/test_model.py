"""Smoke tests for StemFXNet (the nn.Module) — does not require a checkpoint."""

import torch

from stemfx import FXChainTokenizer, StemFXNet

SR = 44100
N = SR * 10  # 10 s


def _stems(B: int):
    return {
        "vocals": torch.randn(B, 2, N) * 0.05,
        "bass":   torch.randn(B, 2, N) * 0.05,
        "drums":  torch.randn(B, 2, N) * 0.05,
        "other":  torch.randn(B, 2, N) * 0.05,
    }


def test_forward_pass_shapes():
    tok = FXChainTokenizer()
    # feature_dim must equal the MixingFeatureExtractor output (64).
    net = StemFXNet(
        encoder_dim=128, feature_dim=64,
        n_mels=40, n_fft=1024, hop_length=512,
        vocab_size=tok.vocab_size, hidden_dim=64, num_layers=2, num_heads=4,
        feedforward_dim=128, max_seq_len=32, dropout=0.0,
        pad_token_idx=tok.pad_token_idx, bos_token_idx=tok.bos_token_idx,
    ).eval()

    target = torch.full((1, 8), tok.pad_token_idx, dtype=torch.long)
    target[0, 0] = tok.bos_token_idx

    with torch.no_grad():
        token_logits, value_preds = net(_stems(1), _stems(1), target)
    assert token_logits.shape == (1, 8, tok.vocab_size)
    assert value_preds.shape == (1, 8)


def test_param_counts_are_reasonable():
    tok = FXChainTokenizer()
    net = StemFXNet(vocab_size=tok.vocab_size,
                    pad_token_idx=tok.pad_token_idx,
                    bos_token_idx=tok.bos_token_idx)
    counts = net.count_parameters()
    # Paper main is ~28M total. Default config should land near that.
    assert counts["total"] > 5_000_000
