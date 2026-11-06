"""Tokenizer round-trip and vocabulary sanity tests."""

import torch

from stemfx import FXChainTokenizer


def test_special_tokens_at_known_indices():
    t = FXChainTokenizer()
    assert t.pad_token_idx == 0
    assert t.bos_token_idx == 1
    assert t.eos_token_idx == 2
    assert t.sep_token_idx == 3


def test_value_token_range_is_contiguous():
    t = FXChainTokenizer()
    assert t.value_token_end - t.value_token_start + 1 == t.NUM_VALUE_BINS
    assert t.is_value_token(t.value_token_start)
    assert t.is_value_token(t.value_token_end)
    assert not t.is_value_token(t.value_token_start - 1)


def test_encode_decode_round_trip():
    t = FXChainTokenizer()
    chain = {
        "stems": {
            "vocals": {
                "fx_chain": [
                    {"effect": "sox_gain", "params": {"gain_db": 3.0}},
                ],
            },
            "drums": {
                "fx_chain": [
                    {"effect": "sox_compand", "params": {
                        "attack_time": 0.1, "decay_time": 0.5, "soft_knee_db": 6.0
                    }},
                ],
            },
        }
    }
    tokens, values, mask = t.encode_fx_chain(chain)
    assert tokens[0] == t.bos_token_idx
    assert tokens[-1] == t.eos_token_idx
    assert len(tokens) == len(values) == len(mask)

    decoded = t.decode_fx_chain(tokens, values)
    assert "vocals" in decoded["stems"]
    assert decoded["stems"]["vocals"]["fx_chain"][0]["effect"] == "sox_gain"
    # value should round-trip approximately within the quantization grid
    assert abs(decoded["stems"]["vocals"]["fx_chain"][0]["params"]["gain_db"] - 3.0) < 1.0


def test_batch_encode_shapes():
    t = FXChainTokenizer()
    chain = {"stems": {"vocals": {"fx_chain": [
        {"effect": "sox_gain", "params": {"gain_db": 0.0}}]}}}
    toks, vals, mask = t.batch_encode([chain, chain], max_length=64)
    assert toks.shape == (2, 64)
    assert vals.shape == (2, 64)
    assert mask.shape == (2, 64)
    assert toks.dtype == torch.long
    assert mask.dtype == torch.bool
