"""End-to-end smoke tests for the StemFX public API.

Skips automatically when no checkpoint is available.
"""

import os

import pytest
import torch

import stemfx

SAMPLE_RATE = 44100
SEG_S = 10
CKPT = os.environ.get("STEMFX_TEST_CKPT")


def _make_dummy_stems():
    return {
        "vocals": torch.randn(2, SAMPLE_RATE * SEG_S) * 0.05,
        "bass":   torch.randn(2, SAMPLE_RATE * SEG_S) * 0.05,
        "drums":  torch.randn(2, SAMPLE_RATE * SEG_S) * 0.05,
        "other":  torch.randn(2, SAMPLE_RATE * SEG_S) * 0.05,
    }


@pytest.mark.skipif(not CKPT, reason="set STEMFX_TEST_CKPT to a real checkpoint to run")
def test_load_and_embed_stems_dict():
    model = stemfx.load(checkpoint=CKPT, device="cpu")
    emb = model.embed(_make_dummy_stems())
    assert emb.shape == (model._net.encoder_dim,)
    assert torch.isclose(emb.norm(), torch.tensor(1.0), atol=1e-3)


@pytest.mark.skipif(not CKPT, reason="set STEMFX_TEST_CKPT to a real checkpoint to run")
def test_transfer_returns_multiafx_chain():
    model = stemfx.load(checkpoint=CKPT, device="cpu")
    e1 = model.embed(_make_dummy_stems())
    e2 = model.embed(_make_dummy_stems())
    chain = model.transfer(e1, e2)
    for stem in ("vocals", "bass", "drums", "other"):
        assert stem in chain
        for step in chain[stem]:
            assert "effect" in step and "params" in step


@pytest.mark.skipif(not CKPT, reason="set STEMFX_TEST_CKPT to a real checkpoint to run")
def test_transfer_audio_returns_audio_and_chain():
    model = stemfx.load(checkpoint=CKPT, device="cpu")
    out = model.transfer_audio(_make_dummy_stems(), _make_dummy_stems(),
                               normalize_loudness=False)
    assert isinstance(out, stemfx.TransferResult)
    assert out.sample_rate == SAMPLE_RATE
    assert out.mix.shape[0] == 2
    for s in ("vocals", "bass", "drums", "other"):
        assert s in out.audio
        assert s in out.chain
    pretty = out.pretty()
    assert "vocals" in pretty
