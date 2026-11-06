# StemFX: Learning Mixing Style Representations via Autoregressive FX Chain Prediction on Source-Separated Stems

**Yuan-Chiao Cheng, Jui-Te Wu, Brian Chen, Yen-Tung Yeh, Yu-Hua Chen, Yi-Hsuan Yang**

The official implementation of the ISMIR 2026 paper *"StemFX: Learning
Mixing Style Representations via Autoregressive FX Chain Prediction on
Source-Separated Stems"*.

[**Paper**](https://arxiv.org/abs/2607.15634) &nbsp;|&nbsp;
[**Demo**](https://barry-mir.github.io/stemfx-demo/) &nbsp;|&nbsp;
[**Weights**](https://huggingface.co/barry-mir/stemfx-bsfilm)

[![PyPI](https://img.shields.io/pypi/v/stemfx)](https://pypi.org/project/stemfx/)
[![arXiv](https://img.shields.io/badge/arXiv-2607.15634-b31b1b)](https://arxiv.org/abs/2607.15634)
[![Code License: MIT](https://img.shields.io/badge/code%20license-MIT-blue)](https://github.com/barry-mir/stemfx/blob/main/LICENSE)
[![Paper License: CC BY 4.0](https://img.shields.io/badge/paper%20license-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

StemFX is a paired encoder–decoder for music mixing:

* **BSFiLM Encoder** — Band-split CNN with FiLM conditioning + temporal
  attention pooling. Maps a 4-stem 10-second audio segment to a single
  L2-normalized 512-d embedding capturing mixing style.
* **FX Chain Generator** — A 6-layer transformer decoder that, given a
  pair of (original, target) embeddings, autoregressively emits a
  per-stem FX chain in [multiafx](https://github.com/barry-mir/multiafx)
  format. The chain can be applied to the original stems to render a new
  mix in the target style.

The two are trained jointly on a Sep-Aug Pipeline: source-separated FMA
stems are randomly remixed across songs and processed through random
multilib FX chains; the model learns to recover the chain that explains
the observed difference.

---

## Install

```bash
pip install stemfx
```

For training and evaluation:

```bash
pip install "stemfx[training,evaluation]"
```

Separating arbitrary mixtures needs no extra setup. The SCNet source is
bundled, and its pretrained weights (~214 MB) are downloaded from the
upstream project's release on first use and cached under
`~/.cache/stemfx` (override with `STEMFX_CACHE_DIR`).

---

## Public API

```python
import stemfx
import soundfile as sf
import torch

model = stemfx.load()                          # downloads the paper checkpoint from HF Hub

# 1a) Embedding a mixture — separated internally with SCNet.
#     The separator checkpoint is fetched and cached on first use.
emb_a = model.embed("song_a.wav")              # → (512,) L2-normalized
emb_b = model.embed("song_b.wav")

# 1b) Or pass pre-separated stems directly (skips the separator entirely).
stems = {
    name: torch.from_numpy(sf.read(f"{name}.wav", dtype="float32", always_2d=True)[0].T)
    for name in ("vocals", "bass", "drums", "other")
}
emb = model.embed(stems)

# To use your own separator checkpoint instead of the default:
model = stemfx.load(scnet_model="path/to/scnet.ckpt",
                    scnet_config="path/to/scnet.yaml")

# 2) Predicting an FX chain from a pair of embeddings (multiafx-format)
chain = model.transfer(emb_a, emb_b)
# chain == {"vocals": [{"effect": "...", "params": {...}}, ...],
#          "bass": [...], "drums": [...], "other": [...]}

# 3) End-to-end audio: render the chain on song_a's stems
result = model.transfer_audio("song_a.wav", "song_b.wav")
print(result.pretty())                         # human-readable chain summary
torchaudio.save("transferred.wav", result.mix, result.sample_rate)
```

`TransferResult` exposes:

| field | type | description |
|------|------|-------------|
| `audio` | `dict[str, Tensor]` | per-stem rendered audio at 44.1 kHz |
| `chain` | `dict[str, list[dict]]` | per-stem FX chain (multiafx format) |
| `mix` | `Tensor` | (2, T) downmix of `audio` |
| `sample_rate` | `int` | always 44100 |
| `pretty()` | `str` | one-line-per-stem chain summary |

---

## Reproducing the paper

### 1 · Preprocess audio (separate into 4 stems)

```bash
python pipeline/preprocess_fma.py \
    --input_dir /path/to/fma_full/ \
    --output_dir /path/to/fma_separated/ \
    --scnet_model third_party/Music-Source-Separation-Training/model_scnet_masked_ep_111_sdr_9.8286.ckpt \
    --scnet_config third_party/Music-Source-Separation-Training/configs/config_musdb18_scnet_xl_ihf.yaml \
    --device cuda:0
```

### 2 · Apply random multilib FX chains (Sep-Aug)

```bash
python pipeline/augment_stems.py \
    --source /path/to/fma_separated/ \
    --output /path/to/fma_separated_augmented/ \
    --min_fx 1 --max_fx 10 \
    --workers 32
```

### 3 · Compute per-stem LUFS and emit `valid_tracks.json`

Required for cross-song stem mixing during training (`mix_stems: true`):

```bash
python pipeline/compute_stem_lufs.py \
    --dataset_dir /path/to/fma_separated_augmented/ \
    --threshold -50.0 \
    --num_workers 32
```

### 4 · Train

```bash
python train.py \
    --config configs/default.yaml \
    data.original_path=/path/to/fma_separated \
    data.augmented_path=/path/to/fma_separated_augmented \
    train.checkpoint_dir=checkpoints/stemfx_main \
    train.log_dir=logs/stemfx_main \
    train.device=cuda:0
```

CLI dotted overrides take precedence over the YAML file. See
`configs/default.yaml` for the full set of knobs.

---

## Evaluation

### Paired FX retrieval

```bash
python evaluate/retrieval.py \
    --dataset /path/to/MUSDB18_paired_fx_corrected_5/ \
    --checkpoint checkpoints/stemfx_main/best_checkpoint.pt \
    --device cuda:0
```

### Style transfer (free + forced MRSTFT)

```bash
python evaluate/style_transfer.py \
    --original_dir /path/to/MUSDB18_normalized/test \
    --target_dir /path/to/MUSDB18_styled/test \
    --checkpoint checkpoints/stemfx_main/best_checkpoint.pt \
    --output_dir outputs/eval_st \
    --num_examples 50
```

---

## Directory layout

```
stemfx/                       # installed Python package
    api.py                    # high-level inference wrapper
    encoder.py                # BSFiLM Encoder
    generator.py              # FX Chain Generator
    model.py                  # StemFX nn.Module (encoder + generator)
    tokenizer.py              # FX-chain tokenization (multiafx vocabulary)
    losses.py                 # FXChainLoss, MultiResolutionSTFTLoss
    separator.py              # SCNetSeparator (bundled SCNet)
    _scnet_infer.py           # chunked overlap-add separation
    weights.py                # checkpoint resolution + download
    third_party/              # vendored SCNet source (MIT, see NOTICE)
    mixing_features.py        # FiLM feature extractor (44.1 kHz)
    normalization.py          # multiafx parameter ranges

configs/default.yaml          # training config (paper main)
data/sepaug_dataset.py        # Sep-Aug paired dataset + DataLoader
pipeline/                     # data preprocessing scripts
    preprocess_fma.py         # SCNet separation
    augment_stems.py          # multilib FX augmentation
    compute_stem_lufs.py      # per-stem LUFS + valid_tracks.json
evaluate/                     # evaluation scripts
    retrieval.py              # paired FX retrieval
    style_transfer.py         # free + forced MRSTFT
train.py                      # config-driven training entry point
tests/                        # pytest smoke tests
```

---

## Citation

```bibtex
@inproceedings{cheng2026stemfx,
  title     = {{StemFX}: Learning Mixing Style Representations via Autoregressive
               {FX} Chain Prediction on Source-Separated Stems},
  author    = {Cheng, Yuan-Chiao and Wu, Jui-Te and Chen, Brian and
               Yeh, Yen-Tung and Chen, Yu-Hua and Yang, Yi-Hsuan},
  booktitle = {Proceedings of the 27th International Society for Music
               Information Retrieval Conference (ISMIR)},
  address   = {Abu Dhabi, UAE},
  year      = {2026},
  eprint    = {2607.15634},
  archivePrefix = {arXiv},
  primaryClass  = {eess.AS},
}
```

## License

- **Code** (this repository): [MIT](LICENSE).
- **Paper** and accompanying documentation: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
