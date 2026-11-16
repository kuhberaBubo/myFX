# testFX

디코더는 StemFX 구조와 가중치를 그대로 쓰고(ft_stemfx와 같은 출발점), **인코더만 바꿔 끼우는** 실험용 모델.
tokenizer / 정답 시퀀스 / 평가(`ft_stemfx/render_eval.py`)는 ft_stemfx와 공유한다.

```
testFx/
├─ tfx_features.py   (dry, wet) → 페달 특징: 프레임 단위 39차원 + 전역 5그룹
├─ tfx_encoders.py   GuitarEncoder("g"), MERTEncoder("mert") → 디코더 메모리 (B, M, 512)
├─ tfx_model.py      MemoryGenerator(StemFX 디코더, 가변 길이 메모리) + StemFX 가중치 warm start
├─ tfx_data.py       mono 15 s 쌍 (10 s 연주 + 5 s 꼬리), 더미 stem 없음
├─ train_tfx.py      학습 (ft_stemfx/train.py와 같은 레시피: early stop, resume, bf16, 기준선 출력)
└─ configs/tfx_g.yaml, tfx_mert.yaml
```

## 디코더 (공통)
StemFX `FXChainGenerator` 6층/512/8헤드. cross-attention 메모리만 인코더가 주는 토큰 열로 바뀜.
StemFX 체크포인트에서 transformer 층·value head·위치 인코딩을 불러오고, 특수/stem/VALUE 토큰 행은 복사, 이펙트·파라미터 행은 새로 초기화 (ft_stemfx와 동일).
→ 세 실험(ft_frozen, ft_full, testFX)의 차이를 인코더 차이로 볼 수 있다.

메모리 = `gate × LayerNorm(인코더 출력)`, gate는 학습되는 스칼라(`model.memory_gate_init`, 기본 0.1). 새로 초기화된 인코더의 잡음 같은 토큰이 사전학습 디코더를 망가뜨리지 않게 처음엔 작게 넣는다. (샘플 3개 과적합: gate 없이 loss 4.66에서 멈춤 → gate 0.1이면 100 step에 0.04)

## GuitarEncoder ("g", 새로 학습, 6.1M)
| 부분 | 내용 |
|---|---|
| 입력 | log-mel 128 [dry, wet, wet−dry], 고정 dB 스케일 (샘플별 정규화 없음 → dry/wet 레벨 차 유지. StemFX는 InstanceNorm으로 레벨을 지움) |
| 대역×시간 | mel 11개 대역(20 bin, 10 겹침) → 대역별 투영(StemFX `BandFeatureProjection`) → axial transformer 4블록 (시간축 attention ↔ 대역축 attention, StemFX 블록 + SDPA) |
| 프레임 특징 → FiLM | 23 ms마다 dry/wet/차이의 RMS, crest factor, 스펙트럼 중심·평탄도·기울기, 옥타브 8대역 에너지 (39차원) → 1D conv → 블록마다 **시간별** γ, β |
| 전역 특징 | tone(1/3옥타브 wet/dry 비), dyn(wet dB vs dry dB 기울기 = 압축비), dist(평탄도·중심·crest 변화), echo(엔벨로프 상호/자기상관 0~2 s), tail(10 s 이후 잔향 에너지·감쇠) |
| 메모리 | 전역 토큰 5 + 프레임 특징 토큰(시간) + 대역×시간 토큰 (~5.4 Hz × 11) ≈ 965개. 시간은 풀링만 하고 없애지 않음 |

## MERTEncoder ("mert", MERT-v1-95M)
24 kHz로 변환, 5 s 조각 3개로 나눠 넣고 이어붙임. 13개 hidden layer 가중합(학습).
위 4층만 학습(lr 1e-5), 나머지 고정. dry의 평균/표준편차로 dry·wet을 같이 스케일 → 레벨 비 유지.
메모리: [dry, wet, wet−dry] × 5 Hz × 15 s = 225 토큰. `pip install transformers nnAudio` 필요.

## 실행
```bash
python testFx/train_tfx.py --config testFx/configs/tfx_g.yaml data.root=<run_20k> \
    train.checkpoint_dir=<ckpt>/tfx_g train.log_dir=<logs>/tfx_g
```
Colab은 `colab/3_tfx_g.ipynb`, `colab/4_tfx_mert.ipynb` (비교는 `colab/5_compare.ipynb`, 노트북 수정은 `colab/make_notebooks.py`에서).
평가는 ft_stemfx와 같다: `ft_stemfx/render_eval.py predict --ckpt <ckpt>/best_model.pt ...` (체크포인트의 `model_kind`로 자동 구분).

## lr
decoder 1e-4 / 새 인코더·투영 3e-4 / 사전학습 인코더(MERT 위층) 1e-5. warmup 1000 step, cosine, early stopping(val loss).
