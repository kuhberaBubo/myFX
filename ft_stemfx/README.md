# ft_stemfx

StemFX 논문 체크포인트(`barry-mir/stemfx-bsfilm`)를 그대로 가져와서, Guitar Rig 쌍 데이터(`dataset/run_20k`)로
decoder 토큰만 새로 바꿔 이어서 학습한다. 모델/학습 루프/loss는 `stemfx` 코드를 그대로 쓰고,
입력 · tokenizer · normalization만 새로 만들었다.

```
myFX/
├─ stemfx/                 원본 클론 (수정 없음, import만 함)
└─ ft_stemfx/
   ├─ rig_normalization.py raw(0~1) ↔ VALUE bin(101), grid snap, 표시값 변환
   ├─ rig_tokenizer.py     프리셋 체인 → 토큰 (꺼진 이펙트는 통째로 생략)
   ├─ rig_dataset.py       DI → other, 렌더 → other, 나머지 stem은 동일한 더미 노이즈
   ├─ load_pretrained.py   체크포인트 부분 로딩 + 특수/stem/VALUE row 복사
   ├─ train.py             학습 (encoder freeze 단계, resume=auto)
   ├─ evaluate.py          greedy 생성 → 체인(이펙트+순서) 일치, 노브 오차
   ├─ infer.py             (DI, 처리된 소리) → 체인 JSON. CPU 가능
   ├─ render_eval.py       예측 체인을 rigkit으로 다시 렌더 → target과 소리 비교
   ├─ configs/ft_default.yaml
   └─ colab_ft.ipynb
```

## 입력
- 15 s 클립의 앞 10 s만 사용 (encoder 학습 길이). 10 s 이후 딜레이/리버브 꼬리는 버림.
- mono → stereo 복제. 레벨 정규화 없음 (타깃 레벨 자체가 체인 정보).
- vocals/bass/drums: −60 dB 핑크 노이즈, original과 augmented에 **같은 신호** → 정답에 체인 없음.
  무음은 안 됨 (`mixing_features`의 `log10(peak/rms)`가 −inf → NaN).

## 토큰
```
[BOS] other  rack rack_input_gain V  tube_compressor <p> V ...  fire_seeker ...  [SEP] [EOS]
```
- 체인 순서는 `param_info/<preset>.json`의 `chain_order`, 맨 앞에 `rack`(입력 게인).
- `<comp>_on_off`가 off면 그 이펙트 블록 전체 생략 (A → B(off) → C ⇒ A → C). 앰프는 항상 있음, 캐비닛은 고정이라 없음.
- 모든 파라미터(bool/str/number/curve)를 `raw` 기준 `VALUE_round(raw*100)`로.
- vocab 195 = 특수 4 + stem 4 + 이펙트 10 + 파라미터 76 + VALUE 101.
  0~7과 VALUE 101개는 체크포인트 row를 복사, 나머지 86개는 새로 초기화.

## 실행
```bash
# 로컬 / Colab 공통
python ft_stemfx/train.py --config ft_stemfx/configs/ft_default.yaml \
    data.root=/content/dataset/run_20k \
    train.checkpoint_dir=/content/drive/MyDrive/myFX/ckpt/ft_v1 \
    train.log_dir=/content/drive/MyDrive/myFX/logs/ft_v1

python ft_stemfx/evaluate.py --ckpt <ckpt_dir>/best_checkpoint.pt --root <data.root> --max-samples 300
```
- `train.resume=auto`(기본): `checkpoint_dir/latest_checkpoint.pt`가 있으면 이어서 학습.
- `train.freeze_encoder=true`(기본): decoder만 학습. `train.unfreeze_at_epoch=N`이면 N epoch부터 encoder도
  `learning_rate × encoder_lr_multiplier`로 학습.
- `stemfx` 패키지가 설치돼 있지 않으면 옆 폴더 `../stemfx`에서 import. `stemfx/__init__`이 `multiafx`를 import하므로 설치 필요.

## 저장 파일 (`train.checkpoint_dir`, Colab에서는 Drive)
| 파일 | 크기 | 용도 |
|---|---|---|
| `latest_checkpoint.pt` | ~322MB | 매 epoch 덮어씀. 이어서 학습(resume)용 |
| `best_checkpoint.pt` | ~322MB | val loss 최저. optimizer 포함 |
| `best_model.pt` | ~114MB | best의 모델만 + tokenizer(param_info 포함). **추론용** |
| `checkpoint_epochN.pt` | ~322MB | `save_interval`마다 스냅샷 |

로그(TensorBoard, config.yaml, tokenizer.json)는 `train.log_dir`.

## 추론 (CPU)
```bash
python ft_stemfx/infer.py --ckpt best_model.pt --di di.wav --wet wet.wav --offset 0 --out pred.json
```
- 임의 샘플레이트 → 44.1kHz mono, `--offset`부터 10 s. DI와 wet은 시간 정렬돼 있어야 함.
- DI를 −32 LUFS로 맞추는 gain을 DI와 wet에 **같이** 적용 (학습 입력과 같은 레벨, 드라이/웨트 비율 유지). `pip install pyloudnorm` 권장.
- 출력: `chain`(순서대로 이펙트 + 파라미터), `values`(dataset/rigkit `values` 형식, 모델이 안 낸 파라미터는 프리셋 기본값으로 채우고 `source:"default"`), `renderable_presets`(이 체인을 rigkit으로 다시 렌더할 수 있는 프리셋).
- 속도: 4스레드 CPU에서 한 쌍당 약 4초 (encoder 0.4 s + 최대 길이 생성 3.6 s). 모델 28M 파라미터.
- dataset 폴더 없이 동작 (param_info가 체크포인트에 들어 있음).

## 다시 렌더해서 비교 (`render_eval.py`)
예측한 체인을 rigkit으로 같은 입력 클립에 렌더해서 target과 소리가 얼마나 다른지 잰다. 체인/순서/노브 지표도 함께 낸다.
```bash
# 데스크톱 (Guitar Rig 7 + ps1~8 프리셋 + 전체 dataset)
$env:RIGKIT_PROJECT = "C:\...\GuitarRig Vst Code"     # rigkit 설정/프리셋 위치 (myFX에서 실행하므로 필요)
python ft_stemfx/render_eval.py all --ckpt best_model.pt --root <dataset>/run_20k --out eval_render --n 100
```
- 단계별로도 실행 가능: `predict`(torch, CPU 가능) → `render`(rigkit 필요) → `score`(numpy만). 끊기면 같은 명령으로 이어서.
- 샘플마다 렌더: `pred` 예측 체인, `rand` 다른 val 샘플의 체인(오디오를 안 보는 기준선), `gt` 정답 체인 재렌더(앞 5개, target과 거의 같아야 정상).
- 예측 체인 → 플러그인 설정: 원래 프리셋이 예측 순서를 표현할 수 있으면 그 프리셋, 아니면 표현 가능한 다른 프리셋,
  둘 다 안 되면 원래 프리셋에서 있는 이펙트만 켬(`forced`). 매번 프리셋을 다시 불러온 뒤 params.toml 고정값 → on/off → 예측 노브 raw.
  모델이 안 낸 노브는 프리셋 저장값.
- 오디오 지표 (target 대비, 15 s 전체): `mrstft` 다중 해상도 STFT 거리, `band_db` 1/3옥타브 장기 스펙트럼 dB 차(톤),
  `rms_db` 레벨 차, `win_rand` pred가 rand보다 가까운 비율. `di_*`는 이펙트 없는 DI와의 거리(참고).
- 결과: `summary.json`, `scores.csv`(샘플별), `audio/<id>_{pred,rand,gt}.flac`(target은 dataset의 `targets/`).
- **듣기 페이지** `listen.html` (`all`의 마지막 단계, 따로는 `listen --k 5`): 샘플마다 원본 DI / target / 예측 렌더 플레이어,
  정답·예측 체인 비교(빠진 이펙트 / 추가된 이펙트 표시), 노브별 정답·예측·오차 표. 기본 5개를 소리 거리 기준으로
  잘 맞춘 것부터 못 맞춘 것까지 고르게 뽑음(`--pick spread|random|first`). 오디오는 `listen/`에 복사돼서 폴더째 옮겨도 열림.

## 지표
- 학습 중 (teacher forcing): `acc_cat` 토큰(이펙트/파라미터 이름) 정확도, `mae_val` VALUE 위치의 bin 오차/100 ≈ 노브 평균 오차.
- `evaluate.py` (자유 생성): 목표는 **어떤 이펙트를 어떤 순서로 어떤 값으로** 썼는지. 프리셋 id는 rigkit이 체인을 만드는 수단일 뿐 예측 대상이 아님.
  - `chain_exact` 이펙트 목록(순서 포함) 완전 일치
  - `effect_acc` / `effect_f1` 이펙트별 사용 여부
  - `order_ok` 양쪽에 다 있는 이펙트끼리 순서 일치
  - `param_mae` 공통 파라미터의 raw 오차, `discrete_acc` bool/선택지 일치, `param_recall` 정답 파라미터 중 예측된 비율
