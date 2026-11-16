"""Generate the Colab notebooks in this folder (edit here, then `python colab/make_notebooks.py`).

1_ft_frozen / 2_ft_full / 3_tfx_g / 4_tfx_mert / 6_tfx_g_soft : one experiment each (setup -> quick test -> train -> predict)
5_compare                                      : status of all runs, predict missing, tables + listening page
All write to the same Drive layout, so the compare notebook (and the desktop) just reads it:
    DRIVE_ROOT (= 26_Research/myFX)/ckpt/<exp>/   checkpoints (best_model.pt = inference)
    DRIVE_ROOT/logs/<exp>/   TensorBoard, status.json
    DRIVE_ROOT/eval/<exp>/   predictions on the shared 100 val samples (+ renders from the desktop)
    DRIVE_ROOT/eval/compare/ compare.md / compare.html
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

EXPERIMENTS = {
    "ft_frozen": ("ft_stemfx/train.py", "ft_stemfx/configs/ft_frozen.yaml",
                  "StemFX 인코더 고정 + 임베딩 캐시, 디코더만 학습, early stopping",
                  "인코더를 한 번만 돌려 임베딩을 저장하고(수 분) 그 뒤 epoch은 수 초. GPU는 T4로도 충분."),
    "ft_full": ("ft_stemfx/train.py", "ft_stemfx/configs/ft_full.yaml",
                "StemFX 인코더까지 처음부터 학습 (인코더 lr ×0.1), early stopping",
                "매 step 인코더를 통과하므로 ft_v1과 비슷한 속도(T4 epoch당 ~24분, L4는 더 빠름)."),
    "tfx_g": ("testFx/train_tfx.py", "testFx/configs/tfx_g.yaml",
              "testFX-G: 기타 인코더(대역×시간 axial + 프레임 FiLM + 전역 특징), 디코더는 StemFX에서 가져옴",
              "인코더는 새로 학습. 15 s 입력, 시간 축 유지."),
    "tfx_mert": ("testFx/train_tfx.py", "testFx/configs/tfx_mert.yaml",
                 "testFX-MERT: MERT-v1-95M 위 4층만 학습, 디코더는 StemFX에서 가져옴",
                 "첫 실행 때 MERT(약 380MB)를 내려받음. 체크포인트가 ~1GB라 latest/best만 저장."),
    "tfx_g_soft": ("testFx/train_tfx.py", "testFx/configs/tfx_g_soft.yaml",
                   "tfx_g + 노브 loss 개선 (주변 칸에도 점수: 가우시안 soft label σ=2칸 + EMD 거리 벌점)",
                   "tfx_g와 loss만 다름 -> tfx_g와 비교. 로그의 `ce_hard`가 다른 실험의 loss와 같은 기준."),
}
ORDER = {"ft_frozen": 1, "ft_full": 2, "tfx_g": 3, "tfx_mert": 4, "tfx_g_soft": 6}   # 5 = compare


class NB:
    def __init__(self):
        self.cells = []

    def md(self, s):
        self.cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})

    def code(self, s):
        self.cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": s.strip("\n")})

    def save(self, path, gpu=True):
        nb = {"cells": self.cells,
              "metadata": {**({"accelerator": "GPU"} if gpu else {}), "colab": {"provenance": []},
                           "kernelspec": {"display_name": "Python 3", "name": "python3"},
                           "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 0}
        for c in nb["cells"]:
            lines = c["source"].split("\n")
            c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", path.name, len(self.cells), "cells")


# ---------------------------------------------------------------- shared cells
PATHS = '''
# ---- 경로 (모든 노트북이 같은 값을 써야 결과가 한 곳에 모임) ----------------
REPO_URL    = "https://github.com/kuhberaBubo/myFX.git"
STEMFX_URL  = "https://github.com/barry-mir/stemfx.git"
RESEARCH    = "/content/drive/MyDrive/26_Research"
DATASET_ZIP = f"{RESEARCH}/dataset.zip"
DRIVE_ROOT  = f"{RESEARCH}/myFX"                       # 결과 전부 여기 (ckpt/ logs/ eval/)
# ft_v1 (지난번 결과)이 DRIVE_ROOT/ckpt/ft_v1에 없으면 예전 위치에서 찾음
OLD_CKPT = {"ft_v1": "/content/drive/MyDrive/myFX/ckpt/ft_v1"}
# -------------------------------------------------------------------------
import os
CODE_DIR  = "/content/myFX"
EXPERIMENTS = {
    "ft_frozen": ("ft_stemfx/train.py", "ft_stemfx/configs/ft_frozen.yaml"),
    "ft_full":   ("ft_stemfx/train.py", "ft_stemfx/configs/ft_full.yaml"),
    "tfx_g":     ("testFx/train_tfx.py", "testFx/configs/tfx_g.yaml"),
    "tfx_mert":  ("testFx/train_tfx.py", "testFx/configs/tfx_mert.yaml"),
    "tfx_g_soft": ("testFx/train_tfx.py", "testFx/configs/tfx_g_soft.yaml"),
}
def ckpt_dir(name):
    new, old = f"{DRIVE_ROOT}/ckpt/{name}", OLD_CKPT.get(name)
    return old if old and not os.path.isdir(new) and os.path.isdir(old) else new
log_dir  = lambda name: f"{DRIVE_ROOT}/logs/{name}"
eval_dir = lambda name: f"{DRIVE_ROOT}/eval/{name}"
!nvidia-smi --query-gpu=name,memory.total --format=csv || echo "no GPU"
'''

MOUNT = '''
from google.colab import drive
drive.mount("/content/drive")
'''

CLONE = '''
import os, shutil, subprocess
try:
    from google.colab import userdata
    tok = userdata.get("GH_TOKEN")
except Exception:
    tok = None
url = REPO_URL.replace("https://", f"https://{tok}@") if tok else REPO_URL
if os.path.isdir(CODE_DIR) and not os.path.isdir(f"{CODE_DIR}/.git"):
    shutil.rmtree(CODE_DIR)
if os.path.isdir(f"{CODE_DIR}/.git"):
    !git -C {CODE_DIR} pull --ff-only
else:
    r = subprocess.run(["git", "clone", "-q", url, CODE_DIR], env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if r.returncode != 0:
        raise SystemExit("git clone failed (private repo -> Colab secret GH_TOKEN)")
if not os.path.isfile(f"{CODE_DIR}/stemfx/stemfx/model.py"):
    !git clone -q {STEMFX_URL} {CODE_DIR}/stemfx
!git -C {CODE_DIR} log -1 --oneline
'''

INSTALL = '''
# stemfx pins numpy<2 -> no deps. transformers + nnAudio: MERT
!pip install -q -e {CODE_DIR}/stemfx --no-deps
!pip install -q multiafx pyloudnorm huggingface_hub soundfile transformers nnAudio
import numpy, torch, transformers
print("numpy", numpy.__version__, "| torch", torch.__version__, "| transformers", transformers.__version__,
      "| bf16", torch.cuda.is_available() and torch.cuda.is_bf16_supported())
'''

DATA = '''
# dataset zip -> local disk (Drive에서 직접 읽으면 느림)
import glob
def find_root():
    hits = sorted(glob.glob("/content/run_20k/train.jsonl") + glob.glob("/content/*/run_20k/train.jsonl"))
    return os.path.dirname(hits[0]) if hits else None
if find_root() is None:
    !cp "{DATASET_ZIP}" /content/dataset.zip
    !unzip -q -o /content/dataset.zip -d /content && rm /content/dataset.zip
DATA_ROOT = find_root()
assert DATA_ROOT, "train.jsonl not found after unzip"
print("DATA_ROOT =", DATA_ROOT)
'''

HELPERS = '''
# ---- helpers --------------------------------------------------------------
import json
def run_exp(name, quick=False, extra=""):
    script, cfg = EXPERIMENTS[name]
    if quick:   # 64/32 samples, 1 epoch, throwaway folder
        dirs = (f"train.checkpoint_dir=/content/tmp/{name}/ckpt train.log_dir=/content/tmp/{name}/logs "
                "data.max_train_samples=64 data.max_val_samples=32 train.num_epochs=1 train.resume=null")
        shutil.rmtree(f"/content/tmp/{name}", ignore_errors=True)
    else:
        dirs = f"train.checkpoint_dir={ckpt_dir(name)} train.log_dir={log_dir(name)}"
    cmd = f"python {CODE_DIR}/{script} --config {CODE_DIR}/{cfg} data.root={DATA_ROOT} {dirs} {extra}"
    print("$", cmd)
    get_ipython().system(cmd)

def status(names=None):
    names = names or ["ft_v1"] + list(EXPERIMENTS)
    print(f"{'exp':10s} {'epoch':>9s} {'best':>8s} {'val loss':>9s} {'val mae':>8s} {'base':>6s} {'min/ep':>7s}  state")
    for name in names:
        p = f"{log_dir(name)}/status.json"
        if not os.path.isfile(p):
            ck = os.path.isfile(f"{ckpt_dir(name)}/best_model.pt")
            print(f"{name:10s} {'-':>9s}   {'(best_model.pt 있음, status 없음)' if ck else ''}"); continue
        s = json.load(open(p))
        state = ("early-stopped" if s["early_stopped"] else
                 "done" if s["finished"] else f"{s['bad_epochs']} w/o improvement")
        print(f"{name:10s} {s['epoch']:>4d}/{s['num_epochs']:<4d} {s['best_val']:8.4f} {s['val']['loss']:9.4f} "
              f"{s['val']['mae_val']:8.4f} {s['baseline_mae']:6.3f} {s['epoch_sec']/60:7.1f}  {state}")

N_EVAL, EVAL_SEED = 100, 0      # 모든 모델이 같은 val 100개를 예측 (바꾸지 말 것)
def predict(name, device="cuda", value_decode="median"):
    """median (기본): 노브 = 예측 분포의 중앙값 -> eval/<name>/
       argmax: 가장 확률 높은 칸 (StemFX 원래 방식) -> eval/<name>_argmax/"""
    ck = f"{ckpt_dir(name)}/best_model.pt"
    if not os.path.isfile(ck):
        print("skip (no best_model.pt):", name); return False
    out = eval_dir(name) if value_decode == "median" else eval_dir(f"{name}_argmax")
    # 예전(argmax) 예측이 eval/<name>/에 있으면 eval/<name>_argmax/로 옮김
    legacy = eval_dir(name)
    if (os.path.isfile(f"{legacy}/preds.jsonl") and not os.path.isfile(f"{legacy}/predict_meta.json")
            and not os.path.exists(eval_dir(f"{name}_argmax"))):
        shutil.move(legacy, eval_dir(f"{name}_argmax"))
        print(f"moved old argmax predictions -> {eval_dir(name + '_argmax')}")
    get_ipython().system(f"python {CODE_DIR}/ft_stemfx/render_eval.py predict --ckpt {ck} --root {DATA_ROOT} "
                         f"--out {out} --n {N_EVAL} --seed {EVAL_SEED} --device {device} --batch-size 16 "
                         f"--value-decode {value_decode}")
    return True
'''

SETUP_MD = '''
### 0) 준비 (런타임을 새로 연결할 때마다 위에서부터 실행)
'''


def setup(nb):
    nb.md(SETUP_MD)
    for c in (PATHS, MOUNT, CLONE, INSTALL, DATA, HELPERS):
        nb.code(c)


# ---------------------------------------------------------------- per experiment
def experiment(i, name):
    script, cfg, title, note = EXPERIMENTS[name]
    nb = NB()
    nb.md(f'''
# {i}. `{name}` — {title}

- 설정: `{cfg}` · 스크립트: `{script}`
- {note}
- 결과 위치 (Drive `26_Research/myFX/`): `ckpt/{name}/`, `logs/{name}/`, `eval/{name}/`
- 런타임 → 런타임 유형 변경 → **GPU (L4 권장)**. 끊기면 0)과 2) 셀만 다시 실행하면 `latest_checkpoint.pt`에서 이어서 학습.
- 다른 실험 노트북과 **동시에** 돌려도 됨 (Drive 폴더가 서로 다름). 대신 컴퓨팅 단위가 그만큼 빨리 줄어듦.
- 모든 실험이 끝나면 `5_compare.ipynb`에서 같이 비교.
''')
    nb.code(f'NAME = "{name}"')
    setup(nb)
    nb.md('''
### 1) 빠른 테스트 (1~3분)
에러가 없는지, 진행바의 `s/it`(한 step 시간)를 확인. 전체 1 epoch = 681 step (batch 16).
''' + ("`ft_frozen`은 인코더를 한 번 돌려 임베딩 캐시를 만드는 시간이 포함됨 (테스트는 96개만).\n"
       if name == "ft_frozen" else ""))
    nb.code("run_exp(NAME, quick=True)")
    if name == "tfx_g":
        nb.md('''
### 1-b) 과적합 검사 (약 5분)
32개로 학습하고 같은 32개로 검증. loss가 계속 내려가고 `mae_val`이 `baseline mae`보다 확실히 작아지면
새 인코더 + 파이프라인이 학습 가능한 상태. 안 내려가면 본 학습 전에 알려주세요.
''')
        nb.code('''
run_exp(NAME, quick=True, extra="data.val_file=train.jsonl data.max_train_samples=32 data.max_val_samples=32 "
        "train.num_epochs=40 train.warmup_steps=20 train.learning_rate=3.0e-4 train.early_stop_patience=null "
        "train.batch_size=8")
''')
    nb.md('''
### 2) 학습
early stopping으로 끝나면 자동 종료. 브라우저 탭은 열어둘 것. 진행 상황은 아래 `status` 셀이나 TensorBoard.
''')
    nb.code('''
%load_ext tensorboard
%tensorboard --logdir {log_dir(NAME)}
''')
    nb.code('''
run_exp(NAME)
status([NAME])
''')
    nb.code("status([NAME])   # 언제든 실행 가능")
    nb.md(f'''
### 3) 평가용 예측 (GPU, 몇 분)
`best_model.pt`로 공통 val 100개(seed 0)의 체인을 예측 → Drive `eval/{name}/preds.jsonl`.
노브 값은 예측 분포의 **중앙값**(기본, `eval/{name}/`)과 가장 확률 높은 칸(`argmax`, StemFX 원래 방식,
`eval/{name}_argmax/`) 두 가지로 예측해서 비교할 수 있게 함. 이미 한 샘플은 건너뜀.
렌더 없이 체인·노브 지표만 먼저 보려면 다음 셀 `score`. 소리 비교는 데스크톱에서 렌더한 뒤.
''')
    nb.code('''
predict(NAME)                          # 노브 = 분포의 중앙값 -> eval/NAME/
predict(NAME, value_decode="argmax")   # 노브 = 가장 확률 높은 칸 -> eval/NAME_argmax/
''')
    nb.code('''
get_ipython().system(f"python {CODE_DIR}/ft_stemfx/render_eval.py score --root {DATA_ROOT} --out {eval_dir(NAME)} | tail -n 3")
s = json.load(open(f"{eval_dir(NAME)}/summary.json"))
print(json.dumps(s["chain"], indent=1))
print(f"{'effect':18s} {'F1':>6s} {'knob':>7s} {'base':>7s}")
for e, v in s["per_effect"].items():
    print(f"{e:18s} {v['f1']:6.3f} {v['knob_mae'] or float('nan'):7.4f} {v['knob_mae_baseline'] or float('nan'):7.4f}")
''')
    nb.save(HERE / f"{i}_{name}.ipynb")


# ---------------------------------------------------------------- compare
def compare():
    nb = NB()
    nb.md('''
# 5. 결과 모으기 / 비교

실험 노트북 1~4가 Drive에 남긴 결과를 한데 모아 비교한다. **GPU 없이(CPU 런타임) 돌려도 됨** — 예측이 빠진 모델만
여기서 예측하는데, CPU면 ft 계열은 샘플당 수 초, testFX는 수십 초가 걸리니 가능하면 각 실험 노트북의 3)에서 미리.

| 단계 | 어디서 | 결과 |
|---|---|---|
| 상태표 / 학습 곡선 | 여기 | 실험별 epoch, best, early stop |
| 예측 (빠진 것만) | 여기 | `eval/<exp>/preds.jsonl` |
| 체인·노브 비교표 | 여기 | `eval/compare/compare.md`, `compare.html`(DI · target · 체인/노브 비교) |
| 렌더 + 소리 비교 | 데스크톱 (Guitar Rig) | 같은 폴더에 `audio/`, 표에 소리 지표·승률 추가, `compare.html`에 모델별 예측 소리 |
''')
    setup(nb)
    nb.md('''
### 1) 상태
`RUNS`: 비교할 실험. `ft_v1` = 지난번 결과 (encoder 고정 13 epoch, early stop 없음).
''')
    nb.code('''
RUNS = ["ft_v1", "ft_frozen", "ft_full", "tfx_g", "tfx_mert", "tfx_g_soft"]
status(RUNS)
''')
    nb.code('''
%load_ext tensorboard
%tensorboard --logdir {DRIVE_ROOT}/logs
''')
    nb.md('''
### 2) 예측 + 점수 (빠진 것만)
이미 예측한 샘플은 건너뜀. 학습이 이어져서 `best_model.pt`가 바뀌었으면 `REDO`에 이름을 넣어 다시 예측.
노브는 중앙값으로 고른 결과가 기본. `ARGMAX`에 넣은 실험은 가장 확률 높은 칸으로 고른 결과(`<이름>_argmax`)도 표에 같이 넣음.
''')
    nb.code('''
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REDO = []                        # 예: ["tfx_g"]
ARGMAX = list(RUNS)              # argmax 결과도 같이 비교할 실험 (공정 비교: 전부)
ready = []
for name in RUNS:
    if name in REDO:
        for d in (eval_dir(name), eval_dir(name + "_argmax")):
            shutil.rmtree(d, ignore_errors=True)
    for mode, run in [("median", name)] + ([("argmax", name + "_argmax")] if name in ARGMAX else []):
        if not predict(name, DEVICE, mode):
            break
        get_ipython().system(f"python {CODE_DIR}/ft_stemfx/render_eval.py score --root {DATA_ROOT} --out {eval_dir(run)} > /dev/null")
        ready.append(run)
print("ready:", ready)
''')
    nb.md('''
### 3) 비교표 + 듣기 페이지
렌더 전이면 체인·노브 표, 같은 샘플에서 노브 오차가 더 작은 비율, DI/target 플레이어 + 체인·노브 비교가 나옴.
데스크톱에서 렌더하고 같은 명령을 다시 돌리면 소리 지표·승률·모델별 예측 소리가 채워짐.
''')
    nb.code('''
from IPython.display import Markdown, display
OUT = eval_dir("compare")
runs = " ".join(f"{n}={eval_dir(n)}" for n in ready)
get_ipython().system(f"python {CODE_DIR}/ft_stemfx/render_eval.py compare --root {DATA_ROOT} --out {OUT} --k 5 --runs {runs} > /dev/null")
display(Markdown(open(f"{OUT}/compare.md", encoding="utf-8").read()))
print("듣기 페이지:", f"{OUT}/compare.html", "(Drive에서 compare 폴더째 내려받아 열기)")
''')
    nb.md(r'''
### 4) 데스크톱 (Guitar Rig): 렌더 → 소리까지 포함한 비교
Drive의 `26_Research/myFX/eval/` 폴더를 데스크톱 `myFX/eval_render/`로 받은 뒤 (PowerShell, myFX 폴더에서):
```powershell
$env:RIGKIT_PROJECT = "C:\...\GuitarRig Vst Code"
$D = "<dataset>/run_20k"
$BASE = "ft_v1","ft_frozen","ft_full","tfx_g","tfx_mert","tfx_g_soft"
$RUNS = $BASE + ($BASE | ForEach-Object { "${_}_argmax" })
foreach ($e in $RUNS) {
  python ft_stemfx/render_eval.py render --root $D --out eval_render/$e
  python ft_stemfx/render_eval.py score  --root $D --out eval_render/$e
  python ft_stemfx/render_eval.py listen --root $D --out eval_render/$e
}
python ft_stemfx/render_eval.py compare --root $D --out eval_render/compare --k 5 `
  --runs ($RUNS | ForEach-Object { "$_=eval_render/$_" })
```
결과: `eval_render/compare/compare.md` (전체 / 이펙트별 / 노브·소리 승률), `compare.html` (같은 샘플을 모델별로 듣기),
각 실험 폴더의 `listen.html`. 렌더는 끊겨도 같은 명령으로 이어서 됨.
''')
    nb.save(HERE / "5_compare.ipynb", gpu=False)


if __name__ == "__main__":
    for name, i in ORDER.items():
        experiment(i, name)
    compare()
