"""
Re-render evaluation: predict the chain for dataset pairs, render the predicted
chain through rigkit (Guitar Rig) on the same input clip, and measure how far
the result sounds from the real target. Chain / order / knob metrics are
reported alongside.

Three steps, each resumable and runnable on a different machine:

  predict   (torch + checkpoint + dataset audio; CPU is fine)
      python ft_stemfx/render_eval.py predict --ckpt best_model.pt \
          --root D:/dataset/run_20k --n 100 --out eval_render
  render    (rigkit + Guitar Rig 7 + presets ps1..ps8; the desktop)
      python ft_stemfx/render_eval.py render --root D:/dataset/run_20k --out eval_render
  score     (numpy + soundfile only)
      python ft_stemfx/render_eval.py score --root D:/dataset/run_20k --out eval_render

  listen    (after score) listening page: listen.html + listen/ audio copies
      python ft_stemfx/render_eval.py listen --root D:/dataset/run_20k --out eval_render --k 5
      per sample: DI / target / predicted-chain render players, true vs predicted
      chain, knob-by-knob table. --pick spread (default) takes samples evenly
      from best to worst sound match.

  `all` runs predict, render, score, listen in a row (desktop with everything installed).

What gets rendered per sample (all on the stored input clip inputs/*.flac,
15 s incl. tail, same as the dataset render):
  pred   the model's chain                                 -> audio/<id>_pred.flac
  rand   the chain of another random val sample            -> audio/<id>_rand.flac
         (baseline: "a plausible chain that ignores the audio")
  gt     the true chain re-rendered, first --check-gt ids  -> audio/<id>_gt.flac
         (sanity check: should match targets/*.flac almost exactly)

How a predicted chain becomes plugin settings
  preset   the sample's own preset if it can express the predicted order,
           else the first preset that can (tokenizer.renderable_presets),
           else the own preset with the predicted effects that exist in it
           (`forced`: order could not be honored)
  state    preset reloaded before every render (no leftovers), fixed values
           from params.toml applied, then <comp>_on_off = predicted presence
           and every predicted knob raw. Knobs of an enabled effect that the
           model did not emit stay at the preset's stored value.

Audio metrics (target vs X, full 15 s, mono)
  mrstft    multi-resolution STFT distance (spectral convergence + log-mag L1,
            FFT 512/1024/2048), lower = closer; the usual FX-matching loss
  band_db   mean |dB difference| of the long-term spectrum in 1/3-octave
            bands 50 Hz-16 kHz: tone / EQ / drive character
  rms_db    |level difference| in dB
  win_rand  fraction of samples where pred is closer (mrstft) than rand
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PREDS = "preds.jsonl"
RENDERS = "renders.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================================================================== predict
def _render_plan(tok, pred: dict, own_preset: str) -> dict:
    """Predicted chain -> {preset, raws, forced}: what to set on the plugin."""
    effects = pred["effects"]
    hits = tok.renderable_presets(effects) if effects else list(tok.preset_layout)
    forced = not hits
    preset = own_preset if (own_preset in hits or forced) else hits[0]
    raws: dict[str, float] = {}
    used = set(effects)
    for comp, keys, onoff in tok.preset_layout[preset]:
        if onoff is not None:
            raws[onoff] = 1.0 if comp in used else 0.0
        if comp in used or onoff is None:
            for k in keys:
                if k in pred["params"]:
                    raws[k] = float(pred["params"][k])
    return {"preset": preset, "raws": raws, "forced": forced}


def cmd_predict(a) -> None:
    import torch

    from evaluate import compare
    from load_pretrained import build_model
    from rig_dataset import RigPairDataset, make_loader
    from rig_tokenizer import RigTokenizer

    if a.threads:
        torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(a.root)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    tok = RigTokenizer.from_checkpoint(ck["tokenizer"])
    model = build_model(ck["args"], tok.vocab_size, dropout=0.0)
    model.load_state_dict(ck["model_state_dict"])
    model.to(a.device).eval()
    dc = ck["config"]["data"]
    ds = RigPairDataset(root, a.split_file, tok, sample_rate=dc["sample_rate"],
                        clip_start_sec=dc["clip_start_sec"], clip_duration=dc["clip_duration"],
                        dummy_level_db=dc["dummy_level_db"],
                        skip_missing_audio=a.skip_missing_audio,
                        seed=ck["config"]["train"]["seed"])

    # random but fixed subset; each sample gets another sample's chain as baseline
    rng = random.Random(a.seed)
    pool = list(ds.rows)
    rng.shuffle(pool)
    rows = pool[:a.n] if a.n else pool
    done = {r["id"] for r in _read_jsonl(out / PREDS)}
    todo = [r for r in rows if r["id"] not in done]
    partner = {r["id"]: pool[(i + len(rows)) % len(pool)] if len(pool) > len(rows)
               else rows[(i + 1) % len(rows)] for i, r in enumerate(rows)}
    print(f"{len(rows)} samples ({len(done & {r['id'] for r in rows})} already predicted)")
    if not todo:
        return
    ds.rows = todo
    loader = make_loader(ds, a.batch_size, False, 0)
    by_id = {r["id"]: r for r in todo}

    t0, n = time.time(), 0
    for batch in loader:
        orig = {k: v.to(a.device) for k, v in batch["original_stems"].items()}
        aug = {k: v.to(a.device) for k, v in batch["augmented_stems"].items()}
        with torch.no_grad():
            gen, _ = model.generate(orig, aug, max_length=dc["max_seq_len"],
                                    bos_token_idx=tok.bos_token_idx,
                                    eos_token_idx=tok.eos_token_idx, top_k=1, top_p=1.0)
        for sid, seq in zip(batch["ids"], gen.cpu().tolist()):
            row, other = by_id[sid], partner[sid]
            pred, true = tok.decode(seq), tok.target_from_row(row)
            m = compare(tok, pred, true)
            _append_jsonl(out / PREDS, {
                "id": sid, "input": row["input"], "target": row["target"],
                "preset_true": row["preset"],
                "effects_true": true["effects"], "effects_pred": pred["effects"],
                "params_true": true["params"], "params_pred": pred["params"],
                "values_pred": pred["values"],
                "metrics": m,
                "plan_pred": _render_plan(tok, pred, row["preset"]),
                "plan_true": {"preset": row["preset"],
                              "raws": {k: float(v["raw"]) for k, v in row["values"].items()}},
                "plan_rand": {"preset": other["preset"], "from": other["id"],
                              "raws": {k: float(v["raw"]) for k, v in other["values"].items()}},
            })
            n += 1
        print(f"\r  predicted {n}/{len(todo)}  ({(time.time() - t0) / n:.1f} s/sample)",
              end="", flush=True)
    print()


# ============================================================================ render
class Renderer:
    """Thin wrapper over rigkit so the plan -> audio step is easy to follow."""

    def __init__(self, plan_header: dict | None, rig=None):
        import rigkit

        self.rigkit = rigkit
        self.rig = rig or rigkit.GuitarRig()
        self.sr = self.rig.cfg.render.sample_rate
        self.presets: dict = {}
        self.fixed: dict = {}
        self.header = plan_header or {}

    def preset(self, pid: str):
        if pid not in self.presets:
            p = self.rigkit.Preset.load(pid)
            want = self.header.get("presets", {}).get(pid)
            have = self.rigkit.render.preset_hashes(p)
            if want and any(want.get(k) != have.get(k) for k in ("state_sha", "specs_sha", "params_sha")):
                print(f"[warn] preset {pid} differs from the one used for the dataset "
                      f"(plan.json {want} vs now {have}); renders may not be comparable")
            self.presets[pid] = p
            self.fixed[pid] = self.rigkit.ParamRules.load(p).fixed_values()
        return self.presets[pid]

    def render(self, x: np.ndarray, plan: dict, with_fixed: bool) -> tuple[np.ndarray, dict]:
        p = self.preset(plan["preset"])
        self.rig.load_preset(p)                               # clean state every time
        if with_fixed and self.fixed[plan["preset"]]:
            self.rig.set(self.fixed[plan["preset"]])
        raws = {k: v for k, v in plan["raws"].items() if k in p.specs}
        skipped = sorted(set(plan["raws"]) - set(raws))
        res = self.rig.set_raw(raws) if raws else None
        y = self.rig.render(x, self.sr)                      # (2, N), same length as x
        y = y.mean(axis=0, keepdims=True)                     # dataset targets are mono
        return y, {"not_ok": res.not_ok if res else [], "skipped": skipped}


def cmd_render(a, rig=None) -> None:
    out, root = Path(a.out), Path(a.root)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    preds = _read_jsonl(out / PREDS)
    if not preds:
        raise SystemExit(f"{out / PREDS} is empty; run `predict` first")
    header_path = root / "plan.json"
    header = json.loads(header_path.read_text(encoding="utf-8")) if header_path.is_file() else None
    r = Renderer(header, rig)
    done = {(x["id"], x["kind"]) for x in _read_jsonl(out / RENDERS)}

    jobs = []
    for i, p in enumerate(preds):
        jobs.append((p, "pred", p["plan_pred"], True))         # fixed values + predicted
        jobs.append((p, "rand", p["plan_rand"], False))        # true raws already include fixed
        if i < a.check_gt:
            jobs.append((p, "gt", p["plan_true"], False))
    jobs = [j for j in jobs if (j[0]["id"], j[1]) not in done]
    # group by preset so the plugin reloads the same state (faster, same as job.run)
    jobs.sort(key=lambda j: j[2]["preset"])
    print(f"{len(jobs)} renders to do")

    t0 = time.time()
    for n, (p, kind, plan, with_fixed) in enumerate(jobs, 1):
        x, sr = sf.read(root / p["input"], dtype="float32", always_2d=True)
        if sr != r.sr:
            raise SystemExit(f"input sr {sr} != rigkit render sr {r.sr}")
        y, info = r.render(np.ascontiguousarray(x.T), plan, with_fixed)
        path = out / "audio" / f"{p['id']}_{kind}.flac"
        sf.write(path, y.T, sr, subtype="PCM_24")
        _append_jsonl(out / RENDERS, {"id": p["id"], "kind": kind, "preset": plan["preset"],
                                      "path": str(path.relative_to(out)).replace("\\", "/"),
                                      "peak_dbfs": float(20 * np.log10(np.abs(y).max() + 1e-12)),
                                      **info})
        print(f"\r  rendered {n}/{len(jobs)}  ({(time.time() - t0) / n:.1f} s/render)",
              end="", flush=True)
    print()


# ============================================================================= score
FFTS = (512, 1024, 2048)


def _stft_mag(x: np.ndarray, n_fft: int) -> np.ndarray:
    hop = n_fft // 4
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    return np.abs(np.fft.rfft(x[idx] * np.hanning(n_fft), axis=1))


def mrstft(ref: np.ndarray, est: np.ndarray) -> float:
    tot = 0.0
    for n_fft in FFTS:
        R, E = _stft_mag(ref, n_fft), _stft_mag(est, n_fft)
        sc = np.linalg.norm(R - E) / (np.linalg.norm(R) + 1e-8)
        lm = np.mean(np.abs(np.log(R + 1e-5) - np.log(E + 1e-5)))
        tot += sc + lm
    return float(tot / len(FFTS))


def band_db(x: np.ndarray, sr: int) -> np.ndarray:
    """Long-term spectrum in 1/3-octave bands (dB)."""
    S = _stft_mag(x, 4096) ** 2
    psd = S.mean(axis=0)
    f = np.fft.rfftfreq(4096, 1 / sr)
    centers = 1000 * 2 ** (np.arange(-14, 13) / 3)            # ~63 Hz .. 16 kHz
    out = []
    for c in centers:
        m = (f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))
        out.append(10 * np.log10(psd[m].mean() + 1e-12) if m.any() else -120.0)
    return np.maximum(np.array(out), -100.0)


def rms_db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(x ** 2) + 1e-12))


def audio_dist(ref: np.ndarray, est: np.ndarray, sr: int) -> dict:
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    return {"mrstft": mrstft(ref, est),
            "band_db": float(np.mean(np.abs(band_db(ref, sr) - band_db(est, sr)))),
            "rms_db": abs(rms_db(ref) - rms_db(est))}


def _mono(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return x.mean(axis=1), sr


def cmd_score(a) -> None:
    out, root = Path(a.out), Path(a.root)
    preds = {p["id"]: p for p in _read_jsonl(out / PREDS)}
    renders: dict = {}
    for r in _read_jsonl(out / RENDERS):
        renders.setdefault(r["id"], {})[r["kind"]] = r

    per = []
    for sid, p in preds.items():
        m = p["metrics"]
        row = {"id": sid, "preset_true": p["preset_true"], "preset_pred": p["plan_pred"]["preset"],
               "forced": p["plan_pred"]["forced"],
               "chain_true": " > ".join(e for e in p["effects_true"] if e != "rack"),
               "chain_pred": " > ".join(e for e in p["effects_pred"] if e != "rack"),
               "chain_exact": m["chain_exact"], "effect_f1": m["effect_f1"],
               "effect_acc": m["effect_acc"], "order_ok": m["order_ok"],
               "param_mae": float(np.mean(m["abs_err"])) if m["abs_err"] else None,
               "discrete_acc": float(np.mean(m["disc_ok"])) if m["disc_ok"] else None,
               "param_recall": m["recall"]}
        rk = renders.get(sid, {})
        if "pred" in rk and "rand" in rk:
            tgt, sr = _mono(root / p["target"])
            di, _ = _mono(root / p["input"])
            for kind in ("pred", "rand", "gt"):
                if kind in rk:
                    y, _ = _mono(out / rk[kind]["path"])
                    for k, v in audio_dist(tgt, y, sr).items():
                        row[f"{kind}_{k}"] = v
            for k, v in audio_dist(tgt, di, sr).items():
                row[f"di_{k}"] = v
            row["win_rand"] = float(row["pred_mrstft"] < row["rand_mrstft"])
        per.append(row)
        print(f"\r  scored {len(per)}/{len(preds)}", end="", flush=True)
    print()

    def mean(key, rows=per):
        v = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.mean(v)), 4) if v else None

    rendered = [r for r in per if "pred_mrstft" in r]
    summary = {
        "n_predicted": len(per), "n_rendered": len(rendered),
        "chain": {k: mean(k) for k in ("chain_exact", "effect_f1", "effect_acc", "order_ok",
                                       "param_mae", "discrete_acc", "param_recall")},
        "forced_preset": mean("forced"),
        "audio": {f"{kind}_{k}": mean(f"{kind}_{k}", rendered)
                  for kind in ("pred", "rand", "di") for k in ("mrstft", "band_db", "rms_db")},
        "win_rand": mean("win_rand", rendered),
        "gt_check": {k: mean(f"gt_{k}") for k in ("mrstft", "band_db", "rms_db")},
    }
    if rendered:
        summary["audio_by_chain_exact"] = {
            str(int(e)): {"n": len(g), "pred_mrstft": mean("pred_mrstft", g),
                          "rand_mrstft": mean("rand_mrstft", g)}
            for e in (0, 1) if (g := [r for r in rendered if r["chain_exact"] == e])}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    keys = sorted({k for r in per for k in r}, key=lambda k: list(per[0]).index(k)
                  if k in per[0] else 1000)
    with open(out / "scores.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(per)
    print(json.dumps(summary, indent=1))
    print(f"-> {out / 'summary.json'}, {out / 'scores.csv'}")


# ============================================================================ listen
LISTEN_CSS = """
:root{--bg:#fafaf9;--card:#fff;--fg:#1c1917;--mute:#78716c;--line:#e7e5e4;
--ok:#15803d;--okbg:#dcfce7;--bad:#b91c1c;--badbg:#fee2e2;--add:#1d4ed8;--addbg:#dbeafe;
--mid:#a16207;--midbg:#fef9c3}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--card:#292524;--fg:#f5f5f4;--mute:#a8a29e;
--line:#44403c;--ok:#86efac;--okbg:#14532d;--bad:#fca5a5;--badbg:#7f1d1d;--add:#93c5fd;--addbg:#1e3a8a;
--mid:#fde047;--midbg:#713f12}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,"Malgun Gothic",sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}.sub{color:var(--mute);margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;margin:0 0 20px}
.head{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:baseline;margin-bottom:12px}
.head h2{font-size:17px;margin:0}.tag{font-size:13px;color:var(--mute)}
.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin:8px 0 16px}
.players div{font-size:13px;color:var(--mute)}audio{width:100%;margin-top:4px}
.chain{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:4px 0}
.chain b{width:44px;font-size:13px;color:var(--mute);font-weight:500}
.chip{padding:2px 9px;border-radius:999px;font-size:13px;border:1px solid var(--line)}
.both{background:var(--okbg);color:var(--ok)}.miss{background:var(--badbg);color:var(--bad);text-decoration:line-through}
.extra{background:var(--addbg);color:var(--add)}.arrow{color:var(--mute)}
.stats{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--mute);margin:10px 0}
.stats span b{color:var(--fg)}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line)}th{color:var(--mute);font-weight:500}
td.fx{color:var(--mute)}td.num{font-variant-numeric:tabular-nums}
.g{background:var(--okbg);color:var(--ok)}.y{background:var(--midbg);color:var(--mid)}.r{background:var(--badbg);color:var(--bad)}
details{margin-top:8px}summary{cursor:pointer;color:var(--mute);font-size:13px}
.legend{font-size:13px;color:var(--mute);margin:-8px 0 20px}
"""


def _pick(rendered: list[dict], k: int, how: str, seed: int) -> list[dict]:
    if how == "first" or not rendered or "pred_mrstft" not in rendered[0]:
        return rendered[:k]
    if how == "random":
        return random.Random(seed).sample(rendered, min(k, len(rendered)))
    s = sorted(rendered, key=lambda r: float(r["pred_mrstft"]))     # spread: best .. worst
    if len(s) <= k:
        return s
    return [s[round(i * (len(s) - 1) / (k - 1))] for i in range(k)] if k > 1 else s[:1]


def cmd_listen(a) -> None:
    import html
    import shutil

    from rig_normalization import raw_to_value
    from rig_tokenizer import RACK, RigTokenizer

    out, root = Path(a.out), Path(a.root)
    if not (out / "scores.csv").is_file():
        cmd_score(a)
    with open(out / "scores.csv", encoding="utf-8-sig") as f:
        scores = {r["id"]: r for r in csv.DictReader(f)}
    preds = {p["id"]: p for p in _read_jsonl(out / PREDS)}
    renders: dict = {}
    for r in _read_jsonl(out / RENDERS):
        renders.setdefault(r["id"], {})[r["kind"]] = r
    rendered = [scores[i] for i in preds if i in scores and "pred" in renders.get(i, {})
                and scores[i].get("pred_mrstft")]
    picked = _pick(rendered, a.k, a.pick, a.seed)
    tok = RigTokenizer(root / "param_info")
    ldir = out / "listen"
    ldir.mkdir(exist_ok=True)

    def val(k, raw):
        v = raw_to_value(tok.specs[k], raw)
        return f"{round(v, 2):g}" if isinstance(v, float) else html.escape(str(v))

    def chain_html(label, mine, other, cls_missing):
        chips = []
        for e in (x for x in mine if x != RACK):
            c = "both" if e in other else cls_missing
            chips.append(f'<span class="chip {c}">{html.escape(e)}</span>')
        body = '<span class="arrow">→</span>'.join(chips) or '<span class="tag">(이펙트 없음)</span>'
        return f'<div class="chain"><b>{label}</b>{body}</div>'

    cards = []
    for n, sc in enumerate(picked, 1):
        sid = sc["id"]
        p, rk = preds[sid], renders[sid]
        files = {}
        for kind, src in (("input", root / p["input"]), ("target", root / p["target"]),
                          ("pred", out / rk["pred"]["path"]),
                          ("rand", out / rk["rand"]["path"]) if "rand" in rk else (None, None)):
            if kind:
                dst = ldir / f"{sid}_{kind}{src.suffix}"
                if not dst.is_file():
                    shutil.copyfile(src, dst)
                files[kind] = f"listen/{dst.name}"
        te, pe = p["effects_true"], p["effects_pred"]
        pt, pp = p["params_true"], p["params_pred"]
        m = p["metrics"]

        rows = []
        order = [e for e in te] + [e for e in pe if e not in te]
        for e in order:
            keys = [k for k in pt if tok.specs[k]["component"] == e]
            keys += [k for k in pp if tok.specs[k]["component"] == e and k not in keys]
            for i, k in enumerate(keys):
                t = val(k, pt[k]) if k in pt else "—"
                q = val(k, pp[k]) if k in pp else "—"
                if k in pt and k in pp:
                    d = abs(pt[k] - pp[k])
                    cls = "g" if d < 0.1 else ("y" if d < 0.3 else "r")
                    dd = f'<td class="num {cls}">{d:.2f}</td>'
                else:
                    dd = '<td class="num r">없음</td>'
                fx = html.escape(e) if i == 0 else ""
                name = html.escape(k.split(e + "_", 1)[-1])
                rows.append(f'<tr><td class="fx">{fx}</td><td>{name}</td><td class="num">{t}</td>'
                            f'<td class="num">{q}</td>{dd}</tr>')

        def f(key, fmt="{:.2f}"):
            v = sc.get(key)
            return fmt.format(float(v)) if v not in (None, "") else "—"

        forced = " · 순서를 표현할 프리셋이 없어 원래 프리셋으로 근사(forced)" if p["plan_pred"]["forced"] else ""
        rand_block = (f'<details><summary>참고: 다른 샘플의 체인으로 렌더한 소리 (오디오를 안 듣는 기준선, '
                      f'{html.escape(p["plan_rand"]["from"])})</summary>'
                      f'<audio controls preload="none" src="{files["rand"]}"></audio></details>'
                      if "rand" in files else "")
        cards.append(f"""
<section class="card">
 <div class="head"><h2>{n}. {html.escape(sid)}</h2>
  <span class="tag">정답 프리셋 {html.escape(p['preset_true'])} · 예측 렌더 프리셋 {html.escape(p['plan_pred']['preset'])}{forced}</span></div>
 <div class="players">
  <div>① 원본 DI<audio controls preload="none" src="{files['input']}"></audio></div>
  <div>② 정답 target<audio controls preload="none" src="{files['target']}"></audio></div>
  <div>③ 예측 체인 렌더<audio controls preload="none" src="{files['pred']}"></audio></div>
 </div>
 {chain_html("정답", te, pe, "miss")}
 {chain_html("예측", pe, te, "extra")}
 <div class="stats">
  <span>체인 완전 일치 <b>{'O' if m['chain_exact'] else 'X'}</b></span>
  <span>이펙트 F1 <b>{m['effect_f1']:.2f}</b></span>
  <span>공통 이펙트 순서 <b>{'일치' if m['order_ok'] else '다름'}</b></span>
  <span>노브 평균 오차 <b>{f('param_mae')}</b></span>
  <span>소리 거리 mrstft: 예측 <b>{f('pred_mrstft')}</b> / 기준선 <b>{f('rand_mrstft')}</b> / DI <b>{f('di_mrstft')}</b></span>
  <span>톤 차 <b>{f('pred_band_db', '{:.1f}')} dB</b></span>
 </div>
 <details open><summary>노브 비교 (값은 표시 단위, 오차는 raw 0~1 기준)</summary>
 <table><tr><th>이펙트</th><th>노브</th><th>정답</th><th>예측</th><th>|오차|</th></tr>{''.join(rows)}</table>
 </details>
 {rand_block}
</section>""")

    sm = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    au, ch = sm.get("audio", {}), sm.get("chain", {})
    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>FX 체인 듣기 비교</title>
<style>{LISTEN_CSS}</style></head><body><main>
<h1>FX 체인 듣기 비교</h1>
<p class="sub">{len(picked)}개 샘플 (전체 {sm.get('n_rendered')}개 중 소리 거리 기준 {'잘 맞춘 것부터 못 맞춘 것까지 고르게' if a.pick == 'spread' else a.pick})
 · 전체 평균: 체인 일치 {ch.get('chain_exact')}, 이펙트 F1 {ch.get('effect_f1')}, 노브 오차 {ch.get('param_mae')},
 mrstft 예측 {au.get('pred_mrstft')} / 기준선 {au.get('rand_mrstft')}, 기준선보다 가까운 비율 {sm.get('win_rand')}</p>
<p class="legend"><span class="chip both">양쪽에 있음</span> <span class="chip miss">예측에서 빠짐</span>
 <span class="chip extra">예측에만 있음</span> · 노브 오차 <span class="chip g">&lt;0.1</span>
 <span class="chip y">&lt;0.3</span> <span class="chip r">≥0.3</span></p>
{''.join(cards)}
</main></body></html>"""
    (out / "listen.html").write_text(page, encoding="utf-8")
    print(f"-> {out / 'listen.html'}  ({len(picked)} samples)")


# ============================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--root", required=True, help="dataset folder (run_20k)")
        p.add_argument("--out", required=True, help="output folder (resumable)")

    def pred_args(p):
        p.add_argument("--ckpt", required=True, help="best_model.pt or any checkpoint")
        p.add_argument("--split-file", default="val.jsonl")
        p.add_argument("--n", type=int, default=100, help="number of samples (0 = all)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--device", default="cpu")
        p.add_argument("--threads", type=int, default=0)
        p.add_argument("--skip-missing-audio", action="store_true")

    def render_args(p):
        p.add_argument("--check-gt", type=int, default=5,
                       help="re-render the true chain for the first N samples (sanity check)")

    def listen_args(p):
        p.add_argument("--k", type=int, default=5, help="samples on the listening page")
        p.add_argument("--pick", choices=("spread", "random", "first"), default="spread",
                       help="spread = evenly from best to worst sound match")
        if not any(a.dest == "seed" for a in p._actions):
            p.add_argument("--seed", type=int, default=0)

    for name, extra in (("predict", [pred_args]), ("render", [render_args]), ("score", []),
                        ("listen", [listen_args]),
                        ("all", [pred_args, render_args, listen_args])):
        p = sub.add_parser(name)
        common(p)
        for f in extra:
            f(p)
    a = ap.parse_args()
    if a.cmd in ("predict", "all"):
        cmd_predict(a)
    if a.cmd in ("render", "all"):
        cmd_render(a)
    if a.cmd in ("score", "all"):
        cmd_score(a)
    if a.cmd in ("listen", "all"):
        cmd_listen(a)


if __name__ == "__main__":
    main()
