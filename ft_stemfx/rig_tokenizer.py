"""
Tokenizer for Guitar Rig FX chains, laid out to be compatible with the
pretrained StemFX vocabulary.

Vocabulary
    [0-3]   [PAD] [BOS] [EOS] [SEP]          same idx as StemFX  (rows copied)
    [4-7]   vocals bass drums other           same idx as StemFX  (rows copied)
    [8-..]  effect tokens (components)        new, random init
    [..-..] param-name tokens <comp>_<param>  new, random init
    [last 101] VALUE_0 .. VALUE_100           same meaning        (rows copied)

Sequence for one sample (guitar lives in the 'other' stem)
    [BOS] other  rack rack_input_gain V
                 <fx_1> <p> V <p> V ...
                 <fx_2> ...
          [SEP] [EOS]

* The model's target is WHICH effects are used, in WHAT order, with WHAT
  values. The preset id is only how rigkit built the chain; it is used here
  to read the chain order from param_info, never emitted as a token.
* Components follow the preset's chain_order, prefixed by the pseudo
  component 'rack' (rack_input_gain sits before the chain).
* An effect whose <comp>_on_off is off is dropped entirely, so
  A -> B(off) -> C is tokenized as A -> C. The amp has no on_off label and
  is always present. The cabinet is fixed and never emitted.
* `<comp>_on_off` itself is not emitted (presence == on).
* Every remaining param (bool / str / number / curve) is emitted as
  <param-name> VALUE_bin with bin = round(raw * 100).
* Leading [BOS] in the target mirrors stemfx.tokenizer.encode_fx_chain, which
  is what the checkpoint was trained on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from rig_normalization import (
    NUM_VALUE_BINS,
    bin_to_raw,
    load_param_specs,
    raw_to_bin,
    raw_to_value,
    snap_raw,
)

SPECIAL = ["[PAD]", "[BOS]", "[EOS]", "[SEP]"]
STEMS = ["vocals", "bass", "drums", "other"]
GUITAR_STEM = "other"
RACK = "rack"
FIXED_COMPONENTS = {"matched_cabinet_pro"}   # locked, not part of the label


def _is_on_off(key: str) -> bool:
    return key.endswith("_on_off")


class RigTokenizer:
    def __init__(self, param_info: str | Path | Dict[str, Any]):
        """param_info: the dataset's param_info/ folder, or the dict stored in a
        checkpoint (see from_checkpoint)."""
        info = load_param_specs(param_info)
        self.param_info_raw: Dict[str, Any] = info["raw"]
        self.presets: Dict[str, Dict] = info["presets"]
        self.specs: Dict[str, Dict] = info["specs"]
        self._build_vocab()
        self._build_preset_lookup()

    # ------------------------------------------------------------------ vocab
    def _build_vocab(self) -> None:
        comps, params = set(), set()
        for ps in self.presets.values():
            for k in ps["keys"]:
                comp = self.specs[k]["component"]
                if comp in FIXED_COMPONENTS:
                    continue
                comps.add(comp)
                if not _is_on_off(k):
                    params.add(k)

        tokens: List[str] = list(SPECIAL) + list(STEMS)
        self.effect_tokens = sorted(comps)
        self.param_tokens = sorted(params)
        tokens += self.effect_tokens + self.param_tokens
        self.value_token_start = len(tokens)
        tokens += [f"VALUE_{i}" for i in range(NUM_VALUE_BINS)]
        self.value_token_end = len(tokens) - 1

        self.idx_to_token = tokens
        self.token_to_idx = {t: i for i, t in enumerate(tokens)}
        assert self.token_to_idx["[PAD]"] == 0 and self.token_to_idx["other"] == 7

    def _build_preset_lookup(self) -> None:
        # Per preset: ordered list of (component, [param keys]) that can appear.
        self.preset_layout: Dict[str, List[Tuple[str, List[str], Optional[str]]]] = {}
        for ps_id, ps in self.presets.items():
            by_comp: Dict[str, List[str]] = {}
            onoff: Dict[str, str] = {}
            for k in ps["keys"]:
                comp = self.specs[k]["component"]
                if _is_on_off(k):
                    onoff[comp] = k
                else:
                    by_comp.setdefault(comp, []).append(k)
            order = [RACK] + [c for c in ps["chain_order"] if c not in FIXED_COMPONENTS]
            self.preset_layout[ps_id] = [(c, by_comp.get(c, []), onoff.get(c)) for c in order]
        # Full chain per preset (all effects on); used only by renderable_presets().
        self._full_chain = {ps: [c for c, _, _ in lay] for ps, lay in self.preset_layout.items()}

    # ------------------------------------------------------------- properties
    @property
    def vocab_size(self) -> int:
        return len(self.idx_to_token)

    pad_token_idx = property(lambda self: 0)
    bos_token_idx = property(lambda self: 1)
    eos_token_idx = property(lambda self: 2)
    sep_token_idx = property(lambda self: 3)
    other_token_idx = property(lambda self: self.token_to_idx[GUITAR_STEM])

    def is_value_token(self, idx: int) -> bool:
        return self.value_token_start <= idx <= self.value_token_end

    # ----------------------------------------------------------------- encode
    def encode(self, row: Dict[str, Any]) -> Tuple[List[int], List[float], List[bool]]:
        """row = one line of train.jsonl / val.jsonl."""
        values = row["values"]
        t2i = self.token_to_idx
        tokens, vals, vmask = [self.bos_token_idx, t2i[GUITAR_STEM]], [0.0, 0.0], [False, False]

        for comp, keys, onoff_key in self.preset_layout[row["preset"]]:
            if onoff_key is not None:
                st = values.get(onoff_key)
                if st is None or float(st["raw"]) < 0.5:
                    continue                     # effect off -> skip whole block
            present = [k for k in keys if k in values]
            if comp == RACK and not present:
                continue                         # no rack param sampled -> nothing to emit
            tokens.append(t2i[comp]); vals.append(0.0); vmask.append(False)
            for k in present:
                raw = float(values[k]["raw"])
                tokens.append(t2i[k]); vals.append(0.0); vmask.append(False)
                tokens.append(self.value_token_start + raw_to_bin(raw))
                vals.append(raw); vmask.append(True)

        tokens += [self.sep_token_idx, self.eos_token_idx]
        vals += [0.0, 0.0]
        vmask += [False, False]
        return tokens, vals, vmask

    # ----------------------------------------------------------------- decode
    def decode(self, tokens: List[int]) -> Dict[str, Any]:
        """
        Token ids -> {"effects": [ordered effect names],
                      "params": {key: raw},  "values": {key: human-readable}}
        Malformed pieces are skipped (useful for free-running generation).
        """
        effects: List[str] = []
        params: Dict[str, float] = {}
        cur_comp, cur_key = None, None
        for idx in tokens:
            idx = int(idx)
            if idx == self.eos_token_idx:
                break
            if idx >= len(self.idx_to_token) or idx in (self.pad_token_idx, self.bos_token_idx):
                continue
            tok = self.idx_to_token[idx]
            if tok in self.effect_tokens:
                cur_comp, cur_key = tok, None
                effects.append(tok)
            elif tok in self.token_to_idx and tok in self.specs and not _is_on_off(tok):
                cur_key = tok if (cur_comp and self.specs[tok]["component"] == cur_comp) else None
            elif self.is_value_token(idx) and cur_key is not None:
                params[cur_key] = snap_raw(self.specs[cur_key],
                                           bin_to_raw(idx - self.value_token_start))
                cur_key = None
        return {
            "effects": effects,
            "params": params,
            "values": {k: raw_to_value(self.specs[k], r) for k, r in params.items()},
        }

    def renderable_presets(self, effects: List[str]) -> List[str]:
        """rigkit presets that can reproduce this chain (their full chain contains
        `effects` as an ordered subsequence; the rest is switched off). Helper for
        rendering a prediction back through rigkit, not a training target."""
        hits = []
        for ps, full in self._full_chain.items():
            it = iter(full)
            if all(e in it for e in effects):
                hits.append(ps)
        return hits

    def target_from_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Ground truth in the same shape as decode() (after bin quantization)."""
        return self.decode(self.encode(row)[0])

    # ------------------------------------------------------------- save/load
    def state_dict(self) -> Dict[str, Any]:
        return {"idx_to_token": self.idx_to_token,
                "value_token_start": self.value_token_start,
                "value_token_end": self.value_token_end,
                "param_info": self.param_info_raw}

    @classmethod
    def from_checkpoint(cls, state: Dict[str, Any]) -> "RigTokenizer":
        """Rebuild from ckpt["tokenizer"] -> no dataset folder needed (inference)."""
        if "param_info" not in state:
            raise ValueError("Checkpoint has no param_info; pass --param-info <dataset>/param_info")
        tok = cls(state["param_info"])
        tok.check_compatible(state)
        return tok

    def check_compatible(self, state: Dict[str, Any]) -> None:
        if state.get("idx_to_token") != self.idx_to_token:
            raise ValueError("Tokenizer vocab differs from the one stored in the checkpoint "
                             "(param_info changed?).")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state_dict(), indent=1), encoding="utf-8")
