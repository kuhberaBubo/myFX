"""
Tokenization system for multilib FX chain prediction.

Converts multilib FX metadata to token sequences for transformer training.
Uses effect-specific parameter names to avoid ambiguity.
"""

from typing import Dict, List, Tuple
import torch

from stemfx.normalization import normalize_param, denormalize_param, quantize_value, get_all_param_keys
from multiafx import registry as _registry

# Build a name->EffectDef dict for compatibility with the original tokenizer code.
EFFECT_REGISTRY = {name: _registry.get(name) for name in _registry.list_effects()}


class FXChainTokenizer:
    """
    Tokenizer for multilib FX chain sequences with effect-specific parameter names.

    Vocabulary structure:
        [0-3]          Special: [PAD], [BOS], [EOS], [SEP]
        [4-7]          Stems: vocals, bass, drums, other
        [8-90]         FX Types: 83 multilib effects (sorted alphabetically)
        [91-N]         Effect-specific parameter names (sorted)
        [N+1-N+101]   Numeric Values: 101 bins (0.00 to 1.00)
    """

    # Special tokens
    PAD_TOKEN = '[PAD]'
    BOS_TOKEN = '[BOS]'
    EOS_TOKEN = '[EOS]'
    SEP_TOKEN = '[SEP]'

    # Stem names
    STEMS = ['vocals', 'bass', 'drums', 'other']

    # FX types from registry (sorted alphabetically)
    FX_TYPES = sorted(EFFECT_REGISTRY.keys())

    # Number of quantization bins for continuous values
    NUM_VALUE_BINS = 101  # 0.00, 0.01, ..., 1.00

    def __init__(self):
        """Initialize tokenizer and build vocabulary."""
        self.token_to_idx = {}
        self.idx_to_token = {}

        # Build vocabulary
        self._build_vocab()

    def _build_vocab(self):
        """Build vocabulary mapping with effect-specific parameter names."""
        idx = 0

        # Special tokens [0-3]
        for token in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.SEP_TOKEN]:
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
            idx += 1

        # Stems [4-7]
        for stem in self.STEMS:
            self.token_to_idx[stem] = idx
            self.idx_to_token[idx] = stem
            idx += 1

        # FX types
        for fx_type in self.FX_TYPES:
            self.token_to_idx[fx_type] = idx
            self.idx_to_token[idx] = fx_type
            idx += 1

        # Effect-specific parameter names (sorted)
        param_keys = get_all_param_keys()
        for param_key in sorted(param_keys):
            self.token_to_idx[param_key] = idx
            self.idx_to_token[idx] = param_key
            idx += 1

        # Numeric value tokens (101 bins)
        self.value_token_start = idx
        for i in range(self.NUM_VALUE_BINS):
            token = f'VALUE_{i}'
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
            idx += 1
        self.value_token_end = idx - 1

        print(f"Built multilib vocabulary with {len(self.token_to_idx)} tokens")

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.token_to_idx)

    @property
    def pad_token_idx(self) -> int:
        """Get PAD token index."""
        return self.token_to_idx[self.PAD_TOKEN]

    @property
    def bos_token_idx(self) -> int:
        """Get BOS token index."""
        return self.token_to_idx[self.BOS_TOKEN]

    @property
    def eos_token_idx(self) -> int:
        """Get EOS token index."""
        return self.token_to_idx[self.EOS_TOKEN]

    @property
    def sep_token_idx(self) -> int:
        """Get SEP token index."""
        return self.token_to_idx[self.SEP_TOKEN]

    def is_value_token(self, token_idx: int) -> bool:
        """Check if token index is a VALUE token."""
        return self.value_token_start <= token_idx <= self.value_token_end

    def encode_fx_chain(self, fx_metadata: Dict) -> Tuple[List[int], List[float], List[bool]]:
        """
        Encode multilib FX metadata to token sequence using effect-specific parameter names.

        Multilib metadata uses 'effect' and 'params' keys (library/category are metadata-only).

        Args:
            fx_metadata: FX metadata dict from JSON file

        Returns:
            tokens: List of token indices
            values: List of normalized continuous values (for VALUE tokens)
            value_mask: Boolean mask indicating which tokens are VALUE tokens
        """
        tokens = [self.bos_token_idx]
        values = [0.0]
        value_mask = [False]

        # Process each stem
        for stem_name in self.STEMS:
            if stem_name not in fx_metadata['stems']:
                continue

            stem_data = fx_metadata['stems'][stem_name]
            fx_chain = stem_data.get('fx_chain', [])

            if len(fx_chain) == 0:
                continue

            # Add stem token
            tokens.append(self.token_to_idx[stem_name])
            values.append(0.0)
            value_mask.append(False)

            # Process each effect in chain
            for effect in fx_chain:
                fx_type = effect['effect']
                params = effect['params']

                # Add FX type token
                tokens.append(self.token_to_idx[fx_type])
                values.append(0.0)
                value_mask.append(False)

                # Add parameter-value pairs
                for param_name, param_value in params.items():
                    # Create effect-specific parameter key
                    param_key = f"{fx_type}_{param_name}"

                    # Add parameter name token
                    if param_key in self.token_to_idx:
                        tokens.append(self.token_to_idx[param_key])
                        values.append(0.0)
                        value_mask.append(False)
                    else:
                        print(f"Warning: Unknown parameter {param_key}, skipping")
                        continue

                    # Continuous value - normalize with effect type
                    try:
                        normalized = normalize_param(param_value, param_name, fx_type)
                        bin_idx = quantize_value(normalized)
                        value_token_idx = self.value_token_start + bin_idx

                        tokens.append(value_token_idx)
                        values.append(normalized)
                        value_mask.append(True)
                    except ValueError as e:
                        print(f"Warning: {e}, skipping")

            # Add separator after each stem
            tokens.append(self.sep_token_idx)
            values.append(0.0)
            value_mask.append(False)

        # Add EOS token
        tokens.append(self.eos_token_idx)
        values.append(0.0)
        value_mask.append(False)

        return tokens, values, value_mask

    def decode_fx_chain(self, tokens: List[int], values: List[float]) -> Dict:
        """
        Decode token sequence back to multilib FX metadata structure.

        Args:
            tokens: List of token indices
            values: List of normalized continuous values (for VALUE tokens)

        Returns:
            fx_metadata: Reconstructed FX metadata dict
        """
        fx_metadata = {'stems': {}}

        i = 0
        if i < len(tokens) and tokens[i] == self.bos_token_idx:
            i += 1

        current_stem = None
        current_fx = None
        current_fx_type = None
        current_param_key = None

        while i < len(tokens):
            token_idx = tokens[i]
            token_str = self.idx_to_token.get(token_idx, None)

            if token_str is None:
                i += 1
                continue

            # Check for special tokens
            if token_str == self.EOS_TOKEN:
                break
            elif token_str == self.SEP_TOKEN:
                current_stem = None
                current_fx = None
                current_fx_type = None
                current_param_key = None
                i += 1
                continue

            # Check for stem token
            if token_str in self.STEMS:
                current_stem = token_str
                fx_metadata['stems'][current_stem] = {'fx_chain': []}
                current_fx = None
                current_fx_type = None
                current_param_key = None
                i += 1
                continue

            # Check for FX type token
            if token_str in self.FX_TYPES:
                if current_stem is not None:
                    current_fx_type = token_str
                    current_fx = {
                        'effect': token_str,
                        'params': {}
                    }
                    fx_metadata['stems'][current_stem]['fx_chain'].append(current_fx)
                current_param_key = None
                i += 1
                continue

            # Check for effect-specific parameter name token
            if '_' in token_str and current_fx_type is not None:
                # Match effect_type prefix (handle multi-word effect names like am_high_shelf_filter)
                if token_str.startswith(f"{current_fx_type}_"):
                    current_param_key = token_str
                    i += 1
                    continue

            # Check for value token
            if current_param_key is not None and current_fx is not None:
                # Extract parameter name by removing the effect type prefix
                param_name = current_param_key[len(current_fx_type) + 1:]

                if self.is_value_token(token_idx):
                    normalized = values[i]
                    param_value = denormalize_param(normalized, param_name, current_fx_type)
                    current_fx['params'][param_name] = param_value

                current_param_key = None

            i += 1

        return fx_metadata

    def batch_encode(self, fx_metadata_list: List[Dict], max_length: int = 256) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batch encode multiple FX chains with padding.

        Args:
            fx_metadata_list: List of FX metadata dicts
            max_length: Maximum sequence length (pad/truncate to this)

        Returns:
            tokens_batch: (B, max_length) tensor of token indices
            values_batch: (B, max_length) tensor of normalized values
            value_mask_batch: (B, max_length) tensor of VALUE token masks
        """
        batch_size = len(fx_metadata_list)

        tokens_batch = torch.full((batch_size, max_length), self.pad_token_idx, dtype=torch.long)
        values_batch = torch.zeros(batch_size, max_length, dtype=torch.float32)
        value_mask_batch = torch.zeros(batch_size, max_length, dtype=torch.bool)

        for i, fx_metadata in enumerate(fx_metadata_list):
            tokens, values, value_mask = self.encode_fx_chain(fx_metadata)

            seq_len = min(len(tokens), max_length)
            tokens_batch[i, :seq_len] = torch.tensor(tokens[:seq_len], dtype=torch.long)
            values_batch[i, :seq_len] = torch.tensor(values[:seq_len], dtype=torch.float32)
            value_mask_batch[i, :seq_len] = torch.tensor(value_mask[:seq_len], dtype=torch.bool)

        return tokens_batch, values_batch, value_mask_batch
