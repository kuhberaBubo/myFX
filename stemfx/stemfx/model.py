"""
Combined BSFiLM Encoder + FX Chain Generator for end-to-end FX chain prediction.

This is the main paper architecture. Encoder and generator are trained jointly
with cross-entropy loss against tokenized FX chains.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from stemfx.encoder import BSFiLMEncoder
from stemfx.mixing_features import MixingFeatureExtractor
from stemfx.generator import FXChainGenerator


class StemFX(nn.Module):
    """
    End-to-end model: BSFiLM Encoder + FX Chain Generator.

    Architecture:
        Original stems  → BSFiLM Encoder → emb_orig
        Augmented stems → BSFiLM Encoder → emb_aug
        [emb_orig, emb_aug] → FXChainGenerator → FX tokens + values
    """

    def __init__(
        self,
        # Encoder config
        encoder_dim: int = 512,
        feature_dim: int = 64,
        n_mels: int = 80,
        n_fft: int = 2048,
        hop_length: int = 512,
        sample_rate: int = 44100,
        use_film: bool = True,
        encoder_backbone: str = 'cnn',
        # Generator config
        vocab_size: int = 358,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        pad_token_idx: int = 0,
        bos_token_idx: int = 1,
        num_memory_tokens: int = 4,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.use_film = use_film
        self.encoder_dim = encoder_dim
        self.feature_dim = feature_dim

        self.encoder = BSFiLMEncoder(
            channels=8,
            embed_dim=encoder_dim,
            feature_dim=feature_dim,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            sample_rate=sample_rate,
            use_film=use_film,
            backbone=encoder_backbone,
        )
        if use_film:
            self.feature_extractor = MixingFeatureExtractor(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            )
        else:
            self.feature_extractor = None

        self.generator = FXChainGenerator(
            vocab_size=vocab_size,
            encoder_dim=self.encoder_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            max_seq_len=max_seq_len,
            pad_token_idx=pad_token_idx,
            bos_token_idx=bos_token_idx,
            num_memory_tokens=num_memory_tokens,
        )

    def _encode_stems(self, stems_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode 4-stem dict via BSFiLM Encoder. Returns (B, encoder_dim)."""
        batch_size = next(iter(stems_dict.values())).size(0)

        if self.use_film:
            features_list = []
            for i in range(batch_size):
                sample = {k: v[i] for k, v in stems_dict.items()}
                features_list.append(self.feature_extractor.extract_all_features(sample))
            features = torch.stack(features_list, dim=0)
        else:
            device = next(iter(stems_dict.values())).device
            features = torch.zeros(batch_size, self.feature_dim, device=device)
        return self.encoder(stems_dict, features)

    def forward(
        self,
        original_stems: Dict[str, torch.Tensor],
        augmented_stems: Dict[str, torch.Tensor],
        target_tokens: torch.Tensor,
        teacher_forcing: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass with teacher forcing.

        Args:
            original_stems: Dict[stem_name, (B, 2, T)] for the 4 stems.
            augmented_stems: Same shape, FX-applied version.
            target_tokens: (B, seq_len) token sequence for teacher forcing.

        Returns:
            token_logits: (B, seq_len, vocab_size)
            value_preds:  (B, seq_len)
        """
        emb_original = F.normalize(self._encode_stems(original_stems), p=2, dim=-1)
        emb_augmented = F.normalize(self._encode_stems(augmented_stems), p=2, dim=-1)
        embedding_pairs = torch.cat([emb_original, emb_augmented], dim=1)

        token_logits, value_preds = self.generator(
            embedding_pairs=embedding_pairs,
            target_tokens=target_tokens,
            teacher_forcing=teacher_forcing,
        )
        return token_logits, value_preds

    @torch.no_grad()
    def generate(
        self,
        original_stems: Dict[str, torch.Tensor],
        augmented_stems: Dict[str, torch.Tensor],
        max_length: int = 256,
        bos_token_idx: int = 1,
        eos_token_idx: int = 2,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive FX-chain generation. Returns (tokens, values)."""
        emb_original = F.normalize(self._encode_stems(original_stems), p=2, dim=-1)
        emb_augmented = F.normalize(self._encode_stems(augmented_stems), p=2, dim=-1)
        embedding_pairs = torch.cat([emb_original, emb_augmented], dim=1)

        return self.generator.generate(
            embedding_pairs=embedding_pairs,
            max_length=max_length,
            bos_token_idx=bos_token_idx,
            eos_token_idx=eos_token_idx,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    def count_parameters(self) -> Dict[str, int]:
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        generator_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        return {
            'encoder': encoder_params,
            'generator': generator_params,
            'total': encoder_params + generator_params,
        }
