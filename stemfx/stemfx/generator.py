"""
Transformer-based FX chain generator.

Takes concatenated embeddings [emb_orig, emb_aug] and autoregressively
predicts FX chain token sequences with dual output heads (token + value).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FXChainGenerator(nn.Module):
    """
    Transformer-based generator for FX chain prediction.

    Architecture:
        - Shared embedding projection: encoder_dim (512) → hidden_dim (512)
        - Applied to emb_orig and emb_aug separately → 1 token each → 2 memory tokens
        - Token embedding: vocab_size → 512
        - Positional encoding
        - Cross-attention memory (encoder embeddings as context)
        - Transformer decoder (6 layers, 8 heads)
        - Dual output heads:
            * Token head: 512 → vocab_size (categorical prediction)
            * Value head: (512 + 512) → 1 (parameter-aware, concat hidden + prev token embedding)
    """

    def __init__(
        self,
        vocab_size: int = 155,
        encoder_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        pad_token_idx: int = 0,
        bos_token_idx: int = 1,
        num_memory_tokens: int = 4
    ):
        """
        Initialize FX chain generator.

        Args:
            vocab_size: Size of token vocabulary
            encoder_dim: Dimension of BSFiLMEncoder output (512)
            hidden_dim: Hidden dimension of transformer
            num_layers: Number of transformer decoder layers
            num_heads: Number of attention heads
            feedforward_dim: Dimension of feedforward network
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            pad_token_idx: Index of PAD token
            bos_token_idx: Index of BOS (beginning of sequence) token
            num_memory_tokens: Number of memory tokens for cross-attention (default: 8)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.pad_token_idx = pad_token_idx
        self.bos_token_idx = bos_token_idx
        self.num_memory_tokens = 2  # 1 token per embedding × 2 = 2 total

        # ===== Embedding Layers =====

        # Shared projection: encoder_dim (512) → hidden_dim (512)
        # Applied separately to emb_orig and emb_aug → 1 memory token each.
        # Total: 2 memory tokens (1 from original + 1 from augmented).
        # Weight sharing ensures both embeddings are projected into the same space.
        self.embedding_proj = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Memory position embedding: distinguishes original (slot 0) from target (slot 1).
        # Without this, cross-attention is permutation-invariant over memory tokens,
        # making the decoder unable to tell which embedding is original vs augmented.
        self.memory_pos_embedding = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)

        # Token embedding for input sequence
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_idx)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len, dropout)

        # ===== Transformer Decoder =====

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=False
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # ===== Output Heads =====

        # Token head: Predict next token from vocabulary
        self.token_head = nn.Linear(hidden_dim, vocab_size)

        # Value head: Parameter-aware continuous value prediction in [0, 1]
        # Concatenates decoder hidden state with previous token's embedding (param name)
        # so the value head knows which parameter it's predicting for.
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Initialize token embedding
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if self.token_embedding.padding_idx is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.token_embedding.padding_idx].fill_(0)

        # Initialize output heads
        nn.init.normal_(self.token_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.token_head.bias)

    def forward(
        self,
        embedding_pairs: torch.Tensor,
        target_tokens: torch.Tensor,
        teacher_forcing: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with teacher forcing for training.

        Args:
            embedding_pairs: (B, encoder_dim * 2) concatenated embeddings [emb_orig, emb_aug]
            target_tokens: (B, seq_len) target token sequence (for teacher forcing)
            teacher_forcing: Whether to use teacher forcing (True for training)

        Returns:
            token_logits: (B, seq_len, vocab_size) token prediction logits
            value_preds: (B, seq_len) continuous value predictions in [0, 1]
        """
        batch_size, seq_len = target_tokens.shape

        # ===== 1. Project encoder embeddings to memory =====
        # Split embedding_pairs back into original and augmented, project each separately
        emb_orig = embedding_pairs[:, :self.encoder_dim]   # (B, encoder_dim)
        emb_aug = embedding_pairs[:, self.encoder_dim:]    # (B, encoder_dim)

        mem_orig = self.embedding_proj(emb_orig).unsqueeze(1)  # (B, 1, hidden_dim)
        mem_aug = self.embedding_proj(emb_aug).unsqueeze(1)    # (B, 1, hidden_dim)
        memory = torch.cat([mem_orig, mem_aug], dim=1)  # (B, 2, hidden_dim)
        memory = memory + self.memory_pos_embedding.unsqueeze(0)  # distinguish orig vs aug

        # ===== 2. Shift target tokens right for teacher forcing =====
        # Create input sequence by prepending BOS and removing last token
        # This prevents the model from seeing the answer at position i when predicting position i
        # Input: [BOS, tok0, tok1, tok2, ..., tok_{n-2}]
        # Target: [tok0, tok1, tok2, tok3, ..., tok_{n-1}]
        bos_token = torch.full(
            (batch_size, 1),
            self.bos_token_idx,
            dtype=torch.long,
            device=target_tokens.device
        )

        # Shift right: [BOS] + target_tokens[:, :-1]
        input_tokens = torch.cat([bos_token, target_tokens[:, :-1]], dim=1)  # (B, seq_len)

        # ===== 3. Embed input tokens =====
        token_emb = self.token_embedding(input_tokens)  # (B, seq_len, hidden_dim)
        token_emb = self.pos_encoding(token_emb)  # Add positional encoding

        # ===== 4. Create causal mask for autoregressive generation =====
        # Prevent attending to future tokens
        tgt_mask = self._generate_square_subsequent_mask(seq_len).to(token_emb.device)

        # Create padding mask (based on INPUT tokens after right-shift)
        tgt_key_padding_mask = (input_tokens == self.pad_token_idx)

        # ===== 5. Transformer decoder =====
        # Cross-attention: query from token_emb, key/value from memory
        decoder_output = self.transformer_decoder(
            tgt=token_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # (B, seq_len, hidden_dim)

        # ===== 6. Output heads =====
        token_logits = self.token_head(decoder_output)  # (B, seq_len, vocab_size)

        # Parameter-aware value head: concat decoder hidden state with
        # the previous token's embedding (the param name at VALUE positions).
        # input_tokens contains the right-shifted sequence, so input_tokens[i]
        # is the token that precedes target position i (i.e., the param name).
        prev_token_emb = self.token_embedding(input_tokens)  # (B, seq_len, hidden_dim)
        value_input = torch.cat([decoder_output, prev_token_emb], dim=-1)  # (B, seq_len, hidden_dim * 2)
        value_preds = self.value_head(value_input).squeeze(-1).clamp(0.0, 1.0)  # (B, seq_len)

        return token_logits, value_preds

    def generate(
        self,
        embedding_pairs: torch.Tensor,
        max_length: int = 256,
        bos_token_idx: int = 1,
        eos_token_idx: int = 2,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive generation of FX chain.

        Args:
            embedding_pairs: (B, encoder_dim * 2) concatenated embeddings
            max_length: Maximum sequence length to generate
            bos_token_idx: Begin-of-sequence token index
            eos_token_idx: End-of-sequence token index
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (0 = disabled)
            top_p: Nucleus sampling threshold

        Returns:
            generated_tokens: (B, seq_len) generated token sequence
            generated_values: (B, seq_len) generated continuous values
        """
        batch_size = embedding_pairs.size(0)
        device = embedding_pairs.device

        # Initialize with BOS token
        generated_tokens = torch.full((batch_size, 1), bos_token_idx, dtype=torch.long, device=device)
        generated_values = torch.zeros(batch_size, 1, device=device)

        # Project encoder embeddings to memory (shared projection, 1 token per embedding)
        emb_orig = embedding_pairs[:, :self.encoder_dim]
        emb_aug = embedding_pairs[:, self.encoder_dim:]
        mem_orig = self.embedding_proj(emb_orig).unsqueeze(1)  # (B, 1, hidden_dim)
        mem_aug = self.embedding_proj(emb_aug).unsqueeze(1)    # (B, 1, hidden_dim)
        memory = torch.cat([mem_orig, mem_aug], dim=1)  # (B, 2, hidden_dim)
        memory = memory + self.memory_pos_embedding.unsqueeze(0)  # distinguish orig vs aug

        for step in range(max_length - 1):
            # Embed current sequence
            token_emb = self.token_embedding(generated_tokens)  # (B, cur_len, hidden_dim)
            token_emb = self.pos_encoding(token_emb)

            # Create causal mask
            cur_len = generated_tokens.size(1)
            tgt_mask = self._generate_square_subsequent_mask(cur_len).to(device)

            # Transformer decoder
            decoder_output = self.transformer_decoder(
                tgt=token_emb,
                memory=memory,
                tgt_mask=tgt_mask
            )  # (B, cur_len, hidden_dim)

            # Get logits for last position
            last_hidden = decoder_output[:, -1, :]  # (B, hidden_dim)
            token_logits = self.token_head(last_hidden)  # (B, vocab_size)

            # Parameter-aware value head: concat with previous token's embedding
            last_input_token = generated_tokens[:, -1]  # (B,) — the token fed at this step
            last_input_emb = self.token_embedding(last_input_token)  # (B, hidden_dim)
            value_input = torch.cat([last_hidden, last_input_emb], dim=-1)  # (B, hidden_dim * 2)
            value_pred = self.value_head(value_input).squeeze(-1).clamp(0.0, 1.0)  # (B,)

            # Sample next token
            next_token = self._sample_token(token_logits, temperature, top_k, top_p)  # (B,)

            # Append to sequence
            generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(1)], dim=1)
            generated_values = torch.cat([generated_values, value_pred.unsqueeze(1)], dim=1)

            # Check for EOS token
            if (next_token == eos_token_idx).all():
                break

        return generated_tokens, generated_values

    def _generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        """
        Generate causal mask for autoregressive generation.

        Args:
            size: Sequence length

        Returns:
            mask: (size, size) boolean mask (True = masked position)
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """
        Sample token from logits with temperature, top-k, and nucleus sampling.

        Args:
            logits: (B, vocab_size) token logits
            temperature: Sampling temperature
            top_k: Top-k sampling (0 = disabled)
            top_p: Nucleus sampling threshold

        Returns:
            sampled_tokens: (B,) sampled token indices
        """
        # Apply temperature
        logits = logits / temperature

        # Top-k sampling
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter back to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        return sampled_tokens

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
