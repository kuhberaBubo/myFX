"""
BSFiLM Encoder — band-split multi-band CNN with FiLM conditioning.

This is the encoder described in the StemFX paper (Section 3.2).
Architecture: 8-channel log-mel spectrogram → split into overlapping mel sub-bands
→ per-band CNN → temporal attention pooling → 512-d mixing-style embedding.

Input:  4 stems (vocals, bass, drums, other) × 2 stereo channels = 8 channels
FiLM:   64-d hand-crafted mixing features modulate each CNN layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math


class MelSpectrogramPreprocessor(nn.Module):
    """
    Converts raw audio stems to 8-channel mel spectrogram.

    Input: Dict of 4 stems (vocals, bass, drums, other), each (B, 2, T)
    Output: 8-channel mel spectrogram (B, 8, n_mels, time_frames)
    """

    def __init__(self, sample_rate=44100, n_fft=1024, hop_length=256, n_mels=128, channels=8):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )

        # Instance normalization on log-mel spectrogram (per-channel, per-sample)
        self.instance_norm = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, stems_dict):
        """
        Convert stems to mel spectrogram.

        Args:
            stems_dict: Dict with keys ['vocals', 'bass', 'drums', 'other']
                       Each value is tensor of shape (B, 2, T) for stereo

        Returns:
            mel_spec: 8-channel mel spectrogram (B, 8, n_mels, time_frames)
        """
        mel_specs = []

        stem_names = list(stems_dict.keys()) if 'mixture' in stems_dict else ['vocals', 'bass', 'drums', 'other']
        for stem_name in stem_names:
            stem_audio = stems_dict[stem_name]  # (B, 2, T)

            # Compute mel spectrogram for both channels
            stem_mel = self.mel_transform(stem_audio)  # (B, 2, n_mels, time_frames)
            mel_specs.append(stem_mel)

        # Stack all stems: N stems × 2 channels = 2N channels
        mel_spec = torch.cat(mel_specs, dim=1)  # (B, channels, n_mels, time_frames)

        # Log scale
        mel_spec = torch.log(mel_spec + 1e-10)

        # Instance normalization (per-channel, per-sample)
        mel_spec = self.instance_norm(mel_spec)

        return mel_spec


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer."""

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

    def forward(self, x, gamma, beta):
        """
        Apply FiLM: x' = gamma * x + beta

        Args:
            x: Input tensor (B, C, ...) where C is the modulated dimension
            gamma: Scale parameter (B, C)
            beta: Shift parameter (B, C)

        Returns:
            Modulated tensor of same shape as x
        """
        # Reshape gamma and beta to match x dimensions
        shape = [x.size(0), self.num_features] + [1] * (x.ndim - 2)
        gamma = gamma.view(*shape)
        beta = beta.view(*shape)

        return gamma * x + beta


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for transformer attention."""

    def __init__(self, dim, theta=10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def apply_rotary(self, x):
        """
        Apply rotary embedding to input tensor.

        Args:
            x: Input tensor (batch, seq_len, heads, head_dim)

        Returns:
            Tensor with rotary position embedding applied, same shape as input
        """
        batch, seq_len, heads, head_dim = x.shape
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, head_dim/2)
        cos_vals = freqs.cos()  # (seq_len, head_dim/2)
        sin_vals = freqs.sin()  # (seq_len, head_dim/2)

        # Reshape for broadcasting: (1, seq_len, 1, head_dim/2)
        cos_vals = cos_vals[None, :, None, :]
        sin_vals = sin_vals[None, :, None, :]

        # Split x into pairs for rotation
        x1 = x[..., 0::2]  # (batch, seq_len, heads, head_dim/2)
        x2 = x[..., 1::2]  # (batch, seq_len, heads, head_dim/2)

        # Apply rotation: rotate_half pattern
        out1 = x1 * cos_vals - x2 * sin_vals
        out2 = x1 * sin_vals + x2 * cos_vals

        # Interleave back
        out = torch.stack([out1, out2], dim=-1)  # (..., head_dim/2, 2)
        out = out.flatten(-2)  # (..., head_dim)
        return out


class BandFeatureProjection(nn.Module):
    """
    Per-band feature projection replacing SubSpectrogramCNN.

    Two-layer conv projection (Conformer/Whisper style):
    1. Conv2d with small kernel learns local frequency patterns within the sub-band
    2. Conv2d collapses remaining frequency bins into 1, producing D features per time frame
    """

    def __init__(self, in_channels, out_dim, split_size):
        super().__init__()
        mid_channels = out_dim // 2  # Intermediate channels

        # Layer 1: local frequency feature extraction (stride 2 halves freq dim)
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels,
            kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # Layer 2: collapse remaining frequency bins
        freq_after_conv1 = (split_size + 2 * 1 - 3) // 2 + 1  # after stride-2 conv
        self.conv2 = nn.Conv2d(
            mid_channels, out_dim,
            kernel_size=(freq_after_conv1, 1),
            stride=(freq_after_conv1, 1)
        )
        self.bn2 = nn.BatchNorm2d(out_dim)

    def forward(self, x, gamma=None, beta=None):
        """
        Project sub-band features.

        Args:
            x: Input sub-band (B, C, split_size, T)
            gamma: Optional FiLM scale (B, D)
            beta: Optional FiLM shift (B, D)

        Returns:
            features: (B, D, T)
        """
        x = self.conv1(x)     # (B, mid, freq/2, T)
        x = self.bn1(x)
        x = F.gelu(x)

        x = self.conv2(x)     # (B, D, 1, T)
        x = self.bn2(x)
        if gamma is not None and beta is not None:
            g = gamma.view(gamma.size(0), -1, 1, 1)
            b = beta.view(beta.size(0), -1, 1, 1)
            x = g * x + b
        x = F.gelu(x)
        x = x.squeeze(2)      # (B, D, T)
        return x


class AxialTransformerBlock(nn.Module):
    """
    Axial transformer block with time-axis and frequency-axis self-attention.

    Pre-norm style (LayerNorm before attention) with RoPE, FFN, residual
    connections, and FiLM conditioning after the freq-axis attention.
    """

    def __init__(self, d_model, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Time-axis attention components
        self.time_norm = nn.LayerNorm(d_model)
        self.time_q_proj = nn.Linear(d_model, d_model)
        self.time_k_proj = nn.Linear(d_model, d_model)
        self.time_v_proj = nn.Linear(d_model, d_model)
        self.time_out_proj = nn.Linear(d_model, d_model)
        self.time_attn_dropout = nn.Dropout(dropout)

        # Time-axis FFN
        self.time_ffn_norm = nn.LayerNorm(d_model)
        self.time_ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

        # Freq-axis attention components
        self.freq_norm = nn.LayerNorm(d_model)
        self.freq_q_proj = nn.Linear(d_model, d_model)
        self.freq_k_proj = nn.Linear(d_model, d_model)
        self.freq_v_proj = nn.Linear(d_model, d_model)
        self.freq_out_proj = nn.Linear(d_model, d_model)
        self.freq_attn_dropout = nn.Dropout(dropout)

        # Freq-axis FFN
        self.freq_ffn_norm = nn.LayerNorm(d_model)
        self.freq_ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

        # RoPE for time and freq axes
        self.time_rope = RotaryPositionEmbedding(self.head_dim)
        self.freq_rope = RotaryPositionEmbedding(self.head_dim)

        # FiLM layer applied after freq-axis attention
        self.film = FiLMLayer(d_model)

    def _attention(self, x, q_proj, k_proj, v_proj, out_proj, attn_dropout, rope):
        """
        Compute multi-head self-attention with RoPE.

        Args:
            x: Input (N, S, D) where N is batch, S is sequence length
            q_proj, k_proj, v_proj: Linear projections
            out_proj: Output projection
            attn_dropout: Dropout for attention weights
            rope: RotaryPositionEmbedding instance

        Returns:
            Output tensor (N, S, D)
        """
        N, S, D = x.shape
        H = self.num_heads
        HD = self.head_dim

        # Project Q, K, V and reshape to (N, S, H, HD)
        q = q_proj(x).view(N, S, H, HD)
        k = k_proj(x).view(N, S, H, HD)
        v = v_proj(x).view(N, S, H, HD)

        # Apply RoPE to Q and K
        q = rope.apply_rotary(q)
        k = rope.apply_rotary(k)

        # Transpose to (N, H, S, HD) for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(HD)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale  # (N, H, S, S)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (N, H, S, HD)

        # Reshape back to (N, S, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(N, S, D)
        return out_proj(attn_output)

    def forward(self, x, gamma=None, beta=None):
        """
        Forward pass through axial transformer block.

        Args:
            x: Input tensor (B, K, T, D)
            gamma: FiLM scale parameter (B, D)
            beta: FiLM shift parameter (B, D)

        Returns:
            Output tensor (B, K, T, D)
        """
        B, K, T, D = x.shape

        # --- Time-axis self-attention ---
        # Reshape to (B*K, T, D)
        xt = x.reshape(B * K, T, D)
        # Pre-norm + attention + residual
        xt = xt + self._attention(
            self.time_norm(xt),
            self.time_q_proj, self.time_k_proj, self.time_v_proj,
            self.time_out_proj, self.time_attn_dropout, self.time_rope
        )
        # FFN + residual
        xt = xt + self.time_ffn(self.time_ffn_norm(xt))
        # Reshape back to (B, K, T, D)
        x = xt.view(B, K, T, D)

        # --- Freq-axis self-attention ---
        # Reshape to (B*T, K, D)
        xf = x.permute(0, 2, 1, 3).reshape(B * T, K, D)
        # Pre-norm + attention + residual
        xf = xf + self._attention(
            self.freq_norm(xf),
            self.freq_q_proj, self.freq_k_proj, self.freq_v_proj,
            self.freq_out_proj, self.freq_attn_dropout, self.freq_rope
        )
        # FFN + residual
        xf = xf + self.freq_ffn(self.freq_ffn_norm(xf))
        # Reshape back to (B, K, T, D)
        x = xf.view(B, T, K, D).permute(0, 2, 1, 3)

        # --- FiLM conditioning ---
        if gamma is not None and beta is not None:
            # x is (B, K, T, D), gamma/beta are (B, D)
            # Reshape gamma/beta to (B, 1, 1, D) for broadcasting
            g = gamma.view(B, 1, 1, D)
            b = beta.view(B, 1, 1, D)
            x = g * x + b

        return x


class SubSpectrogramCNN(nn.Module):
    """
    CNN module for processing a single sub-spectrogram band.

    This is one branch of the band-split architecture.
    Returns 2D features (no flattening) for temporal attention pooling.
    """

    def __init__(self, split_size, channels, out_channels=64):
        super().__init__()
        self.split_size = split_size
        self.channels = channels
        self.out_channels = out_channels

        sub_size = max(1, split_size // 10)  # Vertical pooling size

        # First conv layer
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.film1 = FiLMLayer(32)
        self.pool1 = nn.MaxPool2d((sub_size, 5))
        self.dropout1 = nn.Dropout(0.3)

        # Second conv layer
        self.conv2 = nn.Conv2d(32, out_channels, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.film2 = FiLMLayer(out_channels)
        self.pool2 = nn.MaxPool2d((4, 4))  # Less aggressive time pooling
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, x, gamma1=None, beta1=None, gamma2=None, beta2=None):
        """
        Forward pass through sub-spectrogram CNN.

        Args:
            x: Input sub-spectrogram (B, C, H, W)
            gamma1, beta1: FiLM params for first conv layer
            gamma2, beta2: FiLM params for second conv layer

        Returns:
            features: 2D features (B, out_channels, H', W') where W' is time dimension
        """
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        if gamma1 is not None and beta1 is not None:
            x = self.film1(x, gamma1, beta1)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        if gamma2 is not None and beta2 is not None:
            x = self.film2(x, gamma2, beta2)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        return x  # (B, out_channels, H', W')


class AttentionPooling(nn.Module):
    """
    Attention pooling layer for temporal aggregation.

    Learns to weight different time frames based on their importance
    for mixing style representation.
    """

    def __init__(self, input_dim, hidden_dim=128, output_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Attention mechanism (operates on each time frame)
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Projection to output dimension
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        """
        Apply attention pooling across time dimension.

        Args:
            x: Features (B, C, T) where T is time dimension

        Returns:
            pooled: Attention-pooled features (B, output_dim)
        """
        # Transpose to (B, T, C) for attention computation
        x = x.transpose(1, 2)  # (B, T, C)

        # Compute attention scores for each time frame
        attn_scores = self.attention(x)  # (B, T, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, T, 1)

        # Weighted sum across time dimension
        weighted_features = x * attn_weights  # (B, T, C)
        pooled_features = weighted_features.sum(dim=1)  # (B, C)

        # Project to output dimension
        output = self.projection(pooled_features)  # (B, output_dim)

        return output


class BandSplitEncoder(nn.Module):
    """
    Band-split axial transformer encoder for 8-channel stem input.

    Architecture:
    1. Split mel-spectrogram into K overlapping sub-bands
    2. Per-band projection via shared Conv2d → (B, K, D, T)
    3. N axial transformer blocks (time + freq self-attention with RoPE, FiLM)
    4. Reshape to (B, K*D, T), attention pooling over time
    5. Projection to final embedding
    """

    # Axial transformer hyperparameters (hardcoded per spec)
    _D = 192           # Transformer feature dimension
    _N_BLOCKS = 4      # Number of axial transformer blocks
    _HEADS = 6         # Attention heads
    _FFN_DIM = 768     # FFN hidden dimension (4 * D)
    _DROPOUT = 0.1     # Transformer dropout

    def __init__(
        self,
        sample_rate=44100,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        split_size=20,
        overlap=10,
        channels=8,
        embed_dim=512,
        cnn_out_channels=64,
        backbone='roformer'
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.split_size = split_size
        self.overlap = overlap
        self.channels = channels
        self.cnn_out_channels = cnn_out_channels
        self.backbone = backbone

        D = self._D
        N = self._N_BLOCKS

        # Mel spectrogram preprocessor
        self.mel_preprocessor = MelSpectrogramPreprocessor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            channels=channels
        )

        # Calculate number of sub-bands
        self.n_subbands = 0
        i = 0
        while overlap * i <= n_mels - split_size:
            self.n_subbands += 1
            i += 1

        if backbone == 'cnn':
            # CNN backbone: per-band SubSpectrogramCNN (v1 architecture)
            self.subnet_cnns = nn.ModuleList([
                SubSpectrogramCNN(split_size, channels, out_channels=cnn_out_channels)
                for _ in range(self.n_subbands)
            ])

            # Calculate total feature dim from dummy forward
            dummy_audio_length = int(10.0 * sample_rate)
            dummy_time_frames = (dummy_audio_length // hop_length) + 1
            with torch.no_grad():
                dummy_input = torch.zeros(1, channels, split_size, dummy_time_frames)
                dummy_output = self.subnet_cnns[0](dummy_input)
                self.freq_dim = dummy_output.shape[2]
                self.time_dim = dummy_output.shape[3]

            total_channels = cnn_out_channels * self.n_subbands * self.freq_dim
        else:
            # Roformer backbone: shared projection + axial transformer (v2 architecture)
            self.band_projection = BandFeatureProjection(
                in_channels=channels,
                out_dim=D,
                split_size=split_size
            )

            self.transformer_blocks = nn.ModuleList([
                AxialTransformerBlock(
                    d_model=D,
                    num_heads=self._HEADS,
                    ffn_dim=self._FFN_DIM,
                    dropout=self._DROPOUT
                )
                for _ in range(N)
            ])

            total_channels = self.n_subbands * D

            # Compute T from dummy forward pass
            dummy_audio_length = int(10.0 * sample_rate)
            dummy_time_frames = (dummy_audio_length // hop_length) + 1
            with torch.no_grad():
                dummy_input = torch.zeros(1, channels, split_size, dummy_time_frames)
                dummy_out = self.band_projection(dummy_input)
                self.time_dim = dummy_out.shape[2]

        # Attention pooling layer
        self.attention_pooling = AttentionPooling(
            input_dim=total_channels,
            hidden_dim=256,
            output_dim=embed_dim
        )

    def forward(self, stems_dict, film_params=None):
        """
        Forward pass through band-split axial transformer encoder.

        Args:
            stems_dict: Dict with keys ['vocals', 'bass', 'drums', 'other']
                       Each value is tensor of shape (B, 2, T) for stereo audio
            film_params: Optional dict with FiLM parameters from mixing features
                        Expected keys: 'proj_gamma', 'proj_beta' for projection layer,
                        'block_N_gamma', 'block_N_beta' for each transformer block N

        Returns:
            embedding: Track-level representation (B, embed_dim)
        """
        # Convert stems to mel spectrogram
        x = self.mel_preprocessor(stems_dict)  # (B, 8, n_mels, time_frames)

        # Debug: Check mel spectrogram
        if torch.isnan(x).any():
            print(f"[BandSplitEncoder] NaN in mel spectrogram after preprocessing!")
            print(f"  NaN count: {torch.isnan(x).sum()}/{x.numel()}")

        B = x.size(0)

        if self.backbone == 'cnn':
            return self._forward_cnn(x, B, film_params)
        else:
            return self._forward_roformer(x, B, film_params)

    def _forward_cnn(self, x, B, film_params):
        """CNN backbone forward pass (v1 architecture)."""
        # Process first sub-band
        start_idx = 0
        end_idx = self.split_size
        sub_band = x[:, :, start_idx:end_idx, :]

        if film_params is not None:
            gamma1 = film_params.get('gamma1_0', None)
            beta1 = film_params.get('beta1_0', None)
            gamma2 = film_params.get('gamma2_0', None)
            beta2 = film_params.get('beta2_0', None)
        else:
            gamma1 = beta1 = gamma2 = beta2 = None

        first_features = self.subnet_cnns[0](sub_band, gamma1, beta1, gamma2, beta2)
        _, out_channels, freq_dim, time_dim = first_features.shape

        concat_features = torch.empty(
            B, self.n_subbands * out_channels, freq_dim, time_dim,
            dtype=first_features.dtype, device=first_features.device
        )
        concat_features[:, :out_channels, :, :] = first_features

        for i in range(1, self.n_subbands):
            start_idx = i * self.overlap
            end_idx = start_idx + self.split_size
            sub_band = x[:, :, start_idx:end_idx, :]

            if film_params is not None:
                gamma1 = film_params.get(f'gamma1_{i}', None)
                beta1 = film_params.get(f'beta1_{i}', None)
                gamma2 = film_params.get(f'gamma2_{i}', None)
                beta2 = film_params.get(f'beta2_{i}', None)
            else:
                gamma1 = beta1 = gamma2 = beta2 = None

            features = self.subnet_cnns[i](sub_band, gamma1, beta1, gamma2, beta2)
            concat_features[:, i*out_channels:(i+1)*out_channels, :, :] = features

        # Flatten frequency and channel dims, keep time
        batch_size, channels, freq, time = concat_features.shape
        flattened = concat_features.view(batch_size, channels * freq, time)

        if torch.isnan(flattened).any():
            print(f"[BandSplitEncoder] NaN before attention pooling!")

        embedding = self.attention_pooling(flattened)

        if torch.isnan(embedding).any():
            print(f"[BandSplitEncoder] NaN in embedding after attention pooling!")

        return embedding

    def _forward_roformer(self, x, B, film_params):
        """Roformer backbone forward pass (v2 architecture)."""
        D = self._D

        # Extract per-band features using shared projection
        band_features = []
        for i in range(self.n_subbands):
            start_idx = i * self.overlap
            end_idx = start_idx + self.split_size
            sub_band = x[:, :, start_idx:end_idx, :]

            proj_gamma = None
            proj_beta = None
            if film_params is not None:
                proj_gamma = film_params.get('proj_gamma', None)
                proj_beta = film_params.get('proj_beta', None)

            feat = self.band_projection(sub_band, proj_gamma, proj_beta)
            band_features.append(feat)

        # Stack to (B, K, D, T) then permute to (B, K, T, D)
        x = torch.stack(band_features, dim=1)
        x = x.permute(0, 1, 3, 2)

        # Apply axial transformer blocks with FiLM conditioning
        for block_idx, block in enumerate(self.transformer_blocks):
            block_gamma = None
            block_beta = None
            if film_params is not None:
                block_gamma = film_params.get(f'block_{block_idx}_gamma', None)
                block_beta = film_params.get(f'block_{block_idx}_beta', None)
            x = block(x, block_gamma, block_beta)

        # Reshape to (B, K*D, T) for attention pooling
        B, K, T, D = x.shape
        x = x.permute(0, 1, 3, 2).reshape(B, K * D, T)

        if torch.isnan(x).any():
            print(f"[BandSplitEncoder] NaN before attention pooling!")

        embedding = self.attention_pooling(x)

        if torch.isnan(embedding).any():
            print(f"[BandSplitEncoder] NaN in embedding after attention pooling!")

        return embedding


class MixingFeatureEncoder(nn.Module):
    """
    MLP for encoding mixing features and generating FiLM parameters.

    Generates gamma/beta for:
    - Projection layer: D + D = 2*D
    - N transformer blocks: N * (D + D) = N * 2*D
    Total: 2*D + N*2*D = (1 + N) * 2 * D
    """

    # Must match BandSplitEncoder constants
    _D = 192
    _N_BLOCKS = 4

    def __init__(self, feature_dim, n_subbands, hidden_dim=256, backbone='roformer', cnn_out_channels=64):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_subbands = n_subbands
        self.backbone = backbone

        D = self._D
        N = self._N_BLOCKS

        # Feature embedding MLP
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if backbone == 'cnn':
            # CNN FiLM: per-subband gamma1(32)+beta1(32)+gamma2(C)+beta2(C)
            film_output_dim = n_subbands * (32 + 32 + cnn_out_channels + cnn_out_channels)
            self.cnn_out_channels = cnn_out_channels
        else:
            # Roformer FiLM: projection layer + N transformer blocks
            film_output_dim = 2 * D + N * 2 * D

        self.film_head = nn.Linear(hidden_dim, film_output_dim)

    def forward(self, features):
        """
        Generate FiLM parameters from mixing features.

        Args:
            features: Mixing features (B, feature_dim)

        Returns:
            film_params: Dict with FiLM parameters
        """
        D = self._D
        N = self._N_BLOCKS

        if torch.isnan(features).any():
            print(f"[MixingFeatureEncoder] NaN in input features!")
        if torch.isinf(features).any():
            print(f"[MixingFeatureEncoder] Inf in input features!")

        h = self.feature_mlp(features)

        if torch.isnan(h).any():
            print(f"[MixingFeatureEncoder] NaN after feature_mlp!")
        if torch.isinf(h).any():
            print(f"[MixingFeatureEncoder] Inf after feature_mlp!")

        film_flat = self.film_head(h)

        if torch.isnan(film_flat).any():
            print(f"[MixingFeatureEncoder] NaN after film_head!")

        film_params = {}
        idx = 0

        if self.backbone == 'cnn':
            C = self.cnn_out_channels
            for i in range(self.n_subbands):
                film_params[f'gamma1_{i}'] = film_flat[:, idx:idx + 32]; idx += 32
                film_params[f'beta1_{i}'] = film_flat[:, idx:idx + 32]; idx += 32
                film_params[f'gamma2_{i}'] = film_flat[:, idx:idx + C]; idx += C
                film_params[f'beta2_{i}'] = film_flat[:, idx:idx + C]; idx += C
        else:
            film_params['proj_gamma'] = film_flat[:, idx:idx + D]; idx += D
            film_params['proj_beta'] = film_flat[:, idx:idx + D]; idx += D
            for i in range(N):
                film_params[f'block_{i}_gamma'] = film_flat[:, idx:idx + D]; idx += D
                film_params[f'block_{i}_beta'] = film_flat[:, idx:idx + D]; idx += D

        return film_params


class BSFiLMEncoder(nn.Module):
    """
    Complete mixing style encoder using ResNet50 with FiLM conditioning.

    Combines:
    1. ResNet50 audio encoder (memory efficient)
    2. FiLM conditioning based on mixing features
    """

    def __init__(
        self,
        sample_rate=44100,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        split_size=20,
        overlap=10,
        channels=8,
        embed_dim=512,
        feature_dim=256,
        use_film=True,
        backbone='roformer'
    ):
        super().__init__()
        self.use_film = use_film

        # Audio encoder
        self.audio_encoder = BandSplitEncoder(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            split_size=split_size,
            overlap=overlap,
            channels=channels,
            embed_dim=embed_dim,
            backbone=backbone
        )

        # FiLM parameter generator (disabled in ablation mode)
        if use_film:
            self.film_encoder = MixingFeatureEncoder(
                feature_dim=feature_dim,
                n_subbands=self.audio_encoder.n_subbands,
                backbone=backbone
            )
        else:
            self.film_encoder = None

    def forward(self, stems_dict, mixing_features):
        """
        Forward pass with FiLM conditioning.

        Args:
            stems_dict: Dict with keys ['vocals', 'bass', 'drums', 'other']
                       Each value is tensor of shape (B, 2, T) for stereo audio
            mixing_features: Mixing features (B, feature_dim)

        Returns:
            embedding: Track-level representation (B, embed_dim)
        """
        # Generate FiLM parameters from mixing features (skip if disabled)
        film_params = None
        if self.use_film:
            # Debug: Check mixing features
            if torch.isnan(mixing_features).any():
                print(f"[BSFiLMEncoder] NaN in mixing_features input!")
                print(f"  NaN count: {torch.isnan(mixing_features).sum()}/{mixing_features.numel()}")

            film_params = self.film_encoder(mixing_features)

            # Debug: Check FiLM params
            for key, val in film_params.items():
                if torch.isnan(val).any():
                    print(f"[BSFiLMEncoder] NaN in film_params[{key}]!")
                    break

        # Encode audio (with or without FiLM conditioning)
        embedding = self.audio_encoder(stems_dict, film_params=film_params)  # (B, embed_dim)

        # Debug: Check final embedding
        if torch.isnan(embedding).any():
            print(f"[BSFiLMEncoder] NaN in final embedding!")
            print(f"  NaN count: {torch.isnan(embedding).sum()}/{embedding.numel()}")

        return embedding


class SongIdentityDiscriminator(nn.Module):
    """
    MLP discriminator for predicting song identity from mixing embeddings.

    Used in adversarial training to encourage the mixing encoder to remove
    song-specific content while preserving mixing style information.

    Architecture: input_dim → ReLU → Dropout → hidden_dim → ReLU → Dropout → output_dim
    """

    def __init__(self, input_dim=512, hidden_dim=512, output_dim=512, dropout=0.3):
        """
        Initialize discriminator.

        Args:
            input_dim: Input embedding dimension (mixing embedding size)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (song identity embedding size)
            dropout: Dropout probability
        """
        super(SongIdentityDiscriminator, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input mixing embedding, shape (B, input_dim)

        Returns:
            Predicted song identity embedding, shape (B, output_dim)
        """
        return self.network(x)


class PianoRollDiscriminator(nn.Module):
    """
    Discriminator that predicts piano roll from mixing embeddings.

    Used for adversarial training to remove musical content from embeddings.
    The discriminator learns to predict what notes are being played from the
    mixing style embedding, while the encoder learns (via GRL) to hide this
    information.
    """

    def __init__(self, input_dim=512, hidden_dim=512, dropout=0.3):
        """
        Initialize piano roll discriminator.

        Args:
            input_dim: Dimension of mixing embeddings (default: 512)
            hidden_dim: Hidden layer dimension (default: 512)
            dropout: Dropout probability (default: 0.3)
        """
        super(PianoRollDiscriminator, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 88 * 10)  # Output: 880 logits for (88, 10) piano roll
        )

    def forward(self, x):
        """
        Predict piano roll from mixing embeddings.

        Args:
            x: Mixing embeddings of shape (B, input_dim)

        Returns:
            Piano roll logits of shape (B, 88, 10)
        """
        logits = self.network(x)  # (B, 880)
        return logits.view(-1, 88, 10)  # Reshape to piano roll format


class DiscriminatorFiLMGenerator(nn.Module):
    """Generate FiLM parameters for ResNet50 layers from style embedding."""

    def __init__(self, embed_dim=512, hidden_dim=512):
        super().__init__()
        # ResNet50 layer output channels: [256, 512, 1024, 2048]
        self.total_params = 2 * (256 + 512 + 1024 + 2048)  # gamma + beta = 7680

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.total_params)
        )

    def forward(self, embedding):
        """
        Args:
            embedding: (B, embed_dim)
        Returns:
            Dict with gamma, beta for each layer
        """
        params = self.mlp(embedding)  # (B, 7680)

        # Parse into layer-specific parameters
        idx = 0
        film_params = {}
        for i, channels in enumerate([256, 512, 1024, 2048]):
            film_params[f'layer{i+1}'] = {
                'gamma': params[:, idx:idx+channels],
                'beta': params[:, idx+channels:idx+2*channels]
            }
            idx += 2 * channels

        return film_params


class FiLMResNet50Discriminator(nn.Module):
    """ResNet50 discriminator with FiLM conditioning for audio style transfer."""

    def __init__(self, embed_dim=512, hidden_dim=512, n_mels=80):
        super().__init__()

        # Import ResNet components
        from resnet import ResNet, Bottleneck

        # Create ResNet50 backbone (modify first conv for 1-channel input)
        self.resnet = ResNet(
            ResBlock=Bottleneck,
            layer_list=[3, 4, 6, 3],  # ResNet50 structure
            num_classes=1,  # Binary real/fake classification
            in_channels=1   # Single-channel mel spectrogram
        )

        # FiLM parameter generator
        self.film_generator = DiscriminatorFiLMGenerator(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with small values for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, mel_spec, style_embedding):
        """
        Args:
            mel_spec: (B, 1, n_mels, time_frames) - mono mel spectrogram
            style_embedding: (B, embed_dim) - mixing style embedding

        Returns:
            score: (B, 1) - real/fake logits (no sigmoid)
        """
        # Generate FiLM parameters from embedding
        film_params = self.film_generator(style_embedding)

        # ResNet forward with FiLM modulation
        x = self.resnet.conv1(mel_spec)
        x = self.resnet.batch_norm1(x)
        x = self.resnet.relu(x)
        x = self.resnet.max_pool(x)

        # Layer 1 (3 bottleneck blocks) + FiLM
        x = self.resnet.layer1(x)
        x = self._apply_film(x, film_params['layer1'])

        # Layer 2 (4 bottleneck blocks) + FiLM
        x = self.resnet.layer2(x)
        x = self._apply_film(x, film_params['layer2'])

        # Layer 3 (6 bottleneck blocks) + FiLM
        x = self.resnet.layer3(x)
        x = self._apply_film(x, film_params['layer3'])

        # Layer 4 (3 bottleneck blocks) + FiLM
        x = self.resnet.layer4(x)
        x = self._apply_film(x, film_params['layer4'])

        # Global pooling and classification
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        score = self.resnet.fc(x)

        return score

    def _apply_film(self, x, film_params):
        """Apply FiLM modulation: x = gamma * x + beta"""
        gamma = film_params['gamma'].unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = film_params['beta'].unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)
        return gamma * x + beta
