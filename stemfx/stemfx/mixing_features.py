"""
Utility functions for mixing feature extraction and audio augmentations.
"""

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from scipy.signal import butter, sosfilt


# =============================================================================
# Mixing Feature Extraction
# =============================================================================

class MixingFeatureExtractor:
    """Extract interpretable mixing features from audio stems."""

    def __init__(
        self,
        sample_rate=44100,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        use_detailed_spectral=False,
        n_spectral_bins=32
    ):
        """
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands
            use_detailed_spectral: If True, use detailed frequency curve instead of 3 bands
            n_spectral_bins: Number of frequency bins for detailed spectral features
        """
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.use_detailed_spectral = use_detailed_spectral
        self.n_spectral_bins = n_spectral_bins

        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )

    def get_feature_dim(self):
        """
        Get total feature dimension based on configuration.

        Returns:
            total_dim: Total number of features
        """
        # Per-stem features
        dynamics_dim = 6
        spectral_dim = 5 if not self.use_detailed_spectral else (self.n_spectral_bins + 2)
        stereo_dim = 3
        per_stem_dim = dynamics_dim + spectral_dim + stereo_dim

        # Total: 4 stems × per_stem_dim + 4 rel_loudness + 4 masking
        total_dim = 4 * per_stem_dim + 4 + 4

        return total_dim

    def extract_all_features(self, stems_dict):
        """
        Extract all mixing features from separated stems.

        Args:
            stems_dict: Dict with keys ['vocals', 'bass', 'drums', 'other']
                       Each value is tensor of shape (2, T) for stereo

        Returns:
            features: Dict of extracted features
        """
        features = {}

        # Compute mixture by summing all stems
        mixture = sum(stems_dict.values())  # (2, T)

        # Per-stem features
        for stem_name, stem_audio in stems_dict.items():
            features[f'{stem_name}_dynamics'] = self.extract_dynamics(stem_audio)
            features[f'{stem_name}_spectral'] = self.extract_spectral(stem_audio)
            features[f'{stem_name}_stereo'] = self.extract_stereo(stem_audio)

        # Relative loudness (stem vs mixture)
        mix_loudness = self.compute_loudness(mixture)
        for stem_name, stem_audio in stems_dict.items():
            stem_loudness = self.compute_loudness(stem_audio)
            features[f'{stem_name}_rel_loudness'] = stem_loudness - mix_loudness

        # Inter-stem masking
        features['masking'] = self.extract_masking(stems_dict)

        # Flatten all features into a single vector
        feature_vector = self._flatten_features(features)

        return feature_vector

    def extract_dynamics(self, audio):
        """Extract dynamics features: RMS, crest factor, loudness."""
        # audio: (2, T) stereo
        features = []

        # RMS
        rms = torch.sqrt(torch.mean(audio ** 2, dim=-1))  # (2,)
        # Ensure rms is 1D
        if rms.dim() > 1:
            rms = rms.squeeze()
        features.append(rms)
        if torch.isnan(rms).any():
            print(f"[extract_dynamics] NaN in RMS: {rms}")

        # Crest factor
        peak = torch.max(torch.abs(audio), dim=-1)[0]  # (2,)
        # Ensure peak is 1D
        if peak.dim() > 1:
            peak = peak.squeeze()
        crest = 20 * torch.log10(peak / (rms + 1e-8))
        features.append(crest)
        if torch.isnan(crest).any():
            print(f"[extract_dynamics] NaN in crest factor: {crest}, peak={peak}, rms={rms}")

        # Loudness (simplified LUFS approximation)
        loudness = self.compute_loudness(audio)
        # loudness is a scalar tensor, expand to (2,) for stereo
        loudness_stereo = torch.tensor([loudness.item(), loudness.item()], device=audio.device)
        features.append(loudness_stereo)  # Repeat for stereo
        if torch.isnan(loudness).any():
            print(f"[extract_dynamics] NaN in loudness: {loudness}")

        return torch.cat(features)  # (6,)

    def extract_spectral(self, audio):
        """
        Extract spectral features.

        Old mode (use_detailed_spectral=False):
            Returns 5 features: low_energy, mid_energy, high_energy, tilt, flatness

        New mode (use_detailed_spectral=True):
            Returns n_spectral_bins + 2 features:
            - n_spectral_bins: detailed frequency curve (subsampled mel energies)
            - tilt: spectral tilt
            - flatness: spectral flatness
        """
        # Move mel_transform to same device as audio
        if self.mel_transform.spectrogram.window.device != audio.device:
            self.mel_transform = self.mel_transform.to(audio.device)

        # Compute mel spectrogram
        mel_spec = self.mel_transform(audio)  # (2, n_mels, T)
        mel_spec_db = 10 * torch.log10(mel_spec + 1e-10)
        if torch.isnan(mel_spec_db).any():
            print(f"[extract_spectral] NaN in mel_spec_db after log10")

        # Average over time and channels
        mel_energy = mel_spec_db.mean(dim=(0, 2))  # (n_mels,)
        if torch.isnan(mel_energy).any():
            print(f"[extract_spectral] NaN in mel_energy: {mel_energy}")

        if not self.use_detailed_spectral:
            # OLD MODE: 3-band energy + tilt + flatness (5 features)
            # Spectral statistics
            low_band_bound = mel_energy.shape[0] // 4
            high_band_bound = mel_energy.shape[0] // 4 * 3
            low_energy = mel_energy[:low_band_bound].mean()  # Low frequencies
            mid_energy = mel_energy[low_band_bound:high_band_bound].mean()  # Mid frequencies
            high_energy = mel_energy[high_band_bound:].mean()  # High frequencies
            if torch.isnan(low_energy).any():
                print(f"[extract_spectral] NaN in low_energy: {low_energy}")
            if torch.isnan(mid_energy).any():
                print(f"[extract_spectral] NaN in mid_energy: {mid_energy}")
            if torch.isnan(high_energy).any():
                print(f"[extract_spectral] NaN in high_energy: {high_energy}")

            # Spectral tilt (slope of energy across frequency)
            freq_bins = torch.arange(self.n_mels, dtype=torch.float32, device=audio.device)
            # Handle case where mel_energy is constant (would cause NaN in corrcoef)
            if mel_energy.std() < 1e-6:
                tilt = torch.tensor(0.0, device=audio.device)
            else:
                tilt = torch.corrcoef(torch.stack([freq_bins, mel_energy]))[0, 1]
            if torch.isnan(tilt).any():
                print(f"[extract_spectral] NaN in tilt: {tilt}, mel_energy.std={mel_energy.std()}")

            # Spectral flatness (geometric mean / arithmetic mean)
            flatness = torch.exp(torch.mean(torch.log(mel_spec + 1e-10))) / (torch.mean(mel_spec) + 1e-10)
            if torch.isnan(flatness).any():
                print(f"[extract_spectral] NaN in flatness: {flatness}")

            features = torch.tensor([low_energy, mid_energy, high_energy, tilt, flatness], device=audio.device)
            return features  # (5,)

        else:
            # NEW MODE: Detailed frequency curve + tilt + flatness
            # Subsample mel_energy to n_spectral_bins
            if self.n_spectral_bins >= self.n_mels:
                # If requesting more bins than available, just use all mel bins
                detailed_curve = mel_energy
            else:
                # Interpolate to get n_spectral_bins
                indices = torch.linspace(0, self.n_mels - 1, self.n_spectral_bins, device=audio.device)
                # Use linear interpolation
                detailed_curve = torch.nn.functional.interpolate(
                    mel_energy.unsqueeze(0).unsqueeze(0),  # (1, 1, n_mels)
                    size=self.n_spectral_bins,
                    mode='linear',
                    align_corners=True
                ).squeeze()  # (n_spectral_bins,)

            # Spectral tilt (slope of energy across frequency)
            freq_bins = torch.arange(self.n_spectral_bins, dtype=torch.float32, device=audio.device)
            # Handle case where detailed_curve is constant (would cause NaN in corrcoef)
            if detailed_curve.std() < 1e-6:
                tilt = torch.tensor(0.0, device=audio.device)
            else:
                tilt = torch.corrcoef(torch.stack([freq_bins, detailed_curve]))[0, 1]
            if torch.isnan(tilt).any():
                print(f"[extract_spectral] NaN in tilt: {tilt}, detailed_curve.std={detailed_curve.std()}")

            # Spectral flatness (geometric mean / arithmetic mean)
            flatness = torch.exp(torch.mean(torch.log(mel_spec + 1e-10))) / (torch.mean(mel_spec) + 1e-10)
            if torch.isnan(flatness).any():
                print(f"[extract_spectral] NaN in flatness: {flatness}")

            # Combine: detailed_curve (n_spectral_bins) + tilt + flatness
            features = torch.cat([detailed_curve, torch.tensor([tilt, flatness], device=audio.device)])
            return features  # (n_spectral_bins + 2,)

    def extract_stereo(self, audio):
        """Extract stereo features: ILD, correlation, MSR."""
        L, R = audio[0], audio[1]  # (T,)

        # ILD (Inter-channel Level Difference)
        rms_L = torch.sqrt(torch.mean(L ** 2))
        rms_R = torch.sqrt(torch.mean(R ** 2))
        ild = 20 * torch.log10(rms_L / (rms_R + 1e-8))
        if torch.isnan(ild).any():
            print(f"[extract_stereo] NaN in ILD: {ild}, rms_L={rms_L}, rms_R={rms_R}")

        # Correlation
        L_centered = L - L.mean()
        R_centered = R - R.mean()
        corr = (L_centered * R_centered).sum() / (
            torch.sqrt((L_centered ** 2).sum() * (R_centered ** 2).sum()) + 1e-8
        )
        if torch.isnan(corr).any():
            print(f"[extract_stereo] NaN in correlation: {corr}")

        # Mid-Side Ratio (MSR)
        mid = (L + R) / 2
        side = (L - R) / 2
        E_mid = torch.mean(mid ** 2)
        E_side = torch.mean(side ** 2)
        msr = E_side / (E_mid + 1e-8)
        if torch.isnan(msr).any():
            print(f"[extract_stereo] NaN in MSR: {msr}, E_mid={E_mid}, E_side={E_side}")

        features = torch.tensor([ild, corr, msr], device=L.device)
        return features  # (3,)

    def extract_masking(self, stems_dict):
        """Extract inter-stem masking features."""
        # Move mel_transform to correct device (use first stem's device)
        first_stem_audio = next(iter(stems_dict.values()))
        if self.mel_transform.spectrogram.window.device != first_stem_audio.device:
            self.mel_transform = self.mel_transform.to(first_stem_audio.device)

        # Compute mel spectrograms for all stems
        stem_mels = {}
        for stem_name, stem_audio in stems_dict.items():
            mel = self.mel_transform(stem_audio)  # (2, n_mels, T)
            stem_mels[stem_name] = mel.mean(dim=0)  # Average channels: (n_mels, T)
            if torch.isnan(stem_mels[stem_name]).any():
                print(f"[extract_masking] NaN in mel spectrogram for {stem_name}")

        # Compute masking dominance for each stem
        masking_features = []
        stem_names = ['vocals', 'bass', 'drums', 'other']

        for i, stem_name in enumerate(stem_names):
            stem_energy = stem_mels[stem_name]  # (n_mels, T)

            # Max energy from other stems
            other_energies = [stem_mels[name] for j, name in enumerate(stem_names) if j != i]
            max_other = torch.stack(other_energies).max(dim=0)[0]  # (n_mels, T)

            # Dominance margin
            dominance = stem_energy - max_other

            # Masking indicator (sigmoid-based)
            beta, tau = 0.0, 1.0
            masking = torch.sigmoid((beta - dominance) / tau)

            # Average masking across time and frequency
            avg_masking = masking.mean()
            if torch.isnan(avg_masking).any():
                print(f"[extract_masking] NaN in avg_masking for {stem_name}: {avg_masking}")
            masking_features.append(avg_masking)

        return torch.stack(masking_features)  # (4,)

    def compute_loudness(self, audio):
        """Simplified LUFS loudness computation."""
        # K-weighting approximation (simplified)
        rms = torch.sqrt(torch.mean(audio ** 2))
        loudness = -0.691 + 10 * torch.log10(rms ** 2 + 1e-10)
        if torch.isnan(loudness).any():
            print(f"[compute_loudness] NaN in loudness: {loudness}, rms={rms}")
        return loudness

    def _flatten_features(self, features):
        """Flatten feature dict into a single vector."""
        vectors = []
        device = None
        for key in sorted(features.keys()):
            feat = features[key]
            if isinstance(feat, torch.Tensor):
                vectors.append(feat.flatten())
                if device is None:
                    device = feat.device
            else:
                if device is None:
                    device = torch.device('cpu')  # Default to CPU if no tensor yet
                vectors.append(torch.tensor([feat], device=device))

        result = torch.cat(vectors)

        # Clamp extreme values to prevent overflow
        # Replace inf/-inf with large finite values
        result = torch.clamp(result, min=-100.0, max=100.0)

        # Replace NaN with zeros (NaN can occur for silent/corrupted audio)
        # torch.clamp doesn't handle NaN, so we must replace it explicitly
        if torch.isnan(result).any():
            nan_mask = torch.isnan(result)
            nan_count = nan_mask.sum().item()
            print(f"[MixingFeatureExtractor] WARNING: {nan_count} NaN values detected in features, replacing with 0.0")
            print(f"  This typically indicates silent or corrupted audio")
            result = torch.where(nan_mask, torch.zeros_like(result), result)

        # Final sanity check
        if torch.isinf(result).any():
            raise ValueError(
                f"Inf detected in features after clamping! This should not happen.\n"
                f"Feature values: {result}"
            )

        return result


# =============================================================================
# Audio Augmentations (Degradations)
# =============================================================================

class AudioAugmenter:
    """Apply mixing degradation augmentations to separated stems."""

    def __init__(self, sample_rate=44100, gain_range=9.0, prob=0.5):
        self.sr = sample_rate
        self.gain_range = gain_range
        self.prob = prob

    def augment_stems(self, stems_dict):
        """
        Apply random degradations to stems.

        Args:
            stems_dict: Dict with keys ['vocals', 'bass', 'drums', 'other']
                       Each value is tensor of shape (2, T) for stereo

        Returns:
            augmented_stems: Dict with augmented stems
        """
        aug_stems = {}

        for stem_name, stem_audio in stems_dict.items():
            audio = stem_audio.clone()

            # Gain imbalance
            if torch.rand(1) < self.prob:
                gain_db = torch.rand(1) * 2 * self.gain_range - self.gain_range
                gain_linear = 10 ** (gain_db / 20)
                audio = audio * gain_linear

            # Spectral tilt (simple EQ)
            if torch.rand(1) < self.prob:
                audio = self.apply_spectral_tilt(audio)

            # Compression
            if torch.rand(1) < self.prob:
                audio = self.apply_compression(audio)

            # Bandwidth limiting
            if torch.rand(1) < self.prob:
                audio = self.apply_bandwidth_limit(audio)

            aug_stems[stem_name] = audio

        # Apply reverb to mixture (after summing)
        if torch.rand(1) < self.prob:
            mixture = sum(aug_stems.values())
            mixture = self.apply_reverb(mixture)
            # Redistribute reverb proportionally
            total_energy = sum([torch.mean(s ** 2) for s in aug_stems.values()]) + 1e-8
            for stem_name in aug_stems.keys():
                stem_energy = torch.mean(aug_stems[stem_name] ** 2)
                proportion = stem_energy / total_energy
                aug_stems[stem_name] = aug_stems[stem_name] + mixture * proportion * 0.3

        return aug_stems

    def apply_spectral_tilt(self, audio):
        """Apply spectral tilt using simple shelving filter."""
        # High-pass or low-pass emphasis
        if torch.rand(1) < 0.5:
            # High-frequency boost
            sos = butter(2, 2000, btype='high', fs=self.sr, output='sos')
        else:
            # Low-frequency boost
            sos = butter(2, 500, btype='low', fs=self.sr, output='sos')

        audio_np = audio.cpu().numpy()
        filtered = sosfilt(sos, audio_np, axis=-1)
        return torch.from_numpy(filtered).to(audio.device).float()

    def apply_compression(self, audio, threshold=-20, ratio=4):
        """Apply simple dynamic range compression."""
        # Convert to dB
        audio_db = 20 * torch.log10(torch.abs(audio) + 1e-8)

        # Compress above threshold
        mask = audio_db > threshold
        compressed_db = audio_db.clone()
        compressed_db[mask] = threshold + (audio_db[mask] - threshold) / ratio

        # Convert back to linear
        compressed = torch.sign(audio) * (10 ** (compressed_db / 20))
        return compressed

    def apply_bandwidth_limit(self, audio):
        """Apply bandwidth limiting (low-pass filter)."""
        cutoff = torch.rand(1) * 8000 + 4000  # 4-12 kHz
        sos = butter(4, cutoff.item(), btype='low', fs=self.sr, output='sos')

        audio_np = audio.cpu().numpy()
        filtered = sosfilt(sos, audio_np, axis=-1)
        return torch.from_numpy(filtered).to(audio.device).float()

    def apply_reverb(self, audio, decay=0.5):
        """Apply simple reverb using exponential decay."""
        # Simple reverb impulse response
        reverb_length = int(self.sr * decay)
        t = torch.linspace(0, decay, reverb_length, device=audio.device)
        impulse = torch.exp(-t / (decay / 4)) * torch.randn(reverb_length, device=audio.device) * 0.1

        # Convolve with impulse response (per channel)
        reverb_audio = []
        for ch in range(audio.shape[0]):
            conv = F.conv1d(
                audio[ch:ch+1].unsqueeze(0),
                impulse.unsqueeze(0).unsqueeze(0),
                padding=reverb_length // 2
            )
            reverb_audio.append(conv.squeeze(0))

        reverb_audio = torch.cat(reverb_audio, dim=0)

        # Mix with dry signal
        wet_dry = 0.3
        return audio * (1 - wet_dry) + reverb_audio[:, :audio.shape[1]] * wet_dry
