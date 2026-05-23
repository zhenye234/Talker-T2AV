"""WhisperX-VAE — minimal inference-only audio autoencoder.

This is a stripped-down port of the full lightning_module.CodecLightningModule
in the X-Codec-2.0 training repo, keeping only what's needed to run

    audio  --(encode)-->  32-d latent (25 Hz)  --(decode)-->  audio

i.e. the audio path Talker-T2AV uses. Discriminators, auxiliary semantic
decoders, criteria, optimizers, training-loop methods etc. are all dropped.

Architecture (matches the released checkpoint at HKUSTAudio/Talker-T2AV
under whisperx-vae/model.ckpt):

  CodecEnc  (DAC Encoder, 24 kHz waveform -> 1280-d 25 Hz acoustic feats)
                                       +
  semantic_model  (Whisper-Large-v3 encoder, 50 Hz -> mean-pool to 25 Hz, 1280-d)
                                       |
                                  vae.in_proj   (1280 -> 32)  ── posterior mean only
                                       |
                                latent  (32-d, 25 Hz)
                                       |
                                  vae.out_proj  (32 -> 1280)
                                       |
                                   CodecDec  (-> 24 kHz waveform)

The Whisper backbone is constructed without downloading anything: we use
the hard-coded Large-v3 dims and let the trained weights from our ckpt
overwrite them via `load_state_dict(strict=False)`.

Top-level entry point used by `speech_llm.py`:

    from whisperx_vae import WhisperXVAE
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = WhisperXVAE(ck["hyper_parameters"]["cfg"])
    model.load_state_dict(ck["state_dict"], strict=False)
    model.eval()

    latent = model.encode(wav_b1T, mel_b11ft, return_layout="btc")
    wav_out = model.decode(latent, latent_layout="btc")
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torchaudio
import whisper
from whisper.model import ModelDimensions, Whisper

# We re-use the unmodified weight-norm conv / Snake activation primitives
# from descript-audio-codec, but vendor our own Encoder/Decoder below
# because Talker-T2AV uses non-standard odd strides (e.g. 5) that need an
# `output_padding=1` fix in DecoderBlock that upstream DAC does not have.
from dac.nn.layers import Snake1d, WNConv1d, WNConvTranspose1d


# ---------------------------------------------------------------------------
# DAC Encoder / Decoder — vendored from descript-audio-codec/dac/model/dac.py
# ---------------------------------------------------------------------------
# Modifications relative to upstream:
#   DecoderBlock's WNConvTranspose1d gets `output_padding=0 if stride % 2 == 0 else 1`
#   so that odd strides (used by WhisperX-VAE) produce the expected output length.
# ---------------------------------------------------------------------------

class _ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class _EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            _ResidualUnit(dim // 2, dilation=1),
            _ResidualUnit(dim // 2, dilation=3),
            _ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2, dim,
                kernel_size=2 * stride, stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, d_model: int = 64, strides=(2, 4, 8, 8), d_latent: int = 64):
        super().__init__()
        block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            block += [_EncoderBlock(d_model, stride=stride)]
        block += [Snake1d(d_model), WNConv1d(d_model, d_latent, kernel_size=3, padding=1)]
        self.block = nn.Sequential(*block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class _DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim, output_dim,
                kernel_size=2 * stride, stride=stride,
                padding=math.ceil(stride / 2),
                # Talker-T2AV addition: keep output length aligned for odd strides.
                output_padding=0 if stride % 2 == 0 else 1,
            ),
            _ResidualUnit(output_dim, dilation=1),
            _ResidualUnit(output_dim, dilation=3),
            _ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, input_channel, channels, rates, d_out: int = 1):
        super().__init__()
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]
        for i, stride in enumerate(rates):
            input_dim = channels // 2 ** i
            output_dim = channels // 2 ** (i + 1)
            layers += [_DecoderBlock(input_dim, output_dim, stride)]
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Whisper Large-v3 (128-mel) hard-coded dimensions — avoids needing the
# OpenAI .pt cached on disk. Our ckpt re-supplies all 1259 weight tensors.
_WHISPER_LARGE_V3_DIMS = ModelDimensions(
    n_mels=128,
    n_audio_ctx=1500,
    n_audio_state=1280,
    n_audio_head=20,
    n_audio_layer=32,
    n_vocab=51866,
    n_text_ctx=448,
    n_text_state=1280,
    n_text_head=20,
    n_text_layer=32,
)


class VAEBottleneck(nn.Module):
    """Tiny VAE bottleneck — 1×1 conv in/out around a Gaussian latent.
    For inference we only ever use the posterior mean (no sampling).
    """

    def __init__(self, input_dim: int, codebook_dim: int):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.in_proj = WNConv1d(input_dim, codebook_dim * 2, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)


class WhisperXVAE(nn.Module):
    """Minimal WhisperX-VAE wrapper. `cfg` is the OmegaConf object stored
    inside the ckpt under `hyper_parameters.cfg`."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        enccfg = cfg.model.codec_encoder
        deccfg = cfg.model.codec_decoder

        self.CodecEnc = Encoder(enccfg.in_dim, enccfg.ratios, enccfg.hidden_dim)
        self.CodecDec = Decoder(enccfg.hidden_dim, deccfg.hidden_dim, deccfg.ratios)
        self.vae = VAEBottleneck(input_dim=enccfg.hidden_dim, codebook_dim=deccfg.z_dim)

        # Whisper architecture only — weights come from the ckpt.
        self.semantic_model = Whisper(_WHISPER_LARGE_V3_DIMS)
        self.semantic_model.eval()
        self.semantic_model.requires_grad_(False)

    @torch.no_grad()
    def encode(
        self,
        wav: torch.Tensor,
        feats: torch.Tensor,
        return_layout: str = "btc",
    ) -> torch.Tensor:
        """Wave → 32-d 25 Hz latent.

        Args:
            wav:   (B, 1, T) or (B, T)  raw waveform at cfg.dataset.sr (24 kHz)
            feats: (B, 1, n_mels, n_frames) Whisper log-mel of the *16 kHz*
                   resampled wav, already pad/trimmed to 30 s.
            return_layout: "btc" -> (B, T_lat, z_dim);
                           "bct" -> (B, z_dim, T_lat).
        """
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)

        # codec encoder
        residual_acoustic_emb = self.CodecEnc(wav)  # (B, hidden, T_lat)

        # semantic branch
        feats_in = feats[:, 0, :, :]                # (B, n_mels, n_frames)
        semantic_target = self.semantic_model.embed_audio(feats_in)  # (B, T_50, 1280)

        # crop to twice the codec time axis
        semantic_target = semantic_target[:, : residual_acoustic_emb.shape[2] * 2, :]

        # 50 Hz → 25 Hz via pair-mean
        T = semantic_target.shape[1]
        T2 = (T // 2) * 2
        semantic_target = semantic_target[:, :T2, :]
        semantic_target_25 = semantic_target.view(
            semantic_target.size(0), T2 // 2, 2, semantic_target.size(-1)
        ).mean(dim=2)
        semantic_25_bct = semantic_target_25.transpose(1, 2).contiguous()  # (B, 1280, T_lat)

        vae_in = residual_acoustic_emb + semantic_25_bct

        # posterior mean (deterministic)
        mean, _scale = self.vae.in_proj(vae_in).chunk(2, dim=1)  # (B, z_dim, T_lat)

        if return_layout.lower() == "btc":
            return mean.transpose(1, 2).contiguous()
        elif return_layout.lower() == "bct":
            return mean
        raise ValueError(f"return_layout must be 'btc' or 'bct', got {return_layout!r}")

    @torch.no_grad()
    def decode(self, latent: torch.Tensor, latent_layout: str = "btc") -> torch.Tensor:
        """32-d latent → 24 kHz waveform.

        Args:
            latent: (B, T_lat, z_dim) for "btc" or (B, z_dim, T_lat) for "bct".
        Returns:
            (B, 1, T) waveform.
        """
        if latent_layout.lower() == "btc":
            latent = latent.transpose(1, 2).contiguous()
        elif latent_layout.lower() != "bct":
            raise ValueError(f"latent_layout must be 'btc' or 'bct', got {latent_layout!r}")

        z = self.vae.out_proj(latent)
        return self.CodecDec(z)


# ---------------------------------------------------------------------------
# Convenience helpers (used by both speech_llm.py and standalone reconstruction)
# ---------------------------------------------------------------------------

def load_audio_for_codec(path: str, target_sr: int = 24000) -> torch.Tensor:
    """Load + mono + resample to target_sr; clamp to [-1, 1]. Returns (1, T)."""
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
    return wav.clamp(-1.0, 1.0)


def compute_whisper_mel(wav_1xT: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Whisper expects 16 kHz mono. Returns (1, 1, n_mels=128, n_frames=3000)."""
    audio_1d = wav_1xT.squeeze(0)
    audio_1d = whisper.pad_or_trim(audio_1d)
    mel = whisper.log_mel_spectrogram(audio_1d, n_mels=128)
    return mel.unsqueeze(0).unsqueeze(0).to(device)
