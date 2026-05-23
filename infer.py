#!/usr/bin/env python
"""Talker-T2AV — single-sample inference entry point.

Generates synchronized speech + talking-head video from a text prompt,
conditioned on a reference voice (for speaker identity) and a reference
identity image / motion (for face appearance).

Usage:

    # default: run on bundled sample under ./samples/
    python infer.py

    # custom inputs:
    python infer.py \\
        --text "你好，世界。" \\
        --ref-audio path/to/voice.wav \\
        --ref-motion path/to/lia-x-feature.pt \\
        --ref-video path/to/identity.mp4 \\
        --output out.mp4

Optional env vars (everything else has a sensible default baked in below):
    CHECKPOINT_DIR        Path to checkpoint dir (model.safetensors etc.)
    WHISPERVAE_CKPT       WhisperX-VAE audio autoencoder ckpt
    WHISPERVAE_DIR        WhisperX-VAE source (for `from lightning_module import ...`)
    LIAX_CKPT, LIAX_CODE_DIR    LIA-X video motion autoencoder
    WAVLM_CKPT, SPK_VERIFICATION_DIR    WavLM speaker encoder
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from typing import List

_HERE = os.path.dirname(os.path.abspath(__file__))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import safetensors.torch  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
import whisper  # noqa: E402

sys.path.insert(0, _HERE)

from speech_llm import ModelArguments, init_model  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

# Disable cuDNN — matches eval recipe used to produce paper numbers.
torch.backends.cudnn.enabled = False


# ============================================================================
# Constants & motion-latent normalization stats
# ============================================================================

PS = 4                          # patch size (frames per patch) — overridden from model at runtime
DELAY = 5                       # motion lags speech by DELAY patches
FIXED_MOTION_LEN = 25 * 30      # 750 frames, 25 fps × 30 s
MAX_AUDIO_SAMPLES = 16000 * 30  # 480000 samples, 16 kHz × 30 s
FEAT_DIM = 40                   # 40-dim LIA-X motion latent dimension
SPEECH_DIM = 32                 # 32-dim speech latent dimension

# LIA-X is the video motion autoencoder. The Python sources (networks/),
# CUDA kernel sources (networks/op/*.{cpp,cu}), and motion-stat numpy arrays
# are vendored under ./lia_x/. Upstream:
#     https://github.com/wyhsirius/LIA-X
# Generator weights (lia-x.pt) are not vendored — set LIAX_CKPT or place
# the .pt at ./deps/LIA-X/lia-x.pt to use the default.
LIAX_CKPT = os.environ.get(
    "LIAX_CKPT",
    os.path.join(_HERE, "deps", "LIA-X", "lia-x.pt"),
)

# Motion stats live next to the vendored LIA-X sources.
_LIAX_PKG_DIR = os.path.join(_HERE, "lia_x")
MOTION_MEAN = np.load(os.path.join(_LIAX_PKG_DIR, "motion_mean.npy")).astype(np.float32)  # (40,)
MOTION_STD = np.load(os.path.join(_LIAX_PKG_DIR, "motion_std.npy")).astype(np.float32)    # (40,)
MOTION_STD = np.clip(MOTION_STD, a_min=1e-6, a_max=None)


def normalize_motion(motion_np):
    return (motion_np - MOTION_MEAN) / MOTION_STD


def denormalize_motion(motion_np):
    return motion_np * MOTION_STD + MOTION_MEAN


# ============================================================================
# Tokenization / data loading helpers
# ============================================================================

def build_chat_and_spans(
    tokenizer, model_max_length, speech_len_frames, motion_len_frames,
    patch_size=4, text="",
    speech_tok="<SPEECH_FRAME>", motion_tok="<MOTION_FRAME>",
):
    """Build chat template at patch-level. speech_len_frames /
    motion_len_frames are in frames; placeholder length is in patches
    (one token per patch)."""
    n_speech_patches = math.ceil(speech_len_frames / patch_size)
    n_motion_patches = motion_len_frames // patch_size
    placeholder_len = max(n_speech_patches, n_motion_patches + DELAY)
    speech_placeholder = "".join([speech_tok] * placeholder_len)

    chat = [
        {"role": "user", "content": text},
        {"role": "assistant", "content": speech_placeholder},
    ]

    try:
        ids = tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=False,
            enable_thinking=False, padding=False,
            truncation=True, max_length=model_max_length,
        )
    except TypeError:
        ids = tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=False,
            padding=False, truncation=True, max_length=model_max_length,
        )

    ids = ids.data["input_ids"] if hasattr(ids, "data") else ids
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    speech_id = tokenizer.convert_tokens_to_ids(speech_tok)
    speech_pos = (input_ids[0] == speech_id).nonzero(as_tuple=True)[0]
    speech_span = torch.tensor([int(speech_pos[0]), int(speech_pos[-1] + 1)])
    return input_ids, attention_mask, speech_span


def load_gt_motion(gt_pt_path, Lm=None):
    """Load reference 40-dim LIA-X motion latent from .pt file.
    Returns (gt_raw, gt_norm) — both (T, 40) numpy arrays."""
    alpha = torch.load(gt_pt_path, map_location="cpu", weights_only=True).detach().float().numpy()
    if Lm is None:
        Lm = min(alpha.shape[0], FIXED_MOTION_LEN)
    gt_raw = alpha[:Lm, :FEAT_DIM]
    gt_norm = normalize_motion(gt_raw)
    return gt_raw, gt_norm


def load_audio_and_get_mel(wav_path, device):
    """Load reference audio and return:
        mel:           (1, 1, 128, T)        for WhisperX-VAE encoder
        audio_24k:     (1, 1, T_24k)         for WhisperX-VAE audio path
        audio_16k_pad: (1, 1, T_16k=480000)  for WavLM speaker encoder
        speech_len:    audio frames at 25 fps
    """
    audio_raw, sample_rate = torchaudio.load(wav_path)
    if audio_raw.shape[0] > 1:
        audio_raw = audio_raw.mean(dim=0, keepdim=True)

    audio_16k = (
        torchaudio.transforms.Resample(sample_rate, 16000)(audio_raw)
        if sample_rate != 16000 else audio_raw
    )
    audio_24k = (
        torchaudio.transforms.Resample(sample_rate, 24000)(audio_raw)
        if sample_rate != 24000 else audio_raw
    )

    # Trim to 30 s max
    audio_16k = audio_16k[:, :MAX_AUDIO_SAMPLES]
    audio_24k = audio_24k[:, :int(24000 * 30)]

    audio_duration = audio_16k.size(1) / 16000.0
    speech_len = max(1, min(int(audio_duration * 25), FIXED_MOTION_LEN))

    audio_pad = whisper.pad_or_trim(audio_16k.squeeze(0))
    mel = whisper.log_mel_spectrogram(audio_pad, n_mels=128)

    audio_16k_pad = whisper.pad_or_trim(audio_16k)  # (1, 480000)

    return (
        mel.unsqueeze(0).unsqueeze(0).to(device),
        audio_24k.unsqueeze(0).to(device),
        audio_16k_pad.unsqueeze(0).to(device),
        speech_len,
    )


# ============================================================================
# Autoregressive generation (patch-level continuation)
# ============================================================================

def ar_generate_motion_continuation(
    model, tokenizer, gt_pt_path, wav_path, text="",
    model_max_length=1000, cfm_steps=20,
    cfm_temperature=0.3, cfm_cfg=1.0,
    gt_prefix_seconds=3.0, fps=25,
):
    """Patch-level continuation (matches speech_llm.py:generate()).
    Feeds the reference (GT prefix) through the PatchEncoder, then runs the
    AR loop at patch-level.

    Returns:
        pred_motion (Lm_frames, 40) raw,
        gt_raw      (Lm_frames, 40) raw,
        gen_wav_24k torch.Tensor,
        pred_speech_latents
    """
    device = next(model.parameters()).device
    _PS = getattr(model, "patch_size", PS)
    _DELAY = getattr(model, "delay", DELAY)
    p = getattr(model, "context_patches", 1)
    ctx_frames = p * _PS

    mel, audio_24k, audio_16k_pad, speech_len = load_audio_and_get_mel(wav_path, device)

    gt_raw, gt_motion = load_gt_motion(gt_pt_path)
    motion_len_raw = gt_motion.shape[0]
    motion_len = (motion_len_raw // _PS) * _PS
    if motion_len < _PS:
        motion_len = _PS
    gt_motion = gt_motion[:motion_len]
    gt_raw = gt_raw[:motion_len]

    input_ids, attention_mask, speech_span = build_chat_and_spans(
        tokenizer, model_max_length, speech_len, motion_len,
        patch_size=_PS, text=text,
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    s0, _s1 = speech_span.tolist()

    gt_motion_tensor = torch.from_numpy(gt_motion).to(dtype=torch.bfloat16, device=device)
    first_motion = gt_motion_tensor[0, :].float().unsqueeze(0)  # (1, 40)

    # Compute prefix in patches (aligned to PS)
    prefix_frames = int(gt_prefix_seconds * fps)
    prefix_frames = max(0, min(prefix_frames, motion_len))
    prefix_frames = (prefix_frames // _PS) * _PS
    prefix_patches = prefix_frames // _PS

    if hasattr(model.llm, "config"):
        model.llm.config.use_cache = True

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        text_emb = model.llm.get_input_embeddings()(input_ids)
        D = text_emb.size(-1)

        speech_emb, speech_latent = model.get_speech_embeddings(
            audio_24k, mel, torch.tensor([speech_len])
        )
        Ls_frames = speech_emb.size(1)
        spk_emb = model.get_speaker_embeddings(audio_16k_pad, torch.tensor([speech_len]))

        gt_speech_lat = speech_latent[0].to(dtype=torch.float32, device=device)
        motion_emb = model.projector_motion(gt_motion_tensor.unsqueeze(0))
        Lm_frames = motion_len

        n_speech_patches = math.ceil(Ls_frames / _PS)
        n_motion_patches = Lm_frames // _PS
        total_patches = max(n_speech_patches, n_motion_patches + _DELAY)

        speech_prefix_patches = prefix_patches + _DELAY
        speech_prefix_frames = speech_prefix_patches * _PS

        prefix_total_patches = min(speech_prefix_patches, total_patches)
        prefix_total_frames = prefix_total_patches * _PS

        text_part = text_emb[:, :s0, :]

        if prefix_total_patches > 0:
            speech_only_prefix = speech_emb.new_zeros((prefix_total_frames, D))
            sp_fill = min(Ls_frames, prefix_total_frames)
            if sp_fill > 0:
                speech_only_prefix[:sp_fill] = speech_emb[0, :sp_fill]

            motion_only_prefix = speech_emb.new_zeros((prefix_total_frames, D))
            delay_frames = _DELAY * _PS
            motion_start = delay_frames
            motion_end = min(prefix_total_frames, delay_frames + Lm_frames)
            n_motion_fill = motion_end - motion_start
            if n_motion_fill > 0:
                motion_only_prefix[motion_start:motion_end] = motion_emb[0, :n_motion_fill]

            speech_only_prefix = speech_only_prefix.view(prefix_total_patches, _PS, D)
            motion_only_prefix = motion_only_prefix.view(prefix_total_patches, _PS, D)

            speech_prefix_tokens = model.speech_patch_encoder(speech_only_prefix.unsqueeze(0))
            motion_prefix_tokens = model.motion_patch_encoder(motion_only_prefix.unsqueeze(0))
            prefix_patch_tokens = speech_prefix_tokens + motion_prefix_tokens

            if hasattr(model, "task_embedding"):
                t2sv_tag = torch.ones(1, dtype=torch.long, device=device)
                task_emb = model.task_embedding(t2sv_tag)
                prefix_patch_tokens[:, 0, :] = prefix_patch_tokens[:, 0, :] + task_emb

            full_embeds = torch.cat([text_part, prefix_patch_tokens], dim=1)
        else:
            full_embeds = text_part

        attn_mask = torch.ones((1, full_embeds.size(1)), dtype=torch.long, device=device)
        out = model.llm(
            inputs_embeds=full_embeds,
            attention_mask=attn_mask,
            use_cache=True,
            output_hidden_states=True,
        )
        past_kv = out.past_key_values
        h_cur = out.hidden_states[-1][:, -1, :]

        speech_patch_hist: List[torch.Tensor] = []
        if speech_prefix_frames > 0:
            sp_hist_frames = min(speech_prefix_frames, Ls_frames)
            lat_for_hist = gt_speech_lat[:sp_hist_frames]
            pad_hist = (_PS - sp_hist_frames % _PS) % _PS
            if pad_hist > 0:
                lat_for_hist = torch.nn.functional.pad(lat_for_hist, (0, 0, 0, pad_hist))
            n_hist_patches = lat_for_hist.size(0) // _PS
            for pi in range(n_hist_patches):
                speech_patch_hist.append(lat_for_hist[pi * _PS:(pi + 1) * _PS].detach())

        motion_patch_hist: List[torch.Tensor] = []
        gt_motion_f32 = gt_motion_tensor.float()
        if prefix_frames > 0:
            m_hist_frames = min(prefix_frames, Lm_frames)
            mlat_for_hist = gt_motion_f32[:m_hist_frames]
            pad_mhist = (_PS - m_hist_frames % _PS) % _PS
            if pad_mhist > 0:
                mlat_for_hist = torch.nn.functional.pad(mlat_for_hist, (0, 0, 0, pad_mhist))
            n_mhist_patches = mlat_for_hist.size(0) // _PS
            for pi in range(n_mhist_patches):
                motion_patch_hist.append(mlat_for_hist[pi * _PS:(pi + 1) * _PS].detach())

        remain_patches = total_patches - prefix_total_patches + 15  # extra margin

        gen_speech_latents: List[torch.Tensor] = []
        gen_motion_latents: List[torch.Tensor] = []

        def llm_step_one(past_kv_, attn_mask_, token_embed):
            out_ = model.llm(
                inputs_embeds=token_embed,
                attention_mask=attn_mask_,
                past_key_values=past_kv_,
                use_cache=True,
                output_hidden_states=True,
            )
            return out_.past_key_values, out_.hidden_states[-1][:, -1, :]

        use_stop = hasattr(model, "head_stop")
        stop_streak = 0
        min_gen_patches = 7
        stop_k = 3
        stop_threshold = 0.5

        for step in range(max(0, remain_patches)):
            if use_stop and step >= min_gen_patches:
                logits = model.head_stop(h_cur.float().unsqueeze(1))
                p_stop = torch.sigmoid(logits.float())[0, 0].item()
                if p_stop >= stop_threshold:
                    stop_streak += 1
                    if stop_streak >= stop_k:
                        break
                else:
                    stop_streak = 0

            with torch.autocast(device_type="cuda", enabled=False):
                speech_cond = torch.zeros(
                    (1, SPEECH_DIM, ctx_frames), device=device, dtype=torch.float32
                )
                if len(speech_patch_hist) > 0:
                    k = min(p, len(speech_patch_hist))
                    hist_patches = torch.cat(speech_patch_hist[-k:], dim=0)
                    speech_cond[:, :, ctx_frames - k * _PS:ctx_frames] = (
                        hist_patches.transpose(0, 1).unsqueeze(0)
                    )

                speech_patch = model.speech_cfm(
                    mu=h_cur.float(),
                    global_cond=spk_emb.float(),
                    n_timesteps=int(cfm_steps),
                    patch_size=_PS,
                    cond=speech_cond,
                    temperature=float(cfm_temperature),
                    cfg_value=float(cfm_cfg),
                    sway_sampling_coef=1.0,
                    use_cfg_zero_star=True,
                )

            speech_frames = speech_patch.transpose(1, 2)
            gen_speech_latents.append(speech_frames)
            speech_patch_hist.append(speech_patch[0].transpose(0, 1).detach())

            motion_patch = None
            if step >= _DELAY:
                with torch.autocast(device_type="cuda", enabled=False):
                    motion_cond = torch.zeros(
                        (1, FEAT_DIM, ctx_frames), device=device, dtype=torch.float32
                    )
                    if len(motion_patch_hist) > 0:
                        k = min(p, len(motion_patch_hist))
                        mhist_patches = torch.cat(motion_patch_hist[-k:], dim=0)
                        motion_cond[:, :, ctx_frames - k * _PS:ctx_frames] = (
                            mhist_patches.transpose(0, 1).unsqueeze(0)
                        )

                    motion_patch = model.motion_cfm(
                        mu=h_cur.float(),
                        global_cond=first_motion.float(),
                        n_timesteps=int(cfm_steps),
                        patch_size=_PS,
                        cond=motion_cond,
                        temperature=float(cfm_temperature),
                        cfg_value=float(cfm_cfg),
                        sway_sampling_coef=1.0,
                        use_cfg_zero_star=True,
                    )

                motion_frames = motion_patch.transpose(1, 2)
                gen_motion_latents.append(motion_frames)
                motion_patch_hist.append(motion_patch[0].transpose(0, 1).detach())

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                e_speech = model.projector_speech(speech_frames.to(dtype=full_embeds.dtype))
                speech_token = model.speech_patch_encoder(e_speech.unsqueeze(1))

                if motion_patch is not None:
                    e_motion = model.projector_motion(motion_frames.to(dtype=full_embeds.dtype))
                    motion_token = model.motion_patch_encoder(e_motion.unsqueeze(1))
                    e_next = speech_token + motion_token
                else:
                    e_next = speech_token

            attn_mask = torch.cat(
                [attn_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1
            )
            past_kv, h_cur = llm_step_one(past_kv, attn_mask, e_next)

    prefix_motion_raw = gt_raw[:prefix_frames]

    if gen_motion_latents:
        gen_motion_tensor = torch.cat(gen_motion_latents, dim=1)
        gen_motion_norm = gen_motion_tensor[0].detach().float().cpu().numpy()
        gen_motion_raw = denormalize_motion(gen_motion_norm)
        pred_raw = np.concatenate([prefix_motion_raw, gen_motion_raw], axis=0)
    else:
        pred_raw = prefix_motion_raw

    gen_wav_24k = None
    pred_speech = None
    if gen_speech_latents:
        gen_speech_btc = torch.cat(gen_speech_latents, dim=1)
        prefix_speech_btc = speech_latent[:, :min(speech_prefix_frames, Ls_frames), :].to(device)
        full_speech_latent = torch.cat([prefix_speech_btc, gen_speech_btc], dim=1)

        gen_wav_24k = model.pretrained_speech_encoder.decode(
            full_speech_latent, latent_layout="btc"
        )
        pred_speech = gen_speech_btc[0].detach().float().cpu().numpy()

    return pred_raw, gt_raw, gen_wav_24k, pred_speech


# ============================================================================
# LIA-X video render + ffmpeg mux
# ============================================================================

def init_render_models(liax_ckpt=LIAX_CKPT):
    """Load LIA-X autoencoder for rendering. Returns dict with 'model'."""
    from lia_x.networks.generator import Generator  # noqa: E402

    m = Generator(motion_dim=40, scale=2)
    state_dict = torch.load(liax_ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(state_dict, strict=True)
    m.cuda().eval()
    print(f"[render] LIA-X autoencoder loaded from {liax_ckpt}")
    return {"model": m}


@torch.no_grad()
def render_motion_to_video(motion_40_np, video_path, output_path, render_models, fps=25):
    """Decode 40-d LIA-X motion latent → 1024-d via Direction layer → silent mp4."""
    liax_model = render_models["model"]
    device = next(liax_model.parameters()).device

    cap = cv2.VideoCapture(video_path)
    ret, ref_bgr = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Cannot read video: {video_path}")

    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    ref_rgb = cv2.resize(ref_rgb, (512, 512), interpolation=cv2.INTER_AREA)
    ref_tensor = torch.from_numpy(ref_rgb).permute(2, 0, 1).float() / 127.5 - 1.0
    ref_tensor = ref_tensor.unsqueeze(0).to(device)

    z_s2r, feats = liax_model.enc.enc_2r(ref_tensor)

    motion_40 = torch.from_numpy(motion_40_np).float().to(device)
    r_d = liax_model.dec.direction(motion_40)
    T = r_d.shape[0]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (512, 512))

    for t in range(T):
        s_r_d_t = z_s2r + r_d[t].unsqueeze(0)
        img_t = liax_model.dec(s_r_d_t, alpha=None, feats=feats)
        img = img_t[0].clamp(-1, 1).cpu()
        img = ((img + 1) / 2 * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    writer.release()


def add_audio_to_video(video_path, audio_path, output_path):
    try:
        video_duration = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path]
        ).decode().strip())
    except Exception:
        video_duration = None

    cmd = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path,
           "-c:v", "copy", "-c:a", "aac", "-b:a", "128k"]
    if video_duration:
        cmd.extend(["-t", str(video_duration)])
    cmd.extend(["-shortest", output_path])

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def render_motion_to_video_with_audio(
    motion_40_np, video_path, output_path, render_models, wav_path, fps=25
):
    """Decode motion → silent video, then mux audio with ffmpeg."""
    temp_video_path = output_path.replace(".mp4", "_temp.mp4")
    render_motion_to_video(motion_40_np, video_path, temp_video_path, render_models, fps)
    if wav_path and os.path.exists(wav_path):
        ok = add_audio_to_video(temp_video_path, wav_path, output_path)
        if ok:
            try:
                os.remove(temp_video_path)
            except OSError:
                pass
        else:
            os.rename(temp_video_path, output_path)
    else:
        os.rename(temp_video_path, output_path)


# ============================================================================
# Entry point
# ============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.strip().splitlines()[0],
    )

    # ---- inputs ----
    ap.add_argument(
        "--text", type=str,
        default="对方是不是在拖着自己，或者说是不是对方你已经变心了啊，或者说对方他根本就是不",
        help="Text prompt the talking head will speak.",
    )
    ap.add_argument(
        "--ref-audio", type=str,
        default=os.path.join(_HERE, "samples", "reference_audio.wav"),
        help="Reference 16kHz/24kHz speech clip (used for speaker timbre via WavLM).",
    )
    ap.add_argument(
        "--ref-motion", type=str,
        default=os.path.join(_HERE, "samples", "reference_motion.pt"),
        help="Reference LIA-X motion latent (.pt). First frame is used as motion global cond.",
    )
    ap.add_argument(
        "--ref-video", type=str,
        default=os.path.join(_HERE, "samples", "reference_video.mp4"),
        help="Reference identity video (mp4). First frame supplies the face appearance to LIA-X.",
    )

    # ---- outputs ----
    ap.add_argument(
        "--output", type=str, default="output.mp4",
        help="Path to write the final mp4 (video + audio).",
    )

    # ---- model ----
    ap.add_argument(
        "--checkpoint-dir", type=str,
        default=os.environ.get("CHECKPOINT_DIR", "/dev/shm/talker-t2av-ckpt"),
        help="Talker-T2AV checkpoint directory (containing model.safetensors).",
    )
    ap.add_argument(
        "--llm-name", type=str, default="Qwen/Qwen3-0.6B",
        help="HuggingFace name (or local path) of the AR backbone.",
    )

    # ---- sampling hyperparameters (defaults = paper recipe) ----
    ap.add_argument("--cfm-steps", type=int, default=10,
                    help="Number of OT-CFM Euler steps per patch.")
    ap.add_argument("--cfm-temperature", type=float, default=0.3)
    ap.add_argument("--cfm-cfg", type=float, default=2.0,
                    help="Classifier-free guidance scale.")
    ap.add_argument("--gt-prefix-seconds", type=float, default=0.5,
                    help="Seconds of reference audio+motion fed before AR generation begins. "
                         "0 = speaker/identity-only driving (free style); "
                         ">0 = zero-shot TTS / style cloning (continues the reference's prosody and motion).")
    ap.add_argument("--fps", type=int, default=25)

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    for name, p in [
        ("--ref-audio", args.ref_audio),
        ("--ref-motion", args.ref_motion),
        ("--ref-video", args.ref_video),
    ]:
        if not os.path.exists(p):
            sys.exit(f"[infer] {name} not found: {p}")
    ckpt_safetensors = os.path.join(args.checkpoint_dir, "model.safetensors")
    if not os.path.exists(ckpt_safetensors):
        sys.exit(
            f"[infer] checkpoint not found: {ckpt_safetensors}\n"
            "        Set --checkpoint-dir or CHECKPOINT_DIR env var."
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[infer] device       = {device}")
    print(f"[infer] text         = {args.text!r}")
    print(f"[infer] ref-audio    = {args.ref_audio}")
    print(f"[infer] ref-motion   = {args.ref_motion}")
    print(f"[infer] ref-video    = {args.ref_video}")
    print(f"[infer] checkpoint   = {args.checkpoint_dir}")
    print(f"[infer] sampling     = t={args.cfm_temperature}, cfg={args.cfm_cfg}, "
          f"steps={args.cfm_steps}, prefix={args.gt_prefix_seconds}s")

    t0 = time.time()
    print("[infer] Building Talker-T2AV (this loads Qwen3 + WhisperX-VAE + WavLM)...")
    model_args = ModelArguments(
        llm_model_name_or_path=args.llm_name,
        encoder_projector_ds_rate=12,
        frames_per_second=args.fps,
        speech_embedding_dim=32,
        motion_embedding_dim=40,
        tts_checkpoint_path=None,
        patch_size=4,
        delay=0,
        motion_cfm_weight=10,
        context_patches=1,
        patch_encoder_n_layer=4,
        patch_encoder_n_head=8,
        cfm_k_repeat=2,
        model_max_length=1000,
    )
    model = init_model(model_args)

    sd = safetensors.torch.load_file(ckpt_safetensors)
    result = model.load_state_dict(sd, strict=False)
    print(f"[infer] load_state_dict: missing={len(result.missing_keys)} (expected: codec keys), "
          f"unexpected={len(result.unexpected_keys)}")
    model = model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_dir,
        model_max_length=model_args.model_max_length,
        padding_side="right",
    )
    speech_tok, motion_tok = "<SPEECH_FRAME>", "<MOTION_FRAME>"
    if speech_tok not in tokenizer.get_vocab():
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [speech_tok, motion_tok]}
        )
    print(f"[infer] model ready in {time.time() - t0:.1f}s")

    print("[infer] Loading LIA-X renderer...")
    render_models = init_render_models()

    print("[infer] Generating speech + motion latents (autoregressive)...")
    t1 = time.time()
    pred_motion, _gt_raw, gen_wav_24k, _pred_speech_lat = ar_generate_motion_continuation(
        model,
        tokenizer,
        gt_pt_path=args.ref_motion,
        wav_path=args.ref_audio,
        text=args.text,
        model_max_length=model_args.model_max_length,
        cfm_steps=args.cfm_steps,
        cfm_temperature=args.cfm_temperature,
        cfm_cfg=args.cfm_cfg,
        gt_prefix_seconds=args.gt_prefix_seconds,
        fps=args.fps,
    )
    print(f"[infer] AR generation done in {time.time() - t1:.1f}s "
          f"(motion={pred_motion.shape}, audio={gen_wav_24k.shape})")

    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    audio_path = os.path.splitext(args.output)[0] + ".wav"
    # gen_wav_24k may be (T,), (1,T), or (1,1,T); torchaudio.save wants 2D.
    wav_2d = gen_wav_24k.detach().cpu().float().squeeze().unsqueeze(0)
    torchaudio.save(audio_path, wav_2d, 24000)
    print(f"[infer] wrote audio   -> {audio_path}")

    print("[infer] Rendering video via LIA-X + muxing audio with ffmpeg...")
    t2 = time.time()
    render_motion_to_video_with_audio(
        pred_motion if isinstance(pred_motion, np.ndarray) else pred_motion.detach().cpu().numpy(),
        video_path=args.ref_video,
        output_path=args.output,
        render_models=render_models,
        wav_path=audio_path,
        fps=args.fps,
    )
    print(f"[infer] render done in {time.time() - t2:.1f}s")
    print(f"[infer] wrote video   -> {args.output}")


if __name__ == "__main__":
    main()
