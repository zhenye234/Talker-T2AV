
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from contextlib import nullcontext

import safetensors
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, PreTrainedModel, Wav2Vec2BertModel
from liger_kernel.transformers import AutoLigerKernelForCausalLM
import torch.distributions as D

import os
# WavLM speaker-verification encoder. Vendored under ./speaker_verification/
# from Microsoft UniSpeech (Apache 2.0):
#   https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification
from speaker_verification import init_model as init_spk

@dataclass
class ModelArguments:
    llm_model_name_or_path: Optional[str] = field(default="Qwen/Qwen3-0.6B")

    encoder_ds_rate: int = 2
    encoder_projector_ds_rate: int = 12
    speech_embedding_dim: int = 128
    motion_embedding_dim: int = 40

    projector_model_path: Optional[str] = field(default=None)
    tts_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to TTS checkpoint dir for initializing speech_cfm, projector_speech, head_stop, llm."},
    )
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length"},
    )
    max_speech_seconds: int = 30
    frames_per_second: int = 25

    # Patch-level parameters
    patch_size: int = 4
    delay: int = 5            # in patches (so delay_frames = delay * patch_size)
    motion_cfm_weight: float = 1.0  # weight for loss_motion_cfm in total loss
    context_patches: int = 1
    patch_encoder_n_layer: int = 4
    patch_encoder_n_head: int = 8
    cfm_k_repeat: int = 2

    # TTS data path (txt file, one wav path per line, with sidecar .json for text)
    tts_data_path: Optional[str] = field(default=None)

    # For decode
    decode_instruction: Optional[str] = field(default="")

    @property
    def ds_rate(self):
        return self.encoder_ds_rate * self.encoder_projector_ds_rate

    @property
    def max_mel_size(self):
        return self.max_speech_seconds * self.frames_per_second


class ProjectorConv1d(nn.Module):

    def __init__(self, config, encoder_dim, llm_dim):
        super().__init__()

        self.linear1 = nn.Linear(encoder_dim, llm_dim)
        # self.linear2 = nn.Linear(llm_dim, llm_dim)
        # self.relu2 = nn.ReLU()

    def forward(self, x):

        x = self.linear1(x)
        # x = self.relu2(x)
        # x = self.linear2(x)
        return x

def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


def build_stop_labels(B, T, start_t, len_t, device):
    y = torch.full((B, T), -100, dtype=torch.long, device=device)
    for i in range(B):
        s = int(start_t[i].item()); L = int(len_t[i].item())
        if L <= 0:
            continue
        end = s + L - 1
        s_clip = max(0, s); end_clip = min(T - 1, end)
        if s_clip <= end_clip:
            if end_clip > s_clip:
                y[i, s_clip:end_clip] = 0
            y[i, end_clip] = 1
    return y


def build_stop_labels_ditar(B, T, start_t, len_t, device, fps=25, patch_size=4,
                            ignore_seconds=0.3):
    """DiTAR-style stop labels with tail ignore zone (~300ms before positive).
    positions < ignore zone  → 0 (continue / negative)
    positions in ignore zone → -100 (ignored, tolerate VAD errors)
    last valid patch         → 1 (stop / positive)
    everything else          → -100 (ignored)

    pos_weight = mean(last_true_index) across the batch (like DiTAR code).

    Returns (labels, pos_weight):
      labels: (B, T) long tensor, values in {0, 1, -100}
      pos_weight: scalar tensor for BCE pos_weight
    """
    n_ignore = max(1, round(ignore_seconds * fps / patch_size))  # ~2 patches for 300ms

    y = torch.full((B, T), -100, dtype=torch.long, device=device)
    last_indices = []
    for i in range(B):
        s = int(start_t[i].item())
        L = int(len_t[i].item())
        if L <= 0:
            continue
        s_clip = max(0, s)
        end_clip = min(T, s + L)  # exclusive
 
        if s_clip >= end_clip:
            continue
        if s + L > T:
            # 这些位置只标 negative（continue），不标 positive
            y[i, s_clip:end_clip] = 0
            continue

        if  L > 187:
            # 这些位置只标 negative（continue），不标 positive
            y[i, s_clip:end_clip] = 0
            continue

        last_idx = end_clip - 1

        # Ignore zone: n_ignore patches right before the positive
        ignore_start = max(s_clip, last_idx - n_ignore)

        # Negative: [s_clip, ignore_start) → 0 (continue)
        if ignore_start > s_clip:
            y[i, s_clip:ignore_start] = 0


        # [ignore_start, last_idx) stays -100 (ignored)

        # Positive: last valid patch → 1 (stop)
        y[i, last_idx] = 1

        # record for pos_weight (number of negatives for this sample)
        last_indices.append(float(ignore_start - s_clip))

    # pos_weight = mean number of negative samples
    if last_indices:
        pos_weight = torch.tensor(sum(last_indices) / len(last_indices), device=device)
    else:
        pos_weight = torch.tensor(1.0, device=device)
    return y, pos_weight


class StopPredictor(nn.Module):
    """MLP stop predictor: Linear → SiLU → Linear(1)."""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.actn = nn.SiLU()
        self.head = nn.Linear(dim, 1, bias=False)

    def forward(self, x):
        """
        Args:
            x: (B, T, D) or (B, 1, D) hidden states
        Returns:
            logits: (B, T) stop logits per position
        """
        return self.head(self.actn(self.proj(x))).squeeze(-1)


class SpeechLLM(PreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(
        self,
        llm: nn.Module,
        whisperx_vae: nn.Module,
        projector_speech: nn.Module,
        projector_motion: nn.Module,
        config,
        model_args: ModelArguments,
    ):
        super().__init__(config)
        self.llm = llm

        self.projector_speech = projector_speech
        self.projector_motion = projector_motion

        self.pretrained_speech_encoder = whisperx_vae
        self._keys_to_ignore_on_save = set()
        self.pretrained_speech_encoder.eval()

        for k in self.pretrained_speech_encoder.state_dict().keys():
            self._keys_to_ignore_on_save.add('pretrained_speech_encoder.' + k)

        # Speaker encoder (frozen)
        self.spk = init_spk('wavlm_large', os.environ.get(
            'WAVLM_CKPT',
            '/apdcephfs_tj5/share_302528826/zhenye/deps_eval/models/wavlm_large_finetune.pth',
        ))
        self.spk.eval()
        for k in self.spk.state_dict().keys():
            self._keys_to_ignore_on_save.add('spk.' + k)

        self.model_args = model_args

        from llama4nar import DecoderConfig, PatchEncoder
        from unified_cfm import UnifiedCFM, CfmConfig
        from local_dit import VoxCPMLocDiT

        # Patch-level parameters
        self.patch_size = model_args.patch_size
        self.delay = model_args.delay           # in patches
        self.motion_cfm_weight = model_args.motion_cfm_weight
        self.context_patches = model_args.context_patches
        self.cfm_k_repeat = model_args.cfm_k_repeat

        llm_dim = self.llm.model.config.hidden_size

        # PatchEncoder: compress patch_size frames into 1 LLM token
        # Speech patch encoder (initialized from TTS checkpoint)
        self.speech_patch_encoder = PatchEncoder(
            input_dim=llm_dim,
            hidden_size=llm_dim,
            n_layer=model_args.patch_encoder_n_layer,
            n_head=model_args.patch_encoder_n_head,
        )
        # Motion patch encoder (randomly initialized)
        self.motion_patch_encoder = PatchEncoder(
            input_dim=llm_dim,
            hidden_size=llm_dim,
            n_layer=model_args.patch_encoder_n_layer,
            n_head=model_args.patch_encoder_n_head,
        )

        nar_cfg = DecoderConfig(
            n_layer=8,
            n_head=8,
            feat_dim=1024,
            hidden_size=1024
        )

        cfm_cfg = CfmConfig()

        # Speech CFM DiT (128-d speech latent, 256-d speaker embedding as global_cond)
        speech_dit = VoxCPMLocDiT(config=nar_cfg, in_channels=model_args.speech_embedding_dim, global_cond_channels=256)
        self.speech_cfm = UnifiedCFM(
            in_channels=model_args.speech_embedding_dim,
            cfm_params=cfm_cfg,
            estimator=speech_dit,
            mean_mode=False,
        )

        # Motion CFM DiT (40-d motion latent, 40-d first_motion as global_cond)
        motion_dit = VoxCPMLocDiT(config=nar_cfg, in_channels=model_args.motion_embedding_dim, global_cond_channels=model_args.motion_embedding_dim)
        self.motion_cfm = UnifiedCFM(
            in_channels=model_args.motion_embedding_dim,
            cfm_params=cfm_cfg,
            estimator=motion_dit,
            mean_mode=False,
        )

        # Stop head (MLP predictor, DiTAR style)
        self.head_stop = StopPredictor(dim=self.llm.model.config.hidden_size)

        # Task tag embedding: 0 = TTS, 1 = T2SV
        self.task_embedding = nn.Embedding(2, llm_dim)

        # Learnable pad embedding for TTS mode (replaces zero motion contribution)
        # Initialized to zero so behavior is equivalent to before at start of training
        self.motion_pad_emb = nn.Parameter(torch.zeros(1, 1, llm_dim))

    def get_speech_embeddings(self, audio, feat, mel_len):
        max_mel_len = max(mel_len)
        max_wav_len = max_mel_len * 24000 / 25
        audio = audio[:, :, :int(max_wav_len)]
        with torch.no_grad():
            latent = self.pretrained_speech_encoder.encode(audio, feat, return_layout="btc")
            # latent: (B, T_frames, 128), already 25Hz

        speech_proj = self.projector_speech(latent)  # (B, T, D)

        return speech_proj, latent.detach()

    def get_speaker_embeddings(self, audio16k, mel_len):
        min_mel_len = min(mel_len)
        min_wav_len = min_mel_len * 16000 / 25
        audio16k = audio16k[:, :, :int(min_wav_len)]
        with torch.no_grad():
            speaker_emb = self.spk(audio16k[:, 0, :])
        return speaker_emb.detach()

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        feat: torch.Tensor = None,              # (B,1,128,n_frames)
        mel_len: torch.LongTensor = None,       # (B,)
        audio: torch.Tensor = None,             # (B,1,T_24k)
        audio16k: torch.Tensor = None,          # (B,1,T_16k)
        motion_len: torch.LongTensor = None,
        speech_span: torch.LongTensor = None,   # (B,2) [start,end) in patch tokens
        motion: torch.Tensor = None,            # (B,maxT_frames,40)
        task_tag: torch.LongTensor = None,      # (B,) 0=TTS, 1=T2SV
    ):
        PS = self.patch_size
        DELAY = self.delay  # in patches
        p = self.context_patches  # number of context patches
        K = self.cfm_k_repeat
        SPEECH_DIM = self.model_args.speech_embedding_dim  # 128
        MOTION_DIM = self.model_args.motion_embedding_dim  # 40

        text_emb = self.llm.get_input_embeddings()(input_ids)  # (B, T_tok, D)
        B, T_tok, D = text_emb.shape

        # =============================================
        # 1. Encode speech frames → patches → LLM tokens
        # =============================================
        # speech_emb: (B, L_frames, D), speech_latent: (B, L_frames, 128)
        speech_emb, speech_latent = self.get_speech_embeddings(audio, feat, mel_len)

        # Speaker embedding for speech CFM global condition
        speaker_emb = self.get_speaker_embeddings(audio16k, mel_len)  # (B, 256)

        # Motion embedding (frame-level)
        motion_emb = self.projector_motion(motion)  # (B, max_motion_frames, D)

        # =============================================
        # 2. Build patch-level inputs_embeds
        # =============================================
        # Compute task tag embeddings and prepend to sequence
        if task_tag is not None:
            task_emb = self.task_embedding(task_tag.to(input_ids.device))  # (B, D)
        else:
            _tags = torch.zeros(B, dtype=torch.long, device=input_ids.device)
            if motion_len is not None:
                _tags = (motion_len > 0).long()
            task_emb = self.task_embedding(_tags)  # (B, D)

        # Prepend task_emb as the first token: [task_tag, text_tokens, ..., patch_tokens, ...]
        inputs_embeds = torch.cat([task_emb.unsqueeze(1), text_emb], dim=1)  # (B, 1+T_tok, D)
        # Update attention_mask to account for the prepended token
        attention_mask = torch.cat([
            torch.ones((B, 1), dtype=attention_mask.dtype, device=attention_mask.device),
            attention_mask,
        ], dim=1)  # (B, 1+T_tok)
        # Shift speech_span by +1 to account for the prepended task tag token
        speech_span = speech_span.clone()
        valid_mask = speech_span[:, 0] >= 0
        speech_span[valid_mask] = speech_span[valid_mask] + 1

        B, T_tok, D = inputs_embeds.shape  # T_tok is now 1+original

        # Pre-compute per-sample patch tokens
        for i in range(B):
            s0, s1 = speech_span[i].tolist()
            if s0 < 0:
                continue
            n_patch_slots = max(0, s1 - s0)  # number of patch tokens in the placeholder
            if n_patch_slots <= 0:
                continue

            # Check if this sample has motion data
            has_motion = (motion_len is not None and int(motion_len[i].item()) > 0)

            # Number of speech frames available
            Ls_frames = min(int(mel_len[i].item()), speech_emb.size(1))
            # Pad to PS multiple
            Ls_padded = ((Ls_frames + PS - 1) // PS) * PS
            n_speech_patches = Ls_padded // PS

            if has_motion:
                # Number of motion frames available
                Lm_frames = min(int(motion_len[i].item()), motion_emb.size(1))
                Lm_padded = ((Lm_frames + PS - 1) // PS) * PS
                n_motion_patches = Lm_padded // PS

                # Total patches needed = max(speech, motion+delay)
                total_patches = min(n_patch_slots, max(n_speech_patches, n_motion_patches + DELAY))
            else:
                # TTS: no motion, total patches = speech patches only
                Lm_frames = 0
                total_patches = min(n_patch_slots, n_speech_patches)

            # Build frame-level speech-only and motion-only embeddings
            total_frames = total_patches * PS

            # Speech-only tensor: (total_frames, D), zero-padded where no speech
            speech_only = speech_emb.new_zeros((total_frames, D))
            sp_fill = min(Ls_frames, total_frames)
            if sp_fill > 0:
                speech_only[:sp_fill] = speech_emb[i, :sp_fill]

            # Reshape to patches: (total_patches, PS, D)
            speech_only = speech_only.view(total_patches, PS, D)

            # Run speech patch encoder
            speech_tokens = self.speech_patch_encoder(speech_only.unsqueeze(0))[0]  # (total_patches, D)

            # Motion-only tensor: (total_frames, D), zero-padded where no motion
            motion_only = speech_emb.new_zeros((total_frames, D))
            if has_motion:
                delay_frames = DELAY * PS
                motion_start = delay_frames
                motion_end = min(total_frames, delay_frames + Lm_frames)
                n_motion_fill = motion_end - motion_start
                if n_motion_fill > 0:
                    motion_only[motion_start:motion_end] = motion_emb[i, :n_motion_fill]

            motion_only = motion_only.view(total_patches, PS, D)
            motion_tokens = self.motion_patch_encoder(motion_only.unsqueeze(0))[0]  # (total_patches, D)

            if has_motion:
                patch_tokens = speech_tokens + motion_tokens
            else:
                # TTS: use learnable pad_emb instead of zero, plus dummy forward for DDP
                patch_tokens = speech_tokens + self.motion_pad_emb.squeeze(0).expand(total_patches, -1) + motion_tokens * 0

            # Fill into inputs_embeds
            fill_len = min(total_patches, n_patch_slots)
            inputs_embeds[i, s0:s0+fill_len] = patch_tokens[:fill_len]

        # =============================================
        # 3. LLM forward
        # =============================================
        out = self.llm(inputs_embeds=inputs_embeds,
                       attention_mask=attention_mask,
                       labels=None,
                       output_hidden_states=True)
        hidden = out['hidden_states'][-1]  # (B, T_tok, D)

        B, T_tok, D = hidden.shape

        # =============================================
        # 4. Speech CFM Loss (patch-level: predict next PS frames)
        # =============================================
        speech_target_list = []
        speech_cond_list = []
        speech_gc_list = []
        speech_mu_list = []

        ctx_frames = p * PS  # context = p patches * PS frames

        for i in range(B):
            s0, s1 = speech_span[i].tolist()
            if s0 < 0:
                continue

            Ls_frames = min(int(mel_len[i].item()), speech_latent.size(1))
            Ls_padded = ((Ls_frames + PS - 1) // PS) * PS
            n_speech_patches = Ls_padded // PS

            if n_speech_patches <= 1:
                continue

            # Pad speech_latent to PS multiple
            sl = speech_latent.new_zeros((Ls_padded, SPEECH_DIM))
            sl[:Ls_frames] = speech_latent[i, :Ls_frames]
            # Reshape: (n_speech_patches, PS, 128)
            sl_patches = sl.view(n_speech_patches, PS, SPEECH_DIM)

            # Next-patch prediction: from position t predict patch t+1
            # Additionally predict patch[0] from the last text token (s0-1) with zero cond
            # This teaches the CFM to handle zero conditioning at generation start
            if s0 > 0:
                # Predict patch[0] with zero cond, mu = hidden[s0-1] (last text token)
                target_p0 = sl_patches[0].transpose(0, 1)  # (128, PS)
                speech_target_list.append(target_p0)
                cond_p0 = speech_latent.new_zeros((SPEECH_DIM, ctx_frames))
                speech_cond_list.append(cond_p0)
                speech_gc_list.append(speaker_emb[i])
                speech_mu_list.append(hidden[i, s0 - 1])

            for t in range(n_speech_patches - 1):
                tok_pos = s0 + t
                if tok_pos >= T_tok:
                    break

                # Target: patch t+1 → (SPEECH_DIM, PS)
                target = sl_patches[t + 1].transpose(0, 1)  # (128, PS)
                speech_target_list.append(target)

                # Context: previous p patches flattened → (128, p*PS)
                cond = speech_latent.new_zeros((SPEECH_DIM, ctx_frames))
                # Gather context from patches [max(0,t+1-p)..t+1)
                ctx_start_patch = max(0, t + 1 - p)
                ctx_end_patch = t + 1
                n_ctx = ctx_end_patch - ctx_start_patch
                ctx_start_frame = ctx_start_patch * PS
                ctx_end_frame = ctx_end_patch * PS
                actual_frames = min(ctx_end_frame, Ls_frames) - ctx_start_frame
                if actual_frames > 0:
                    cond[:, ctx_frames - n_ctx * PS:ctx_frames - n_ctx * PS + actual_frames] = \
                        speech_latent[i, ctx_start_frame:ctx_start_frame + actual_frames].transpose(0, 1)
                speech_cond_list.append(cond)

                # Global condition: speaker embedding
                speech_gc_list.append(speaker_emb[i])

                # Mu: hidden state at position t
                speech_mu_list.append(hidden[i, tok_pos])

        # Compute speech CFM loss
        if len(speech_target_list) > 0:
            s_x1 = torch.stack(speech_target_list, dim=0)      # (N, 128, PS)
            s_cond = torch.stack(speech_cond_list, dim=0)      # (N, 128, p*PS)
            s_gc = torch.stack(speech_gc_list, dim=0)          # (N, 256)
            s_mu = torch.stack(speech_mu_list, dim=0)          # (N, D)
            s_tgt_mask = torch.ones((s_x1.size(0), 1, s_x1.size(2)), device=s_x1.device, dtype=s_x1.dtype)

            s_mu_k = s_mu.repeat_interleave(K, dim=0)
            s_x1_k = s_x1.repeat_interleave(K, dim=0)
            s_cond_k = s_cond.repeat_interleave(K, dim=0)
            s_gc_k = s_gc.repeat_interleave(K, dim=0)
            s_tgt_mask_k = s_tgt_mask.repeat_interleave(K, dim=0)

            loss_speech_cfm = self.speech_cfm.compute_loss(
                x1=s_x1_k, mu=s_mu_k, cond=s_cond_k,
                global_cond=s_gc_k, tgt_mask=s_tgt_mask_k, progress=0.0,
            )
        else:
            loss_speech_cfm = hidden.new_zeros(())

        # =============================================
        # 5. Motion CFM Loss (patch-level, delayed by DELAY patches)
        # =============================================
        motion_target_list = []
        motion_cond_list = []
        motion_gc_list = []
        motion_mu_list = []

        for i in range(B):
            s0, s1 = speech_span[i].tolist()
            if s0 < 0:
                continue

            Lm_frames = min(int(motion_len[i].item()), motion.size(1)) if motion_len is not None else motion.size(1)
            Lm_padded = ((Lm_frames + PS - 1) // PS) * PS
            n_motion_patches = Lm_padded // PS

            if n_motion_patches <= 1:
                continue

            # Pad motion to PS multiple
            ml = motion.new_zeros((Lm_padded, MOTION_DIM))
            ml[:Lm_frames] = motion[i, :Lm_frames]
            ml_patches = ml.view(n_motion_patches, PS, MOTION_DIM)

            # First motion (first frame) as global condition
            first_motion_vec = motion[i, 0]  # (40,)

            # Motion patches are at LLM positions [s0+DELAY, s0+DELAY+n_motion_patches)
            # Additionally predict motion patch[0] with zero cond
            # mu = hidden[s0+DELAY-1] (the position right before motion starts)
            motion_tok_start = s0 + DELAY
            if motion_tok_start > 0:
                target_m0 = ml_patches[0].transpose(0, 1)  # (40, PS)
                motion_target_list.append(target_m0)
                cond_m0 = motion.new_zeros((MOTION_DIM, ctx_frames))
                motion_cond_list.append(cond_m0)
                motion_gc_list.append(first_motion_vec)
                motion_mu_list.append(hidden[i, motion_tok_start - 1])

            for t in range(n_motion_patches - 1):
                tok_pos = s0 + DELAY + t
                if tok_pos >= T_tok:
                    break

                # Target: patch t+1 → (40, PS)
                target = ml_patches[t + 1].transpose(0, 1)
                motion_target_list.append(target)

                # Context: previous p patches flattened → (40, p*PS)
                cond = motion.new_zeros((MOTION_DIM, ctx_frames))
                ctx_start_patch = max(0, t + 1 - p)
                ctx_end_patch = t + 1
                n_ctx = ctx_end_patch - ctx_start_patch
                ctx_start_frame = ctx_start_patch * PS
                ctx_end_frame = ctx_end_patch * PS
                actual_frames = min(ctx_end_frame, Lm_frames) - ctx_start_frame
                if actual_frames > 0:
                    cond[:, ctx_frames - n_ctx * PS:ctx_frames - n_ctx * PS + actual_frames] = \
                        motion[i, ctx_start_frame:ctx_start_frame + actual_frames].transpose(0, 1)
                motion_cond_list.append(cond)

                # Global condition: first motion
                motion_gc_list.append(first_motion_vec)

                # Mu: hidden state at tok_pos
                motion_mu_list.append(hidden[i, tok_pos])

        # Compute motion CFM loss
        if len(motion_target_list) > 0:
            m_x1 = torch.stack(motion_target_list, dim=0)      # (N, 40, PS)
            m_cond = torch.stack(motion_cond_list, dim=0)      # (N, 40, p*PS)
            m_gc = torch.stack(motion_gc_list, dim=0)          # (N, 40)
            m_mu = torch.stack(motion_mu_list, dim=0)          # (N, D)
            m_tgt_mask = torch.ones((m_x1.size(0), 1, m_x1.size(2)), device=m_x1.device, dtype=m_x1.dtype)

            m_mu_k = m_mu.repeat_interleave(K, dim=0)
            m_x1_k = m_x1.repeat_interleave(K, dim=0)
            m_cond_k = m_cond.repeat_interleave(K, dim=0)
            m_gc_k = m_gc.repeat_interleave(K, dim=0)
            m_tgt_mask_k = m_tgt_mask.repeat_interleave(K, dim=0)

            loss_motion_cfm = self.motion_cfm.compute_loss(
                x1=m_x1_k, mu=m_mu_k, cond=m_cond_k,
                global_cond=m_gc_k, tgt_mask=m_tgt_mask_k, progress=0.0,
            )
        else:
            # Dummy forward keeps motion_cfm in DDP graph, *0 ensures zero contribution
            _d_x1 = hidden.new_zeros((1, MOTION_DIM, PS))
            _d_cond = hidden.new_zeros((1, MOTION_DIM, ctx_frames))
            _d_gc = hidden.new_zeros((1, MOTION_DIM))
            _d_mu = hidden.new_zeros((1, D))
            _d_mask = hidden.new_ones((1, 1, PS))
            loss_motion_cfm = self.motion_cfm.compute_loss(
                x1=_d_x1, mu=_d_mu, cond=_d_cond,
                global_cond=_d_gc, tgt_mask=_d_mask, progress=0.0,
            ) * 0

        # =============================================
        # 6. Stop Loss (DiTAR style, patch-level)
        # =============================================
        # Compute number of active patches per sample for stop labels
        stop_start = speech_span[:, 0]  # (B,)
        stop_len = speech_span[:, 1] - speech_span[:, 0]  # (B,) number of patch tokens
        stop_labels, stop_pos_weight = build_stop_labels_ditar(
            B=B, T=T_tok,
            start_t=stop_start,
            len_t=stop_len,
            device=input_ids.device,
            fps=self.model_args.frames_per_second,
            patch_size=self.patch_size,
        )
        stop_logits = self.head_stop(hidden)  # (B, T)
        stop_mask = (stop_labels != -100)
        if stop_mask.any():
            stop_loss = F.binary_cross_entropy_with_logits(
                stop_logits[stop_mask].float(),
                stop_labels[stop_mask].float(),
                pos_weight=stop_pos_weight* 0.3,
            )
        else:
            stop_loss = torch.tensor(0.0, device=input_ids.device)

        # =============================================
        # 7. Total Loss
        # =============================================
        # Dummy forward for projector_motion: when motion is (B,0,40), Linear
        # has no real computation, so we run a (1,1,40) dummy to keep it in graph.
        _d_proj = self.projector_motion(motion.new_zeros((1, 1, MOTION_DIM))).sum() * 0
        _d_pad = self.motion_pad_emb.sum() * 0  # keep motion_pad_emb in DDP graph
        total_loss = loss_speech_cfm + self.motion_cfm_weight * loss_motion_cfm  +  stop_loss + _d_proj + _d_pad

        return transformers.modeling_outputs.CausalLMOutput(
            loss=total_loss,
            logits=out.logits
        )

    # =========================================================
    # generate_speech: TTS-only generation (from dar_baseline)
    # =========================================================
    @torch.no_grad()
    def generate_speech(
        self,
        input_ids: torch.LongTensor,
        feat: torch.Tensor,                 # (B,1,128,n_frames)
        mel_len: torch.LongTensor,          # (B,)
        audio: torch.Tensor,                # (B,1,T24k)
        audio16k: torch.Tensor,             # (B,1,T16k)
        speech_span: torch.LongTensor,      # (B,2) [s0,s1)
        tokenizer=None,

        prefix_seconds: float = 3.0,
        fps: int = 25,

        # CFM sampling
        cfm_steps: int = 10,
        cfm_temperature: float = 1.0,
        cfm_cfg: float = 1.5,

        # stopping
        use_stop: bool = False,
        stop_threshold: float = 0.5,
        min_gen_frames: int = 25,
        stop_k: int = 3,

        # amp
        amp_bf16: bool = True,

        # return control
        return_latents: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        TTS-only speech generation (no motion). Directly ported from dar_baseline.
        Returns:
          wav_prefix_24k, wav_rest_24k, wav_full_24k, latents_btc (optional)
        """
        device = next(self.parameters()).device
        assert input_ids.dim() == 2, "input_ids should be (B,T)"
        B = input_ids.size(0)
        if B != 1:
            raise RuntimeError("generate_speech() currently supports B=1.")

        if hasattr(self.llm, "config"):
            self.llm.config.use_cache = True

        inputs_embeds = self.llm.get_input_embeddings()(input_ids.to(device))
        s0, s1 = speech_span[0].tolist()

        total_frames = int(mel_len[0].item())
        total_frames = max(1, total_frames)

        prefix_frames = int(round(prefix_seconds * fps))
        prefix_frames = max(0, min(prefix_frames, total_frames))

        samples_per_frame = int(round(24000 / fps))
        audio = audio.to(device)
        audio16k = audio16k.to(device)
        feat = feat.to(device)

        no_prefix = (prefix_frames == 0)

        if no_prefix:
            with torch.no_grad():
                spk_emb = self.spk(audio16k[:, 0, :])
            spk_emb = spk_emb.detach()
            _dummy = self.pretrained_speech_encoder.encode(
                audio[:, :, :samples_per_frame], feat, return_layout="btc"
            )
            latent_dim = _dummy.size(-1)
            Tp = 0
            lat_btc = torch.zeros((1, 0, latent_dim), device=device, dtype=_dummy.dtype)
        else:
            prefix_samples = int(prefix_frames * samples_per_frame)
            wav_prefix_24k = audio[:, :, :prefix_samples]

            prefix_samples_16k = int(prefix_frames * 16000 / fps)
            wav_prefix_16k = audio16k[:, :, :prefix_samples_16k]

            with torch.no_grad():
                spk_emb = self.spk(wav_prefix_16k[:, 0, :])
            spk_emb = spk_emb.detach()

            lat_btc = self.pretrained_speech_encoder.encode(wav_prefix_24k, feat, return_layout="btc")
            lat_btc = lat_btc[:, :prefix_frames, :].contiguous()
            Tp = lat_btc.size(1)
            latent_dim = lat_btc.size(-1)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if (amp_bf16 and device.type == "cuda")
            else nullcontext()
        )

        def llm_step_one(past_kv, attn_mask, token_embed_1x1d):
            with amp_ctx:
                out = self.llm(
                    inputs_embeds=token_embed_1x1d,
                    attention_mask=attn_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True,
                )
                h = out.hidden_states[-1][:, -1, :]
                return out.past_key_values, h

        PS = self.patch_size
        D_llm = inputs_embeds.size(-1)
        p_ctx = self.context_patches

        # Task tag embedding for TTS (tag=0)
        tts_tag = torch.zeros(1, dtype=torch.long, device=device)
        task_emb_tts = self.task_embedding(tts_tag)  # (1, D_llm)

        with amp_ctx:
            prefill_embeds = inputs_embeds[:, :s0, :]
            # Prepend task tag token to match training layout: [task_tag, text, ...]
            prefill_embeds = torch.cat([task_emb_tts.unsqueeze(0).to(dtype=prefill_embeds.dtype), prefill_embeds], dim=1)

            if no_prefix:
                full_embeds = prefill_embeds
                prefix_patch_hist: List[torch.Tensor] = []
            else:
                prefix_proj = self.projector_speech(lat_btc.to(dtype=prefill_embeds.dtype))
                Lp = prefix_proj.size(1)
                pad_pf = (PS - Lp % PS) % PS
                if pad_pf > 0:
                    prefix_proj = F.pad(prefix_proj, (0, 0, 0, pad_pf))
                n_prefix_patches = prefix_proj.size(1) // PS
                prefix_patched = prefix_proj.reshape(1, n_prefix_patches, PS, D_llm)
                prefix_patch_emb = self.speech_patch_encoder(prefix_patched)
                # TTS: add motion_pad_emb to compensate missing motion signal
                prefix_patch_emb = prefix_patch_emb + self.motion_pad_emb
                full_embeds = torch.cat([prefill_embeds, prefix_patch_emb], dim=1)

                lat_for_hist = lat_btc
                if pad_pf > 0:
                    lat_for_hist = F.pad(lat_for_hist, (0, 0, 0, pad_pf))
                prefix_patch_hist = []
                for pi in range(n_prefix_patches):
                    prefix_patch_hist.append(lat_for_hist[0, pi*PS:(pi+1)*PS, :].detach())

            attn_mask = torch.ones((1, full_embeds.size(1)), dtype=torch.long, device=device)

            out = self.llm(
                inputs_embeds=full_embeds,
                attention_mask=attn_mask,
                use_cache=True,
                output_hidden_states=True,
            )
            past_kv = out.past_key_values
            h_cur = out.hidden_states[-1][:, -1, :]

        latent_hist: List[torch.Tensor] = prefix_patch_hist

        n_prefix_pat = len(latent_hist)
        total_patches = math.ceil(total_frames / PS)
        remain = total_patches - n_prefix_pat + 25
        gen_latents: List[torch.Tensor] = []

        stop_streak = 0

        for step in range(max(0, remain)):
            if use_stop and step >= (min_gen_frames // PS):
                logits = self.head_stop(h_cur.float().unsqueeze(1))
                p_stop = torch.sigmoid(logits.float())[0, 0].item()
                if p_stop >= stop_threshold:
                    stop_streak += 1
                    if stop_streak >= stop_k:
                        break
                else:
                    stop_streak = 0

            with torch.autocast(device_type="cuda", enabled=False):
                cond = torch.zeros((1, latent_dim, p_ctx * PS), device=device, dtype=torch.float32)
                if len(latent_hist) > 0:
                    k = min(p_ctx, len(latent_hist))
                    hist_patches = torch.cat(latent_hist[-k:], dim=0)
                    cond[:, :, p_ctx * PS - k * PS:p_ctx * PS] = hist_patches.transpose(0, 1).unsqueeze(0)

                x_next = self.speech_cfm(
                    mu=h_cur.float(),
                    global_cond=spk_emb.float(),
                    n_timesteps=int(cfm_steps),
                    patch_size=PS,
                    cond=cond,
                    temperature=float(cfm_temperature),
                    cfg_value=float(cfm_cfg),
                    sway_sampling_coef=1.0,
                    use_cfg_zero_star=True,
                )

            patch_latent = x_next[0].transpose(0, 1).detach()
            gen_latents.append(patch_latent.unsqueeze(0))
            latent_hist.append(patch_latent)

            with amp_ctx:
                proj_frames = self.projector_speech(
                    patch_latent.unsqueeze(0).to(dtype=full_embeds.dtype)
                )
                patch_4d = proj_frames.unsqueeze(0)
                next_token = self.speech_patch_encoder(patch_4d)
                # TTS: add motion_pad_emb to compensate missing motion signal
                next_token = next_token + self.motion_pad_emb

            attn_mask = torch.cat([attn_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1)
            past_kv, h_cur = llm_step_one(past_kv, attn_mask, next_token)

        if len(gen_latents) > 0:
            gen_btc = torch.cat(gen_latents, dim=1)
            latents_btc = torch.cat([lat_btc, gen_btc], dim=1)
        else:
            latents_btc = lat_btc
            gen_btc = lat_btc[:, :0, :]

        wav_full_24k = self.pretrained_speech_encoder.decode(latents_btc, latent_layout="btc")
        if no_prefix:
            wav_prefix_dec_24k = wav_full_24k[:, :, :0]
        else:
            wav_prefix_dec_24k = self.pretrained_speech_encoder.decode(lat_btc, latent_layout="btc")
        wav_rest_dec_24k = (
            self.pretrained_speech_encoder.decode(gen_btc, latent_layout="btc")
            if gen_btc.size(1) > 0
            else wav_full_24k[:, :, :0]
        )

        if return_latents:
            return wav_prefix_dec_24k, wav_rest_dec_24k, wav_full_24k, latents_btc
        return wav_prefix_dec_24k, wav_rest_dec_24k, wav_full_24k, None

    # =========================================================
    # generate_speech_motion: T2SV generation (speech + motion)
    # =========================================================
    @torch.no_grad()
    def generate_speech_motion(
        self,
        input_ids: torch.LongTensor,
        feat: torch.Tensor,                 # (B,1,128,n_frames)
        mel_len: torch.LongTensor,          # (B,)
        audio: torch.Tensor,                # (B,1,T24k)
        audio16k: torch.Tensor,             # (B,1,T16k)
        speech_span: torch.LongTensor,      # (B,2) [s0,s1)
        first_motion: torch.Tensor,         # (40,) first frame motion (normalized)
        tokenizer=None,

        prefix_seconds: float = 3.0,
        fps: int = 25,

        # CFM sampling
        cfm_steps: int = 10,
        cfm_temperature: float = 1.0,
        cfm_cfg: float = 1.5,

        # stopping
        use_stop: bool = False,
        stop_threshold: float = 0.5,
        min_gen_patches: int = 7,
        stop_k: int = 3,

        # amp
        amp_bf16: bool = True,

        # return control
        return_latents: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Patch-level dual autoregressive generation: speech latents + motion latents.
        Each step generates patch_size frames of speech and (after delay) motion.

        Returns:
          wav_prefix_24k: decoded prefix speech
          wav_rest_24k  : decoded generated speech
          wav_full_24k  : full decoded speech
          motion_latents: (1, T_motion, 40) generated motion latents (normalized)
        """
        device = next(self.parameters()).device
        assert input_ids.dim() == 2, "input_ids should be (B,T)"
        B = input_ids.size(0)
        if B != 1:
            raise RuntimeError("generate() currently supports B=1.")

        if hasattr(self.llm, "config"):
            self.llm.config.use_cache = True

        PS = self.patch_size
        DELAY = self.delay           # in patches
        p = self.context_patches     # context patches for CFM
        SPEECH_DIM = self.model_args.speech_embedding_dim  # 128
        MOTION_DIM = self.model_args.motion_embedding_dim  # 40

        inputs_embeds = self.llm.get_input_embeddings()(input_ids.to(device))
        s0, s1 = speech_span[0].tolist()

        total_frames = int(mel_len[0].item())
        total_frames = max(1, total_frames)
        total_patches = (total_frames + PS - 1) // PS

        prefix_frames = int(round(prefix_seconds * fps))
        prefix_frames = max(0, min(prefix_frames, total_frames))
        # Align prefix to PS boundary
        prefix_frames = (prefix_frames // PS) * PS
        prefix_patches = prefix_frames // PS

        samples_per_frame_24k = int(round(24000 / fps))
        audio = audio.to(device)
        audio16k = audio16k.to(device)
        feat = feat.to(device)
        first_motion = first_motion.to(device).float()

        no_prefix = (prefix_frames == 0)

        # Speaker embedding
        with torch.no_grad():
            if no_prefix:
                spk_emb = self.spk(audio16k[:, 0, :])
            else:
                prefix_samples_16k = int(prefix_frames * 16000 / fps)
                wav_prefix_16k = audio16k[:, :, :prefix_samples_16k]
                spk_emb = self.spk(wav_prefix_16k[:, 0, :])
        spk_emb = spk_emb.detach()

        # Encode prefix speech latents
        if no_prefix:
            _dummy = self.pretrained_speech_encoder.encode(
                audio[:, :, :samples_per_frame_24k], feat, return_layout="btc"
            )
            latent_dim = _dummy.size(-1)
            Tp_frames = 0
            speech_lat_prefix = torch.zeros((1, 0, latent_dim), device=device, dtype=_dummy.dtype)
        else:
            prefix_samples_24k = int(prefix_frames * samples_per_frame_24k)
            wav_prefix_24k = audio[:, :, :prefix_samples_24k]
            speech_lat_prefix = self.pretrained_speech_encoder.encode(wav_prefix_24k, feat, return_layout="btc")
            speech_lat_prefix = speech_lat_prefix[:, :prefix_frames, :].contiguous()
            Tp_frames = speech_lat_prefix.size(1)
            latent_dim = speech_lat_prefix.size(-1)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if (amp_bf16 and device.type == "cuda")
            else nullcontext()
        )

        def llm_step_one(past_kv, attn_mask, token_embed_1x1d):
            with amp_ctx:
                out = self.llm(
                    inputs_embeds=token_embed_1x1d,
                    attention_mask=attn_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True,
                )
                h = out.hidden_states[-1][:, -1, :]
                return out.past_key_values, h

        # Prefill: text + prefix patches
        # Task tag embedding for T2SV (tag=1)
        t2sv_tag = torch.ones(1, dtype=torch.long, device=device)
        task_emb_t2sv = self.task_embedding(t2sv_tag)  # (1, D_llm)

        with amp_ctx:
            prefill_embeds = inputs_embeds[:, :s0, :]  # text part
            # Prepend task tag token to match training layout: [task_tag, text, ...]
            prefill_embeds = torch.cat([task_emb_t2sv.unsqueeze(0).to(dtype=prefill_embeds.dtype), prefill_embeds], dim=1)

            if no_prefix:
                full_embeds = prefill_embeds
            else:
                # Convert prefix frames to patch tokens via patch_encoder
                prefix_proj = self.projector_speech(speech_lat_prefix.to(dtype=prefill_embeds.dtype))
                # (1, Tp_frames, D) → pad to PS multiple → reshape → patch_encoder
                Lp = prefix_proj.size(1)
                pad_pf = (PS - Lp % PS) % PS
                if pad_pf > 0:
                    prefix_proj = F.pad(prefix_proj, (0, 0, 0, pad_pf))
                n_prefix_patches = prefix_proj.size(1) // PS
                prefix_patched = prefix_proj.reshape(1, n_prefix_patches, PS, prefix_proj.size(-1))
                prefix_patch_embeds = self.speech_patch_encoder(prefix_patched)  # (1, n_prefix_patches, D)
                full_embeds = torch.cat([prefill_embeds, prefix_patch_embeds], dim=1)

            attn_mask = torch.ones((1, full_embeds.size(1)), dtype=torch.long, device=device)

            out = self.llm(
                inputs_embeds=full_embeds,
                attention_mask=attn_mask,
                use_cache=True,
                output_hidden_states=True,
            )
            past_kv = out.past_key_values
            h_cur = out.hidden_states[-1][:, -1, :]

        # History for CFM context (patch-level: each entry is (PS, dim))
        speech_patch_hist: List[torch.Tensor] = []
        if Tp_frames > 0:
            # Build patch-level history from prefix latents
            lat_for_hist = speech_lat_prefix
            pad_hist = (PS - Tp_frames % PS) % PS
            if pad_hist > 0:
                lat_for_hist = F.pad(lat_for_hist, (0, 0, 0, pad_hist))
            n_hist_patches = lat_for_hist.size(1) // PS
            for pi in range(n_hist_patches):
                speech_patch_hist.append(lat_for_hist[0, pi*PS:(pi+1)*PS, :].detach())  # (PS, 128)
        motion_patch_hist: List[torch.Tensor] = []

        ctx_frames = p * PS  # context frames for CFM

        # Generation loop (patch-level)
        remain_patches = total_patches - prefix_patches + 15  # extra margin
        gen_speech_latents: List[torch.Tensor] = []
        gen_motion_latents: List[torch.Tensor] = []

        stop_streak = 0

        for step in range(max(0, remain_patches)):
            # Stop check
            if use_stop and step >= min_gen_patches:
                logits = self.head_stop(h_cur.float().unsqueeze(1))  # (1,1,D) → (1,1)
                p_stop = torch.sigmoid(logits.float())[0, 0].item()
                if p_stop >= stop_threshold:
                    stop_streak += 1
                    if stop_streak >= stop_k:
                        break
                else:
                    stop_streak = 0

            with torch.autocast(device_type="cuda", enabled=False):
                # Speech CFM sample: generate PS frames
                speech_cond = torch.zeros((1, SPEECH_DIM, ctx_frames), device=device, dtype=torch.float32)
                if len(speech_patch_hist) > 0:
                    k = min(p, len(speech_patch_hist))
                    # Each entry is (PS, 128), concat last k patches -> (k*PS, 128)
                    hist_patches = torch.cat(speech_patch_hist[-k:], dim=0)
                    speech_cond[:, :, ctx_frames - k * PS:ctx_frames] = hist_patches.transpose(0, 1).unsqueeze(0)

                speech_patch = self.speech_cfm(
                    mu=h_cur.float(),
                    global_cond=spk_emb.float(),
                    n_timesteps=int(cfm_steps),
                    patch_size=PS,
                    cond=speech_cond,
                    temperature=float(cfm_temperature),
                    cfg_value=float(cfm_cfg),
                    sway_sampling_coef=1.0,
                    use_cfg_zero_star=True,
                )  # (1, 128, PS)

            # Store generated speech frames
            speech_frames = speech_patch.transpose(1, 2)  # (1, PS, 128)
            gen_speech_latents.append(speech_frames)
            # x_next is (1, 128, PS) -> store as (PS, 128)
            speech_patch_hist.append(speech_patch[0].transpose(0, 1).detach())  # (PS, 128)

            # Motion CFM sample (only after DELAY patches from start of generation)
            motion_patch = None
            if step >= DELAY:
                with torch.autocast(device_type="cuda", enabled=False):
                    motion_cond = torch.zeros((1, MOTION_DIM, ctx_frames), device=device, dtype=torch.float32)
                    if len(motion_patch_hist) > 0:
                        k = min(p, len(motion_patch_hist))
                        mhist_patches = torch.cat(motion_patch_hist[-k:], dim=0)
                        motion_cond[:, :, ctx_frames - k * PS:ctx_frames] = mhist_patches.transpose(0, 1).unsqueeze(0)

                    motion_patch = self.motion_cfm(
                        mu=h_cur.float(),
                        global_cond=first_motion.unsqueeze(0).float() if first_motion.dim() == 1 else first_motion.float(),
                        n_timesteps=int(cfm_steps),
                        patch_size=PS,
                        cond=motion_cond,
                        temperature=float(cfm_temperature),
                        cfg_value=float(cfm_cfg),
                        sway_sampling_coef=1.0,
                        use_cfg_zero_star=True,
                    )  # (1, 40, PS)

                motion_frames = motion_patch.transpose(1, 2)  # (1, PS, 40)
                gen_motion_latents.append(motion_frames)
                motion_patch_hist.append(motion_patch[0].transpose(0, 1).detach())  # (PS, 40)

            # Build next LLM token: project generated speech+motion → separate patch encoders → sum
            with amp_ctx:
                # Speech projection: (1, PS, 128) → (1, PS, D)
                e_speech = self.projector_speech(speech_frames.to(dtype=full_embeds.dtype))
                # Speech patch encoder: (1, 1, PS, D) → (1, 1, D)
                speech_token = self.speech_patch_encoder(e_speech.unsqueeze(1))  # (1, 1, D)

                if motion_patch is not None:
                    e_motion = self.projector_motion(motion_frames.to(dtype=full_embeds.dtype))
                    # Motion patch encoder: (1, 1, PS, D) → (1, 1, D)
                    motion_token = self.motion_patch_encoder(e_motion.unsqueeze(1))  # (1, 1, D)
                    e_next = speech_token + motion_token
                else:
                    e_next = speech_token

            attn_mask = torch.cat([attn_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1)
            past_kv, h_cur = llm_step_one(past_kv, attn_mask, e_next)

        # Decode speech latents to audio
        if len(gen_speech_latents) > 0:
            gen_speech_btc = torch.cat(gen_speech_latents, dim=1)  # (1, N*PS, 128)
            speech_latents_btc = torch.cat(
                [speech_lat_prefix, gen_speech_btc], dim=1
            )
        else:
            speech_latents_btc = speech_lat_prefix
            gen_speech_btc = speech_lat_prefix[:, :0, :]

        wav_full_24k = self.pretrained_speech_encoder.decode(speech_latents_btc, latent_layout="btc")
        if no_prefix:
            wav_prefix_dec_24k = wav_full_24k[:, :, :0]
        else:
            wav_prefix_dec_24k = self.pretrained_speech_encoder.decode(speech_lat_prefix, latent_layout="btc")
        wav_rest_dec_24k = (
            self.pretrained_speech_encoder.decode(gen_speech_btc, latent_layout="btc")
            if gen_speech_btc.size(1) > 0
            else wav_full_24k[:, :, :0]
        )

        # Collect motion latents
        if len(gen_motion_latents) > 0:
            motion_latents = torch.cat(gen_motion_latents, dim=1)  # (1, T_motion, 40)
        else:
            motion_latents = torch.zeros((1, 0, MOTION_DIM), device=device)

        return wav_prefix_dec_24k, wav_rest_dec_24k, wav_full_24k, motion_latents

    def enable_input_require_grads(self):
        self.llm.enable_input_require_grads()

    def freeze_encoder(self):
        freeze_model(self.pretrained_speech_encoder)
        self.pretrained_speech_encoder.eval()
        freeze_model(self.spk)
        self.spk.eval()
        # lm_head is never used (no text-token prediction), freeze it
        freeze_model(self.llm.lm_head)

    def load_projector(self, projector_dir: str):
        """
        Load all safetensors shards from a directory and merge them.
        """
        import safetensors.torch
        import os
        import re
        files = [
            os.path.join(projector_dir, f)
            for f in os.listdir(projector_dir)
            if f.endswith('.safetensors')
        ]
        if not files:
            raise ValueError(f"No .safetensors files found in {projector_dir}")

        def shard_key(fp):
            name = os.path.basename(fp)
            m = re.search(r'(\d+)-of-(\d+)\.safetensors$', name)
            if m:
                return (int(m.group(2)), int(m.group(1)))
            return (0, 0)

        files.sort(key=shard_key)

        merged = {}
        for fp in files:
            sd = safetensors.torch.load_file(fp)
            merged.update(sd)

        # Remap old checkpoint keys (first_motion_proj -> global_cond_proj, patch_encoder -> speech_patch_encoder)
        remap = {}
        for k in list(merged.keys()):
            new_k = k.replace('first_motion_proj', 'global_cond_proj')
            if new_k.startswith('patch_encoder.'):
                new_k = 'speech_patch_encoder.' + new_k[len('patch_encoder.'):]
            if new_k != k:
                remap[new_k] = merged.pop(k)
        merged.update(remap)

        aa = self.load_state_dict(merged, strict=False)

    def load_tts_checkpoint(self, tts_ckpt_dir: str):
        """
        Load TTS checkpoint to initialize speech-related components.

        TTS model key mapping to our model:
          motion_cfm.*           -> speech_cfm.*        (TTS "motion_cfm" is actually speech CFM)
          *.spk_proj.*           -> *.global_cond_proj.* (speaker proj renamed)
          projector_speech.*     -> projector_speech.*   (only linear1 matches; linear2 is new)
          head_stop.*            -> head_stop.*
          llm.*                  -> llm.*
        """
        import safetensors.torch
        import os
        import re

        files = [
            os.path.join(tts_ckpt_dir, f)
            for f in os.listdir(tts_ckpt_dir)
            if f.endswith('.safetensors')
        ]
        if not files:
            raise ValueError(f"No .safetensors files found in {tts_ckpt_dir}")

        def shard_key(fp):
            name = os.path.basename(fp)
            m = re.search(r'(\d+)-of-(\d+)\.safetensors$', name)
            if m:
                return (int(m.group(2)), int(m.group(1)))
            return (0, 0)

        files.sort(key=shard_key)

        merged = {}
        for fp in files:
            sd = safetensors.torch.load_file(fp)
            merged.update(sd)

        # Key remapping: TTS checkpoint -> our model
        remapped = {}
        for k, v in merged.items():
            new_k = k
            # 1) motion_cfm -> speech_cfm (TTS's "motion_cfm" is the speech CFM)
            if new_k.startswith('motion_cfm.'):
                new_k = 'speech_cfm.' + new_k[len('motion_cfm.'):]
            # 2) spk_proj -> global_cond_proj
            new_k = new_k.replace('.spk_proj.', '.global_cond_proj.')
            # 3) patch_encoder.in_proj -> speech_patch_encoder.input_proj
            #    patch_encoder.* -> speech_patch_encoder.*
            new_k = new_k.replace('patch_encoder.in_proj.', 'patch_encoder.input_proj.')
            if new_k.startswith('patch_encoder.'):
                new_k = 'speech_patch_encoder.' + new_k[len('patch_encoder.'):]
            remapped[new_k] = v

        # Filter: only load keys that exist in our model and have matching shapes
        our_sd = self.state_dict()
        to_load = {}
        skipped_missing = []
        skipped_shape = []
        for k, v in remapped.items():
            if k not in our_sd:
                skipped_missing.append(k)
                continue
            if our_sd[k].shape != v.shape:
                skipped_shape.append(f"{k}: ckpt {v.shape} vs model {our_sd[k].shape}")
                continue
            to_load[k] = v

        print(f"[TTS Init] Loading {len(to_load)} keys from TTS checkpoint: {tts_ckpt_dir}")
        if skipped_missing:
            print(f"[TTS Init] Skipped {len(skipped_missing)} keys (not in model): {skipped_missing[:10]}...")
        if skipped_shape:
            print(f"[TTS Init] Skipped {len(skipped_shape)} keys (shape mismatch):")
            for s in skipped_shape:
                print(f"  {s}")

        # Count loaded keys by component
        component_counts = {}
        for k in to_load:
            prefix = k.split('.')[0]
            component_counts[prefix] = component_counts.get(prefix, 0) + 1
        print(f"[TTS Init] Keys per component: {component_counts}")

        result = self.load_state_dict(to_load, strict=False)
        if result.unexpected_keys:
            print(f"[TTS Init] Unexpected keys: {result.unexpected_keys}")
        c=1

def init_model(model_args):

    # Load WhisperX-VAE as speech encoder
    from whisperx_vae import WhisperXVAE
    _whispervae_ckpt = os.environ.get(
        "WHISPERVAE_CKPT",
        '/apdcephfs_tj5/share_302528826/zhenye/deps_eval/models/whispervae/epoch=0-step=480000.ckpt',
    )
    ckpt = torch.load(_whispervae_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    cfg = ckpt['hyper_parameters']['cfg']
    whisperx_vae = WhisperXVAE(cfg)
    whisperx_vae.load_state_dict(state_dict, strict=False)
    whisperx_vae.eval()

    config = transformers.AutoConfig.from_pretrained(
        model_args.llm_model_name_or_path)
    config.use_cache = False

    llm_model = AutoLigerKernelForCausalLM.from_pretrained(
        model_args.llm_model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype='bfloat16'
    )
    # from transformers import AutoModelForCausalLM

    # llm_model = AutoModelForCausalLM.from_pretrained(
    #     model_args.llm_model_name_or_path,
    #     attn_implementation="flash_attention_2",
    #     torch_dtype='bfloat16'
    # )
    llm_dim = config.hidden_size

    projector_speech = ProjectorConv1d(model_args, model_args.speech_embedding_dim, llm_dim)
    projector_motion = ProjectorConv1d(model_args, model_args.motion_embedding_dim, llm_dim)

    model = SpeechLLM(llm_model, whisperx_vae, projector_speech, projector_motion, config, model_args)
    if model_args.tts_checkpoint_path is not None:
        model.load_tts_checkpoint(model_args.tts_checkpoint_path)
    if model_args.projector_model_path is not None:
        model.load_projector(model_args.projector_model_path)
    return model
