
import math
import json
import random
import csv
from dataclasses import dataclass, field
from typing import Dict
import os
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
import torch
import torch.nn.functional as F
import torchaudio
import transformers
import whisper
from io import BytesIO
import numpy as np

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."})
    test_data_path: str = field(default=None,
                                metadata={"help": "Path to the test data."})


# Default paths — override via env vars when launching training.
TRAIN_CSV = os.environ.get("TRAIN_CSV", "./data/csv/train.csv")
TRAIN_TXT = os.environ.get("TRAIN_TXT", "./data/txt/train_pt_list.txt")
VIDEO_DIR = os.environ.get("VIDEO_DIR", "./data/video")
# DATA_ROOT prefixes any relative wav/motion/video paths from the CSV.
# Set this to the dataset root after extracting Talker-T2AV-Data shards.
DATA_ROOT = os.environ.get("DATA_ROOT", "")


def _resolve(path: str) -> str:
    """Prefix `path` with DATA_ROOT if it is relative (and DATA_ROOT is set)."""
    if not path or os.path.isabs(path) or not DATA_ROOT:
        return path
    return os.path.join(DATA_ROOT, path)

# Motion normalization stats (40-dim) — reused from the vendored LIA-X
# package at the repo root (../lia_x/ relative to this file).
_LIAX_PKG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lia_x")
MOTION_MEAN = np.load(os.path.join(_LIAX_PKG_DIR, "motion_mean.npy")).astype(np.float32)  # (40,)
MOTION_STD = np.load(os.path.join(_LIAX_PKG_DIR, "motion_std.npy")).astype(np.float32)    # (40,)
MOTION_STD = np.clip(MOTION_STD, a_min=1e-6, a_max=None)


class SpeechDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Supports two tasks:
    - T2SV (text-to-speech-video): CSV with motion_pt_path, wav_path, text
    - TTS  (text-to-speech only):  TXT file with wav paths + sidecar .json for text

    When both are provided, they are mixed in a single dataset.
    TTS samples have motion_len=0 and empty motion tensor.
    """

    def __init__(
        self,
        data_path,
        tokenizer: transformers.PreTrainedTokenizer,
        config,
        inference: bool = False,
        use_csv: bool = True,
    ):
        super(SpeechDataset, self).__init__()
        print("Formatting inputs...")
        self.tokenizer = tokenizer
        self.config = config
        self.inference = inference
        self.data_path = data_path
        self.use_csv = use_csv

        # Patch-level parameters
        self.patch_size = getattr(config, 'patch_size', 4)
        self.delay = getattr(config, 'delay', 5)  # in patches

        self.SPEECH_TOK = "<SPEECH_FRAME>"
        self.MOTION_TOK = "<MOTION_FRAME>"

        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.SPEECH_TOK, self.MOTION_TOK]})

        # Load data from CSV or TXT
        if use_csv:
            csv_path = TRAIN_CSV
            print(f"Loading T2SV data from CSV: {csv_path}")
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                t2sv_list = list(reader)

            # Mark T2SV samples
            for item in t2sv_list:
                item['task'] = 't2sv'

            self.data_list = t2sv_list
            print(f"  T2SV 数据: {len(t2sv_list)} 条")
        else:
            txt_path = TRAIN_TXT
            print(f"Loading data from TXT: {txt_path}")
            with open(txt_path, 'r') as f:
                pt_paths = [line.strip() for line in f if line.strip()]

            self.data_list = []
            for pt_path in pt_paths:
                name = os.path.splitext(os.path.basename(pt_path))[0]
                wav_path = os.path.join(VIDEO_DIR, f'{name}.wav')
                self.data_list.append({
                    'motion_pt_path': pt_path,
                    'wav_path': wav_path,
                    'pt_length': '-1',
                    'dataset_source': 'dh_facevid',
                    'task': 't2sv',
                })

            print(f"总共有 {len(self.data_list)} 条 (TXT格式)")

        # Load TTS data (TXT file with wav paths, text from sidecar .json)
        tts_data_path = getattr(config, 'tts_data_path', None)
        if tts_data_path:
            print(f"Loading TTS data from TXT: {tts_data_path}")
            with open(tts_data_path, 'r', encoding='utf-8') as f:
                wav_paths = [line.strip() for line in f if line.strip()]
            tts_list = []
            for wp in wav_paths:
                tts_list.append({
                    'wav_path': wp,
                    'task': 'tts',
                })

            # Upsample the smaller set so both tasks have roughly equal data
            n_t2sv = len(self.data_list)
            n_tts = len(tts_list)
            if n_t2sv > 0 and n_tts > 0:
                if n_tts > n_t2sv:
                    # T2SV is smaller → repeat T2SV to match TTS
                    repeat = max(1, round(n_tts / n_t2sv))
                    self.data_list = self.data_list * repeat
                    print(f"  T2SV 上采样 {repeat}x: {n_t2sv} → {len(self.data_list)} 条")
                elif n_t2sv > n_tts:
                    # TTS is smaller → repeat TTS to match T2SV
                    repeat = max(1, round(n_t2sv / n_tts))
                    tts_list = tts_list * repeat
                    print(f"  TTS 上采样 {repeat}x: {n_tts} → {len(tts_list)} 条")

            self.data_list.extend(tts_list)
            print(f"  TTS 数据: {n_tts} 条 (原始)")

        print(f"总共有 {len(self.data_list)} 条 (混合)")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            return self._load_item(i)
        except Exception as e:
            print(f"[WARNING] Failed to load item {i}: {e}, fallback to random item")
            return self._load_item(random.randint(0, len(self.data_list) - 1))

    def _load_item(self, i) -> Dict[str, torch.Tensor]:
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index

        data_item = self.data_list[i]
        task = data_item.get('task', 't2sv')
        wav_path = _resolve(data_item['wav_path'])
        is_tts = (task == 'tts')

        # =============================================
        # Load motion (T2SV only)
        # =============================================
        if is_tts:
            motion = np.zeros((0, 40), dtype=np.float32)
            motion_len = 0
            start_frame = 0
            # Read text from sidecar .json file
            json_path = os.path.splitext(wav_path)[0] + ".json"
            with open(json_path, "r", encoding="utf-8") as f:
                text = json.load(f)["text"]
        else:
            pt_path = _resolve(data_item['motion_pt_path'])
            text = data_item.get('text', '')
            # Load motion: [T, 40] float16 -> float32, then normalize
            motion = torch.load(pt_path, map_location='cpu', weights_only=True).detach().float().numpy()
            motion = (motion - MOTION_MEAN) / MOTION_STD
            start_frame = 0

        # Variable length: use full length, clamp to MAX_MOTION_LEN (750 frames / 30s)
        MAX_MOTION_LEN = 25 * 30
        if not is_tts:
            total_frames = motion.shape[0]
            if total_frames > MAX_MOTION_LEN:
                if not self.inference:
                    start_frame = random.randint(0, total_frames - MAX_MOTION_LEN)
                else:
                    start_frame = 0
                motion = motion[start_frame:start_frame + MAX_MOTION_LEN, :]

        # Load audio
        audio_raw, sample_rate = torchaudio.load(wav_path)

        # Resample to 24kHz for WhisperVAE encoder
        if sample_rate != 24000:
            audio_24k = torchaudio.transforms.Resample(sample_rate, 24000)(audio_raw)
        else:
            audio_24k = audio_raw

        # Resample to 16kHz for Whisper mel spectrogram and speaker encoder
        if sample_rate != 16000:
            audio_16k = torchaudio.transforms.Resample(sample_rate, 16000)(audio_raw)
        else:
            audio_16k = audio_raw

        if is_tts:
            # TTS: use full audio length (no motion alignment), clamp to 30s
            max_samples_16k = 16000 * 30
            max_samples_24k = 24000 * 30
            audio_16k = audio_16k[:, :max_samples_16k]
            audio_24k = audio_24k[:, :max_samples_24k]
        else:
            # T2SV: crop audio aligned with motion (variable length)
            # 16kHz: 640 samples per frame; 24kHz: 960 samples per frame
            audio_start_16k = start_frame * 640
            audio_end_16k = audio_start_16k + motion.shape[0] * 640
            audio_16k = audio_16k[:, audio_start_16k:audio_end_16k]

            audio_start_24k = start_frame * 960
            audio_end_24k = audio_start_24k + motion.shape[0] * 960
            audio_24k = audio_24k[:, audio_start_24k:audio_end_24k]

        audio_pad = whisper.pad_or_trim(audio_16k)

        # audio16k for speaker encoder (pad to 30s)
        audio_16k_pad = whisper.pad_or_trim(audio_16k)

        feat = whisper.log_mel_spectrogram(audio_pad, n_mels=128)

        fps = 25
        num_samples = int(audio_16k.size(1)) if audio_16k.size(1) is not None else 0
        mel_len_raw = math.floor((num_samples / 16000.0) * fps) if num_samples > 0 else 0
        mel_len = int(max(1, min(mel_len_raw, int(30 * fps))))

        PS = self.patch_size
        DELAY = self.delay  # in patches

        if is_tts:
            # TTS: no motion, placeholder = speech patches only
            motion_len = 0
            n_speech_patches = math.ceil(mel_len / PS)
            placeholder_len = n_speech_patches
        else:
            # T2SV: motion present
            motion_len = motion.shape[0]
            # Truncate motion_len to patch_size multiple
            motion_len = (motion_len // PS) * PS
            if motion_len < PS:
                motion_len = PS
            motion = motion[:motion_len, :]

            # Compute patch counts
            n_speech_patches = math.ceil(mel_len / PS)
            n_motion_patches = motion_len // PS
            # Placeholder length in patch tokens
            placeholder_len = max(n_speech_patches, n_motion_patches + DELAY)

        speech_placeholder = "".join([self.SPEECH_TOK] * placeholder_len)

        chat = [
            {"role": "user", "content": text},
            {"role": "assistant", "content": speech_placeholder},
        ]

        kwargs = {
            'padding': False,
            'max_length': self.config.model_max_length,
            'truncation': True,
            'add_generation_prompt': False,
        }

        ids_text = self.tokenizer.apply_chat_template(chat,
                                                      tokenize=True,
                                                      enable_thinking=False,
                                                      **kwargs)
        ids_text = ids_text.data['input_ids'] if hasattr(ids_text, 'data') else ids_text
        input_ids = torch.tensor(ids_text, dtype=torch.int)

        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        speech_id = self.tokenizer.convert_tokens_to_ids(self.SPEECH_TOK)

        speech_pos = (input_ids == speech_id).nonzero(as_tuple=True)[0]

        speech_span = (int(speech_pos[0]), int(speech_pos[-1] + 1)) if speech_pos.numel() else (-1, -1)

        ret = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'motion': torch.tensor(motion).float(),
            'feat': feat,
            'audio': audio_24k,
            'audio16k': audio_16k_pad,
            "speech_span": torch.tensor(speech_span),
            "mel_len": mel_len,
            "motion_len": motion_len,
            "task_tag": 0 if is_tts else 1,  # 0=TTS, 1=T2SV
        }

        return ret


class DynamicPadCollator:
    """Collator that dynamically pads each batch to the longest sample.
    Ensures max_motion_len is a multiple of patch_size."""

    def __init__(self, pad_token_id, patch_size=4):
        self.pad_token_id = pad_token_id
        self.patch_size = patch_size

    def __call__(self, batch):
        max_seq_len = max(sample['input_ids'].size(0) for sample in batch)
        max_motion_len = max(sample['motion'].size(0) for sample in batch)
        # Align max_motion_len to patch_size multiple (handle 0 case for TTS-only batches)
        PS = self.patch_size
        if max_motion_len > 0:
            max_motion_len = ((max_motion_len + PS - 1) // PS) * PS
        max_audio_len = max(sample['audio'].size(1) for sample in batch)
        max_audio16k_len = max(sample['audio16k'].size(1) for sample in batch)
        B = len(batch)

        input_ids = torch.full((B, max_seq_len), self.pad_token_id, dtype=torch.int)
        attention_mask = torch.zeros((B, max_seq_len), dtype=torch.bool)
        motion = torch.zeros((B, max_motion_len, 40), dtype=torch.float)
        audio = torch.zeros((B, 1, max_audio_len), dtype=torch.float)
        audio16k = torch.zeros((B, 1, max_audio16k_len), dtype=torch.float)

        feat = torch.stack([sample['feat'] for sample in batch], dim=0)
        speech_span = torch.stack([sample['speech_span'] for sample in batch], dim=0)
        mel_len = torch.tensor([sample['mel_len'] for sample in batch], dtype=torch.long)
        motion_len = torch.tensor([sample['motion_len'] for sample in batch], dtype=torch.long)
        task_tag = torch.tensor([sample['task_tag'] for sample in batch], dtype=torch.long)

        for i, sample in enumerate(batch):
            seq_i = sample['input_ids'].size(0)
            input_ids[i, :seq_i] = sample['input_ids']
            attention_mask[i, :seq_i] = sample['attention_mask']

            m_i = sample['motion'].size(0)
            motion[i, :m_i, :] = sample['motion']

            a_i = sample['audio'].size(1)
            audio[i, :, :a_i] = sample['audio']

            a16k_i = sample['audio16k'].size(1)
            audio16k[i, :, :a16k_i] = sample['audio16k']

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'motion': motion,
            'feat': feat,
            'audio': audio,
            'audio16k': audio16k,
            'speech_span': speech_span,
            'mel_len': mel_len,
            'motion_len': motion_len,
            'task_tag': task_tag,
        }


if __name__ == "__main__":

    aaa = SpeechDataset()

    c = 1
