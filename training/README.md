# Training Talker-T2AV

Minimal training entry point for the AR backbone + dual diffusion heads
+ Patch Transformer Encoders.

```
training/
├── train.py        # Trainer entry point
├── dataset.py      # SpeechDataset (T2SV CSV + optional TTS .txt) + DynamicPadCollator
├── config.json     # example training-run config
└── README.md
```

`train.py` adds the repo root to `sys.path` at import time, so the
shared model code (`../speech_llm.py`, `../llama4nar.py`,
`../local_dit.py`, `../unified_cfm.py`, `../whisperx_vae.py`,
`../speaker_verification/`, `../lia_x/`) is reused as-is.

## Data layout

Override any of these via env vars when launching:

| Env var | Default | Contents |
|---|---|---|
| `TRAIN_CSV` | `./data/csv/train.csv` | Training rows: `motion_pt_path,wav_path,text,duration,...` |
| `TRAIN_TXT` | `./data/txt/train_pt_list.txt` | Alternative TXT format: one `.pt` motion path per line, with sidecar `.json` for the text |
| `VIDEO_DIR` | `./data/video` | Root for `{name}.wav` lookups (TXT mode only) |

Motion-normalization stats are loaded from the vendored
`../lia_x/motion_mean.npy` and `../lia_x/motion_std.npy` automatically.

## Pretrained inputs

Set the following env vars before launching:

```bash
export WHISPERVAE_CKPT=$(pwd)/ckpts/hf_weights/whisperx-vae/model.ckpt
export WAVLM_CKPT=/path/to/wavlm_large_finetune.pth
```

`config.json` further references:

- `llm_model_name_or_path` — `Qwen/Qwen3-0.6B`, downloaded by
  `transformers` on first run (set `HF_HOME` to redirect the cache).
- `tts_checkpoint_path` — small TTS-only Talker-T2AV checkpoint that
  warm-starts the speech CFM head + Patch Transformer Encoder + LLM
  stop predictor (the paper's Stage-1 weights).

## Launch

Single node, 8 GPUs:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    training/train.py training/config.json
```

`Trainer` resumes from `<output_dir>/checkpoint-*` if any exists; a
fresh run starts from `tts_checkpoint_path`.
