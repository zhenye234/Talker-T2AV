# Training Talker-T2AV

This folder holds the bare minimum to train the Talker-T2AV model on
your own speech-video corpus. The five mid-training eval callbacks from
the original training run (Teacher-Forcing × {DH-FaceVid, Hallo3},
Continuation × {DH-FaceVid, Hallo3}, TTS-WER) — and the
`eval_teacherforcing.py` they imported — have been removed so this
folder is purely a training entry point. To reproduce the paper
benchmarks at the end of training, use the inference-side scripts at
the repo root (`infer.py` for single samples, plus whatever offline
benchmark you bring).

## Layout

```
training/
├── train.py             # entry point — imports speech_llm + dataset, runs Trainer
├── dataset.py           # SpeechDataset (T2AV CSV + TTS .txt) + DynamicPadCollator
├── ds_config_zero3.json # DeepSpeed Zero-3 config (mirrors the paper's setup)
├── config.json          # example training-run config (overrides via JSON)
└── README.md            # this file
```

`train.py` adds the repo root to `sys.path` at import time, so the
shared model code (`../speech_llm.py`, `../llama4nar.py`,
`../local_dit.py`, `../unified_cfm.py`, `../whisperx_vae.py`,
`../speaker_verification/`, `../lia_x/`) is reused exactly as-is, with
no copy.

## Data layout

`dataset.py` expects either a CSV (T2AV) or a TXT (TTS-only) source.
Paths are read from env vars at module import time; defaults are below.

| Env var | Default | What it lists |
|---|---|---|
| `TRAIN_CSV` | `./data/csv/unified_all_dataset_more_filtered.csv` | Training rows: `motion_pt_path,wav_path,text,duration,...` (text column is required). |
| `EVAL_CSV_DH_FACEVID` | `./data/csv/dh_facevid_dataset_128.csv` | Held-out DH-FaceVid eval rows (used for the `eval_dataset` Subset). |
| `EVAL_CSV_HALLO3` | `./data/csv/hallo3_dataset_128.csv` | Held-out Hallo3 rows. |
| `TRAIN_TXT` / `EVAL_TXT` | `./data/txt/dh_facevid_pt_list_*.txt` | Legacy TXT splits (one `.pt` motion path per line; sidecar `.json` carries the text). Keep when training the TTS-auxiliary Emilia mix. |
| `VIDEO_DIR` | `./data/video` | Root for the `{name}.wav` lookups inside `SpeechDataset` (DH-FaceVid). |

LIA-X motion-normalization stats are loaded from the vendored
`../lia_x/motion_mean.npy` and `../lia_x/motion_std.npy` automatically.

## Required pretrained inputs

`init_model(model_args)` (in `../speech_llm.py`) instantiates and freezes:

- **Qwen3-0.6B** — pulled from `model_args.llm_model_name_or_path` via
  `AutoModelForCausalLM.from_pretrained` (downloaded by `transformers`
  on first run; set `HF_HOME` to redirect the cache).
- **WhisperX-VAE** — checkpoint path comes from the env var
  `WHISPERVAE_CKPT`; the model code is vendored in
  `../whisperx_vae.py`.
- **WavLM-Large speaker encoder** — `WAVLM_CKPT` env var; code in
  `../speaker_verification/`. The s3prl WavLM upstream is also fetched
  via `torch.hub` (cached at `~/.cache/torch/hub` or `$S3PRL_CACHE_DIR`).
- **TTS pre-trained Talker-T2AV checkpoint** — pointed at by
  `tts_checkpoint_path` in `config.json`. Set this to the small
  TTS-only run that warm-starts the speech CFM head + Patch Transformer
  Encoder + LLM stop predictor (the paper's "Stage-1" weights).

Set the corresponding env vars before launch — e.g.:

```bash
export WHISPERVAE_CKPT=$(pwd)/ckpts/hf_weights/whisperx-vae/model.ckpt
export WAVLM_CKPT=/path/to/wavlm_large_finetune.pth
export HF_HOME=$(pwd)/.cache/huggingface       # optional
```

## Launch

Single-node, 8 GPUs (DeepSpeed Zero-3, bf16, gradient checkpointing on):

```bash
deepspeed --num_gpus 8 training/train.py training/config.json
```

Multi-node: launch with whatever multi-node mechanism your cluster uses
(e.g. `torchrun --nnodes ...` or your scheduler-provided `deepspeed
--hostfile`). Make sure all ranks see the same `TRAIN_CSV`,
`WHISPERVAE_CKPT`, `WAVLM_CKPT`, etc.

`Trainer` resumes from `<output_dir>/checkpoint-*` if any exists, so
re-running the same command after a preempt continues training; a
fresh run starts from `tts_checkpoint_path`.

## Notes

- `train.py` removed the original mid-training eval callbacks because
  they pulled in the full SyncNet / LIA-X render / Whisper-WER stack
  via `eval_teacherforcing.py`. Loss curves in W&B (per-step
  `train/loss`, `train/learning_rate`, ...) are still emitted by the
  HuggingFace `Trainer` itself.
- All `model.freeze_encoder()` logic stays inside
  `speech_llm.SpeechLLM`; nothing in `train.py` currently un-freezes
  any sub-module.
- DeepSpeed Zero-3 is wired in via `config.json["deepspeed"]`. To
  switch to Zero-2 or DDP-only, drop that key (and `--deepspeed`) and
  fall back to `torchrun --nproc_per_node 8 training/train.py ...`.
