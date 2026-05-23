"""Talker-T2AV training entry point.

Phase-1 release: this script trains the AR backbone + dual diffusion
heads + Patch Transformer Encoders on the joint speech-video corpus.

The original `train.py` shipped with five custom `TrainerCallback`s
(TeacherForcingEvalCallback × {dh_facevid, hallo3}, ContinuationEvalCallback
× {dh_facevid, hallo3}, EvalContinueWERCallback) that ran rich
mid-training metrics — render + SyncNet + Whisper-WER — by importing
`eval_teacherforcing.py`. Those have been removed for a minimal
training-only release; reproduce the paper benchmarks separately via the
inference-side `eval_metrics/` tooling.

Usage (single node, 8 GPUs):
    deepspeed --num_gpus 8 training/train.py training/config.json

The repo-root modules (`speech_llm`, `unified_cfm`, `local_dit`,
`llama4nar`, `whisperx_vae`, ...) are imported by adding the parent
directory to `sys.path` below.
"""
import os
import sys
import pathlib
from dataclasses import field

# --- repo-root + this-dir on sys.path so peer + parent .py files import cleanly ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)   # for ./dataset.py
sys.path.insert(0, _REPO)   # for ../speech_llm.py and friends

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("WANDB_PROJECT", "talker-t2av")

import torch  # noqa: E402
torch.backends.cudnn.enabled = False
from transformers import logging  # noqa: E402
logging.set_verbosity_error()

import transformers  # noqa: E402
from transformers import AutoTokenizer, Trainer  # noqa: E402

from dataset import DataArguments, SpeechDataset, DynamicPadCollator  # noqa: E402
from speech_llm import init_model, ModelArguments  # noqa: E402


class TrainingArguments(transformers.TrainingArguments):
    """Same as transformers.TrainingArguments with a fused-AdamW default
    and DDP find-unused-parameters off. Eval-callback-specific fields from
    the original training run have been removed alongside the callbacks."""

    optim: str = field(default="adamw_torch_fused")
    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={"help": "All trainable params now participate via dummy "
                          "forwards in speech_llm; set False."},
    )


def main() -> None:
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    (model_args, data_args, training_args) = parser.parse_json_file(
        json_file=os.path.abspath(sys.argv[1])
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.llm_model_name_or_path,
        model_max_length=model_args.model_max_length,
        padding_side="right",
    )

    print("Loading data...")
    train_dataset = SpeechDataset("train", tokenizer, model_args)
    eval_dataset = SpeechDataset("val", tokenizer, model_args, inference=True)
    train_dataset[0]  # touch first sample so any decode error fails loud

    from torch.utils.data import Subset
    eval_dataset = Subset(eval_dataset, indices=list(range(256)))
    eval_dataset_loss = Subset(train_dataset, indices=list(range(64)))

    model = init_model(model_args)
    model.freeze_encoder()
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    collator = DynamicPadCollator(
        pad_token_id=tokenizer.pad_token_id,
        patch_size=model_args.patch_size,
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_loss,
        data_collator=collator,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    main()
