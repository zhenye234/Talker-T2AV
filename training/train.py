"""Talker-T2AV training entry point.

Usage (single node, 8 GPUs):
    torchrun --standalone --nnodes=1 --nproc_per_node=8 \\
        training/train.py training/config.json
"""
import os
import sys
import pathlib
from dataclasses import field

# Repo root + this dir on sys.path so peer + parent .py files import cleanly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

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
    optim: str = field(default="adamw_torch_fused")
    ddp_find_unused_parameters: bool = field(default=False)


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
    train_dataset[0]  # touch first sample so any decode error fails loud

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
        data_collator=collator,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    main()
