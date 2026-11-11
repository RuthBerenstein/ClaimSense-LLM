"""
QLoRA fine-tuning for ClaimSense-LLM.
Usage: python src/train.py 
       config configs/train_config.yaml
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)

sys.path.append(str(Path(__file__).parent))
from prompts import build_example  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_config.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    # Llama 3.1 ships a dedicated padding token; using EOS as pad would stop the model learning when to end.
    if "<|finetune_right_pad_id|>" in tok.get_vocab():
        tok.pad_token = "<|finetune_right_pad_id|>"
    else:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], quantization_config=bnb, device_map="auto")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files={"train": cfg["train_file"], "eval": cfg["eval_file"]})

    def tokenize(ex):
        return tok(build_example(ex["task"], ex["input"], ex["output"]),
                   truncation=True, max_length=cfg["max_len"])

    ds = ds.map(tokenize, remove_columns=ds["train"].column_names)

    steps_per_epoch = max(1, len(ds["train"]) // (cfg["batch_size"] * cfg["grad_accum"]))
    warmup = max(1, int(0.03 * steps_per_epoch * cfg["epochs"]))

    args = TrainingArguments(
        output_dir=cfg["output_dir"], num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"], gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"], warmup_steps=warmup, lr_scheduler_type="cosine",
        logging_steps=20, eval_strategy="epoch", save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
        report_to="none")
    Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["eval"],
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()

    model.save_pretrained(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])
    print("Saved adapter to", cfg["output_dir"])


if __name__ == "__main__":
    main()
