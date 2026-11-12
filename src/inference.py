# Run ClaimSense on a single input.

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.append(str(Path(__file__).parent))
from prompts import build_prompt  # noqa: E402

BASE = "meta-llama/Llama-3.1-8B-Instruct"

def load(adapter: str, base: str = BASE):
    tok = AutoTokenizer.from_pretrained(adapter)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map="auto")
    return PeftModel.from_pretrained(model, adapter).eval(), tok

@torch.no_grad()
def generate(model, tok, task: str, text: str, max_new_tokens: int = 300) -> str:
    ids = tok(build_prompt(task, text), return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id,
                         eos_token_id=tok.convert_tokens_to_ids("<|eot_id|>"))
    #return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    raw = tok.decode(out[0][ids["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False).strip()
        while raw.endswith("</s>"):
            raw = raw[:-4].rstrip()
        return raw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--task", choices=["extract", "coverage", "fraud"], required=True)
    ap.add_argument("--text", required=True)
    a = ap.parse_args()
    model, tok = load(a.adapter)
    raw = generate(model, tok, a.task, a.text)
    try:
        print(json.dumps(json.loads(raw), indent=2))
    except json.JSONDecodeError:
        print(raw)

if __name__ == "__main__":
    main()
