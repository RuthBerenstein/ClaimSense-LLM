"""Evaluate ClaimSense on a JSONL test set.

Usage: python src/evaluate.py --adapter outputs/claimsense-lora --data data/test.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from inference import generate, load  # noqa: E402

KEY_FIELD = {"coverage": "decision", "fraud": "risk"}


def field_f1(pred: dict, gold: dict) -> float:
    if not gold:
        return 0.0
    hits = sum(1 for k, v in gold.items() if k != "notes" and pred.get(k) == v)
    return hits / max(1, len([k for k in gold if k != "notes"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--limit", type=int, default=300)
    a = ap.parse_args()
    model, tok = load(a.adapter)
    stats = defaultdict(lambda: {"n": 0, "valid": 0, "score": 0.0})
    for i, line in enumerate(open(a.data)):
        if i >= a.limit:
            break
        ex = json.loads(line)
        gold = json.loads(ex["output"])
        s = stats[ex["task"]]
        s["n"] += 1
        try:
            pred = json.loads(generate(model, tok, ex["task"], ex["input"]))
            s["valid"] += 1
        except json.JSONDecodeError:
            continue
        if ex["task"] == "extract":
            s["score"] += field_f1(pred, gold)
        else:
            s["score"] += float(pred.get(KEY_FIELD[ex["task"]]) == gold[KEY_FIELD[ex["task"]]])
    print(f"{'task':<10}{'n':>5}{'valid_json':>12}{'score':>8}")
    for t, s in stats.items():
        print(f"{t:<10}{s['n']:>5}{s['valid']/s['n']:>12.2%}{s['score']/s['n']:>8.3f}")


if __name__ == "__main__":
    main()
