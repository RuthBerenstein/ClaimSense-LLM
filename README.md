# ClaimSense-LLM 🛡️
**A domain-adapted Large Language Model for Property & Casualty insurance claims triage, coverage reasoning, and fraud-signal detection.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Status](https://img.shields.io/badge/status-research-orange)


## 1. The Problem
Insurance carriers receive millions of First Notice of Loss (FNOL) reports every year as free-text: call-centre notes, emails, adjuster comments. Today:

- **Adjusters spend 30–40% of their time** reading and re-keying unstructured notes into claim systems.
- **Coverage questions** ("Is water damage from a burst pipe covered under my HO-3 policy?") require reading long policy wordings.
- **Fraud indicators** (late reporting, inconsistent dates, recent policy changes) are easy to miss when notes are long and inconsistent.

General-purpose LLMs are fluent but hallucinate policy terms, don't output reliable structured data, and are not calibrated for insurance terminology.

## 2. The Solution
ClaimSense-LLM fine-tunes an open-weight base model (default: `meta-llama/Llama-3.1-8B-Instruct`, swappable for Mistral / Qwen 2.5) using **QLoRA** on a synthetic, privacy-safe insurance corpus to perform three tasks:

| Task | Input | Output |
|------|-------|--------|
| **1. FNOL Extraction** | Free-text claim note | Strict JSON: line of business, loss date, loss type, location, estimated severity, injuries |
| **2. Coverage Reasoning** | Policy clause + loss scenario | Covered / Not covered / Needs review + cited clause + explanation |
| **3. Fraud-Signal Flagging** | Claim note + policy metadata | Risk score (low/medium/high) + list of red-flag indicators with justification |

> ⚠️ ClaimSense is a **decision-support** tool. It never makes final claim decisions; a licensed adjuster stays in the loop.

## 3. Architecture
```
 Synthetic data generator ──► JSONL instruction dataset ──► QLoRA fine-tune (4-bit)
                                                                   │
            Evaluation (JSON validity, field F1, accuracy) ◄───────┤
                                                                   ▼
                                               Inference CLI / Gradio demo
``` 

## 4. Repository Structure
```
claimsense-llm/
├── README.md
├── LICENSE
├── MODEL_CARD.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── train_config.yaml
├── data/
│   └── sample/sample_train.jsonl
├── src/
│   ├── generate_data.py     # synthetic insurance dataset generator
│   ├── train.py             # QLoRA fine-tuning
│   ├── evaluate.py          # metrics
│   ├── inference.py         # CLI inference
│   └── prompts.py           # shared prompt templates
├── app/
│   └── demo.py              # Gradio web demo
└── tests/
    └── test_data.py
```

## 5. Quick Start
```bash
git clone https://github.com/ruthberenstein/claimsense-llm.git
cd claimsense-llm
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Generate synthetic training + test examples (no real customer data)
python src/generate_data.py --n 1000 --out data/train.jsonl --seed 42
python src/generate_data.py --n 100  --out data/test.jsonl  --seed 123

# 2) Fine-tune (free Colab T4 works: 1,000 examples, 1 epoch, ~1.5 h)
python src/train.py --config configs/train_config.yaml

# 3) Evaluate
python src/evaluate.py --adapter outputs/claimsense-lora --data data/test.jsonl

# 4) Try it
python src/inference.py --adapter outputs/claimsense-lora --task extract \
  --text "Customer called 3 Nov, kitchen pipe burst overnight, water through ceiling at 12 Elm St, Winnipeg. No injuries."

# 5) Web demo
python app/demo.py
```

## 6. Example
**Input (FNOL note):**
> Insured reports rear-end collision on Hwy 1 near Brandon on 2026-10-14 around 5pm. Other driver fled. Insured has neck pain, went to ER. Bumper and trunk heavily damaged.

**ClaimSense output:**
```json
{
  "line_of_business": "auto",
  "loss_type": "collision",
  "loss_date": "2026-10-14",
  "location": "Hwy 1 near Brandon",
  "injuries": true,
  "severity": "high",
  "third_party_involved": true,
  "notes": "Hit-and-run; bodily injury reported, ER visit."
}
```

## 7. Evaluation Results

ClaimSense-LLM is a QLoRA fine-tuned adapter for `meta-llama/Llama-3.1-8B-Instruct`. Evaluation used 100 synthetic test examples generated with seed 123. The `n` column shows the number of examples for each task.

### Task scores

| Task | n | Base Llama-3.1-8B-Instruct¹ | ClaimSense-LoRA² |
|---|---:|---:|---:|
| Claim details extraction | 34 | 0.689 | **0.987** |
| Coverage decision | 33 | 0.788 | **1.000** |
| Fraud risk classification | 33 | 0.424 | **0.727** |

Scores are reported on a 0–1 scale using the task-specific scoring rules in `src/evaluate.py`; higher is better. Invalid JSON outputs receive a score of zero.

### JSON parsing results

| Task | Base: relaxed parsing¹ | ClaimSense: strict parsing² |
|---|---:|---:|
| Claim details extraction | 100.00% | 100.00% |
| Coverage decision | 100.00% | 100.00% |
| Fraud risk classification | 96.97% | 100.00% |

**How these results were measured**

1. **Base model:** the pretrained Llama model without the ClaimSense adapter. A temporary evaluation helper accepted a single JSON object enclosed in Markdown code fences, including outputs with commentary outside the fenced block. The original strict evaluation rejected these formatted responses, producing 0% valid JSON and zero scores for all three tasks. Those zeros reflect failure to meet the strict output format, not proof that the model could not answer the tasks.
2. **ClaimSense-LoRA:** measured in the earlier evaluation using strict JSON parsing, after removing trailing end-of-sequence markers such as `</s>` from generated text. All 100 outputs parsed successfully.

**Comparison limitation:** the two columns were measured with different parsing rules. The attempt to rerun ClaimSense with the relaxed helper was interrupted by a Colab GPU-session limit. Because its earlier outputs were already valid JSON, the helper is expected to leave them unchanged, but that expectation has not been confirmed by a completed rerun. These are measured results with a disclosed parsing difference, not a verified identical-protocol comparison.

### Training and test setup

- Base model: `meta-llama/Llama-3.1-8B-Instruct`
- Fine-tuning method: QLoRA
- Training environment: Google Colab with a T4 GPU
- Training data: 1,000 synthetic examples, seed 42
- Test data: 100 synthetic examples, seed 123
- Training epochs: 1
- Maximum training sequence length: 512 tokens
- JSON validity measured after removing trailing `</s>` markers.
- n = the number of test examples for that task.

### Limitations

These results come from a small synthetic test set created by the same generator used for training data. They do not establish performance on real insurance claims, unfamiliar policy wording, or independent datasets. A coverage score of 1.000 means all 33 synthetic coverage examples were scored correctly—not that real-world coverage decisions will always be correct.

ClaimSense-LLM is a portfolio research prototype. It is not validated for autonomous claim decisions, coverage determinations, or fraud accusations. Production use would require independent evaluation, privacy and security controls, and qualified human review.


## 8. Data & Privacy
All training data is **100% synthetic**, generated by `src/generate_data.py`. No real policyholder or PII data is used. Do not upload real claims data to this repository.

## 9. Responsible AI
- Human-in-the-loop required for all coverage and fraud outputs.
- Fraud flags are *indicators*, not accusations; they must not be used as the sole basis for denial.
- Monitor for bias by region, age and claim type before any production use.

## 10. Next Roadmap - To Do
- [ ] RAG over full policy wordings (FAISS + embeddings)
- [ ] Multilingual FNOL (English/French for Canadian market)
- [ ] Subrogation opportunity detection
- [ ] Hugging Face Hub model release

## 11. Author
Ruth Berenstein — LLM / AI engineer focused on insurance & financial services.
LinkedIn: https://www.linkedin.com/in/ruthberenstein/ · Email: rutha603@gmail.com

## License
MIT - Ruth Berenstein see [LICENSE]
