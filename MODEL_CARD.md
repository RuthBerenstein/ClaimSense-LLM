# Model Card: ClaimSense-LLM (LoRA adapter)

- **Base model:** mistralai/Mistral-7B-Instruct-v0.3
- **Method:** QLoRA (4-bit NF4, r=16, alpha=32), 3 epochs
- **Training data:** 3,000 synthetic instruction examples (FNOL extraction, coverage reasoning, fraud signals)
- **Intended use:** Decision support for P&C claims adjusters and insurance analytics teams.
- **Out of scope:** Automated claim denial, underwriting decisions, legal advice, use on real PII without governance.
- **Limitations:** Trained on synthetic templates; real-world notes are messier. Policy clauses are simplified and do not reflect any specific insurer's wording.
- **Ethical considerations:** Fraud indicators may correlate with protected attributes; audit for disparate impact before deployment. Human review is mandatory.
- **Evaluation:** See README §7 and `src/evaluate.py`.
