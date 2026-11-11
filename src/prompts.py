# Shared prompt templates for ClaimSense-LLM

SYSTEM = (
    "You are ClaimSense, an insurance claims assistant. "
    "Be precise, never invent policy terms, and answer only with valid JSON."
)

TASK_INSTRUCTIONS = {
    "extract": (
        "Extract structured First Notice of Loss data from the claim note. Return JSON with keys: "
        "line_of_business, loss_type, loss_date (YYYY-MM-DD), location, injuries (bool), "
        "severity (low|medium|high), third_party_involved (bool), notes."
    ),
    "coverage": (
        "Given the policy clause and the loss scenario, decide coverage. Return JSON with keys: "
        "decision (covered|not_covered|needs_review), cited_clause, explanation."
    ),
    "fraud": (
        "Review the claim note and policy metadata for fraud indicators. Return JSON with keys: "
        "risk (low|medium|high), red_flags (list of strings), justification."
    ),
}


# Llama 3.x chat format. The tokenizer adds <|begin_of_text|> itself.
def build_prompt(task: str, text: str) -> str:
    return (
        f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n### Task\n{TASK_INSTRUCTIONS[task]}"
        f"\n\n### Input\n{text}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def build_example(task: str, text: str, answer: str) -> str:
    return build_prompt(task, text) + f"{answer}<|eot_id|>"

    return f"<s>[INST] {SYSTEM}\n\n### Task\n{TASK_INSTRUCTIONS[task]}\n\n### Input\n{text} [/INST]"


def build_example(task: str, text: str, answer: str) -> str:
    return build_prompt(task, text) + f" {answer}</s>"
